#!/usr/bin/env python3
"""Cross-machine one-step probe for the DNA RWKV training path.

Run the identical command on each machine, then compare the emitted JSON files.
The script deliberately stops before an optimizer update: it isolates data loading,
forward computation, and backward computation without creating a checkpoint.
"""

import argparse
import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("RWKV_JIT_ON", "1")
os.environ.setdefault("RWKV_MY_TESTING", "x070")
os.environ.setdefault("RWKV_HEAD_SIZE", "64")

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("out/L5-D512-CTXLEN2048-TIE1-NNCPDATA1-x070-PREP20260817-RUN1024/rwkv-init.pth"),
    )
    parser.add_argument("--data-file", type=Path, default=Path("data/enwik9_tokens.bin"))
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def seed_worker(worker_id, base_seed):
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def update_digest_with_tensor(digest, name, tensor):
    digest.update(name.encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(str(tensor.dtype).encode("ascii"))
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
    digest.update(raw.tobytes())


def tensor_hash(name, tensor):
    digest = hashlib.sha256()
    update_digest_with_tensor(digest, name, tensor)
    return digest.hexdigest()


def collection_hash(named_tensors):
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        update_digest_with_tensor(digest, name, tensor)
    return digest.hexdigest()


def tensor_summary(tensor):
    values = tensor.detach().float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": tensor_hash("tensor", tensor),
        "mean": float(values.mean().item()),
        "std": float(values.std().item()),
        "abs_max": float(values.abs().max().item()),
    }


def model_args():
    return SimpleNamespace(
        vocab_size=16384,
        ctx_len=2048,
        n_layer=5,
        n_embd=512,
        dim_att=512,
        dim_ffn=1792,
        head_size=64,
        weight_tying=1,
        my_testing="x070",
        accelerator="gpu",
        grad_cp=0,
        weight_decay=0.001,
        lr_init=0.001,
        betas=(0.9, 0.99),
        adam_eps=1e-18,
    )


def cuda_driver_version():
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return None


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This probe requires one visible CUDA device.")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.data_file.is_file():
        raise FileNotFoundError(args.data_file)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    from src.dataset import MyDataset_NNCP
    from src.training.model import RWKV

    dataset_args = model_args()
    dataset_args.data_file = str(args.data_file)
    dataset_args.epoch_steps = 1680
    dataset_args.micro_bsz = args.batch_size
    dataset_args.real_bsz = args.batch_size
    dataset = MyDataset_NNCP(dataset_args)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=partial(seed_worker, base_seed=args.seed),
        generator=generator,
        drop_last=True,
    )
    inputs, targets = next(iter(loader))

    model = RWKV(model_args())
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(device="cuda", dtype=torch.bfloat16).train()
    inputs = inputs.cuda(non_blocking=True)
    targets = targets.cuda(non_blocking=True)

    logits = model(inputs)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    loss.backward()
    torch.cuda.synchronize()

    gradients = [(name, parameter.grad) for name, parameter in model.named_parameters() if parameter.grad is not None]
    grad_norm_squared = sum(gradient.detach().float().square().sum() for _, gradient in gradients)
    sample_positions = ((0, 0, 0), (0, 17, 31), (0, 1024, 1024), (args.batch_size - 1, 2047, 16383))
    report = {
        "probe": "dna-rwkv-forward-backward-v1",
        "host": socket.gethostname(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "cuda_driver": cuda_driver_version(),
        "environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG", "TORCH_CUDA_ARCH_LIST")
        },
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.file_digest(args.checkpoint.open("rb"), "sha256").hexdigest(),
        "data_file": str(args.data_file),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "inputs": tensor_summary(inputs),
        "targets": tensor_summary(targets),
        "logits": tensor_summary(logits),
        "cross_entropy": float(loss.detach().float().item()),
        "logit_samples": [float(logits[index].float().item()) for index in sample_positions],
        "gradients_sha256": collection_hash(gradients),
        "gradient_l2_norm": float(grad_norm_squared.sqrt().item()),
        "gradient_tensor_count": len(gradients),
    }
    output = args.output or Path("tmp") / f"compare_one_step_{socket.gethostname()}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
