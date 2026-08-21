#!/usr/bin/env python3
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

HEAD_SIZE = 64
DTYPE = torch.bfloat16
THIS_DIR = Path(__file__).resolve().parent
CUDA_DIR = THIS_DIR / "cuda"

WKV_MODE = "fp32io16"
EMB_DEVICE = "gpu"
RKV_MODE = "auto"
CMIX_SPARSE = "auto"
CMIX_B1T1_SPARSE = "b1t1_sparse"
CMIX_ROWS2_SPARSE = "rows2_sparse"
CMIX_B1T1_NOFC = "b1t1_nofc"
CMIX_ROWS2_NOFC = "rows2_nofc"
CMIX_DENSE = "dense"


def log(message: str) -> None:
    print(f"[rwkv7_fast_v3] {message}", flush=True)

def cuda_mem() -> str:
    if not torch.cuda.is_available():
        return "cuda=unavailable"
    free, total = torch.cuda.mem_get_info()
    used = total - free
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    return f"gpu_mem used={used/2**30:.2f}GiB allocated={allocated/2**30:.2f}GiB reserved={reserved/2**30:.2f}GiB total={total/2**30:.2f}GiB"

@dataclass(frozen=True)
class PathConfig:
    rows: int
    use_batched_rkv: bool
    cmix_mode: str

def select_path(B: int, T: int) -> PathConfig:
    """All B/T dependent fast-path choices live here."""
    rows = B*T
    if CMIX_SPARSE == "off":
        cmix_mode = CMIX_DENSE
    elif CMIX_SPARSE == "no-fc":
        cmix_mode = CMIX_B1T1_NOFC if rows == 1 else (CMIX_ROWS2_NOFC if rows == 2 else CMIX_DENSE)
    elif rows == 1:
        cmix_mode = CMIX_B1T1_SPARSE
    elif rows == 2:
        cmix_mode = CMIX_ROWS2_SPARSE
    else:
        cmix_mode = CMIX_DENSE
    if RKV_MODE == "auto":
        use_batched_rkv = 4 <= rows <= 64
    elif RKV_MODE == "on":
        use_batched_rkv = True
    else:
        use_batched_rkv = False
    return PathConfig(rows=rows, use_batched_rkv=use_batched_rkv, cmix_mode=cmix_mode)

def load_extensions(wkv_mode: str = "fp16") -> None:
    t0 = time.perf_counter()
    log(f"loading CUDA extensions fast_ops + wkv={wkv_mode}")
    cuda_flags = ["-O3", "--use_fast_math", "--extra-device-vectorization"] + ([] if os.name == "nt" else ["-Xptxas", "-O3"])
    load(name="rwkv7_fast_ops_bf16", sources=[str(CUDA_DIR / "rwkv7_fast_ops_bf16.cpp"), str(CUDA_DIR / "rwkv7_fast_ops_bf16.cu")], is_python_module=False, verbose=False, extra_cflags=["-O3"], extra_cuda_cflags=cuda_flags)
    if wkv_mode == "bf16":
        load(name="rwkv7_wkv_bf16_v2", sources=[str(CUDA_DIR / "rwkv7_wkv_bf16_v2.cpp"), str(CUDA_DIR / "rwkv7_wkv_bf16_v2.cu")], is_python_module=False, verbose=False, extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "-res-usage", "--extra-device-vectorization", "-Xptxas", "-O3"])
    elif wkv_mode == "fp32io16":
        load(name="rwkv7_wkv_fp32_v2", sources=[str(CUDA_DIR / "rwkv7_wkv_fp32_v2.cpp"), str(CUDA_DIR / "rwkv7_wkv_fp32_v2.cu")], is_python_module=False, verbose=False, extra_cflags=["-O3", "-D_IO_FP16_"], extra_cuda_cflags=["-O3", "--use_fast_math", "-Xptxas", "-O3", "-D_IO_FP16_"])
    else:
        raise ValueError(f"unknown wkv_mode: {wkv_mode}")
    log(f"CUDA extensions loaded in {time.perf_counter() - t0:.3f}s")
load_extensions(WKV_MODE)

class RWKV7:
    def __init__(self, z) -> None:
        torch.set_float32_matmul_precision("highest")
        torch._C._jit_set_autocast_mode(False)

        self.H, self.N = z["blocks.0.att.r_k"].shape
        self.C, self.V = self.H * self.N, z["emb.weight"].shape[0]
        assert self.N == HEAD_SIZE
        log(f"detected model C={self.C} H={self.H} N={self.N} V={self.V}")

        emb_cpu = z["emb.weight"].squeeze() if EMB_DEVICE == "cpu" else None
        max_layer = -1
        t0 = time.perf_counter()
        log(f"moving and preprocessing weights to CUDA emb={EMB_DEVICE}")
        for key in list(z.keys()):
            if key == "emb.weight" and emb_cpu is not None:
                continue
            value = z[key].squeeze()
            if ".ffn.key.weight" in key and CMIX_SPARSE == "auto":
                z[key + ".fc"] = value.to(device="cuda", dtype=DTYPE).contiguous()
            if (
                "key.weight" in key
                or "value.weight" in key
                or "receptance.weight" in key
                or "output.weight" in key
                or "head.weight" in key
            ):
                value = value.t()
            value = value.to(device="cuda", dtype=DTYPE).contiguous()
            if key.endswith("att.r_k"):
                value = value.flatten().contiguous()
            z[key] = value
            parts = key.split(".")
            if parts[0] == "blocks":
                max_layer = max(max_layer, int(parts[1]))

        self.L = max_layer + 1
        if emb_cpu is None:
            z["emb.weight"] = F.layer_norm(z["emb.weight"], (self.C,), weight=z["blocks.0.ln0.weight"], bias=z["blocks.0.ln0.bias"]).contiguous()
        else:
            emb = torch.empty((self.V, self.C), dtype=DTYPE, pin_memory=True)
            for start in range(0, self.V, 4096):
                end = min(start + 4096, self.V)
                chunk = emb_cpu[start:end].to(device="cuda", dtype=DTYPE)
                chunk = F.layer_norm(chunk, (self.C,), weight=z["blocks.0.ln0.weight"], bias=z["blocks.0.ln0.bias"])
                emb[start:end].copy_(chunk)
            z["emb.weight"] = emb
        if RKV_MODE != "off":
            for layer in range(self.L):
                p = f"blocks.{layer}.att."
                z[p+"rkv.weight"] = torch.stack((z[p+"receptance.weight"], z[p+"key.weight"], z[p+"value.weight"])).contiguous()
        self.z = z
        self.emb_cpu = EMB_DEVICE == "cpu"
        self.emb_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        torch.cuda.synchronize()
        log(f"model ready in {time.perf_counter() - t0:.3f}s L={self.L} C={self.C} H={self.H} N={self.N} V={self.V}")
        log(cuda_mem())

    def zero_state(self, B: int) -> list[torch.Tensor]:
        return [
            torch.zeros((self.L,2,B,self.C), dtype=DTYPE, device="cuda"),
            torch.zeros((self.L,B,self.H,self.N,self.N), dtype=torch.float32 if WKV_MODE == "fp32io16" else DTYPE, device="cuda"),
            torch.zeros((B,), dtype=torch.int32, device="cuda"),
        ]

    def forward(self, tokens: torch.Tensor, state: list[torch.Tensor]) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        B, T = tokens.shape
        path = select_path(B, T)
        x = self.embed(tokens)
        return self.forward_from_x(x, state, path)

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self.emb_cpu:
            return self.z["emb.weight"][tokens]
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        B, T = tokens.shape
        host, dev = self.emb_cache.get((B, T), (None, None))
        if host is None:
            host = torch.empty((B*T,self.C), dtype=DTYPE, pin_memory=True)
            dev = torch.empty((B,T,self.C), dtype=DTYPE, device="cuda")
            self.emb_cache[(B, T)] = (host, dev)
        flat = tokens.reshape(-1)
        if flat.device.type != "cpu":
            flat = flat.cpu()
        torch.index_select(self.z["emb.weight"], 0, flat, out=host)
        dev.copy_(host.view(B,T,self.C), non_blocking=True)
        return dev

    def forward_from_x(self, x: torch.Tensor, state: list[torch.Tensor], path: PathConfig, all_logits: bool = False) -> torch.Tensor:
        z = self.z
        B, T, _ = x.shape
        v_first = torch.empty_like(x)

        for layer in range(self.L):
            p = f"blocks.{layer}."
            xx = F.layer_norm(x, (self.C,), weight=z[p+"ln1.weight"], bias=z[p+"ln1.bias"])
            xx, v_first = self.tmix(layer, xx, state[0][layer], state[1][layer], state[2], v_first, p+"att.", path)
            x = x + xx
            xx = F.layer_norm(x, (self.C,), weight=z[p+"ln2.weight"], bias=z[p+"ln2.bias"])
            x = x + self.cmix(xx, state[0][layer], p+"ffn.", path)

        if not all_logits:
            x = x[:, -1, :]
            
        x = F.layer_norm(x, (self.C,), weight=z["ln_out.weight"], bias=z["ln_out.bias"])
        state[2].add_(T) # !!! IMPORTANT FOR WKV16 DITHERING !!!
        return x @ z["head.weight"]

    def forward_all_logits(self, tokens: torch.Tensor, state: list[torch.Tensor]) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        B, T = tokens.shape
        path = select_path(B, T)
        x = self.embed(tokens)
        return self.forward_from_x(x, state, path, all_logits=True)

    def tmix(self, layer: int, x: torch.Tensor, shift_state: torch.Tensor, wkv_state: torch.Tensor, elapsed_t: torch.Tensor, v_first: torch.Tensor, p: str, path: PathConfig) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.z
        ops = torch.ops.rwkv7_fast_ops_bf16
        B, T, _ = x.shape
        xr, xw, xk, xv, xa, xg = ops.tmix_mix6(B, T, self.C, x.contiguous(), shift_state[0], z[p+"x_r"], z[p+"x_w"], z[p+"x_k"], z[p+"x_v"], z[p+"x_a"], z[p+"x_g"])
        if path.use_batched_rkv:
            flat = torch.stack((xr.reshape(-1,self.C), xk.reshape(-1,self.C), xv.reshape(-1,self.C)))
            rkv = torch.bmm(flat, z[p+"rkv.weight"])
            r, k, v = [t.view(B,T,self.C) for t in rkv.unbind(0)]
        else:
            r = xr @ z[p+"receptance.weight"]
            k = xk @ z[p+"key.weight"]
            v = xv @ z[p+"value.weight"]

        w = ops.act_tanh((xw @ z[p+"w1"]).contiguous()) @ z[p+"w2"]
        a = (xa @ z[p+"a1"]) @ z[p+"a2"]
        g = ops.act_sigmoid((xg @ z[p+"g1"]).contiguous()) @ z[p+"g2"]
        k, neg_kk, kka = ops.tmix_kk_a_gate(B, T, self.C, self.H, k.contiguous(), z[p+"k_k"], z[p+"a0"], a.contiguous(), z[p+"k_a"])

        if layer == 0:
            v_first = v
        else:
            v12 = (xv @ z[p+"v1"]) @ z[p+"v2"]
            v = ops.tmix_vres_gate(B, T, self.C, v.contiguous(), v_first.contiguous(), z[p+"v0"], v12.contiguous())

        w_raw = ops.add_vec(self.C, w.contiguous(), z[p+"w0"])
        y = torch.empty_like(r)
        if WKV_MODE == "fp32io16":
            torch.ops.rwkv7_wkv_fp32_v2.forward(B, T, self.C, self.H, wkv_state, r.contiguous(), w_raw.contiguous(), k.contiguous(), v.contiguous(), neg_kk.contiguous(), kka.contiguous(), y)
        else:
            torch.ops.rwkv7_wkv_bf16_v2.wkv_seq(B, T, self.C, self.H, wkv_state, r.contiguous(), w_raw.contiguous(), k.contiguous(), v.contiguous(), neg_kk.contiguous(), kka.contiguous(), y, elapsed_t)
        y = ops.tmix_lnx_rkvres_xg(B, T, self.C, self.H, y.contiguous(), r.contiguous(), k.contiguous(), v.contiguous(), z[p+"r_k"], z[p+"ln_x.weight"], z[p+"ln_x.bias"], g.contiguous())
        return y @ z[p+"output.weight"], v_first

    def cmix(self, x: torch.Tensor, shift_state: torch.Tensor, p: str, path: PathConfig) -> torch.Tensor:
        z = self.z
        ops = torch.ops.rwkv7_fast_ops_bf16
        B, T, _ = x.shape

        if path.cmix_mode == CMIX_B1T1_SPARSE:
            return ops.cmix_sparse_one(self.C, z[p+"key.weight.fc"].size(0), x.contiguous(), shift_state[1], z[p+"x_k"], z[p+"key.weight.fc"], z[p+"value.weight"])
        if path.cmix_mode == CMIX_ROWS2_SPARSE:
            return ops.cmix_sparse_rows(B, T, self.C, z[p+"key.weight.fc"].size(0), x.contiguous(), shift_state[1], z[p+"x_k"], z[p+"key.weight.fc"], z[p+"value.weight"])

        mixed = ops.cmix_mix(B, T, self.C, x.contiguous(), shift_state[1], z[p+"x_k"])
        hid = mixed @ z[p+"key.weight"]
        if path.cmix_mode == CMIX_B1T1_NOFC:
            return ops.cmix_sparse_down_relu_one(self.C, z[p+"value.weight"].size(0), hid.view(-1).contiguous(), z[p+"value.weight"])
        if path.cmix_mode == CMIX_ROWS2_NOFC:
            return ops.cmix_sparse_down_relu_rows(B, T, self.C, z[p+"value.weight"].size(0), hid.contiguous(), z[p+"value.weight"])

        k = ops.relu_square(hid.contiguous())
        return k @ z[p+"value.weight"]
