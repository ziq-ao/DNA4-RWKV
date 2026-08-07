import os
import math
import copy
import time
import datetime
import sys
import numpy as np

import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.cpp_extension import load

try:
    from deepspeed.ops.adam import FusedAdam
except Exception as e:
    print(f"[DNA-4] DeepSpeed FusedAdam unavailable ({e}); falling back to torch.optim.AdamW")
    from torch.optim import AdamW as FusedAdam

from src.inference.batch_inf import RWKV7

def enforce_determinism(seed=1024):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

def configure_optimizers(model, args):
    lr_decay = set()
    lr_1x = set()
    lr_2x = set()
    
    for n, p in model.named_parameters():
        if ("att.w0" in n):
            lr_2x.add(n)
        elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0) and (".weight" in n):
            lr_decay.add(n)
        else:
            lr_1x.add(n)

    lr_decay = sorted(list(lr_decay))
    lr_1x = sorted(list(lr_1x))
    lr_2x = sorted(list(lr_2x))

    param_dict = {n: p for n, p in model.named_parameters()}
    
    optim_groups = [
        {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
        {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
    ]

    if args.weight_decay > 0:
        optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": args.weight_decay, "my_lr_scale": 1.0}]
        return FusedAdam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)
    else:
        return FusedAdam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps, bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)

def run_engine(model, args, mode, input_file, output_file, total_tokens=None):
    enforce_determinism(1024)
    
    is_compress = (mode == "compress")
    
    logger_dir = os.path.join(args.log_dir, "batch_inf")
    os.makedirs(logger_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    logger_path = os.path.join(logger_dir, f"info_dna4_{mode}_{timestamp}.txt")

    if is_compress:
        file_size_bytes = os.path.getsize(input_file)
        total_tokens = file_size_bytes // 2
        tokens_np = np.memmap(input_file, dtype='>u2', mode='r', shape=(total_tokens,)) - 1
        tokens = torch.tensor(tokens_np, dtype=torch.int16).to(args.device)
    else:
        if total_tokens is None:
            raise ValueError("total_tokens must be provided for decompression")
        tokens = torch.zeros(total_tokens, dtype=torch.int16, device=args.device)

    pad_len = (args.batch_size_inf - (total_tokens % args.batch_size_inf)) % args.batch_size_inf

    if pad_len > 0:
        padded_tokens = torch.cat([tokens, torch.zeros(pad_len, dtype=tokens.dtype, device=args.device)])
        mask_1d = torch.cat([
            torch.ones(total_tokens, dtype=torch.bool, device=args.device),
            torch.zeros(pad_len, dtype=torch.bool, device=args.device)
        ])
    else:
        padded_tokens = tokens
        mask_1d = torch.ones(total_tokens, dtype=torch.bool, device=args.device)

    block_stride = len(padded_tokens) // args.batch_size_inf
    blocks = padded_tokens.view(args.batch_size_inf, block_stride).to(args.device)
    mask_blocks = mask_1d.view(args.batch_size_inf, -1).to(args.device)
    
    if not is_compress:
        decoded_blocks = torch.zeros_like(blocks)

    num_chunks = (block_stride + args.chunk_size - 1) // args.chunk_size
    optimizer = configure_optimizers(model, args)
    T_max = int(0.05 * num_chunks * (args.batch_size_inf // args.train_bs)) 

    ac_module = load(
        name="arithmetic_coder_jit",
        sources=["src/AC/ac_wrapper_torch.cpp", "src/AC/ArithmeticCoder.cpp", "src/AC/BitIoStream.cpp"],
        extra_include_paths=["src/AC"], 
        extra_cflags=["-O3", "-march=native", "-std=c++17", "-w"], 
        verbose=False
    )
    
    cumul_cpu_buffer = torch.empty([args.batch_size_inf, args.vocab_size+1], dtype=torch.int32, pin_memory=True)

    if is_compress:
        codec = ac_module.StreamingEncoder(output_file)
    else:
        codec = ac_module.StreamingDecoder(input_file)

    model.train()
    global_training_step = 0
    global_inf_step = 0
    sum_loss = 0.0
    num_processed_tokens = 0
    print_interval = 100

    base_cumul = torch.arange(args.vocab_size + 1, dtype=torch.int32)
    uniform_cumul = base_cumul.unsqueeze(0).expand(args.batch_size_inf, -1).contiguous()

    if is_compress:
        target_token = blocks[:, 0].to(torch.int32).cpu().contiguous()
        codec.encode_batch(uniform_cumul, target_token)
    else:
        decoded_tokens = codec.decode_batch(uniform_cumul).to(torch.int16).to(args.device)
        decoded_blocks[:, 0] = decoded_tokens
        blocks[:, 0] = decoded_tokens

    sum_loss += torch.log(args.vocab_size * torch.ones(args.batch_size_inf)).sum().item()
    num_processed_tokens += args.batch_size_inf
    global_inf_step += 1

    for i in range(0, block_stride, args.chunk_size):
        window_end = min(i + args.chunk_size + 1, block_stride)
        window_start = max(0, window_end - args.ctx_len - 1)
        current_eval_len = window_end - i
        window_len = window_end - window_start

        current_window = blocks[:, window_start : window_end]
        current_mask = mask_blocks[:, window_start : window_end]
        
        weights = copy.deepcopy(model.state_dict())
        model_inf = RWKV7(weights)
        state = model_inf.zero_state(args.batch_size_inf)
        
        for t in range(window_len - 1):
            infer_input = current_window[:, t:t+1].long()
            target_mask = current_mask[:, t+1:t+2]

            with torch.no_grad():
                output = model_inf.forward(infer_input, state)

            if t >= window_len - current_eval_len:
                prob = F.softmax(output.detach(), -1)
                cumul = torch.round(prob * 10000000).to(torch.int32)
                cumul = torch.max(cumul, cumul.new_ones(cumul.size()))
                cumul = F.pad(cumul, (1, 0), value=0)
                cumul = torch.cumsum(cumul, -1)
                _ = cumul_cpu_buffer.copy_(cumul, non_blocking=True)

                valid_len = target_mask.sum().item()
                torch.cuda.current_stream().synchronize()
                valid_cumuls = cumul_cpu_buffer[:valid_len]

                if is_compress:
                    target_token = current_window[:, t+1:t+2].long()
                    valid_tokens = target_token.squeeze(1)[:valid_len].to(torch.int32).cpu().contiguous()
                    codec.encode_batch(valid_cumuls, valid_tokens)
                    
                    step_loss = F.cross_entropy(output.reshape(-1, output.size(-1)), target_token.reshape(-1), reduction='none')
                    masked_loss = step_loss.view_as(target_token) * target_mask
                    sum_loss += masked_loss.sum().item()
                else:
                    decoded_tokens = codec.decode_batch(valid_cumuls)
                    global_col = window_start + t + 1
                    
                    decoded_blocks[:valid_len, global_col] = decoded_tokens.to(torch.int16).to(args.device)
                    blocks[:valid_len, global_col] = decoded_blocks[:valid_len, global_col]

                num_processed_tokens += valid_len
                global_inf_step += 1

                if global_inf_step % print_interval == 0 and is_compress:
                    mean_loss_overall = sum_loss / num_processed_tokens
                    bpc = mean_loss_overall / math.log(2)
                    est_current_mb = (bpc * num_processed_tokens) / 8 / (1024 * 1024)
                    est_total_mb = (bpc * total_tokens) / 8 / (1024 * 1024)
                    
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(logger_path, "a") as f:
                        f.write(
                            f"[{current_time}] "
                            f"[block {i // args.chunk_size}] inf_step {global_inf_step:5d}, "
                            f"lr: {optimizer.param_groups[0]['lr']:.6f}, "
                            f"BPC: {bpc:.4f}, "
                            f"EstCur: {est_current_mb:.2f}MB, EstTot: {est_total_mb:.2f}MB\n"
                        )

        model.train()
        if current_window.size(1) > 1: 
            permuted_indices = torch.randperm(args.batch_size_inf, device=args.device)
            shuffled_window = current_window[permuted_indices, :]

            for b in range(0, args.batch_size_inf, args.train_bs):
                train_seq = shuffled_window[b : b + args.train_bs, :]
                train_input = train_seq[:, :-1].long()
                train_target = train_seq[:, 1:].long()
                
                train_output = model(train_input) 
                loss = F.cross_entropy(train_output.reshape(-1, train_output.size(-1)), train_target.reshape(-1), reduction='mean')
                
                if global_training_step <= T_max:
                    progress = global_training_step / T_max
                    current_base_lr = args.lr_final + 0.5 * (args.lr_init - args.lr_final) * (1.0 + math.cos(math.pi * progress))
                    for param_group in optimizer.param_groups:
                        scale = param_group.get("my_lr_scale", 1.0) 
                        param_group['lr'] = current_base_lr * scale
                        
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()
                global_training_step += 1

    if is_compress:
        codec.close()
    else:
        flat_decoded = decoded_blocks.view(-1)[:total_tokens]
        flat_decoded_np = (flat_decoded.cpu().numpy() + 1).astype('>u2')
        flat_decoded_np.tofile(output_file)
