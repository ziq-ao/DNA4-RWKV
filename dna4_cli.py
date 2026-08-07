import os
import argparse
import shutil
import subprocess
from typing import Optional
import torch
import torch.nn.functional as F

# 导入底层模块
from src.model_codec import compress_and_save_model, load_and_decompress_to_state_dict

# 导入我们的核心模块
try:
    from src.nf4_rotation_optimizer import process_and_optimize
except ImportError:
    def process_and_optimize(*args, **kwargs):
        print("[DNA-4] modules.nf4_rotation_optimizer not found. Using placeholder.")

try:
    from src.dna4_engine import run_engine
except ImportError:
    def run_engine(*args, **kwargs):
        print("[DNA-4] modules.dna4_engine not found. Using placeholder.")

# ==============================================================================
# Utility: Verification Function
# ==============================================================================
def file_ok(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0

def describe_reuse(path: str) -> str:
    size = os.path.getsize(path)
    if size >= 1024 * 1024:
        return f"{path} ({size / 1024 / 1024:.2f} MiB)"
    if size >= 1024:
        return f"{path} ({size / 1024:.2f} KiB)"
    return f"{path} ({size} bytes)"

def resolve_preprocess_binary(preprocess_bin: Optional[str]) -> str:
    candidates = []
    if preprocess_bin:
        candidates.append(preprocess_bin)
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "preprocess"))
    path_binary = shutil.which("preprocess")
    if path_binary:
        candidates.append(path_binary)

    for candidate in candidates:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    raise FileNotFoundError(
        "Cannot find an executable preprocess binary. Compile it with "
        "`gcc -O3 -Wall -DCONFIG_STANDALONE preprocess.c -o preprocess -lm`, "
        "or pass --preprocess_bin."
    )

def default_raw_output_path(token_output_path: str) -> str:
    root, extension = os.path.splitext(token_output_path)
    return root if extension == ".bin" else f"{token_output_path}.raw"

def verify_restored_state_dict(original_dict: dict, restored_dict: dict, verbose: bool = True) -> bool:
    print(f"\n{'='*20} Starting Model Consistency Verification {'='*20}")
    
    orig_keys = set(original_dict.keys())
    rest_keys = set(restored_dict.keys())
    expected_missing = {k for k in orig_keys if k.endswith('.0.att.v1') or k.endswith('.0.att.v2')}
    
    actual_missing = orig_keys - rest_keys
    unexpected_extra = rest_keys - orig_keys
    
    passed_structure = True
    if unexpected_extra:
        print(f"❌ Structure Error: Unknown tensors in restored_dict: {unexpected_extra}")
        passed_structure = False
    if actual_missing != expected_missing:
        unaccounted_missing = actual_missing - expected_missing
        if unaccounted_missing:
            print(f"❌ Structure Error: Critical tensors missing in restored_dict: {unaccounted_missing}")
            passed_structure = False
            
    if passed_structure:
        print(f"✅ Structure Verification Passed! Stripped {len(expected_missing)} dead weights.")
    else:
        print("🚨 Structure Verification Failed. Aborting!")
        return False

    total_layers = 0
    total_mse = 0.0
    total_cos_sim = 0.0
    exact_match_count = 0
    quantized_count = 0

    for name in rest_keys:
        orig_t = original_dict[name].to(torch.float32).cuda()
        rest_t = restored_dict[name].to(torch.float32).cuda()
        
        if orig_t.shape != rest_t.shape:
            print(f"❌ Shape Error: [{name}] Original {orig_t.shape} -> Restored {rest_t.shape}")
            return False
            
        if torch.equal(orig_t, rest_t):
            exact_match_count += 1
            if verbose:
                print(f"💎 [Lossless] {name:<40} | Exact Match")
            cos_sim = 1.0
        else:
            quantized_count += 1
            mse = F.mse_loss(orig_t, rest_t).item()
            cos_sim = F.cosine_similarity(orig_t.flatten().unsqueeze(0), rest_t.flatten().unsqueeze(0)).item()
            total_mse += mse
            if verbose:
                print(f"📉 [NF4 Quantized] {name:<40} | MSE: {mse:.4e} | CosSim: {cos_sim:.6f}")

        total_cos_sim += cos_sim
        total_layers += 1
        del orig_t, rest_t

    avg_mse = total_mse / max(quantized_count, 1)
    avg_cos_sim = total_cos_sim / total_layers
    
    print(f"\n{'='*20} Global Verification Report {'='*20}")
    print(f"📊 Total Layers Checked: {total_layers} (Lossless: {exact_match_count}, NF4: {quantized_count})")
    print(f"🎯 Average MSE (Quantized Layers): {avg_mse:.4e}")
    print(f"📐 Global Average Cosine Similarity: {avg_cos_sim:.6f}")
    
    return True

# ==============================================================================
# Pipeline: Compression
# ==============================================================================
def run_compression(args):
    print(f"\n[DNA-4] === Starting Compression Pipeline ===")
    os.makedirs(args.archive_dir, exist_ok=True)
    
    # ---------------------------------------------------------
    # Step 1: NNCP Preprocessing / Reuse
    # ---------------------------------------------------------
    print(f"\n[Step 1] NNCP Preprocessing / Reuse...")
    nncp_dict_path = os.path.join(args.archive_dir, "nncp.dict")
    if file_ok(nncp_dict_path):
        print(f"  -> Reusing existing NNCP dictionary: {describe_reuse(nncp_dict_path)}")
    elif args.input_is_preprocessed:
        print("  -> Input is marked as preprocessed; no NNCP dictionary was found in archive.")
        print("  -> Compression can continue, but decompression to the original raw file will need the matching dictionary.")
    else:
        raise FileNotFoundError(
            "NNCP dictionary is missing and this CLI was given only a token .bin input. "
            f"Expected: {nncp_dict_path}. Run ./preprocess first, or pass --input_is_preprocessed "
            "if you intentionally want to compress tokens without bundling a dictionary."
        )

    # ---------------------------------------------------------
    # Step 2: Model Quantization & Entropy Coding
    # ---------------------------------------------------------
    print(f"\n[Step 2] Model Quantization & Entropy Coding...")
    seeds_path = os.path.join(args.archive_dir, "rotation_seeds.pt")
    model_ac_prefix = os.path.join(args.archive_dir, "dna4_compressed_model")
    model_ac_path = f"{model_ac_prefix}.ac"
    model_meta_path = f"{model_ac_prefix}.meta"
    
    if file_ok(seeds_path) and not args.force_rotation:
        print(f"  -> Reusing existing NF4 rotation seeds: {describe_reuse(seeds_path)}")
    else:
        if args.force_rotation and file_ok(seeds_path):
            print("  -> --force_rotation set; regenerating NF4 rotation seeds...")
        else:
            print("  -> NF4 rotation seeds not found; finding optimal rotation seeds...")
        process_and_optimize(
            model_path=args.model_path,
            save_config_path=seeds_path,
            num_trials=args.seed_trials,
            device=args.device
        )
    
    state_dict = torch.load(args.model_path, map_location="cpu", mmap=True)
    seeds = torch.load(seeds_path, map_location="cpu")
    
    model_codec_ready = file_ok(model_ac_path) and file_ok(model_meta_path)
    if model_codec_ready and not args.force_model_codec:
        print(f"  -> Reusing compressed model weights: {describe_reuse(model_ac_path)}, {describe_reuse(model_meta_path)}")
    else:
        if args.force_model_codec and model_codec_ready:
            print("  -> --force_model_codec set; recompressing model weights...")
        else:
            print("  -> Compressed model weights not found; compressing and saving model weights...")
        compress_and_save_model(state_dict, model_ac_prefix, seeds)
        print(f"  -> Model compressed and saved to {model_ac_prefix}.*")

    # ---------------------------------------------------------
    # Step 3: DNA-4 Core Processing
    # ---------------------------------------------------------
    print(f"\n[Step 3] DNA-4 Core Adaptive Compression...")
    print("  -> Loading quantized model for absolute symmetric compression...")
    # 【核心！】使用解压方法读回权重，确保压缩时的模型带有量化误差，保证端到端一致！
    restored_state_dict = load_and_decompress_to_state_dict(model_ac_prefix)
    
    model = RWKV(args)
    model.load_state_dict(restored_state_dict, strict=False)
    model = model.to(args.device).to(torch.bfloat16)
    
    dna4_stream_path = os.path.join(args.archive_dir, "dna4_stream.ac")
    
    run_engine(
        model=model,
        args=args,
        mode="compress",
        input_file=args.input_file, # 这里的 input_file 应该是预处理后的 .bin
        output_file=dna4_stream_path
    )

    print(f"\n[DNA-4] === Compression Complete! Archive saved in {args.archive_dir} ===")

# ==============================================================================
# Pipeline: Decompression
# ==============================================================================
def run_decompression(args):
    print(f"\n[DNA-4] === Starting Decompression Pipeline ===")
    
    if not os.path.exists(args.archive_dir):
        raise FileNotFoundError(f"Archive directory not found: {args.archive_dir}")

    # ---------------------------------------------------------
    # Step 1: Model Decoding & Verification
    # ---------------------------------------------------------
    print(f"\n[Step 1] Model Decoding...")
    model_ac_prefix = os.path.join(args.archive_dir, "dna4_compressed_model")
    
    print("  -> Decompressing model weights from AC stream...")
    restored_state_dict = load_and_decompress_to_state_dict(model_ac_prefix)
    
    if args.verify and getattr(args, 'model_path', None):
        print("  -> Running verification against original model...")
        original_state_dict = torch.load(args.model_path, map_location="cpu", mmap=True)
        verify_restored_state_dict(original_state_dict, restored_state_dict, verbose=args.verbose)

    print("  -> Initializing RWKV model...")
    model = RWKV(args)
    model.load_state_dict(restored_state_dict, strict=False)
    model = model.to(args.device).to(torch.bfloat16)

    # ---------------------------------------------------------
    # Step 2: DNA-4 Core Decoding
    # ---------------------------------------------------------
    print(f"\n[Step 2] DNA-4 Core Decoding...")
    if args.total_tokens is None:
        raise ValueError("You must provide --total_tokens for decompression!")
        
    dna4_stream_path = os.path.join(args.archive_dir, "dna4_stream.ac")
    token_output_dir = os.path.dirname(args.output_file)
    if token_output_dir:
        os.makedirs(token_output_dir, exist_ok=True)
    
    run_engine(
        model=model,
        args=args,
        mode="decompress",
        input_file=dna4_stream_path,   # 从压缩包里的 ac 流读取
        output_file=args.output_file,  # 解压输出 .bin 
        total_tokens=args.total_tokens
    )

    # ---------------------------------------------------------
    # Step 3: NNCP Postprocessing
    # ---------------------------------------------------------
    print(f"\n[Step 3] NNCP Postprocessing...")
    nncp_dict_path = os.path.join(args.archive_dir, "nncp.dict")
    if not file_ok(nncp_dict_path):
        raise FileNotFoundError(
            "Archive does not contain nncp.dict, so the restored token stream "
            "cannot be converted back to the original byte file."
        )

    raw_output_file = args.raw_output_file or default_raw_output_path(args.output_file)
    raw_output_dir = os.path.dirname(raw_output_file)
    if raw_output_dir:
        os.makedirs(raw_output_dir, exist_ok=True)

    preprocess_bin = resolve_preprocess_binary(args.preprocess_bin)
    print(f"  -> Restoring raw bytes with {preprocess_bin}...")
    subprocess.run(
        [preprocess_bin, "d", nncp_dict_path, args.output_file, raw_output_file],
        check=True,
    )
    
    print("\n[DNA-4] === Decompression Complete! "
          f"Token output: {args.output_file}; raw output: {raw_output_file} ===")

# ==============================================================================
# CLI Entry Point
# ==============================================================================
def add_engine_args(parser):
    """向解析器统一注册架构和引擎的超参数"""
    # 模型架构参数
    parser.add_argument("--n_layer", type=int, default=5)
    parser.add_argument("--n_embd", type=int, default=512)
    parser.add_argument("--ctx_len", type=int, default=2048)
    parser.add_argument("--vocab_size", type=int, default=16384)
    parser.add_argument("--head_size", type=int, default=64)
    parser.add_argument("--weight_tying", type=int, default=0)
    
    # 引擎训练/推断参数
    parser.add_argument("--batch_size_inf", type=int, default=4096)
    parser.add_argument("--train_bs", type=int, default=16)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--lr_init", type=float, default=2.5e-4)
    parser.add_argument("--lr_final", type=float, default=0.5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--grad_clip", type=float, default=0.05)
    parser.add_argument("--grad_cp", type=bool, default=False)
    parser.add_argument("--betas", nargs=2, type=float, default=(0.0, 0.9999))
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    
    # 系统参数
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_dir", type=str, default="logs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DNA-4 End-to-End Neural Compressor CLI")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to run")

    # ----- Compress Parser -----
    compress_parser = subparsers.add_parser("compress", help="Compress a file into a DNA-4 archive")
    compress_parser.add_argument("--input_file", type=str, required=True, help="Original file to compress (.bin)")
    compress_parser.add_argument("--model_path", type=str, required=True, help="Path to base RWKV model (.pth)")
    compress_parser.add_argument("--archive_dir", type=str, default="enwik9.dna4", help="Output directory for compressed assets")
    compress_parser.add_argument("--seed_trials", type=int, default=20, help="Number of trials for Hadamard seed search")
    compress_parser.add_argument("--input_is_preprocessed", action="store_true", help="Treat --input_file as an already-preprocessed token .bin even if archive_dir/nncp.dict is missing")
    compress_parser.add_argument("--force_rotation", action="store_true", help="Regenerate archive_dir/rotation_seeds.pt even if it already exists")
    compress_parser.add_argument("--force_model_codec", action="store_true", help="Recompress model weights even if dna4_compressed_model.ac/.meta already exist")
    add_engine_args(compress_parser)

    # ----- Decompress Parser -----
    decompress_parser = subparsers.add_parser("decompress", help="Decompress a DNA-4 archive")
    decompress_parser.add_argument("--archive_dir", type=str, default="enwik9.dna4", help="Input directory of compressed assets")
    decompress_parser.add_argument("--output_file", type=str, default="data/enwik9_restored.bin", help="Path to save the restored token .bin file")
    decompress_parser.add_argument("--raw_output_file", type=str, help="Path to save the restored raw file (default: --output_file without .bin)")
    decompress_parser.add_argument("--preprocess_bin", type=str, help="Path to the compiled preprocess executable")
    decompress_parser.add_argument("--total_tokens", type=int, default=200608961, help="Target total tokens to decode (Must match original file)")
    
    # Debug / Verification arguments
    decompress_parser.add_argument("--verify", action="store_true", help="Verify restored model against original (requires --model_path)")
    decompress_parser.add_argument("--model_path", type=str, help="Path to original base RWKV model (used only for verification)")
    decompress_parser.add_argument("--verbose", action="store_true", help="Print detailed verification logs")
    add_engine_args(decompress_parser)

    args = parser.parse_args()

    # 注入 RWKV 所需的环境变量与隐式参数
    args.my_testing = "x070"
    args.dim_att = args.n_embd
    args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)
    os.environ["RWKV_MY_TESTING"] = args.my_testing
    os.environ["RWKV_CTXLEN"] = str(args.ctx_len)
    os.environ["RWKV_HEAD_SIZE"] = str(args.head_size)
    os.environ["RWKV_FLOAT_MODE"] = "bf16"
    os.environ["RWKV_JIT_ON"] = "0"
    from src.training.model import RWKV  # 确保这个路径指向你的 RWKV 模型定义文件

    if args.command == "compress":
        run_compression(args)
    elif args.command == "decompress":
        run_decompression(args)
