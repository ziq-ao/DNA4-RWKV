import torch
import bitsandbytes.functional as bnbf
import pickle, os
from torch.utils.cpp_extension import load

def unpack_4bit_to_indices(w_4bit_packed: torch.Tensor) -> torch.Tensor:
    low_nibble = w_4bit_packed & 0x0F
    high_nibble = (w_4bit_packed >> 4) & 0x0F
    return torch.stack([low_nibble, high_nibble], dim=-1).flatten()

def repack_indices_to_4bit(unpacked_indices: torch.Tensor, packed_shape: torch.Size) -> torch.Tensor:
    pairs = unpacked_indices.view(-1, 2)
    packed = (pairs[:, 0] | (pairs[:, 1] << 4)).to(torch.uint8)
    return packed.view(packed_shape)

def extract_and_quantize_model(state_dict, blocksize=64, use_hadamard=True, rotation_seeds=None):
    non_quantized_weights = {}
    quantized_indices = {}
    quantization_states = {}
    tied_mappings = {}
    packed_shapes = {}

    if "head.weight" in state_dict and "emb.weight" in state_dict:
        head_w, emb_w = state_dict["head.weight"], state_dict["emb.weight"]
        if torch.equal(head_w, emb_w):
            tied_mappings["head.weight"] = {"target": "emb.weight", "transpose": False}
        elif head_w.shape == emb_w.t().shape and torch.equal(head_w, emb_w.t()):
            tied_mappings["head.weight"] = {"target": "emb.weight", "transpose": True}

    sensitive_suffixes = ['w0', 'a0', 'v0', 'ln']
    dead_weight_suffixes = ['.0.att.v1', '.0.att.v2']

    for name, tensor in state_dict.items():
        if any(name.endswith(suffix) for suffix in dead_weight_suffixes):
            continue

        if name in tied_mappings:
            continue

        if tensor.numel() <= 512 or any(name.endswith(suffix) for suffix in sensitive_suffixes):
            non_quantized_weights[name] = tensor.detach().cpu().clone()
            continue

        w_original = tensor.cuda()

        if use_hadamard and rotation_seeds is not None:
            seed = rotation_seeds.get(name, -1)
            dim = w_original.shape[-1]
            if seed == -1:
                H = torch.eye(dim, device='cuda', dtype=w_original.dtype)
            else:
                from src.nf4_rotation_optimizer import get_random_orthogonal_matrix
                H = get_random_orthogonal_matrix(dim, seed, device='cuda').to(w_original.dtype)
            w_trans = torch.matmul(w_original, H)
        else:
            w_trans = w_original

        w_4bit, quant_state = bnbf.quantize_4bit(w_trans, blocksize=blocksize, compress_statistics=True, quant_type='nf4')
        
        quantized_indices[name] = unpack_4bit_to_indices(w_4bit).cpu()
        quantization_states[name] = quant_state
        packed_shapes[name] = w_4bit.shape

        del w_original, w_trans, w_4bit

    torch.cuda.empty_cache()
    return non_quantized_weights, quantized_indices, quantization_states, tied_mappings, packed_shapes

def compress_and_save_model(state_dict, out_file_path: str, rotation_seeds: dict):
    non_quant_w, nf4_indices, q_states, tied_mappings, packed_shapes = extract_and_quantize_model(
        state_dict, use_hadamard=True, rotation_seeds=rotation_seeds
    )
    
    all_indices = torch.cat(list(nf4_indices.values()), dim=0).to(torch.int32).contiguous()
    counts = torch.bincount(all_indices, minlength=16)
    counts = torch.clamp_min(counts, 1) 
    
    cumul_freq = torch.zeros(17, dtype=torch.int32)
    cumul_freq[1:] = torch.cumsum(counts, dim=0)
    
    ac_module = load(
        name="arithmetic_coder_jit",
        sources=["src/AC/ac_wrapper_torch.cpp", "src/AC/ArithmeticCoder.cpp", "src/AC/BitIoStream.cpp"],
        extra_include_paths=["src/AC"], 
        extra_cflags=["-O3", "-march=native", "-std=c++17", "-w"], 
        verbose=False
    )
    
    ac_bin_path = f"{out_file_path}.ac"
    if os.path.dirname(ac_bin_path):
        os.makedirs(os.path.dirname(ac_bin_path), exist_ok=True)
        
    encoder = ac_module.StreamingEncoder(ac_bin_path)
    
    chunk_size = 10_000_000
    total_len = all_indices.size(0)
    uniform_cumul_giant = cumul_freq.unsqueeze(0).expand(chunk_size, -1).contiguous()

    for i in range(0, total_len, chunk_size):
        chunk = all_indices[i : i + chunk_size]
        current_len = chunk.size(0)

        if current_len == chunk_size:
            encoder.encode_batch(uniform_cumul_giant, chunk)
        else:
            tail_cumul = cumul_freq.unsqueeze(0).expand(current_len, -1).contiguous()
            encoder.encode_batch(tail_cumul, chunk)

    encoder.close()
    
    meta_path = f"{out_file_path}.meta"
    with open(meta_path, 'wb') as f:
        pickle.dump({
            'non_quant_w': non_quant_w,
            'q_states': q_states,
            'cumul_freq': cumul_freq,
            'layer_shapes': {k: v.shape for k, v in nf4_indices.items()},
            'packed_shapes': packed_shapes,
            'tied_mappings': tied_mappings,
            'rotation_seeds': rotation_seeds
        }, f)


def load_and_decompress_to_state_dict(model_file_prefix: str) -> dict:
    meta_path = f"{model_file_prefix}.meta"
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
        
    ac_module = load(
        name="arithmetic_coder_jit",
        sources=["src/AC/ac_wrapper_torch.cpp", "src/AC/ArithmeticCoder.cpp", "src/AC/BitIoStream.cpp"],
        extra_include_paths=["src/AC"], 
        extra_cflags=["-O3", "-march=native", "-std=c++17", "-w"], 
        verbose=False
    )
    
    ac_bin_path = f"{model_file_prefix}.ac"
    decoder = ac_module.StreamingDecoder(ac_bin_path)
    
    total_len = sum(shape[0] for shape in meta['layer_shapes'].values())
    cumul_freq = meta['cumul_freq']
    
    decoded_stream = torch.zeros(total_len, dtype=torch.int32)
    pointer = 0
    chunk_size = 10_000_000 
    
    uniform_cumul_giant = cumul_freq.unsqueeze(0).expand(chunk_size, -1).contiguous()
    
    while pointer < total_len:
        current_len = min(chunk_size, total_len - pointer)
        
        if current_len == chunk_size:
            chunk_result = decoder.decode_batch(uniform_cumul_giant)
        else:
            tail_cumul = cumul_freq.unsqueeze(0).expand(current_len, -1).contiguous()
            chunk_result = decoder.decode_batch(tail_cumul)
            
        decoded_stream[pointer : pointer + current_len] = chunk_result
        pointer += current_len

    restored_state_dict = {}
    
    for name, tensor in meta['non_quant_w'].items():
        restored_state_dict[name] = tensor.cuda()

    stream_pointer = 0
    for name, shape in meta['layer_shapes'].items():
        layer_len = shape[0]
        layer_unpacked_idx = decoded_stream[stream_pointer : stream_pointer + layer_len]
        stream_pointer += layer_len
        
        quant_state = meta['q_states'][name]
        packed_shape = meta['packed_shapes'][name]
        
        packed_4bit = repack_indices_to_4bit(layer_unpacked_idx, packed_shape)
        w_recon = bnbf.dequantize_4bit(packed_4bit.cuda(), quant_state, quant_type='nf4')
        
        seed = meta['rotation_seeds'].get(name, -1)
        if seed != -1:
            from src.nf4_rotation_optimizer import get_random_orthogonal_matrix
            H = get_random_orthogonal_matrix(w_recon.shape[-1], seed, device='cuda').to(w_recon.dtype)
            w_recon = torch.matmul(w_recon, H.t())
            
        restored_state_dict[name] = w_recon

    for name, tie_info in meta.get('tied_mappings', {}).items():
        target_tensor = restored_state_dict[tie_info["target"]]
        restored_state_dict[name] = target_tensor.t() if tie_info["transpose"] else target_tensor

    return restored_state_dict
