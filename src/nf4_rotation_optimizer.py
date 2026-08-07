import torch
import scipy.linalg
import numpy as np
import math
import os
from tqdm import tqdm

def get_random_orthogonal_matrix(dim: int, seed: int, device='cpu') -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    
    is_power_of_two = (dim & (dim - 1) == 0) and dim > 0
    
    if is_power_of_two:
        try:
            H_scipy = scipy.linalg.hadamard(dim)
            H = torch.from_numpy(H_scipy).to(device=device, dtype=torch.float32)
            scale = 1.0 / math.sqrt(dim)
            H *= scale
            signs = torch.randint(0, 2, (dim,), generator=g, device=device).float() * 2 - 1
            H = H * signs.view(1, -1)
        except Exception:
            return None
    else:
        try:
            X = torch.randn(dim, dim, generator=g, device=device, dtype=torch.float32)
            Q, R = torch.linalg.qr(X)
            d = torch.diag(R, 0)
            ph = d.sign()
            Q *= ph
            H = Q
        except RuntimeError:
            return None
    return H

def calc_kurtosis_tensor(tensor):
    if tensor.numel() == 0: return float('inf')
    t = tensor.float()
    mean = torch.mean(t)
    std = torch.std(t)
    if std < 1e-9: return float('inf')
    fourth_moment = torch.mean((t - mean)**4)
    kurtosis = fourth_moment / (std**4)
    return kurtosis.item()

def find_best_seed_for_layer(layer_weight, num_trials=20, device='cuda'):
    try:
        weight = layer_weight.to(device).float()
    except:
        weight = layer_weight.float() 
        device = 'cpu'

    dim = weight.shape[-1]
    
    best_seed = -1
    best_kurtosis = calc_kurtosis_tensor(weight)
    
    base_seed = 2026
    
    for i in range(num_trials):
        current_seed = base_seed + i
        
        H = get_random_orthogonal_matrix(dim, seed=current_seed, device=device)
        if H is None: continue
        
        w_trans = torch.matmul(weight, H)
        k = calc_kurtosis_tensor(w_trans)
        
        if k < best_kurtosis:
            best_kurtosis = k
            best_seed = current_seed
            
    return best_seed, best_kurtosis

def process_and_optimize(model_path, 
                         save_config_path="rotation_seeds.pt",
                         num_trials=20,
                         device='cuda'):
    
    if not torch.cuda.is_available() and device == 'cuda':
        print("CUDA not available, falling back to CPU (might be slow)")
        device = 'cpu'
        
    print(f"Seed config will be saved to: {save_config_path}")
    
    seed_config = {}

    print(f"Loading model: {model_path} ...")
    try:
        state_dict = torch.load(model_path, map_location='cpu')
    except Exception as e:
        print(f"Failed to load: {e}")
        return

    target_items = []
    for name, param in state_dict.items():
        if not torch.is_tensor(param): continue
        if param.numel() <= 768: continue 
        if param.dtype not in [torch.float32, torch.float16, torch.bfloat16]: continue
        target_items.append((name, param))

    print(f"Found {len(target_items)} eligible weight matrices. Starting optimization (Trials={num_trials})...")

    for name, param in tqdm(target_items):
        try:
            best_seed, k_best = find_best_seed_for_layer(
                param, num_trials=num_trials, device=device
            )
            
            if best_seed == -1 and k_best == float('inf'):
                 print(f"  [Skip] {name}: Failed to generate rotation matrix (dimension too large?)")
                 continue
                 
            seed_config[name] = best_seed
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  [Error] Failed to process {name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nSaving seed config to {save_config_path} ...")
    os.makedirs(os.path.dirname(save_config_path), exist_ok=True)
    torch.save(seed_config, save_config_path)
    print("All done!")

if __name__ == "__main__":
    MODEL_PATH = "out/L5-D512-CTXLEN2048-TIE1-NNCPDATA1-x070/rwkv-200.pth"
    
    process_and_optimize(
        MODEL_PATH, 
        save_config_path="enwik9.dna4/rotation_seeds.pt",
        num_trials=100,  
        device='cuda'
    )