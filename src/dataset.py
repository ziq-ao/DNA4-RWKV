########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json, math, random, os, sys
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info
from .binidx import MMapIndexedDataset

def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

class MyDataset(Dataset):
    def __init__(self, args):
        self.args = args

        self.vocab_size = args.vocab_size
        rank_zero_info(f"Current vocab size = {self.vocab_size} (make sure it's correct)")

        self.data = MMapIndexedDataset(args.data_file)
        self.data_size = len(self.data._bin_buffer) // self.data._index._dtype_size
        rank_zero_info(f"Data has {self.data_size} tokens.")

        self.samples_per_epoch = args.epoch_steps * args.real_bsz
        assert self.samples_per_epoch == 40320
        rank_zero_info(f"########## train stage {args.train_stage} ##########")
        dataset_slot = self.data_size // args.ctx_len

        assert is_prime(args.magic_prime)
        assert args.magic_prime % 3 == 2
        assert args.magic_prime / dataset_slot > 0.9 and args.magic_prime / dataset_slot <= 1

    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx):
        args = self.args
        rank = self.global_rank
        epoch = self.real_epoch
        world_size = self.world_size
        # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size}")

        ctx_len = args.ctx_len
        req_len = ctx_len + 1
        magic_prime = args.magic_prime

        ii = 1 + epoch * self.samples_per_epoch + (idx * world_size) + rank

        factor = (math.sqrt(5) - 1) / 2
        factor = int(magic_prime * factor)
        i = ((factor * ii * ii * ii) % magic_prime) * ctx_len
        # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size} ii {ii} pos {round(i / self.data_size, 3)}")

        dix = self.data.get(idx=0, offset=i, length=req_len).astype(int)

        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)

        return x, y


class MyDataset_NNCP(Dataset):
    def __init__(self, args):
        self.args = args
        bin_path = args.data_file
        self.ctx_len = args.ctx_len
        self.vocab_size = args.vocab_size
        
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"找不到文件: {bin_path}")
            
        file_size_bytes = os.path.getsize(bin_path)

        self.total_tokens = file_size_bytes // 2
        rank_zero_info(f"[Info] 文件大小: {file_size_bytes / 1024 / 1024:.2f} MB")
        rank_zero_info(f"[Info] Token 总数: {self.total_tokens}")

        self.data = np.memmap(bin_path, dtype='>u2', mode='r', shape=(self.total_tokens,)) - 1
        self.data_size = len(self.data)
        rank_zero_info(f"Data has {self.data_size} tokens.")
        
        # 简单的校验：确保没有读出乱码
        max_id = np.max(self.data[:10000]) # 只检查前1万个，节省时间
        if max_id >= args.vocab_size:
            rank_zero_info(f"[Warning] 发现 Token ID {max_id} 超过了词表大小 {args.vocab_size}！")
            
    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx):
        req_len = self.ctx_len + 1
        i = np.random.randint(0, self.data_size - req_len)
        
        # 从 memmap 中读取一段，numpy 会自动处理字节序转换
        chunk = self.data[i : i + self.ctx_len + 1]
        
        # 6. 转换为 PyTorch Tensor
        # Embedding 层通常需要 Int64 (Long) 类型
        chunk_tensor = torch.from_numpy(chunk.astype(np.int64))
        
        x = chunk_tensor[:-1]
        y = chunk_tensor[1:]
        
        return x, y
