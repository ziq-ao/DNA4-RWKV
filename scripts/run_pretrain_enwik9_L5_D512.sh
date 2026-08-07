#!/usr/bin/env bash
set -euo pipefail

source /home/aoziqiao/miniconda3/etc/profile.d/conda.sh
conda activate llm

cd /home/aoziqiao/azq/DNA_RWKV_release_v1

export CUDA_VISIBLE_DEVICES=0

python train.py \
  --load_model out/L5-D512-CTXLEN2048-TIE1-NNCPDATA1-x070/rwkv-init.pth \
  --wandb "" \
  --proj_dir out/L5-D512-CTXLEN2048-TIE1-NNCPDATA1-x070 \
  --random_seed 1024 \
  --data_file data/enwik9_tokens.bin \
  --data_type binidx \
  --vocab_size 16384 \
  --ctx_len 2048 \
  --epoch_steps 1680 \
  --epoch_count 72 \
  --epoch_begin 0 \
  --epoch_save 20 \
  --micro_bsz 24 \
  --n_layer 5 \
  --n_embd 512 \
  --weight_tying 1 \
  --dim_att 512 \
  --dim_ffn 1792 \
  --nncp_data 1 \
  --lr_init 0.001 \
  --lr_final 0.0001 \
  --warmup_steps 10 \
  --beta1 0.9 \
  --beta2 0.99 \
  --adam_eps 1e-18 \
  --grad_cp 1 \
  --weight_decay 0.001 \
  --grad_clip 1.0 \
  --train_stage 3 \
  --ds_bucket_mb 2 \
  --head_size 64 \
  --load_partial 0 \
  --magic_prime 2926181 \
  --my_testing x070 \
  --my_exit_tokens 20480000000 \
  --devices 1 \
  --enable_progress_bar True \
  --accelerator gpu \
  --strategy deepspeed_stage_2 \
  --precision bf16 \
  --num_sanity_val_steps 0
