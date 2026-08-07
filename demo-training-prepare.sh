#!/usr/bin/env bash
set -euo pipefail

#######################################################################################################################
#
# Generate the initial RWKV-7 prior for the enwik9 DNA-4 run.
#
# Required input:
#   data/enwik9_tokens.bin
#
# Output:
#   out/L5-D512-CTXLEN2048-TIE1-NNCPDATA1-x070/rwkv-init.pth
#
#######################################################################################################################

source /home/aoziqiao/miniconda3/etc/profile.d/conda.sh
conda activate llm

export CUDA_VISIBLE_DEVICES=0

MODEL_TYPE="x070"
DATA="enwik9"
VOCAB_SIZE=16384
N_LAYER=5
N_EMBD=512
CTX_LEN=2048
WEIGHT_TYING=1
NNCP_DATA=1
PRECISION="bf16"
HEAD_SIZE=64
MAGIC_PRIME=2926181

PROJ_DIR="out/L${N_LAYER}-D${N_EMBD}-CTXLEN${CTX_LEN}-TIE${WEIGHT_TYING}-NNCPDATA${NNCP_DATA}-${MODEL_TYPE}"
DATA_FILE="data/${DATA}_tokens.bin"

echo "Using Data File: ${DATA_FILE}"
echo "Writing init prior to: ${PROJ_DIR}/rwkv-init.pth"

python train.py --wandb "" --proj_dir "${PROJ_DIR}" \
  --data_file "${DATA_FILE}" --data_type "binidx" --vocab_size "${VOCAB_SIZE}" --my_testing "${MODEL_TYPE}" \
  --ctx_len "${CTX_LEN}" --train_stage 1 --epoch_count 1 --epoch_begin 0 \
  --epoch_save 1 --weight_decay 0 --head_size "${HEAD_SIZE}" --weight_tying "${WEIGHT_TYING}" --nncp_data "${NNCP_DATA}" \
  --num_nodes 1 --micro_bsz 1 --n_layer "${N_LAYER}" --n_embd "${N_EMBD}" --my_exit_tokens 1498226207 --magic_prime "${MAGIC_PRIME}" \
  --lr_init 1e-5 --lr_final 1e-5 --warmup_steps 10 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
  --accelerator cpu --devices 1 --precision "${PRECISION}" --strategy deepspeed_stage_2 --grad_cp 1
