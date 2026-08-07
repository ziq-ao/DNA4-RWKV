#!/usr/bin/env bash
set -euo pipefail

#######################################################################################################################
#
# Train the enwik9 L5-D512 RWKV-7 prior for DNA-4.
#
# Run demo-training-prepare.sh first to create:
#   out/L5-D512-CTXLEN2048-TIE1-NNCPDATA1-x070/rwkv-init.pth
#
# Note: train.py with train_stage >= 2 automatically loads the latest rwkv-*.pth in PROJ_DIR.
# If you want to start from rwkv-init.pth, keep old rwkv-N.pth checkpoints outside PROJ_DIR
# or move them into a subdirectory such as _previous_checkpoints/.
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
INIT_MODEL="${PROJ_DIR}/rwkv-init.pth"

M_BSZ=24
LR_INIT=0.001
LR_FINAL=0.0001
GRAD_CP=1
EPOCH_SAVE=20
N_NODE=1
GPU_PER_NODE=1
DS_BUCKET_MB=2
MY_EXIT_TOKENS=20480000000
RANDOM_SEED=1024

echo "Using Data File: ${DATA_FILE}"
echo "Using Project Dir: ${PROJ_DIR}"
echo "Requested init model: ${INIT_MODEL}"

python train.py --load_model "${INIT_MODEL}" --wandb "" --proj_dir "${PROJ_DIR}" --my_testing "${MODEL_TYPE}" \
  --random_seed "${RANDOM_SEED}" \
  --ctx_len "${CTX_LEN}" --train_stage 3 --epoch_count 72 --epoch_begin 0 \
  --data_file "${DATA_FILE}" --my_exit_tokens "${MY_EXIT_TOKENS}" --magic_prime "${MAGIC_PRIME}" \
  --num_nodes "${N_NODE}" --micro_bsz "${M_BSZ}" --n_layer "${N_LAYER}" --n_embd "${N_EMBD}" \
  --dim_att "${N_EMBD}" --dim_ffn 1792 \
  --lr_init "${LR_INIT}" --lr_final "${LR_FINAL}" --warmup_steps 10 --beta1 0.9 --beta2 0.99 --adam_eps 1e-18 \
  --data_type "binidx" --vocab_size "${VOCAB_SIZE}" \
  --weight_decay 0.001 --grad_clip 1.0 --epoch_save "${EPOCH_SAVE}" --head_size "${HEAD_SIZE}" \
  --weight_tying "${WEIGHT_TYING}" --nncp_data "${NNCP_DATA}" \
  --accelerator gpu --devices "${GPU_PER_NODE}" --precision "${PRECISION}" --strategy deepspeed_stage_2 \
  --grad_cp "${GRAD_CP}" --enable_progress_bar True --ds_bucket_mb "${DS_BUCKET_MB}"
