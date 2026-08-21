#!/usr/bin/env bash
set -eo pipefail

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

# Activate the required Python environment before running this script.
set -u

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${ROOT}/demo-model-config.sh"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python - <<'PY'
import sys
import torch

assert sys.version_info[:3] == (3, 12, 1), sys.version
assert torch.__version__ == "2.7.0+cu126", torch.__version__
print(f"Environment verified: {sys.executable} | Python {sys.version.split()[0]} | torch {torch.__version__}")
PY

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RANDOM_SEED=20260817

echo "Using Data File: ${DATA_FILE}"
echo "Writing init prior to: ${PROJ_DIR}/rwkv-init.pth"

python train.py --wandb "" --proj_dir "${PROJ_DIR}" \
  --data_file "${DATA_FILE}" --data_type "binidx" --vocab_size "${VOCAB_SIZE}" --my_testing "${MODEL_TYPE}" \
  --random_seed "${RANDOM_SEED}" \
  --ctx_len "${CTX_LEN}" --train_stage 1 --epoch_count 1 --epoch_begin 0 \
  --epoch_save 1 --weight_decay 0 --head_size "${HEAD_SIZE}" --weight_tying "${WEIGHT_TYING}" --nncp_data "${NNCP_DATA}" \
  --num_nodes 1 --micro_bsz 1 --n_layer "${N_LAYER}" --n_embd "${N_EMBD}" --my_exit_tokens 1498226207 --magic_prime "${MAGIC_PRIME}" \
  --lr_init 1e-5 --lr_final 1e-5 --warmup_steps 10 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
  --accelerator cpu --devices 1 --precision "${PRECISION}" --strategy deepspeed_stage_2 --grad_cp 1
