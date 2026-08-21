#!/usr/bin/env bash
set -euo pipefail

# Compress enwik9 with the model produced by demo-training-run.sh.

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${ROOT}/demo-model-config.sh"

# Activate the required Python environment before running this script.

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=1024
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

INPUT_FILE="${DATA_FILE}"
MODEL_PATH="${PROJ_DIR}/rwkv-200.pth"
ARCHIVE_DIR="enwik9.dna4"
SEED_TRIALS=100

cd "$ROOT"

[[ -f "$INPUT_FILE" ]] || { echo "Missing input file: $INPUT_FILE" >&2; exit 1; }
[[ -f "$MODEL_PATH" ]] || {
  echo "Missing model checkpoint: $MODEL_PATH" >&2
  echo "Run demo-training-run.sh to completion, or update MODEL_PATH deliberately." >&2
  exit 1
}
[[ -f "${ARCHIVE_DIR}/nncp.dict" ]] || {
  echo "Missing ${ARCHIVE_DIR}/nncp.dict required for raw-file restoration." >&2
  exit 1
}

exec python dna4_cli.py compress \
  --input_file "$INPUT_FILE" \
  --model_path "$MODEL_PATH" \
  --archive_dir "$ARCHIVE_DIR" \
  --seed_trials "$SEED_TRIALS" \
  --weight_tying "$WEIGHT_TYING" \
  --device cuda
