#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}"
cd "${REPO_DIR}"
VENV_PY="${REPO_DIR}/.venv/bin/python"

# Offline + dead-proxy guards (override by exporting beforehand).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true

# Tunables (env or trailing CLI flags; first positional is exp_name).
EXP_NAME="${EXP_NAME:-${1:-libero_$(date +%Y%m%d_%H%M%S)}}"
NUM_GPUS="${NUM_GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
if [[ $# -ge 1 && "${1}" != --* ]]; then shift; fi

ARGS=(
  pi05_libero
  --exp_name="${EXP_NAME}"
  --batch_size="${BATCH_SIZE}"
  --num_train_steps="${NUM_TRAIN_STEPS}"
  --save_interval="${SAVE_INTERVAL}"
  --checkpoint_base_dir="${CHECKPOINT_BASE_DIR}"
  --data.repo_id="${OPENPI_LIBERO_REPO_ID}"
  --pytorch_weight_path="${PI05_BASE_PYTORCH_PATH}"
  "$@"   # user flags win
)

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  exec "${VENV_PY}" -m torch.distributed.run --standalone --nnodes=1 \
    --nproc_per_node="${NUM_GPUS}" scripts/train_pytorch.py "${ARGS[@]}"
else
  exec "${VENV_PY}" scripts/train_pytorch.py "${ARGS[@]}"
fi

    