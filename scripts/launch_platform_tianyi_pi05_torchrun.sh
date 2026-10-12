#!/usr/bin/env bash
# Platform-native launcher for Tianyi PI05 fine-tuning.
# Submit this script once as the cloud job startup command. The platform runs
# the same command in every worker; HOSTNAME determines the torchrun node rank.
set -euo pipefail

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

OPENPI_ROOT="${OPENPI_ROOT:-/iag_ad_01/ad/dushuyue/openpi}"
OPENPI_RUNTIME="${OPENPI_RUNTIME:-$OPENPI_ROOT/.venv}"
OPENPI_DATA_ROOT="${OPENPI_DATA_ROOT:-$OPENPI_ROOT/data}"
NUM_MACHINES="${OPENPI_NUM_MACHINES:-2}"
GPUS_PER_MACHINE="${OPENPI_GPUS_PER_MACHINE:-8}"
MASTER_PORT="${OPENPI_MAIN_PROCESS_PORT:-29630}"

CONFIG_NAME="${OPENPI_CONFIG_NAME:-pi05_tianyi_finetune}"
DATASET_REPO_ID="${OPENPI_DATASET_REPO_ID:-tianyi}"
DATASET_ROOT="${OPENPI_DATASET_ROOT:-${OPENPI_TIANYI_DATASET_ROOT:-$OPENPI_DATA_ROOT/tianyi_lerobot_v21_overlay}}"
EPISODE_INDICES_PATH="${OPENPI_EPISODE_INDICES_PATH:-}"
ACTION_SPACE="${OPENPI_ACTION_SPACE:-}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-$OPENPI_DATA_ROOT/checkpoints}"
PI05_BASE_PYTORCH_PATH="${PI05_BASE_PYTORCH_PATH:-$OPENPI_DATA_ROOT/checkpoints/pi05_base_pytorch}"
EXP_NAME="${OPENPI_EXP_NAME:-tianyi_stationary_pi05_2x8}"
BATCH_SIZE="${OPENPI_BATCH_SIZE:-32}"
NUM_TRAIN_STEPS="${OPENPI_NUM_TRAIN_STEPS:-30000}"
SAVE_INTERVAL="${OPENPI_SAVE_INTERVAL:-1000}"
MAX_CHECKPOINTS="${OPENPI_MAX_CHECKPOINTS:-5}"
NUM_WORKERS="${OPENPI_NUM_WORKERS:-}"
WARMUP_STEPS="${OPENPI_WARMUP_STEPS:-}"
DECAY_STEPS="${OPENPI_DECAY_STEPS:-}"
RESUME="${OPENPI_RESUME:-0}"

case "${HOSTNAME:-}" in
  *-worker-*) ;;
  *) echo "Cannot derive worker index from HOSTNAME=${HOSTNAME:-}" >&2; exit 2 ;;
esac
BASE="${HOSTNAME%-worker-*}"
NODE_INDEX="${HOSTNAME##*-worker-}"
MASTER_FILE="${OPENPI_TORCH_MASTER_FILE:-$OPENPI_DATA_ROOT/.pi05_tianyi_torch_master_${BASE}}"
[[ "$NODE_INDEX" =~ ^[0-9]+$ ]] || { echo "Invalid worker index: $NODE_INDEX" >&2; exit 2; }
(( NODE_INDEX < NUM_MACHINES )) || {
  echo "NODE_INDEX=$NODE_INDEX >= NUM_MACHINES=$NUM_MACHINES" >&2
  exit 2
}

# Worker 0 publishes an address on the shared data volume. Other workers wait
# for it, so no node-specific command or manual rank assignment is needed.
if [[ -n "${OPENPI_TORCH_MASTER_ADDR:-}" ]]; then
  MASTER_ADDR="$OPENPI_TORCH_MASTER_ADDR"
elif [[ "$NODE_INDEX" == "0" ]]; then
  MASTER_IP="$(hostname -I | tr ' ' '\n' | awk '/^172\.31\./ {print; found=1; exit} /^192\.18\./ {fallback=$0} END {if (!found && fallback != "") print fallback}')"
  if [[ -z "$MASTER_IP" ]]; then
    MASTER_IP="$(ip -4 -o addr show scope global | awk '$4 !~ /^127\./ {split($4,a,"/"); print a[1]; exit}')"
  fi
  [[ -n "$MASTER_IP" ]] || { echo "TORCHRUN_MASTER_IP_NOT_FOUND" >&2; exit 3; }
  tmp="${MASTER_FILE}.$$"
  printf '%s\n' "$MASTER_IP" > "$tmp"
  mv -f "$tmp" "$MASTER_FILE"
  MASTER_ADDR="$MASTER_IP"
else
  MASTER_ADDR=""
  for _ in $(seq 1 120); do
    if [[ -s "$MASTER_FILE" && "$(wc -l < "$MASTER_FILE" | tr -d '[:space:]')" == "1" ]]; then
      candidate="$(head -n 1 "$MASTER_FILE" | tr -d '[:space:]')"
      if [[ "$candidate" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        MASTER_ADDR="$candidate"
        break
      fi
    fi
    sleep 1
  done
  [[ -n "$MASTER_ADDR" ]] || { echo "TORCHRUN_MASTER_FILE_TIMEOUT=$MASTER_FILE" >&2; exit 3; }
fi

if [[ -n "${OPENPI_NCCL_SOCKET_IFNAME:-}" ]]; then
  NCCL_SOCKET_IFNAME="$OPENPI_NCCL_SOCKET_IFNAME"
else
  NCCL_SOCKET_IFNAME="$(ip -4 -o addr show scope global | awk -v ip="$MASTER_ADDR" '$4 ~ "^" ip "/" {print $2; exit}')"
  if [[ -z "$NCCL_SOCKET_IFNAME" || "$NCCL_SOCKET_IFNAME" == "lo" ]]; then
    NCCL_SOCKET_IFNAME="$(ip route get "$MASTER_ADDR" 2>/dev/null | sed -n 's/.* dev \([^ ]*\).*/\1/p' | head -n 1)"
  fi
fi
[[ -n "$NCCL_SOCKET_IFNAME" && "$NCCL_SOCKET_IFNAME" != "lo" ]] || {
  echo "Cannot infer NCCL interface for MASTER_ADDR=$MASTER_ADDR; set OPENPI_NCCL_SOCKET_IFNAME" >&2
  exit 4
}

export NCCL_SOCKET_IFNAME
export GLOO_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME"
export NCCL_NET="${NCCL_NET:-IB}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_IB_PCI_RELAXED_ORDERING="${NCCL_IB_PCI_RELAXED_ORDERING:-1}"
export HF_HOME="${HF_HOME:-$OPENPI_DATA_ROOT/.hf_cache}"
export HF_DATASETS_CACHE="${OPENPI_HF_DATASETS_CACHE:-$OPENPI_DATA_ROOT/.hf_cache/tianyi_stationary_v21}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$OPENPI_DATA_ROOT/.openpi_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
# The cloud worker image has no usable FFmpeg shared libraries, so torchcodec
# cannot load there. PyAV is bundled with the existing LeRobot runtime and does
# not use torchcodec. It must run in the rank process (no DataLoader workers),
# which also avoids known decoder-worker crashes.
export OPENPI_VIDEO_BACKEND="${OPENPI_VIDEO_BACKEND:-pyav}"
if [[ -z "$NUM_WORKERS" ]]; then
  NUM_WORKERS=0
fi
# PyAV is CPU-decoded and can run in persistent DataLoader workers. The
# caller controls the worker count; use 0 only when worker-side decoder
# failures are observed.
if [[ "$OPENPI_VIDEO_BACKEND" == "torchcodec" ]]; then
  TORCH_LIB_DIR="$($OPENPI_RUNTIME/bin/python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / "lib")')"
  export LD_LIBRARY_PATH="$TORCH_LIB_DIR:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
  "$OPENPI_RUNTIME/bin/python" -c 'import torchcodec; print("TORCHCODEC_FFMPEG_OK")'
fi

printf '%s\n' "OPENPI_TORCH_LAUNCH HOSTNAME=$HOSTNAME NODE_RANK=$NODE_INDEX NNODES=$NUM_MACHINES NPROC_PER_NODE=$GPUS_PER_MACHINE"
printf '%s\n' "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
printf '%s\n' "CONFIG=$CONFIG_NAME DATASET_REPO_ID=$DATASET_REPO_ID DATASET_ROOT=$DATASET_ROOT EPISODE_INDICES_PATH=$EPISODE_INDICES_PATH ACTION_SPACE=$ACTION_SPACE EXP_NAME=$EXP_NAME GLOBAL_BATCH=$BATCH_SIZE STEPS=$NUM_TRAIN_STEPS VIDEO_BACKEND=$OPENPI_VIDEO_BACKEND NUM_WORKERS=$NUM_WORKERS"

if [[ "${OPENPI_DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

cd "$OPENPI_ROOT"
DATA_ARGS=(
  "--data.repo-id=$DATASET_REPO_ID"
  "--data.dataset-root=$DATASET_ROOT"
)
if [[ -n "$EPISODE_INDICES_PATH" ]]; then
  DATA_ARGS+=("--data.episode-indices-path=$EPISODE_INDICES_PATH")
fi
if [[ -n "$ACTION_SPACE" ]]; then
  DATA_ARGS+=("--data.action-space=$ACTION_SPACE")
fi
TRAIN_ARGS=(
  "$CONFIG_NAME"
  "--exp-name=$EXP_NAME"
  "--batch-size=$BATCH_SIZE"
  "--num-train-steps=$NUM_TRAIN_STEPS"
  "--save-interval=$SAVE_INTERVAL"
  "--max-checkpoints=$MAX_CHECKPOINTS"
  "--num-workers=$NUM_WORKERS"
  "--checkpoint-base-dir=$CHECKPOINT_BASE_DIR"
  "--pytorch-weight-path=$PI05_BASE_PYTORCH_PATH"
  "--no-wandb-enabled"
)
if [[ -n "$WARMUP_STEPS" ]]; then
  TRAIN_ARGS+=("--lr-schedule.warmup-steps=$WARMUP_STEPS")
fi
if [[ -n "$DECAY_STEPS" ]]; then
  TRAIN_ARGS+=("--lr-schedule.decay-steps=$DECAY_STEPS")
fi
if [[ "$RESUME" == "1" ]]; then
  TRAIN_ARGS+=("--resume")
fi
TORCH_LOG_DIR="${OPENPI_TORCH_LOG_DIR:-$OPENPI_DATA_ROOT/training_logs/$EXP_NAME/$BASE}/worker-$NODE_INDEX"
mkdir -p "$TORCH_LOG_DIR"
printf '%s\n' "TORCH_LOG_DIR=$TORCH_LOG_DIR"
exec "$OPENPI_RUNTIME/bin/python" -m torch.distributed.run \
  --log-dir="$TORCH_LOG_DIR" \
  --tee=3 \
  --nnodes="$NUM_MACHINES" \
  --nproc_per_node="$GPUS_PER_MACHINE" \
  --node_rank="$NODE_INDEX" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  scripts/train_pytorch.py "${TRAIN_ARGS[@]}" \
  "${DATA_ARGS[@]}"
