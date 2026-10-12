#!/usr/bin/env bash
# Run held-out Tianyi action evaluation and annotated video export on one GPU.
set -euo pipefail

OPENPI_ROOT="${OPENPI_ROOT:-/iag_ad_01/ad/dushuyue/openpi}"
OPENPI_RUNTIME="${OPENPI_RUNTIME:-$OPENPI_ROOT/.venv}"
OPENPI_DATA_ROOT="${OPENPI_DATA_ROOT:-$OPENPI_ROOT/data}"
CHECKPOINT="${OPENPI_EVAL_CHECKPOINT:-$OPENPI_DATA_ROOT/checkpoints/pi05_tianyi_finetune/tianyi_stationary_pi05_2x8/30000}"
EVAL_ROOT="${OPENPI_TIANYI_EVAL_ROOT:-$OPENPI_DATA_ROOT/tianyi-eval_lerobot_v21_overlay}"
OUTPUT_DIR="${OPENPI_EVAL_OUTPUT_DIR:-$OPENPI_DATA_ROOT/eval/pi05_tianyi_step30000}"
MAX_EPISODES="${OPENPI_EVAL_MAX_EPISODES:-730}"
MAX_FRAMES="${OPENPI_EVAL_MAX_FRAMES:-}"
VIDEO_EVERY="${OPENPI_EVAL_VIDEO_EVERY:-100}"
DEVICE="${OPENPI_EVAL_DEVICE:-cuda}"
EPISODES_PER_TASK="${OPENPI_EVAL_EPISODES_PER_TASK:-}"
SELECTION_FILE="${OPENPI_EVAL_SELECTION_FILE:-}"
NORM_STATS_DIR="${OPENPI_EVAL_NORM_STATS_DIR:-}"
MODEL_LABEL="${OPENPI_EVAL_MODEL_LABEL:-PI0.5}"
COMPARISON_VIDEO="${OPENPI_EVAL_COMPARISON_VIDEO:-0}"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$OPENPI_DATA_ROOT/.openpi_cache}"
export HF_HOME="${HF_HOME:-$OPENPI_DATA_ROOT/.hf_cache}"
export HF_DATASETS_CACHE="${OPENPI_HF_DATASETS_CACHE:-$OPENPI_DATA_ROOT/.hf_cache/tianyi_eval_v21}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export OPENPI_VIDEO_BACKEND="${OPENPI_VIDEO_BACKEND:-pyav}"

args=(
  --checkpoint "$CHECKPOINT"
  --eval-root "$EVAL_ROOT"
  --output-dir "$OUTPUT_DIR"
  --device "$DEVICE"
  --video-backend "$OPENPI_VIDEO_BACKEND"
  --seed "${OPENPI_EVAL_SEED:-0}"
  --max-episodes "$MAX_EPISODES"
  --video-every "$VIDEO_EVERY"
  --model-label "$MODEL_LABEL"
  --openpi-data-home "$OPENPI_DATA_HOME"
)
if [[ -n "$EPISODES_PER_TASK" ]]; then
  args+=(--episodes-per-task "$EPISODES_PER_TASK")
fi
if [[ -n "$SELECTION_FILE" ]]; then
  args+=(--selection-file "$SELECTION_FILE")
fi
if [[ -n "$NORM_STATS_DIR" ]]; then
  args+=(--norm-stats-dir "$NORM_STATS_DIR")
fi
if [[ "$COMPARISON_VIDEO" == "1" ]]; then
  args+=(--comparison-video)
fi
if [[ -n "$MAX_FRAMES" ]]; then
  args+=(--max-frames "$MAX_FRAMES")
fi

cd "$OPENPI_ROOT"
exec "$OPENPI_RUNTIME/bin/python" -u scripts/eval_tianyi.py "${args[@]}"
