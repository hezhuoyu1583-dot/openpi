#!/bin/bash
# 启动推理服务器（附带环境标识）
# 特点：自动激活主 venv，修改终端标题和提示符，显示环境信息

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 参数
OPENPI_DATA_ROOT="${SCRIPT_DIR}/data"
# Support both trained and base checkpoints
CHECKPOINT_TYPE="${CHECKPOINT_TYPE:-libero}"  # libero | base
if [[ "${CHECKPOINT_TYPE}" == "base" ]]; then
  PI05_LIBERO_CHECKPOINT="${OPENPI_DATA_ROOT}/checkpoints/pi05_base_pytorch"
else
  PI05_LIBERO_CHECKPOINT="${OPENPI_DATA_ROOT}/checkpoints/pi05_libero/my_run/30000"
fi
PORT="${PORT:-8000}"

# 激活主 venv
source "${SCRIPT_DIR}/.venv/bin/activate"

# 设置数据根目录和缓存（与训练脚本保持一致）
export OPENPI_DATA_ROOT="${SCRIPT_DIR}/data"
export OPENPI_DATA_HOME="${OPENPI_DATA_ROOT}/.openpi_cache"
export HF_HOME="${OPENPI_DATA_ROOT}/.hf_cache"
export HF_DATASETS_CACHE="${OPENPI_DATA_ROOT}/.hf_cache"

# Offline + dead-proxy guards（与训练脚本保持一致）
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true

# 修改终端标题（适用于大多数终端）
echo -ne "\033]0;[推理服务器] OpenPI Inference - Port ${PORT}\007"

# 修改提示符，添加环境标识（红色背景 = 推理）
export PS1="\[\033[41m\]\[\033[37m\] INFERENCE (Port:${PORT}) \[\033[0m\] \[\033[1;32m\]\w\[\033[0m\] $ "

# 验证 checkpoint
if [[ ! -f "${PI05_LIBERO_CHECKPOINT}/model.safetensors" ]]; then
  echo "❌ Checkpoint 不存在: ${PI05_LIBERO_CHECKPOINT}/model.safetensors"
  echo ""
  echo "可用的 checkpoints:"
  ls -1 "${OPENPI_DATA_ROOT}/checkpoints/pi05_libero/my_run/"
  exit 1
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║          🟥 INFERENCE SERVER (推理服务器)                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📋 环境信息:"
echo "  - Python 版本: $(.venv/bin/python --version)"
echo "  - Venv: ${SCRIPT_DIR}/.venv"
echo "  - 模块: $(python -c 'import openpi; print("✓ openpi")')"
echo ""
echo "📌 推理配置:"
echo "  - Checkpoint: ${PI05_LIBERO_CHECKPOINT}"
echo "  - Port: ${PORT}"
echo "  - WebSocket: ws://0.0.0.0:${PORT}"
echo ""
echo "⏱️  启动服务器..."
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

export OPENPI_DATA_ROOT="${OPENPI_DATA_ROOT}"
export PI05_LIBERO_CHECKPOINT="${PI05_LIBERO_CHECKPOINT}"
export PORT="${PORT}"

python scripts/serve_policy.py \
  --env LIBERO \
  --port "${PORT}" \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir "${PI05_LIBERO_CHECKPOINT}" \
  "$@"
