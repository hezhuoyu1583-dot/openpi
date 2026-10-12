#!/bin/bash
# 快速验证脚本 — 推理服务器测试客户端
# 无需仿真 venv，可直接使用主 venv 验证推理服务器是否正常工作

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}"
cd "${REPO_DIR}"
VENV_PY="${REPO_DIR}/.venv/bin/python"

# 设置数据根目录和缓存（与训练和推理脚本保持一致）
export OPENPI_DATA_ROOT="${REPO_DIR}/data"
export OPENPI_DATA_HOME="${OPENPI_DATA_ROOT}/.openpi_cache"
export HF_HOME="${OPENPI_DATA_ROOT}/.hf_cache"
export HF_DATASETS_CACHE="${OPENPI_DATA_ROOT}/.hf_cache"

# Offline + dead-proxy guards（与训练脚本保持一致）
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true

# 连接参数
SERVE_HOST="${SERVE_HOST:-127.0.0.1}"
SERVE_PORT="${SERVE_PORT:-8000}"

echo "[INFO] 推理服务器验证客户端"
echo "  目标: ws://${SERVE_HOST}:${SERVE_PORT}"
echo "  缓存: ${OPENPI_DATA_HOME}"
echo ""

# 检查推理脚本是否存在
if [[ ! -f "examples/simple_client/main.py" ]]; then
  echo "[ERROR] 示例客户端不存在: examples/simple_client/main.py" >&2
  exit 1
fi

exec "${VENV_PY}" examples/simple_client/main.py \
  --env LIBERO \
  --host "${SERVE_HOST}" \
  --port "${SERVE_PORT}" \
  "$@"
