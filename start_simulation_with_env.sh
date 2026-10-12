#!/bin/bash
# 启动仿真客户端（附带环境标识）
# 特点：自动激活仿真 venv，修改终端标题和提示符，显示环境信息

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 参数
TASK_SUITE="${1:-libero_spatial}"
NUM_TRIALS="${2:-2}"
TASK_ID="${3:-}"          # 可选：只跑这个task_id对应的环境
PROMPT_TASK_ID="${4:-}"   # 可选：喂给模型的instruction改用这个task_id的（反事实测试）
SERVE_HOST="${SERVE_HOST:-127.0.0.1}"
SERVE_PORT="${SERVE_PORT:-8000}"

# 检查仿真 venv 存在
SIM_VENV_PY="${SCRIPT_DIR}/examples/libero/.venv/bin/python"
if [[ ! -x "${SIM_VENV_PY}" ]]; then
  echo "❌ 仿真 venv 未搭建"
  echo "请先运行: bash setup_libero_sim.sh"
  exit 1
fi

# 激活仿真 venv
source "${SCRIPT_DIR}/examples/libero/.venv/bin/activate"

# 设置数据根目录和缓存（与训练和推理脚本保持一致）
export OPENPI_DATA_ROOT="${SCRIPT_DIR}/data"
export OPENPI_DATA_HOME="${OPENPI_DATA_ROOT}/.openpi_cache"
export HF_HOME="${OPENPI_DATA_ROOT}/.hf_cache"
export HF_DATASETS_CACHE="${OPENPI_DATA_ROOT}/.hf_cache"

# Offline + dead-proxy guards（与训练脚本保持一致）
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true

# 修改终端标题
echo -ne "\033]0;[仿真客户端] OpenPI Simulation - ${TASK_SUITE}\007"

# 修改提示符，添加环境标识（蓝色背景 = 仿真）
export PS1="\[\033[44m\]\[\033[37m\] SIMULATION (${TASK_SUITE}) \[\033[0m\] \[\033[1;32m\]\w\[\033[0m\] $ "

# 检查推理服务连接
if ! timeout 2 bash -c "echo > /dev/tcp/${SERVE_HOST}/${SERVE_PORT}" 2>/dev/null; then
  echo ""
  echo "╔════════════════════════════════════════════════════════════╗"
  echo "║  ⚠️  警告：无法连接到推理服务器                             ║"
  echo "╚════════════════════════════════════════════════════════════╝"
  echo ""
  echo "❌ 连接失败: ws://${SERVE_HOST}:${SERVE_PORT}"
  echo ""
  echo "💡 请确保在另一个终端已启动推理服务器:"
  echo "   bash start_inference_with_env.sh"
  echo ""
  read -p "继续尝试连接? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
  fi
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║          🟦 SIMULATION CLIENT (仿真客户端)                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📋 环境信息:"
echo "  - Python 版本: $(python --version)"
echo "  - Venv: ${SCRIPT_DIR}/examples/libero/.venv"
echo "  - 模块: ✓ mujoco, robosuite (initialized on startup)"
echo ""
echo "📌 仿真配置:"
echo "  - 服务器: ws://${SERVE_HOST}:${SERVE_PORT}"
echo "  - 任务套件: ${TASK_SUITE}"
echo "  - 试验次数: ${NUM_TRIALS}"
[[ -n "${TASK_ID}" ]] && echo "  - 只跑 task_id: ${TASK_ID}"
[[ -n "${PROMPT_TASK_ID}" ]] && echo "  - ⚠️  反事实prompt: 使用 task_id=${PROMPT_TASK_ID} 的instruction"
echo ""
echo "⏱️  连接到推理服务器..."
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

export SERVE_HOST="${SERVE_HOST}"
export SERVE_PORT="${SERVE_PORT}"
export TASK_SUITE="${TASK_SUITE}"
export NUM_TRIALS="${NUM_TRIALS}"
export PYTHONPATH="${PYTHONPATH:-}:${SCRIPT_DIR}/third_party/libero:${SCRIPT_DIR}/packages/openpi-client/src"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="osmesa"

EXTRA_ARGS=()
[[ -n "${TASK_ID}" ]] && EXTRA_ARGS+=(--task-id "${TASK_ID}")
[[ -n "${PROMPT_TASK_ID}" ]] && EXTRA_ARGS+=(--prompt-task-id "${PROMPT_TASK_ID}")

python examples/libero/main_osmesa.py \
  --host "${SERVE_HOST}" \
  --port "${SERVE_PORT}" \
  --task-suite-name "${TASK_SUITE}" \
  --num-trials-per-task "${NUM_TRIALS}" \
  --video-out-path "data/libero/videos" \
  "${EXTRA_ARGS[@]}"
