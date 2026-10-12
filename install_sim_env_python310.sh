#!/bin/bash
# LIBERO 仿真环境搭建 (Python 3.10 - 简化版)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}"
cd "${REPO_DIR}"

SIM_VENV_DIR="${REPO_DIR}/examples/libero/.venv"
SIM_VENV_PY="${SIM_VENV_DIR}/bin/python"
SIM_VENV_PIP="${SIM_VENV_DIR}/bin/pip"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║      LIBERO 仿真环境搭建 (Python 3.10 - 实用版)             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

echo "[STEP 1/5] 检查 Python 3.10"
if ! command -v python3.10 &> /dev/null; then
  echo "❌ Python 3.10 不可用"
  exit 1
fi
python3.10 --version
echo ""

echo "[STEP 1.5/5] 安装系统依赖..."
echo "  - 安装 libosmesa6 (OSMesa 渲染库)..."
sudo apt-get update -qq && sudo apt-get install -y libosmesa6 libosmesa6-dev 2>&1 | grep -E "Setting up|already" || true
echo "  ✓ 系统依赖已安装"
echo ""

# 创建 venv
if [[ -d "${SIM_VENV_DIR}" ]]; then
  echo "[STEP 2/3] 清理旧 venv..."
  rm -rf "${SIM_VENV_DIR}"
fi

echo "[STEP 2/3] 创建 Python 3.10 venv..."
python3.10 -m venv "${SIM_VENV_DIR}"
echo "  ✓ venv 创建完成"
echo ""

# 升级 pip
echo "[STEP 3/3] 安装仿真依赖..."
"${SIM_VENV_PIP}" install --upgrade pip setuptools wheel -q

# 安装依赖包
packages=(
  "numpy"
  "scipy"
  "pyyaml"
  "mujoco>=2.3.0"
  "robosuite>=1.4.0"
  "torch"
  "torchvision"
  "websockets"
  "imageio"
  "imageio-ffmpeg"
  "matplotlib"
  "opencv-python"
  "tqdm"
  "tyro"
  "libero"
  "msgpack"
)

for pkg in "${packages[@]}"; do
  echo "  - 安装 $pkg..."
  if [[ "$pkg" == "torch" ]] || [[ "$pkg" == "torchvision" ]]; then
    "${SIM_VENV_PIP}" install "$pkg" --index-url https://download.pytorch.org/whl/cpu -q 2>/dev/null || true
  else
    "${SIM_VENV_PIP}" install "$pkg" -q 2>/dev/null || true
  fi
done

# 修复 EGL 问题（无 libEGL.so 的系统）
OPENGL_EGL_INIT="${SIM_VENV_DIR}/lib/python3.10/site-packages/OpenGL/EGL/__init__.py"
if [[ ! -f "${OPENGL_EGL_INIT}" ]] || [[ ! -s "${OPENGL_EGL_INIT}" ]]; then
  mkdir -p "$(dirname "${OPENGL_EGL_INIT}")"
  cat > "${OPENGL_EGL_INIT}" << 'EOFPATCH'
"""OpenGL.EGL the portable interface to GL environments - Stubbed for headless systems"""
import ctypes

def _stub_func(*args, **kwargs):
    return None

_STUB_TYPE = ctypes.c_void_p

class _StubModule:
    """Stub module for missing EGL"""
    def __getattr__(self, name):
        if name.startswith('egl') or name.startswith('Egl'):
            return _stub_func
        return _STUB_TYPE
    def __call__(self, *args, **kwargs):
        return _STUB_TYPE

_stub = _StubModule()

EGLBoolean = _STUB_TYPE
EGLint = _STUB_TYPE
EGLDisplay = _STUB_TYPE
EGLContext = _STUB_TYPE
EGLSurface = _STUB_TYPE
EGLConfig = _STUB_TYPE
EGLClientBuffer = _STUB_TYPE
EGLDeviceEXT = _STUB_TYPE
EGLNativeDisplayType = _STUB_TYPE
EGLNativeWindowType = _STUB_TYPE
EGLNativePixmapType = _STUB_TYPE
EGLSyncKHR = _STUB_TYPE

eglGetProcAddress = _stub_func
eglGetDisplay = _stub_func
eglInitialize = _stub_func
eglChooseConfig = _stub_func
eglCreateContext = _stub_func
eglCreateWindowSurface = _stub_func
eglMakeCurrent = _stub_func
eglSwapBuffers = _stub_func

class EGLError(Exception):
    pass

__getattr__ = lambda name: _stub.__getattr__(name)
EOFPATCH
  echo "  ✓ OpenGL.EGL 修复已应用"
fi

# 修复 robosuite EGL 导入（禁用 EGL，使用 osmesa 降级）
ROBOSUITE_EGL="${SIM_VENV_DIR}/lib/python3.10/site-packages/robosuite/renderers/context/egl_context.py"
if [[ -f "${ROBOSUITE_EGL}" ]]; then
  if ! grep -q "_SKIP_EGL" "${ROBOSUITE_EGL}"; then
    sed -i '17a\\n# Check if we should actually use EGL or skip it\n_SKIP_EGL = os.environ.get("MUJOCO_GL", "").lower().strip() in ("osmesa", "glfw", "glx", "disable")\n\nif _SKIP_EGL:\n    raise ImportError("EGL disabled - using alternate rendering backend")\n' "${ROBOSUITE_EGL}"
    echo "  ✓ robosuite EGL 绕过已应用"
  fi
fi

# 修复 binding_utils.py 添加 EGL 降级处理
BINDING_UTILS="${SIM_VENV_DIR}/lib/python3.10/site-packages/robosuite/utils/binding_utils.py"
if [[ -f "${BINDING_UTILS}" ]]; then
  if ! grep -q "falling back to osmesa" "${BINDING_UTILS}"; then
    sed -i '/elif _SYSTEM == "Linux" and _MUJOCO_GL == "egl":/i\        try:\n            from robosuite.renderers.context.egl_context import EGLGLContext as GLContext\n        except ImportError:\n            print("[WARNING] EGL not available, falling back to osmesa")\n            from robosuite.renderers.context.osmesa_context import OSMesaGLContext as GLContext\n            GLContext = OSMesaGLContext\n            GLContext = None\n        if GLContext is not None:\n            pass\n        else:' "${BINDING_UTILS}"
    echo "  ✓ robosuite 降级处理已应用"
  fi
fi

# 创建 main_osmesa.py 包装脚本（确保环境变量在导入前设置）
MAIN_OSMESA="${SIM_VENV_DIR}/../main_osmesa.py"
if [[ ! -f "${MAIN_OSMESA}" ]]; then
  echo "[STEP 4/5] 创建 OSMesa 包装脚本..."
  cat > "${MAIN_OSMESA}" << 'EOFMAIN'
#!/usr/bin/env python3
"""OSMesa wrapper for headless mujoco/libero rendering"""
import os
import sys

# Set rendering backend BEFORE any imports
os.environ.setdefault('MUJOCO_GL', 'osmesa')
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

# Add packages to path
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(script_dir, 'third_party', 'libero'))
sys.path.insert(0, os.path.join(script_dir, 'packages', 'openpi-client', 'src'))

# Now import and run main
from examples.libero import main

if __name__ == '__main__':
    main.main()
EOFMAIN
  chmod +x "${MAIN_OSMESA}"
  echo "  ✓ OSMesa 包装脚本已创建"
fi

# 修复 robosuite egl_context.py 以支持 osmesa 优雅降级
ROBOSUITE_EGL="${SIM_VENV_DIR}/lib/python3.10/site-packages/robosuite/renderers/context/egl_context.py"
if [[ -f "${ROBOSUITE_EGL}" ]]; then
  if ! grep -q "# Check if we should actually use EGL" "${ROBOSUITE_EGL}"; then
    echo "[STEP 4.5/5] 修复 robosuite EGL 降级..."
    # Prepend EGL check to egl_context.py
    temp_file=$(mktemp)
    cat > "${temp_file}" << 'EOFCHECK'
import os

# Check if we should actually use EGL or skip it for osmesa
_SKIP_EGL = os.environ.get("MUJOCO_GL", "").lower().strip() in ("osmesa", "glfw", "glx", "disable")
if _SKIP_EGL:
    raise ImportError("EGL disabled - using alternate rendering backend")

EOFCHECK
    cat "${ROBOSUITE_EGL}" >> "${temp_file}"
    mv "${temp_file}" "${ROBOSUITE_EGL}"
    echo "  ✓ robosuite EGL 降级已应用"
  fi
fi

echo ""
echo "╔══��═════════════════════════════════════════════════════════════╗"
echo "║              ✅ 仿真 venv 搭建完成！                           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "环境信息："
echo "  - Python: $("${SIM_VENV_PY}" --version)"
echo "  - Venv: ${SIM_VENV_DIR}"
echo "  - 渲染后端: osmesa (CPU软件渲染)"
echo ""

# 验证
echo "验证安装..."
"${SIM_VENV_PY}" -c "
import sys
modules = []
try:
    import mujoco
    modules.append('mujoco')
except:
    pass
try:
    import robosuite
    modules.append('robosuite')
except:
    pass
try:
    import torch
    modules.append('torch')
except:
    pass
if modules:
    print(f'  ✓ 已安装: {\" \".join(modules)}')
else:
    print('  ⚠️  某些依赖可能未安装')
" 2>/dev/null || echo "  ⚠️  验证脚本执行有问题"

echo ""
echo "✅ 仿真环境已就绪！"
echo ""
echo "现在可以运行仿真评估："
echo "  bash start_simulation_with_env.sh libero_spatial 2"
