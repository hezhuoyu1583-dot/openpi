# 终端环境区分指南

## 问题

推理和仿真使用不同的 Python 环境，在两个终端中容易混淆：

```
推理服务器:      主 venv (.venv) - Python 3.11
仿真客户端:      仿真 venv (examples/libero/.venv) - Python 3.8
```

---

## 方案 1：使用增强启动脚本（推荐）✅

### 特点
- 自动激活正确的 venv
- 修改终端标题（显示环境和功能）
- 修改提示符（用颜色区分）
- 检查依赖和连接

### 使用方法

**终端 1 - 启动推理（红色标识）：**

```bash
bash start_inference_with_env.sh
```

**预期输出：**
```
╔════════════════════════════════════════════════════════════╗
║          🟥 INFERENCE SERVER (推理服务器)                   ║
╚════════════════════════════════════════════════════════════╝

📋 环境信息:
  - Python 版本: Python 3.11.15
  - Venv: /iag_ad_01/ad/dushuyue/openpi/.venv
  - 模块: ✓ openpi

📌 推理配置:
  - Checkpoint: .../my_run/11000
  - Port: 8000
  - WebSocket: ws://0.0.0.0:8000

⏱️  启动服务器...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**终端标题：** `[推理服务器] OpenPI Inference - Port 8000`

**提示符（红色背景）：** `INFERENCE (Port:8000) /path/to/openpi $`

---

**终端 2 - 启动仿真（蓝色标识）：**

```bash
bash start_simulation_with_env.sh libero_spatial 2
```

**预期输出：**
```
╔════════════════════════════════════════════════════════════╗
║          🟦 SIMULATION CLIENT (仿真客户端)                   ║
╚════════════════════════════════════════════════════════════╝

📋 环境信息:
  - Python 版本: Python 3.8.19
  - Venv: /iag_ad_01/ad/dushuyue/openpi/examples/libero/.venv
  - 模块: ✓ mujoco, robosuite

📌 仿真配置:
  - 服务器: ws://127.0.0.1:8000
  - 任务套件: libero_spatial
  - 试验次数: 2

⏱️  连接到推理服务器...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**终端标题：** `[仿真客户端] OpenPI Simulation - libero_spatial`

**提示符（蓝色背景）：** `SIMULATION (libero_spatial) /path/to/openpi $`

---

## 方案 2：手动激活环境（可选）

如果需要在终端中手动操作，可以显式激活不同的 venv：

### 推理服务器环境

```bash
# 激活主 venv
source /iag_ad_01/ad/dushuyue/openpi/.venv/bin/activate

# 验证环境
python --version  # 应该是 Python 3.11.x
python -c "import openpi; print('✓ openpi')"

# 启动推理
bash eval_libero.sh
```

**提示符（自动更新）：** `(.venv) /path/to/openpi $`

### 仿真客户端环境

```bash
# 激活仿真 venv
source /iag_ad_01/ad/dushuyue/openpi/examples/libero/.venv/bin/activate

# 验证环境
python --version  # 应该是 Python 3.8.x
python -c "import mujoco, robosuite; print('✓')"

# 启动仿真
bash sim_libero.sh
```

**提示符（自动更新）：** `(.venv) /path/to/openpi $`

---

## 方案 3：自定义提示符（高级）

编辑 `~/.bashrc` 添加自定义提示符识别脚本：

```bash
# 添加到 ~/.bashrc

# 识别当前 Python 环境
show_venv_info() {
    if [[ -n "$VIRTUAL_ENV" ]]; then
        local venv_name=$(basename "$VIRTUAL_ENV")
        local py_version=$(python --version 2>&1 | awk '{print $2}')
        
        if [[ "$VIRTUAL_ENV" == *"/.venv"* ]]; then
            echo "🟥 主环境 (Python $py_version)"
        elif [[ "$VIRTUAL_ENV" == *"/libero/.venv"* ]]; then
            echo "🟦 仿真环境 (Python $py_version)"
        else
            echo "? 其他环境"
        fi
    else
        echo "⚫ 未激活任何环境"
    fi
}

# 在每个新终端显示
echo "当前环境: $(show_venv_info)"
```

---

## 方案 4：命令行一览

### 快速参考

| 操作 | 命令 | 环境 | Python |
|------|------|------|--------|
| 启动推理（推荐） | `bash start_inference_with_env.sh` | 主 venv | 3.11 |
| 启动仿真（推荐） | `bash start_simulation_with_env.sh libero_spatial 2` | 仿真 venv | 3.8 |
| 手动激活推理 | `source .venv/bin/activate` | 主 venv | 3.11 |
| 手动激活仿真 | `source examples/libero/.venv/bin/activate` | 仿真 venv | 3.8 |
| 查看当前环境 | `python --version && which python` | 当前 | - |
| 检查 openpi | `python -c "import openpi"` | 必须主 venv | 3.11 |
| 检查 mujoco | `python -c "import mujoco"` | 必须仿真 venv | 3.8 |

---

## 终端显示对比

### 使用方案 1（推荐）

```
╔═══════════════════════════════════════════════════════════════╗
║ 终端 1（推理）                  │ 终端 2（仿真）              ║
├─────────────────────────────────┼───────────────────────────┤
║ 标题: [推理服务器]              │ 标题: [仿真客户端]         ║
║ 背景: 红色 🟥                   │ 背景: 蓝色 🟦             ║
║ 提示符: INFERENCE (Port:8000)   │ 提示符: SIMULATION (...)   ║
║ Python: 3.11                    │ Python: 3.8               ║
║ Venv: .venv                     │ Venv: examples/libero     ║
║                                 │ .venv                     ║
╚═══════════════════════════════════════════════════════════════╝
```

### 使用方案 2（手动）

```
终端 1（推理）
$ source .venv/bin/activate
(.venv) $ python --version
Python 3.11.15

终端 2（仿真）
$ source examples/libero/.venv/bin/activate
(.venv) $ python --version
Python 3.8.19
```

---

## 常见问题

### Q1: 怎么确认我在正确的环境中？

```bash
# 检查 Python 版本
python --version

# 检查 venv 路径
which python

# 检查模块是否可用
python -c "import openpi"  # 应该在主 venv 中成功
python -c "import mujoco"  # 应该在仿真 venv 中成功
```

### Q2: 我不小心在仿真环境中运行了推理脚本怎么办？

```bash
# 激活正确的环境
source /iag_ad_01/ad/dushuyue/openpi/.venv/bin/activate

# 验证
python -c "import openpi; print('✓ 现在在正确的环境中')"

# 重新运行
bash eval_libero.sh
```

### Q3: 如何在一个终端中切换环境？

```bash
# 当前在推理环境
source .venv/bin/activate
python --version  # Python 3.11

# 切换到仿真环境
deactivate  # 退出当前环境
source examples/libero/.venv/bin/activate
python --version  # Python 3.8

# 切换回推理环境
deactivate
source .venv/bin/activate
python --version  # Python 3.11
```

---

## 总结

| 方案 | 优点 | 缺点 | 推荐度 |
|------|------|------|--------|
| **方案 1**（增强脚本） | 自动化，清晰易区分 | 需要特定脚本 | ⭐⭐⭐⭐⭐ |
| **方案 2**（手动激活） | 灵活，学习 venv 原理 | 需要手动操作 | ⭐⭐⭐ |
| **方案 3**（自定义提示符） | 持久化，可扩展 | 配置复杂 | ⭐⭐ |
| **方案 4**（无区分） | 简单 | 容易混淆环境 | ⭐ |

**建议：使用方案 1（增强脚本）+ 方案 2（必要时手动）**
