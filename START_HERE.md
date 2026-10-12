# 🚀 OpenPI LIBERO 本地测试 - 快速开始

## 📌 推荐路径

**第一次阅读**: [OPERATION_GUIDE.md](OPERATION_GUIDE.md) ← 核心指南  
**遇到问题**: [PROXY_CONFIG_GUIDE.md](PROXY_CONFIG_GUIDE.md) 或 [TERMINAL_ENV_GUIDE.md](TERMINAL_ENV_GUIDE.md)

---

## 📋 3 分钟快速上手

### 前置要求检查

```bash
cd /iag_ad_01/ad/dushuyue/openpi

# ✅ 检查环境
.venv/bin/python -c "import openpi; print('✓ 主 venv 就绪')"
examples/libero/.venv/bin/python -c "import mujoco; print('✓ 仿真 venv 就绪')"
ls data/checkpoints/pi05_libero/my_run/11000/model.safetensors
# 输出: 文件存在
```

### 第一次运行（仅需一次）

```bash
# ⚠️ 重要：确保代理已配置
export http_proxy=http://proxy.hs.com:3128
export https_proxy=http://proxy.hs.com:3128

# 仿真环境已搭建（无需重新安装）
```

### 每次测试都这样做

**终端 1 - 启动推理服务器（红色标识，Python 3.11）：**

```bash
cd /iag_ad_01/ad/dushuyue/openpi
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

INFO: Uvicorn running on http://0.0.0.0:8000
```

**终端 2 - 启动仿真评估（蓝色标识，Python 3.10）：**

```bash
cd /iag_ad_01/ad/dushuyue/openpi
bash start_simulation_with_env.sh libero_spatial 2
```

**预期输出：**
```
╔════════════════════════════════════════════════════════════╗
║          🟦 SIMULATION CLIENT (仿真客户端)                   ║
╚════════════════════════════════════════════════════════════╝

Connected to server
Starting libero_spatial evaluation
Task 1/10: pick_place_within_drawer
  Trial 1/2: Success=0.95
...
```

**完成！** ✅ 看到评估结果输出

💡 **提示：** 两个终端会自动显示不同的颜色（红色=推理，蓝色=仿真）和 Python 版本信息

---

## 📊 评估结论

### ✅ 可以在本地同时运行推理和仿真

你的系统配置：
- GPU: NVIDIA H20 (48GB) - ✅ 充足
- CPU: 180 核 - ✅ 充足
- 内存: 111GB - ✅ 充足

资源占用：
- 推理: ~12-15GB 显存
- 仿真: ~1-2GB 内存
- **总计: <35% 利用率** ✅

---

## 🎯 下一步

### 快速验证（5 分钟）
```bash
bash start_inference_with_env.sh &
sleep 10
bash verify_inference.sh
```

### 轻量测试（20 分钟）
```bash
bash start_simulation_with_env.sh libero_spatial 2
```

### 完整评估（2+ 小时）
```bash
bash start_simulation_with_env.sh libero_spatial 10
bash start_simulation_with_env.sh libero_object 10
```

---

## 🆘 遇到问题？

### 代理配置
→ 查看 [PROXY_CONFIG_GUIDE.md](PROXY_CONFIG_GUIDE.md)

### 终端环境混淆
→ 查看 [TERMINAL_ENV_GUIDE.md](TERMINAL_ENV_GUIDE.md)

### 完整操作说明
→ 查看 [OPERATION_GUIDE.md](OPERATION_GUIDE.md)

---

## 💡 常用命令速查

| 任务 | 命令 |
|------|------|
| 启动推理 | `bash start_inference_with_env.sh` |
| 启动仿真 | `bash start_simulation_with_env.sh libero_spatial 2` |
| 快速验证 | `bash verify_inference.sh` |
| 查看 GPU | `nvidia-smi` |
| 查看视频 | `ls -lh data/libero/videos/` |
| 重新安装 | `bash install_sim_env_python310.sh` |

---

**准备就绪！现在就开始测试吧 🚀**

### 每次测试都这样做

**终端 1 - 启动推理服务器（红色标识，Python 3.11）：**

```bash
cd /iag_ad_01/ad/dushuyue/openpi
bash start_inference_with_env.sh
```

等待看到:
```
╔════════════════════════════════════════════════════════════╗
║          🟥 INFERENCE SERVER (推理服务器)                   ║
╚════════════════════════════════════════════════════════════╝
...
INFO: Uvicorn running on http://0.0.0.0:8000
```

**终端 2 - 启动仿真评估（蓝色标识，Python 3.8）：**

```bash
cd /iag_ad_01/ad/dushuyue/openpi
bash start_simulation_with_env.sh libero_spatial 2
```

等待看到:
```
╔════════════════════════════════════════════════════════════╗
║          🟦 SIMULATION CLIENT (仿真客户端)                   ║
╚════════════════════════════════════════════════════════════╝
...
评估结果输出
```

**完成！** ✅ 看到评估结果输出

💡 **提示：** 两个终端自动区分（红色=推理，蓝色=仿真），还会显示 Python 版本和 venv 路径

---

## 📊 评估结论

### ✅ 可以在本地同时运行推理和仿真

你的系统配置：
- GPU: NVIDIA H20 (48GB) - ✅ 充足
- CPU: 180 核 - ✅ 充足
- 内存: 111GB - ✅ 充足

资源占用：
- 推理: ~12-15GB 显存
- 仿真: ~1-2GB 内存
- **总计: <35% 利用率** ✅

---

## 🎯 下一步

### 快速验证（5 分钟）
```bash
bash start_inference.sh &        # 后台启动
sleep 10                         # 等待服务启动
bash start_simulation.sh libero_spatial 1
```

### 轻量测试（20 分钟）
```bash
bash start_simulation.sh libero_spatial 2
```

### 完整评估（2 小时）
```bash
bash start_simulation.sh libero_spatial 10
bash start_simulation.sh libero_object 10
```

---

## 📚 完整文档

- **[LIBERO_LOCAL_TEST_GUIDE.md](LIBERO_LOCAL_TEST_GUIDE.md)** - 详细的测试指南
- **[LIBERO_FEASIBILITY_REPORT.md](LIBERO_FEASIBILITY_REPORT.md)** - 完整的可行性评估报告
- **[INFERENCE_SIMULATION_GUIDE.md](INFERENCE_SIMULATION_GUIDE.md)** - 推理和仿真的架构说明

---

## 🆘 遇到问题？

### 推理服务无法启动
```bash
# 确保主 venv 有 openpi 模块
.venv/bin/pip install -e .
```

### 仿真连接失败
```bash
# 确保推理服务已启动并在监听 8000 端口
netstat -an | grep 8000
```

### 仿真 venv 搭建失败
```bash
# 检查 Python 3.8
python3.8 --version
# 如果不存在，安装:
apt-get install python3.8 python3.8-venv
```

---

## 💡 常用命令速查

| 任务 | 命令 |
|------|------|
| 启动推理（推荐） | `bash start_inference_with_env.sh` |
| 启动仿真（推荐） | `bash start_simulation_with_env.sh libero_spatial 2` |
| 快速验证 | `bash verify_inference.sh` |
| 查看 GPU | `nvidia-smi` |
| 查看视频 | `ls -lh data/libero/videos/` |
| 搭建仿真环境 | `bash setup_libero_sim.sh` |
| **环境区分指南** | **📖 见 [TERMINAL_ENV_GUIDE.md](TERMINAL_ENV_GUIDE.md)** |

---

**准备就绪！现在就开始测试吧 🚀**
