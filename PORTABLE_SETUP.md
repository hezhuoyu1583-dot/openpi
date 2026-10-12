# 便携式快速部署指南

## 概述

所有启动脚本、安装脚本已完全本地化，**不包含硬编码的绝对路径**。可以直接复制整个项目到任何位置使用。

## 快速开始（任何位置）

### 第一次使用

```bash
# 1. 复制项目到你的位置
cp -r openpi ~/my-projects/

# 2. 进入项目
cd ~/my-projects/openpi

# 3. 安装推理环境（Python 3.11）
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 4. 安装仿真环境（Python 3.10）
bash install_sim_env_python310.sh

# 5. 准备数据（确保 checkpoint 存在）
# data/checkpoints/pi05_libero/my_run/30000/model.safetensors
```

### 日常使用

```bash
# 终端 1：启动推理服务
cd ~/my-projects/openpi
bash start_inference_with_env.sh

# 终端 2：启动仿真客户端
cd ~/my-projects/openpi
bash start_simulation_with_env.sh libero_spatial 2
```

## 移植检查清单

- [x] 所有脚本使用 `SCRIPT_DIR` 动态获取项目路径
- [x] 所有路径相对于项目根目录（`$SCRIPT_DIR`）
- [x] 没有硬编码的绝对路径（如 `/iag_ad_01/...`）
- [x] 文档中的路径已改为 `<openpi-project-root>`
- [x] 相对路径 `data/`、`examples/libero/` 等自动相对于项目根

## 关键脚本位置

| 脚本 | 功能 | 相对路径 |
|------|------|--------|
| `install_sim_env_python310.sh` | 安装仿真环境 | 项目根目录 |
| `start_inference_with_env.sh` | 启动推理服务 | 项目根目录 |
| `start_simulation_with_env.sh` | 启动仿真客户端 | 项目根目录 |
| `.venv/` | 推理环境 | 项目根目录 |
| `examples/libero/.venv/` | 仿真环境 | 项目根目录下 |

## 如果遇到问题

### "无法找到 checkpoint"

确保 checkpoint 存在于：
```
data/checkpoints/pi05_libero/my_run/30000/model.safetensors
```

### "无法找到 libero assets"

仿真脚本会自动从 `.venv` 中读取，无需手动操作。

### "路径错误"

检查你在哪个目录运行脚本。所有脚本都应该从项目根目录运行：
```bash
cd <project-root>
bash start_simulation_with_env.sh ...
```

## 完全移植示例

```bash
# 场景：想在另一台机器上运行

# 机器 A（原始位置）
cd /iag_ad_01/ad/dushuyue/openpi
# ... 项目已经过测试

# 机器 B（新位置）
mkdir -p /home/user/projects
scp -r /iag_ad_01/ad/dushuyue/openpi /home/user/projects/

# 在机器 B 上
cd /home/user/projects/openpi
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash install_sim_env_python310.sh
bash start_inference_with_env.sh  # 同样有效！
```

---

**版本**: 1.0  
**最后更新**: 2026-07-02  
**状态**: ✅ 完全便携式
