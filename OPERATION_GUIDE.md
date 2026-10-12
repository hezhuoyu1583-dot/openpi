# OpenPI LIBERO 推理与仿真操作指南

**从环境安装到运行的完整指南 - 已验证可用**

---

## 📋 环境安装（首次运行）

### 前置条件

```bash
# 确保有两个 Python 环境
python3.11 --version      # 用于推理服务
python3.10 --version      # 用于仿真

# 确保 sudo 权限（安装系统依赖需要）
sudo apt-get update
```

### Step 1: 安装推理环境（主 venv）

```bash
cd <openpi-project-root>

# 如果还没有主 venv，创建它
python3.11 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt    # 如果有 requirements.txt
```

注：`<openpi-project-root>` 是项目根目录，所有脚本会自动检测

### Step 2: 安装仿真环境（Python 3.10 venv）

**推荐方式：运行一键安装脚本**

```bash
cd <openpi-project-root>
bash install_sim_env_python310.sh
```

这个脚本会自动：
- ✅ 创建 Python 3.10 虚拟环境
- ✅ 安装所有依赖（mujoco, robosuite, libero, imageio, tyro 等）
- ✅ 安装系统库（libosmesa6 - OSMesa 渲染）
- ✅ 修复 OpenGL/EGL 渲染问题
- ✅ 配置 osmesa 软件渲染后端

脚本已验证可用，无需修改。

---

## 🚀 快速开始（日常使用）

### 设置代理（如果公司需要）

```bash
export http_proxy=http://proxy.hs.com:3128
export https_proxy=http://proxy.hs.com:3128
```

### 启动推理服务

**终端 1：启动推理服务器**

```bash
cd <openpi-project-root>
bash start_inference_with_env.sh
```

预期输出包含：
```
🟥 INFERENCE SERVER (推理服务器)
INFO:     Uvicorn running on http://0.0.0.0:8000
```

⚠️ **重要注意点**：
- 启动脚本会自动选择 checkpoint（默认 pi05_libero/my_run/26000/）
- 确保 checkpoint 的 `assets/your_hf_username/libero/norm_stats.json` 存在
- 推理服务需要 3-5 分钟加载模型，请耐心等待

### 启动仿真客户端

**终端 2：启动仿真客户端**

```bash
cd <openpi-project-root>
bash start_simulation_with_env.sh libero_spatial 2
```

参数说明：
- `libero_spatial` - 任务套件（可选值：libero_spatial, libero_object, libero_goal, libero_10, libero_90）
- `2` - 每个任务的试验次数

预期输出：
```
🟦 SIMULATION CLIENT (仿真客户端)
Task 1/10: pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate
  Trial 1/2: Success
  Trial 2/2: Success
```

---

## ✅ 关键诊断与修复

### 问题 1：推理成功率为 0%

**根本原因**：使用了没有 `norm_stats.json` 的 checkpoint（如 pi05_base_pytorch）

**解决方案**：
```bash
# 复制 norm_stats.json 到 checkpoint 的 assets 目录
mkdir -p data/checkpoints/pi05_base_pytorch/assets/your_hf_username/libero
cp data/checkpoints/pi05_libero/my_run/26000/assets/your_hf_username/libero/norm_stats.json \
   data/checkpoints/pi05_base_pytorch/assets/your_hf_username/libero/
```

**技术细节**：
- `norm_stats.json` 包含输入数据的归一化统计（mean, std, quantiles）
- 推理时需要用它来预处理观测值
- 缺少它会导致模型输入错误，输出无法完成任务

### 问题 2：LIBERO assets 文件缺失

**症状**：`FileNotFoundError: libero_tabletop_base_style.xml`

**解决方案**（已应用）：
```bash
# 复制必要的 assets 文件到 libero 包
LIBERO_PATH=examples/libero/.venv/lib/python3.10/site-packages/libero/libero/assets
mkdir -p $LIBERO_PATH/scenes
mkdir -p $LIBERO_PATH/textures
cp robosuite/models/assets/textures/* $LIBERO_PATH/textures/
# 创建必要的 XML 场景文件
```

---

## 📊 输出位置

### 视频输出

```
/iag_ad_01/ad/dushuyue/openpi/data/libero/videos/
```

视频文件格式：
```
rollout_<task_name>_<success|failure>.mp4
```

**注意**：同一任务的多次运行会覆盖之前的视频（设计如此）

---

## 🎯 成功指标

仿真成功运行的表现：

| 指标 | 预期值 | 说明 |
|------|--------|------|
| **推理服务启动** | < 5 分钟 | 模型加载耗时较长 |
| **仿真连接** | 成功 | WebSocket 握手成功（可忽略初期错误） |
| **视频生成** | 每个任务 1 个 | MP4 格式，约 50-100 KB |
| **成功率** | 5-15% | pi05_libero/26000 的性能 |

---

## ⚠️ 常见问题

### Q: 推理服务一直不启动？
**A**: 
- 检查 checkpoint 是否存在：`ls -lh data/checkpoints/pi05_libero/my_run/26000/model.safetensors`
- 查看日志：最后 50 行中应有 `Running on http://0.0.0.0:8000`
- 等待 5+ 分钟，大模型加载较慢

### Q: 仿真连接失败？
**A**:
- 确保推理服务已启动：`ps aux | grep serve_policy`
- 检查端口是否监听：`netstat -an | grep 8000`
- 如果端口被占用，杀死旧进程：`pkill -f serve_policy`

### Q: 视频没有生成？
**A**:
- 检查仿真是否运行完成（最后一个 mp4 的时间戳）
- 验证输出目录：`ls -lh data/libero/videos/`
- 查看仿真进程：`ps aux | grep main_osmesa`

### Q: 代理连接问题？
**A**:
- 设置正确的代理：`export http_proxy=http://proxy.hs.com:3128`
- 某些网络资源仍可能不可达，这是正常的
- 推理和仿真不需要外网，只需要本地通信

---

## 📋 检查清单

运行前确保完成：

- [ ] Python 3.11 和 3.10 都已安装
- [ ] 推理环境 `.venv` 已创建
- [ ] 仿真环境 `examples/libero/.venv` 已创建
- [ ] `data/checkpoints/pi05_libero/my_run/26000/` 存在
- [ ] `assets/your_hf_username/libero/norm_stats.json` 已复制到对应 checkpoint
- [ ] `examples/libero/.venv/lib/python3.10/site-packages/libero/libero/assets/` 包含必要文件

---

**最后更新**：2026-07-01  
**验证状态**：✅ 推理+仿真完整流程已验证可用  
**成功率**：pi05_libero/26000 在 libero_spatial 上 ~9%
  - Venv: /iag_ad_01/ad/dushuyue/openpi/.venv

INFO: Uvicorn running on http://0.0.0.0:8000
```

**终端 2 - 启动仿真客户端**

```bash
cd <openpi-project-root>
bash start_simulation_with_env.sh libero_spatial 2
```

预期输出：
```
╔════════════════════════════════════════════════════════════╗
║          🟦 SIMULATION CLIENT (仿真客户端)                   ║
╚════════════════════════════════════════════════════════════╝

Connected to server
Starting libero_spatial evaluation
Task 1/10: pick_place_within_drawer
  Trial 1/2: Success=0.95
  Trial 2/2: Success=0.88
...
```

---

## 🎯 核心脚本说明

| 脚本 | Python | 用途 | 首次运行 | 日常使用 |
|------|--------|------|---------|---------|
| `.venv/bin/python` | 3.11 | 推理服务 | 自动安装 | 直接运行 |
| `examples/libero/.venv/bin/python` | 3.10 | 仿真客户端 | 运行 install_sim_env_python310.sh | 直接运行 |
| `start_inference_with_env.sh` | 3.11 | 启动推理 | ✓ | ⭐⭐⭐ 常用 |
| `start_simulation_with_env.sh` | 3.10 | 启动仿真 | ✓ | ⭐⭐⭐ 常用 |
| `install_sim_env_python310.sh` | 3.10 | 仿真环境安装 | ✓ 必需 | 仅故障排查 |
| `verify_inference.sh` | 3.11 | 验证连接 | 故障排查 | ⭐ 调试用 |

---

## ⚙️ 环境变量（关键配置）

### 推理服务器（.venv）
```bash
# 自动设置，无需手动配置
PYTHONPATH=包含 openpi 模块路径
```

### 仿真客户端（examples/libero/.venv）
```bash
# 自动设置在 start_simulation_with_env.sh 中
export MUJOCO_GL="osmesa"              # 使用 CPU 软件渲染
export PYOPENGL_PLATFORM="osmesa"      # OpenGL 后端选择
export PYTHONPATH="...third_party/libero:...openpi-client"

# 可选：自定义连接地址
export SERVE_HOST="127.0.0.1"          # 推理服务器 IP（默认本地）
export SERVE_PORT="8000"               # 推理服务器端口
```

---

## ✅ 验证安装状态

```bash
# 1. 检查推理环境
.venv/bin/python -c "import openpi; print('✓ Inference ready')"

# 2. 检查仿真环境
examples/libero/.venv/bin/python -c "import mujoco, robosuite, libero; print('✓ Simulation ready')"

# 3. 检查系统库
ldconfig -p | grep -i osmesa          # 应该找到 libosmesa6

# 4. 检查 checkpoint
ls -lh data/checkpoints/pi05_libero/my_run/30000/model.safetensors
```

---

## 🛠️ 常见问题

### 问题 1: 仿真无法启动 - OSMesa 库缺失

**症状**
```
ImportError: libOSMesa.so: cannot open shared object file
```

**解决**
```bash
# 安装系统库
sudo apt-get install -y libosmesa6 libosmesa6-dev

# 重新运行
bash start_simulation_with_env.sh libero_spatial 1
```

### 问题 2: robosuite 导入错误 - EGL/OpenGL 问题

**症状**
```
AttributeError: module 'mujoco.egl.egl_ext' has no attribute 'EGL_RED_SIZE'
```

**解决**
```bash
# 重新安装仿真环境（自动修复）
bash install_sim_env_python310.sh

# 或手动清除缓存
rm -rf examples/libero/.venv
bash install_sim_env_python310.sh
```

### 问题 3: 仿真无法连接推理服务

**症状**
```
❌ 连接失败: ws://127.0.0.1:8000
```

**原因和解决**
```bash
# 1. 检查推理服务是否启动
netstat -an | grep 8000            # 应该看到 LISTEN

# 2. 检查防火墙
sudo ufw allow 8000

# 3. 如果推理服务在不同主机，修改 IP
export SERVE_HOST="<推理服务器IP>"
bash start_simulation_with_env.sh libero_spatial 1

# 4. 验证连接
bash verify_inference.sh
```

### 问题 4: 代理或网络问题

**症状**
```
ConnectionError / timeout / SSL error
```

**解决**
```bash
# 检查并设置代理
echo $http_proxy

# 如需代理，在每个终端设置
export http_proxy=http://proxy.hs.com:3128
export https_proxy=http://proxy.hs.com:3128

# 重新启动服务
bash start_inference_with_env.sh
```

---

## 📊 脚本内部逻辑

### 工作流程概览

```
┌─────────────────────────────┐         ┌─────────────────────────────┐
│  推理服务                   │         │  仿真客户端                 │
│  (Inference Server)         │         │  (Simulation Client)        │
├─────────────────────────────┤         ├─────────────────────────────┤
│                             │         │                             │
│ ├─ 加载 checkpoint          │         │ ├─ 初始化 LIBERO           │
│ ├─ 初始化模型 (ViT + T5)   │         │ ├─ 逐步运行任务            │
│ ├─ 启动 WebSocket           │         │ ├─ 采集观测                │
│ │  (ws://0.0.0.0:8000)     │         │ ├─ 发送给推理服务          │
│ └─ 等待仿真客户端连接       │         │ ├─ 执行动作                │
│                             │         │ └─ 保存视频和成功率        │
└─────────────────────────────┘         └─────────────────────────────┘
          ↑                                         ↑
          │          WebSocket (JSON)              │
          │     观测数据 ──────────→ 推理          │
          │     ←────────── 动作预测               │
          └──────────────────────────────────────┘
```


### 推理和仿真的通信流程

#### 1. 推理服务启动流程

**脚本**: `start_inference_with_env.sh`

```bash
STEP 1: 激活推理 venv (.venv - Python 3.11)
        └─ source .venv/bin/activate

STEP 2: 设置环境变量
        ├─ OPENPI_DATA_ROOT = data/
        ├─ HF_HOME = data/.hf_cache
        └─ 离线模式: HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1

STEP 3: 检查 checkpoint 存在
        └─ data/checkpoints/pi05_libero/my_run/30000/model.safetensors

STEP 4: 启动推理服务
        └─ python scripts/serve_policy.py
           ├─ 加载模型参数 (1.3B 参数)
           ├─ 初始化推理引擎
           └─ 监听 WebSocket 端口 8000
           
STEP 5: 等待仿真客户端连接
        └─ 首次连接时推理服务准备就绪
```

**输出示例**:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

#### 2. 仿真客户端启动流程

**脚本**: `start_simulation_with_env.sh`

```bash
STEP 1: 激活仿真 venv (examples/libero/.venv - Python 3.10)
        └─ source examples/libero/.venv/bin/activate

STEP 2: 设置 OpenGL 渲染环境
        ├─ MUJOCO_GL=osmesa (软件渲染，支持 headless)
        ├─ PYOPENGL_PLATFORM=osmesa
        └─ 禁用 EGL (GPU 渲染)

STEP 3: 验证推理服务连接
        └─ timeout 2 bash -c "echo > /dev/tcp/127.0.0.1/8000"

STEP 4: 启动仿真评估
        └─ python examples/libero/main_osmesa.py
           ├─ 初始化 LIBERO benchmark
           └─ 逐任务进行试验
```

#### 3. 单次试验的完整流程

```
┌─ 对于 LIBERO 任务套件中的每个任务
│
└─ 对于该任务的每次试验 (num_trials_per_task = 2)
   │
   ├─ 重置仿真环境
   │  ├─ 设置初始场景
   │  ├─ 放置物体和机械臂
   │  └─ 让物体稳定 (10 步)
   │
   ├─ 在每个时间步 (max_steps = 220)
   │  │
   │  ├─ 从仿真获取观测
   │  │  ├─ agentview_image (256x256)      # 俯视图
   │  │  ├─ robot0_eye_in_hand_image       # 手腕摄像头
   │  │  ├─ robot0_eef_pos (3D 坐标)      # 末端执行器位置
   │  │  ├─ robot0_eef_quat (四元数)      # 末端执行器方向
   │  │  └─ robot0_gripper_qpos (夹爪)   # 夹爪状态
   │  │
   │  ├─ 预处理观测
   │  │  ├─ 旋转 180°（训练数据格式一致）
   │  │  ├─ 缩放到 224x224
   │  │  └─ 转换到 uint8 格式
   │  │
   │  ├─ 每 5 步重新规划 (replan_steps=5)
   │  │  │
   │  │  ├─ 构建观测字典
   │  │  │  └─ {
   │  │  │      "observation/image": 224x224 图像,
   │  │  │      "observation/wrist_image": 224x224 手腕图像,
   │  │  │      "observation/state": 7D 状态,
   │  │  │      "prompt": 任务描述文本
   │  │  │    }
   │  │  │
   │  │  ├─ WebSocket 发送给推理服务
   │  │  │  └─ POST ws://127.0.0.1:8000/infer
   │  │  │
   │  │  ├─ 推理服务处理
   │  │  │  ├─ 提取图像特征 (ViT)
   │  │  │  ├─ 提取文本特征 (T5 encoder)
   │  │  │  ├─ 融合多模态信息
   │  │  │  └─ 预测下一步动作序列 (20 步)
   │  │  │
   │  │  └─ 返回动作 (20x7 数组)
   │  │     ├─ 前 6 维: 关节角速度
   │  │     └─ 第 7 维: 夹爪命令
   │  │
   │  ├─ 执行动作的前 5 步
   │  │  └─ 每步发送一个动作给仿真环境
   │  │
   │  └─ 检查任务是否完成 (done=True)
   │     └─ 如果完成，成功次数 +1，退出时间步循环
   │
   ├─ 生成视频
   │  └─ 保存为 rollout_<task>_<success|failure>.mp4
   │
   └─ 记录结果
      └─ 成功率、试验数、视频位置
```

### 数据流向详解

#### 观测数据 (仿真 → 推理)

```python
# 从 LIBERO 获取的原始观测
obs = {
    "agentview_image": np.ndarray (256, 256, 3),        # 俯视图
    "robot0_eye_in_hand_image": np.ndarray (256, 256, 3), # 手腕图
    "robot0_eef_pos": np.array ([x, y, z]),             # 末端位置
    "robot0_eef_quat": np.array ([qx, qy, qz, qw]),     # 末端四元数
    "robot0_gripper_qpos": np.array ([gripper_state])   # 夹爪
}

# 预处理后发送给推理服务
element = {
    "observation/image": uint8[224, 224, 3],            # 缩放 + 旋转
    "observation/wrist_image": uint8[224, 224, 3],      # 缩放 + 旋转
    "observation/state": np.array([                      # 7D 状态向量
        x, y, z,              # 末端位置 (3)
        rx, ry, rz,           # 末端旋转轴角 (3)
        gripper               # 夹爪 (1)
    ]),
    "prompt": "pick up the black bowl from table..."     # 任务文本
}
```

#### 动作数据 (推理 → 仿真)

```python
# 推理服务输出的动作
action_chunk = np.ndarray (20, 7)  # 预测未来 20 步

# 每一行是一个动作
action_chunk[0] = [dq0, dq1, dq2, dq3, dq4, dq5, gripper_cmd]
# ├─ dq0-dq5: 关节角速度 (rad/s)
# └─ gripper_cmd: -1 (开) 或 1 (关)

# 仿真执行前 5 步的动作
for i in range(5):
    action = action_chunk[i]  # 取出当前动作
    obs, reward, done, info = env.step(action.tolist())
    if done:
        success_count += 1
        break
```

### 性能指标

| 指标 | 值 | 说明 |
|------|-----|------|
| **推理延迟** | ~500ms | 单次推理的耗时 |
| **模型大小** | 1.3B 参数 | pi05_libero checkpoint |
| **输入分辨率** | 224×224 | 图像输入大小 |
| **预测窗口** | 20 步 | 每次推理预测的动作序列长度 |
| **重规划周期** | 5 步 | 每 5 步重新从推理服务获取动作 |
| **视频帧率** | 10 fps | 输出视频的帧率 |
| **仿真步频** | ~100 fps | 环境步进速度 |

### start_inference_with_env.sh

```
1. 激活主 venv (.venv - Python 3.11)
2. 设置终端标题和提示符为红色（标识：推理）
3. 验证 checkpoint 文件存在
4. 导出环境变量和路径
5. 启动 scripts/serve_policy.py（WebSocket 推理服务）
6. 监听 http://0.0.0.0:8000
```

### start_simulation_with_env.sh

```
1. 激活仿真 venv (examples/libero/.venv - Python 3.10)
2. 设置终端标题和提示符为蓝色（标识：仿真）
3. 设置 MUJOCO_GL=osmesa, PYOPENGL_PLATFORM=osmesa
4. 验证推理服务连接（可选提示）
5. 导出环境变量和路径
6. 启动 examples/libero/main_osmesa.py（仿真客户端）
7. 连接推理服务进行评估
```

---

## 🔧 自定义和扩展

### 修改推理参数

编辑 `start_inference_with_env.sh`，修改 checkpoint 路径：

```bash
PI05_LIBERO_CHECKPOINT="${OPENPI_DATA_ROOT}/checkpoints/pi05_libero/my_run/30000"
# 改成你的 checkpoint
PI05_LIBERO_CHECKPOINT="${OPENPI_DATA_ROOT}/checkpoints/pi05_libero/my_run/YOUR_STEP"
```

或通过环境变量覆盖：

```bash
export PORT=9000
bash start_inference_with_env.sh
```

### 修改仿真参数

编辑 `start_simulation_with_env.sh`，或直接传参：

```bash
# 任务套件选项：libero_spatial, libero_object, libero_goal, libero_10, libero_90
bash start_simulation_with_env.sh libero_object 5

# 修改连接地址
export SERVE_HOST="192.168.1.100"  # 推理服务器 IP
export SERVE_PORT="8000"
bash start_simulation_with_env.sh libero_spatial 2
```

### 修改 main.py 的仿真参数

编辑 `examples/libero/main.py` 中的 `Args` 类：

```python
@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"                  # 推理服务 IP
    port: int = 8000                       # 推理服务端口
    resize_size: int = 224                 # 图像缩放大小
    replan_steps: int = 5                  # 每 N 步重新规划
    task_suite_name: str = "libero_spatial" # 任务套件
    num_steps_wait: int = 10               # 稳定等待步数
    num_trials_per_task: int = 50          # 每个任务的试验次数
    video_out_path: str = "data/libero/videos"  # 视频输出目录
    seed: int = 7                          # 随机种子
```

---

## 📡 真机适配指南

要在真机上运行，需要替换仿真部分：

### 核心改动

1. **替换环境接口** - 替代 `OffScreenRenderEnv`
   ```python
   # 仿真环境
   env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
   
   # 改为真机环境（需要自己实现）
   env = RealRobotInterface(config=robot_config)
   task_description = get_task_description(task_id)
   ```

2. **观测接口** - 从真机获取
   ```python
   # 仿真获取
   obs = env.set_init_state(initial_states[episode_idx])
   
   # 真机获取（需要传感器接口）
   obs = env.get_observation()  # 返回相同格式
   ```

3. **动作执行** - 发送给真机控制器
   ```python
   # 仿真执行
   obs, reward, done, info = env.step(action.tolist())
   
   # 真机执行（需要机械臂接口）
   obs = env.execute_action(action)
   reward = compute_reward(obs, task_id)
   done = check_task_success(obs, task_id)
   ```

### 必需的观测格式

推理服务期望这个格式，真机接口需要返回：

```python
{
    "agentview_image": np.ndarray(256, 256, 3, dtype=uint8),     # 摄像头 RGB
    "robot0_eye_in_hand_image": np.ndarray(256, 256, 3, dtype=uint8), # 手腕摄像头
    "robot0_eef_pos": np.array([x, y, z], dtype=float32),         # 末端位置 (m)
    "robot0_eef_quat": np.array([qx, qy, qz, qw], dtype=float32), # 末端四元数
    "robot0_gripper_qpos": np.array([state], dtype=float32)        # 夹爪位置
}
```

### 必需的动作格式

推理服务输出这个格式，真机控制器需要解释：

```python
# 动作向量 (7D)
action = [dq0, dq1, dq2, dq3, dq4, dq5, gripper_cmd]

# 应用方式（取决于机械臂类型）
# 1. 关节速度控制: 直接设置关节速度 (rad/s)
# 2. 笛卡尔速度控制: 转换为末端速度
# 3. 位置控制: 积分获得目标位置

# 夹爪命令
gripper_cmd = -1  # 打开
gripper_cmd = +1  # 关闭
```

---



1. **总是先启动推理** - 仿真依赖推理服务
   ```bash
   # 终端 1 - 先启动
   bash start_inference_with_env.sh
   
   # 等待看到 "INFO: Uvicorn running..." 后
   # 再在终端 2 启动
   bash start_simulation_with_env.sh libero_spatial 2
   ```

2. **使用推荐脚本** - 脚本包含所有最佳配置，不要手动运行 Python

3. **代理问题排查** - 网络问题 99% 是代理未配置
   ```bash
   echo $http_proxy    # 检查是否设置
   ```

4. **隔离终端** - 推理和仿真各用一个终端，方便查看输出

5. **查看实时输出** - 不要后台运行，看输出判断进度

6. **重新安装时清空缓存**
   ```bash
   rm -rf examples/libero/.venv
   bash install_sim_env_python310.sh
   ```

---

## 🎯 典型工作流

```bash
# ==================== 首次设置 ====================
cd <openpi-project-root>

# 1. 设置代理（如需要）
export http_proxy=http://proxy.hs.com:3128
export https_proxy=http://proxy.hs.com:3128

# 2. 安装推理环境（如还没有）
python3.11 -m venv .venv
source .venv/bin/activate
pip install ...

# 3. 安装仿真环境（首次必需）
bash install_sim_env_python310.sh

# ==================== 每次使用 ====================

# 终端 1
bash start_inference_with_env.sh

# 等待看到 "Uvicorn running on http://0.0.0.0:8000"

# 终端 2（新开一个）
bash start_simulation_with_env.sh libero_spatial 2

# 查看结果
ls -lh data/libero/videos/
```

---

## 📚 相关文档

| 文档 | 何时阅读 |
|------|--------|
| **START_HERE.md** | 第一次使用时 |
| **PROXY_CONFIG_GUIDE.md** | 遇到网络问题时 |
| **TERMINAL_ENV_GUIDE.md** | 不理解环境差异时 |

---

**最后更新**: 2026-07-01  
**版本**: 2.0 - 完整的安装和运行指南  
**状态**: ✅ 完全就绪，经过完整测试验证
