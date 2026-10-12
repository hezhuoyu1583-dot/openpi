# TianYi × PI0.5 × Isaac 适配状态与实施方案

更新时间：2026-09-07。本文核对了已有 Isaac 仿真仓库、OpenPI PI0.5
checkpoint 与当前评测脚本，区分“代码存在”“协议 smoke”和“正式成绩”。

## 结论

已有工作已经把 TianYi2-BrainCo2 的 Isaac 本体、USD、5 个任务、头部相机、
ZMQ 通信、固定 seed 和本地录像 runner 搭好；OpenPI 侧也已有 TianYi stationary
16D 数据适配、PI0.5 后训练模型和 WebSocket 服务基础设施。但是，当前尚无
PI0.5 的 Isaac 闭环 rollout 或官方成功率。还不能把历史 ACT/WALL 运行、MuJoCo
proxy 的 scripted/kinematic 结果称为 PI0.5/Isaac 成绩。

正式评测前的 P0 阻塞是：

1. PI0.5 输出为 16D（双臂 14D + 两个 scalar gripper），Isaac 要求 26D（双臂
   14D + 左右 BrainCo 手各 6D）；没有可靠的 scalar↔hand6 标定就不能安全接入。
2. 后训练 TianYi 是 36 个 household task，Isaac benchmark 是 5 个工业/实验室
   task；若没有五任务的 26D 示教或明确接受 zero-shot，结果是跨任务迁移，不能
   用离线 MAE/RMSE 代替任务成功率。
3. benchmark 本地 runner 只有 practice 模式（固定 timeout、录像、无 scorer），
   OpenPI 专用 policy entry、scheduler、安全层和官方 scoring 均未完成。

## 1. 已找到的 Isaac 工作和证据

仓库：

```text
/iag_ad_01/ad/dushuyue/x-humanoid-vla-simulation-benchmark
```

| 项目 | 证据 | 状态 |
|---|---|---|
| TianYi2-BrainCo2 USD 和关节 | `usds/tianyi_2_6d_force_brainco2_hand_v1/`、`robots/TianYi2_Brainco2.py` | 已有 |
| 任务 | `benchmark.py`、`tasks/TianYi2_Brainco2_tasks/` | 5 个已注册：`ind_task_01/02/03`、`lab_task_01/03` |
| 相机 | `common/robot_config/cfg_TianYi2_Brainco2.toml` | `camera_head` RGB/depth，其他相机默认关闭 |
| 通信 | `common/utils/zmq_utils.py`、`tasks/.../TianYi2_Brainco2_task_base.py` | `task/task_cbd/start/obs/action/reset` 已有 |
| 时序防护 | `ZmqReceiver.is_old_action()` | episode/step 旧动作过滤已实现 |
| 可复现采样 | `tasks/task_manager.py`、`benchmark.py --seed` | 固定 seed、预选 USD 已有 |
| 录像 | `tools/run_sim_local.sh`、`common/utils/video_recorder.py` | 本地 MP4 practice 输出 |
| policy 插件 | `tools/policies/runner.py` | ACT、ACT-v1、WALL；无 PI0.5 |
| IK/proxy | `mujoco_proxy/`、`docs/wall_tianyi_deployment.md` | 工程辅助，不是官方 scorer |
| Isaac scorer | `benchmark.py::run_benchmark()` | 明确为 practice/no scoring |

历史日志：

```text
x-humanoid-vla-simulation-benchmark/logs/isaac_sim/
```

2026-07-27 日志证明过 Isaac 启动、USD/相机初始化和 ZMQ warm-up；没有证明
PI0.5 动作执行或成功。历史还出现两类问题：相机首帧 `None` 导致旧版
`sim_cam.py` 的 `AttributeError`，以及 H20 上 Vulkan/RTX renderer 崩溃。当前
`get_rgb()` 已对首帧空值保留预分配 buffer；`tools/run_sim_local.sh` 仍会主动
拒绝 H20/H100/A100 等无 RTX RT Core 节点。正式运行应换 RTX/RTX PRO/L40(S)/A40/
Ada/Blackwell RTX 节点。

当前配置路径为：

```text
/iag_ad_01/ad/dushuyue/envs/isaacsim-5.1/bin/python
```

该 executable 存在；设置 `OMNI_KIT_ACCEPT_EULA=YES` 后 `import isaacsim, omni`
预检通过。但本机 GPU 是 NVIDIA H20，不能据此宣称可完成相机 rollout。

benchmark 工作区当前没有发现已挂载的 `data/` challenge HDF5。README 所说的
“每任务约 200 条 26D challenge 数据”需另行下载/挂载并核对版本。

## 2. OpenPI PI0.5 当前状态

OpenPI：

```text
/iag_ad_01/ad/dushuyue/openpi
```

后训练 checkpoint：

```text
data/checkpoints/pi05_tianyi_finetune/tianyi_stationary_pi05_2x8/30000
```

base 权重：

```text
data/checkpoints/pi05_base_pytorch
```

配置/实现已经具备：

- PI0.5 `action_dim=32`、`action_horizon=10`；有效 TianYi stationary action 为
  前 16D，后 16D padding，并由逐样本 mask 排除 loss。
- layout：`[left_arm(7), right_arm(7), left_gripper_scalar(1), right_gripper_scalar(1)]`。
- 首版是 absolute joint position；Isaac `update_command()` 也是 joint position target，
  不应重复减 state 或重复加 state。
- 数据是 LeRobot v2.1、15 fps、单 `camera_top`；映射到 `base_0_rgb`，两个 wrist
  用零图但 `mask=False`。
- `scripts/serve_policy.py`、`create_trained_policy()`、WebSocket server/client 已有。

现有 OpenPI 服务命令（仅启动模型服务，不包含 Isaac adapter）：

```bash
cd /iag_ad_01/ad/dushuyue/openpi
export OPENPI_DATA_HOME=/iag_ad_01/ad/dushuyue/openpi/data/.openpi_cache
export HF_HOME=/iag_ad_01/ad/dushuyue/openpi/data/.hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python scripts/serve_policy.py \
  --policy.config=pi05_tianyi_finetune \
  --policy.dir=/iag_ad_01/ad/dushuyue/openpi/data/checkpoints/pi05_tianyi_finetune/tianyi_stationary_pi05_2x8/30000 \
  --port=8000
```

## 3. 动作/状态/相机接口与映射

PI0.5 模型张量（32D）：

```text
0:7    left arm absolute joint position
7:14   right arm absolute joint position
14:15  left scalar gripper
15:16  right scalar gripper
16:32  padding（loss mask=False）
```

Isaac ZMQ action（26D）：

```text
[left_arm(7), left_hand(6), right_arm(7), right_hand(6)]
```

必须显式实现以下映射，不能靠隐式 `concatenate` 猜顺序：

```text
src = [L_arm7, R_arm7, L_scalar, R_scalar]
dst[0:7]   = src[0:7]
dst[7:13]  = left_hand_mapper(src[14])
dst[13:20] = src[7:14]
dst[20:26] = right_hand_mapper(src[15])
```

反向 observation 也必须实现：

```text
[L_arm7, L_hand6, R_arm7, R_hand6]
 -> [L_arm7, R_arm7, L_scalar, R_scalar]
```

`hand6 -> scalar` 不能使用未经标定的六角度平均。需要 BrainCo/TianYi 提供左右
手 `q_open/q_close`、scalar 范围/方向、单位和限位；当前仓库只有安全 home pose：

```text
left  = [1.571, -0.093, -0.172, -0.172, -0.173, -0.173]
right = [1.571,  0.079, -0.179, -0.175, -0.178, -0.177]
```

这些是 open/home fallback，不是 close synergy。没有标定时只能做协议 smoke，
不能作为正式成绩。

## 4. 尚缺少部分（按优先级）

### P0：正式 PI0.5 Isaac 前必须完成

1. **PI0.5 policy entry/bridge**：在 benchmark 增加 `pi05_tianyi`，或新增
   `tools/tianyi_pi05_bridge.py`；通过 `openpi_client.websocket_client_policy.WebsocketClientPolicy`
   请求 `[10,16]` action chunk，再发送 ZMQ 26D。
2. **BrainCo 标定**：scalar↔hand6 的双向、左右独立、限位、单位和开合方向；标定文件
   和来源版本随实验归档。
3. **任务/数据对齐**：确认云端是否就是这 5 个任务；若要公平比较，应提供对应
   26D demonstrations 或把结果明确标记为跨任务 zero-shot。
4. **官方 scorer**：取得组织方 success predicate/API；本地若自行实现，必须标成
   `internal`，不能写 official success rate。

### P1：可以并行

1. **调度**：训练 15 fps，Isaac physics 120 Hz、每 4 physics step 发布一次 obs（约
   30 Hz），PI0.5 horizon=10。首版每个模型 action 保持两个 obs（等效 15 Hz），
   每 2–5 个动作 receding-horizon 重规划；再比较 hold、线性插值、30 Hz 重采样。
2. **安全层**：finite/shape/order 检查、NaN/Inf 拒绝、USD joint limit、单步位置/速度/
   加速度限制、hand limit、超时 hold、reset 清 queue、clip/延迟/丢帧日志。
3. **相机域**：确认 RGB 而非 BGR、分辨率、裁剪、内参/视场、曝光和坐标系。不要把
   `camera_head` 复制成 wrist 视角；保持 wrist mask=False。
4. **可观测性**：记录 episode/step、WebSocket latency、inference p50/p95、queue depth、
   action clipping、NaN/timeout、旧 step 丢弃和每关节最大变化。

### P2：移动平台/混合训练接口

OpenPI 已预留 mobile 25D：stationary 前 16D 加 `chassis_twist[6]`、`head_position[3]`，
再 padding 到 32D；但当前 Isaac robot 是固定底座 26D，没有 mobile chassis/head 执行器。
mobile 数据还有两个任务 prompt 错误，须先修正任务元数据；stationary/mobile 要分别
计算 norm stats 并使用 embodiment-aware sampler，不能直接合并统计量。

## 5. 推荐架构

```text
Isaac Sim 5.1
  ZMQ obs: camera_head + [L_arm7,L_hand6,R_arm7,R_hand6]
        │
        ▼
TianYi PI0.5 bridge（独立进程）
  26D obs -> arm14 + calibrated scalar2
  camera_head -> observation.images.camera_top
  task name -> 固定 English prompt
  WebSocket -> OpenPI PI0.5 server
  [10,16] absolute chunk -> queue/scheduler
  scalar -> calibrated BrainCo hand6
  safety limiter -> ZMQ 26D action
        │
        ▼
Isaac position controller
```

独立 bridge 的好处是 Isaac Python 和 OpenPI/Torch/transformers 依赖隔离，也便于
在云端把 policy server 放到另一台机器。ZMQ 只暴露给 bridge，WebSocket 端口按内网
访问控制；任务 prompt 必须与训练/评测约定一致。

## 6. 分阶段实施与验收

### 阶段 0：RTX 节点只读预检（约半天）

- `OMNI_KIT_ACCEPT_EULA=YES`，import Isaac、加载 USD、读取一帧 `camera_head`。
- 固定 `seed=0`，每任务 1 episode，确认 RGB shape、26D 字段、step_id 单调。
- 暂不调用模型，回传当前关节/手部目标，验证不抖动、不越限。

验收：每任务至少 100 个连续 obs、无 `None` 图像、无 renderer crash、无旧 action。

### 阶段 1：纯 Python adapter/mock（约 1 天）

实现建议：

```text
x-humanoid-vla-simulation-benchmark/tools/policies/pi05_tianyi_policy.py
x-humanoid-vla-simulation-benchmark/tools/tianyi_pi05_bridge.py
```

测试 16D→26D 左右单关节脉冲、左右交换、绝对动作 round-trip、标定 hand round-trip、
错误 shape/NaN/Inf、joint clip，以及 mock WebSocket `[10,16]` chunk 的消费/重规划。

### 阶段 2：Isaac no-motion smoke（约半天）

用 base 或微调 checkpoint 接收图像和状态，但回传 current-state action；检查通信、
reset、视频、延迟和 shape。此阶段只能证明桥接，不证明任务成功。

### 阶段 3：低幅动作（约 1 天）

输出乘小 gain，先只放开一个 arm joint，随后双臂，最后 hand synergy；每步启用限位、
速度/加速度约束和超时 hold。任一振荡/越限立即退回 no-motion。

### 阶段 4：本地五任务 rollout（RTX 节点）

扩展 runner 注册 `pi05_tianyi` 后，使用统一 seed、scheduler、bridge 分别跑 base 和
fine-tuned checkpoint，例如：

```bash
cd /iag_ad_01/ad/dushuyue/x-humanoid-vla-simulation-benchmark
export OMNI_KIT_ACCEPT_EULA=YES
export OPENPI_PI05_WS_URL=ws://<policy-host>:8000
bash tools/run_sim_local.sh --task ind_task_01 --policy pi05_tianyi \
  --loop 2 --seed 0 --timeout 120 --headless
```

当前命令会因 `pi05_tianyi` 尚未注册而失败；这是待办，不是已有 PI0.5 结果。录像
默认在 `logs/task_videos/`，启动日志在 `logs/sim_local/<timestamp>/`。

### 阶段 5：云端正式评测

只有在手部标定、动作顺序/单位/限位、scorer、seed、timeout、reset 和 bridge 版本均
归档后提交。base 与 fine-tuned 必须使用完全相同的 adapter/scheduler/seed，并分别
报告 offline teacher-forced、internal Isaac practice、official cloud score。

## 7. 结果目录与指标

建议在 OpenPI 下保存每次 run，不覆盖历史：

```text
openpi/data/eval/isaac_pi05/<run_id>/
  manifest.json                 # checkpoint/hash/config/seed
  adapter_config.json           # order/calibration/limits/scheduler
  episodes.csv                  # task,episode,seed,status,success(optional)
  bridge_metrics.jsonl          # latency, clips, timeout, dropped steps
  videos/<task>/episode_*.mp4
  logs/benchmark.log
  logs/bridge.log
  logs/policy_server.log
```

应至少报告：每 task/seed 的 official success（仅 scorer 提供时）、internal predicate
success（注明版本）、timeout/reset/error rate、inference p50/p95、action clipping rate、
NaN/Inf 和旧 step 数。离线 arm MAE/RMSE 只反映 teacher-forced action 误差，不能替代
闭环成功率。

## 8. 最终状态矩阵

```text
Isaac USD/本体/五任务/相机/ZMQ/固定 seed/视频       已有
ACT/WALL practice adapter                           已有
PI0.5 TianYi stationary checkpoint                  已有
PI0.5 WebSocket 服务基础设施                        已有
PI0.5 TianYi Isaac policy entry                     缺失
16D↔26D BrainCo hand 标定/双向映射                  缺失（P0）
15↔30 Hz scheduler + PI0.5 safety layer             缺失（P1）
五任务 PI0.5 本地 rollout                            尚未完成
Isaac 官方 scorer/云端 success rate                 尚未获得
```

当前最准确的项目表述是：

> 已有可用的 TianYi2 Isaac benchmark 和 PI0.5 TianYi 离线后训练模型，正在补齐
> 16D→26D BrainCo 控制适配；尚无 PI0.5 Isaac 闭环成功率或正式 rollout 结论。
