# Tianyi 数据适配 pi0.5 后训练与云端评测方案

## 1. 核查结论

### 1.1 原始数据

原始数据位于：

```text
/iag_ad_01/ad/xiejinxia/data/tianyi
/iag_ad_01/ad/xiejinxia/data/tianyi-eval
```

实际统计：

| 数据集 | Episode 数 | Task 数 | Frame 数 |
|---|---:|---:|---:|
| `tianyi` | 6415 | 36 | 1,012,390 |
| `tianyi-eval` | 730 | 36 | 116,665 |

已对全部 7145 个 HDF5 做 schema 扫描。所有 episode 的相机、master、puppet 数据长度一致，没有发现字段维度异常。训练集与评测集的 `(task, episode_id)` 没有重叠。

原始 HDF5 的有效控制字段为：

```text
arm_left_position_align/data           (T, 7)
arm_right_position_align/data          (T, 7)
end_effector_left_position_align/data  (T,)
end_effector_right_position_align/data (T,)
```

因此原始 Tianyi stationary 数据为 16D：

```text
[left_arm(7), right_arm(7), left_gripper(1), right_gripper(1)]
```

原始数据不是 26D。两个 end-effector 字段是标量夹爪值，不是 6D 灵巧手关节。

已有转换脚本：

```text
/iag_ad_01/ad/xiejinxia/code/test/RLinf/toolkits/lerobot/
  convert_tianyi_hdf5_to_lerobot.py
```

其默认定义为：

```text
state  = puppet
action = master
```

state/action 均为 absolute position。已有 LeRobot 数据是 v2.1、15 fps、单路 `camera_top`、16D state/action。

### 1.2 Tianyi-mobile 原始数据

移动平台数据位于：

```text
/iag_ad_01/ad/xiejinxia/data/tianyi-mobile
/iag_ad_01/ad/xiejinxia/data/tianyi-mobile-eval
```

全量扫描结果：

| 数据集 | HDF5 Episode 数 | 原始任务目录数 | LeRobot Task 数 | Frame 数 |
|---|---:|---:|---:|---:|
| `tianyi-mobile` | 1598 | 10 | 9 | 384,338 |
| `tianyi-mobile-eval` | 181 | 10 | 9 | 44,626 |

全部 1779 个 HDF5 的字段维度和流长度一致，train/eval 的 `(task_folder, episode_id)` 没有重叠。移动平台为 25D：

```text
0:7    left arm
7:14   right arm
14:15  left scalar gripper
15:16  right scalar gripper
16:22  chassis twist [vx, vy, vz, wx, wy, wz]
22:25  head position
```

当前数据中 chassis twist 仅第 0、5 维变化，其余四维始终为零；chassis/head 的 master 与 puppet 数组在全部 episode 中逐字节一致。因此 chassis 6D 应保留为已有速度指令/状态语义，不能套用关节 absolute-to-delta 变换。双臂 14D 可单独进行 delta ablation，夹爪、底盘和头部保持原语义。

现有移动平台 LeRobot 数据也是 v2.1、15 fps、单路 `camera_top`、25D state/action，图像为 720x1280。工作区另有 v3 转换：

```text
/iag_ad_01/ad/dushuyue/data/tianyi-mobile_lerobot_v3
/iag_ad_01/ad/dushuyue/data/tianyi-mobile-eval_lerobot_v3
```

当前 OpenPI pinned loader 应直接使用 xiejinxia 目录下的 v2.1 版本，不能直接切到上述 v3 目录。

移动数据存在 P0 级语言元数据问题：10 个原始任务目录被压成 9 条 LeRobot task string，其中两类 HDF5 指令明显错误：

| 原始任务目录 | HDF5 中的错误指令 | 建议修正语义 |
|---|---|---|
| `place_storage_box_into_left_shelf_container` | `take the circuit breaker switch from the desktop and place it on the shelf` | `place the storage box into the left shelf container` |
| `stack_pink_ikea_bowl_on_shelf_middle_layer` | `grab a starfruit from the basket on the right and place it in the shelf on the left` | `stack the pink IKEA bowl on the middle shelf` |

移动或混合训练前必须由任务负责人确认最终英文 prompt，建立显式 `task_folder -> corrected_prompt` 映射并重新生成 `tasks.jsonl`、episode task index 及相关元数据。不能根据错误 prompt 做全局字符串替换，因为相同字符串可能属于另一个真实任务。

### 1.3 Isaac 评测动作空间

当前仿真仓库：

```text
/iag_ad_01/ad/dushuyue/x-humanoid-vla-simulation-benchmark
```

Isaac TianYi2-BraInCo 接口使用 26D：

```text
[left_arm(7), left_hand(6), right_arm(7), right_hand(6)]
```

原始 16D 与仿真 26D 不是同一 action schema，不能直接连接。映射关系应为：

```text
dst[0:7]   = src[0:7]
dst[7:13]  = left_hand_mapper(src[14])
dst[13:20] = src[7:14]
dst[20:26] = right_hand_mapper(src[15])
```

目前仓库只有 BrainCo 手部安全 home pose，没有正式的 scalar-to-six close synergy 标定。正式评测前需要取得官方标定，或使用仿真提供的 26D 示教数据重新训练。

## 2. LeRobot 与 OpenPI 兼容性

当前 OpenPI 使用本地 pinned LeRobot：

```text
openpi/data/lerobot-0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
```

其 Python 包版本是 0.3.3，但数据格式常量是：

```text
CODEBASE_VERSION = v2.1
```

四套 xiejinxia LeRobot 数据的 `info.json` 都声明为 v2.1，schema 与当前 OpenPI 相符，但 metadata 完整性不同：

| 数据目录 | `episodes_stats.jsonl` | 当前 pinned loader |
|---|---|---|
| `tianyi/lerobot` | 缺失 | 不能直接加载 |
| `tianyi-eval/lerobot` | 缺失 | 不能直接加载 |
| `tianyi-mobile/lerobot` | 存在 | 可加载 |
| `tianyi-mobile-eval/lerobot` | 存在 | 可加载 |

缺失时，原 LeRobot 构造器会把本地目录误判为不完整并访问 Hugging Face。OpenPI 现已增加 fail-fast 校验，明确报出缺失文件而不联网。由于原始目录可能没有写权限，已提供 overlay 工具：只复制小型 `meta/` 文件，从 parquet 生成 `episodes_stats.jsonl`，并将大型 `data/`、`videos/` 建为软链接；不会写入或改动 xiejinxia 原目录。

已在本机生成：

```text
openpi/data/tianyi_lerobot_v21_overlay
openpi/data/tianyi-eval_lerobot_v21_overlay
```

如需重建 overlay：

```bash
python scripts/create_tianyi_lerobot_overlay.py \
  --src /iag_ad_01/ad/xiejinxia/data/tianyi/lerobot \
  --dst /iag_ad_01/ad/dushuyue/openpi/data/tianyi_lerobot_v21_overlay

python scripts/create_tianyi_lerobot_overlay.py \
  --src /iag_ad_01/ad/xiejinxia/data/tianyi-eval/lerobot \
  --dst /iag_ad_01/ad/dushuyue/openpi/data/tianyi-eval_lerobot_v21_overlay
```

目标目录默认已存在就拒绝覆盖；`--overwrite` 只允许删除并重建 overlay 目录，不会触碰 source。

如果“LeRobot v3”指数据集 `codebase_version=v3.x`，当前 loader 不能保证直接兼容；应转换为 v2.1，或同步升级 LeRobot 依赖和 OpenPI loader 后重新验证。

当前适配已在 `DataConfig` 中加入可选 `dataset_root`，并同时传给 `LeRobotDatasetMetadata` 和 `LeRobotDataset`。因此无需依赖全局环境变量，当前训练配置直接使用：

```text
repo_id     = tianyi
dataset_root = /iag_ad_01/ad/dushuyue/openpi/data/tianyi_lerobot_v21_overlay
```

离线评测应另建只读配置并指向：

```text
/iag_ad_01/ad/dushuyue/openpi/data/tianyi-eval_lerobot_v21_overlay
```

未来移动平台对应根目录分别是 `tianyi-mobile/lerobot` 和 `tianyi-mobile-eval/lerobot`，但当前 `pi05_tianyi_finetune` 没有引用这两个目录。

## 3. 动作空间方案

### 3.1 推荐路线 A：训练 16D，部署映射到 26D

模型有效动作：

```text
[L_arm7, R_arm7, L_gripper1, R_gripper1]
```

部署 adapter 再将两个 scalar gripper 转为左右各 6 个 BrainCo 手指关节。

优点：不伪造训练标签，保留原始数据语义。缺点：依赖可靠的手部标定。

### 3.2 路线 B：标定后生成 26D 训练数据

在数据转换阶段根据官方手部 synergy 标定，把两个 scalar 展开为左右各 6D，再直接训练 Isaac 需要的 26D 顺序。

优点：训练输出和仿真接口统一。缺点：标签质量完全依赖手部标定，没有官方映射时不应作为正式方案。

### 3.3 统一 32D 本体布局与混合训练接口

stationary/mobile 共用前 16 维，移动底盘和头部追加在后面，模型维度保持 checkpoint 的 32D：

```text
0:7    left arm
7:14   right arm
14:15  left scalar gripper
15:16  right scalar gripper
16:22  mobile chassis twist
22:25  mobile head position
25:32  reserved padding
```

有效维度 mask 为：

```text
stationary: [True x 16, False x 16]
mobile:     [True x 25, False x 7]
```

`TianyiInputs` 已按样本输出 `action_dim_mask`，JAX 和 PyTorch loss 已按该 mask 对有效维度求均值；`TianyiOutputs` 根据本体截取 16D 或 25D。这样保留了未来同 batch 混合本体的 loss 接口。

当前只注册 `pi05_tianyi_finetune`，其 embodiment 固定为 `STATIONARY`，数据根目录指向 OpenPI 下的 `/iag_ad_01/ad/dushuyue/openpi/data/tianyi_lerobot_v21_overlay`。未来真正混合训练仍需增加多 repo/weighted sampler、每个 repo 独立 norm stats、逐样本 embodiment 标识，并先修正 mobile prompt；不能把两套统计量合并后直接归一化。

### 3.4 绝对值与相对值

OpenPI 模型本身不限定 absolute 或 delta。动作语义由 transform 决定：

```text
DeltaActions:    action -= current_state
AbsoluteActions: action += current_state
```

Tianyi 首版推荐使用 absolute joint position：

```text
训练 target：absolute
推理输出：absolute
Isaac 控制：joint position target
```

原因是原始 puppet/master 和 Isaac `update_command()` 都是位置语义，可避免重复减 state 或重复加 state。

后续再做 ablation：双臂 14D 使用 delta，gripper/hand 保持 absolute。使用 delta 前必须完成 transform round-trip 测试。

## 4. pi0.5 Base Checkpoint

本地 checkpoint：

```text
/iag_ad_01/ad/dushuyue/openpi/data/checkpoints/pi05_base_pytorch
```

配置为：

```json
{
  "action_dim": 32,
  "action_horizon": 10,
  "paligemma_variant": "gemma_2b",
  "action_expert_variant": "gemma_300m",
  "precision": "bfloat16"
}
```

该目录可作为 PyTorch pi0.5 base 初始化来源，正式文件是 `model.safetensors`。`model.safetensors.bak` 不应作为默认权重。

必须保持模型 `action_dim=32`，否则 action input/output projection 与 checkpoint 形状不匹配：

```text
16D 路线：[16D valid action, 16D padding]
25D mobile：[25D valid action, 7D padding]
26D Isaac：[26D deployment action, 仅用于仿真 adapter 或另行训练]
```

现已增加逐样本 `action_dim_mask`，JAX 与 PyTorch 训练都只对有效动作维度计算 loss。stationary 的 16 个 padding 维度及 mobile 的 7 个 padding 维度不进入 loss。

checkpoint 自带的 norm stats 不能直接视为 Tianyi stats。当前已将 stationary train overlay 的全量 `meta/stats.json` 转成 OpenPI 格式，写入 `assets/pi05_tianyi_finetune/tianyi/norm_stats.json`；不能混入 `tianyi-eval`。若原始 stats 更新，应重新转换或用 `scripts/compute_norm_stats.py` 重算。

该 checkpoint 是 PyTorch safetensors，应优先使用 `scripts/train_pytorch.py` 和 `pytorch_weight_path`。JAX trainer 需要 Orbax params，不能直接把该 safetensors 当作 JAX checkpoint。

## 5. SFT 与超参数

当前训练行为：

- JAX 默认 `freeze_filter=nnx.Nothing`，即全参数微调。
- PyTorch optimizer 直接使用 `model.parameters()`，也是全参数微调，且当前没有实现基于 `freeze_filter` 的冻结。

如果要采用 LoRA、只训练 action expert 或冻结视觉塔，需要显式设置 `requires_grad=False`，并让 optimizer 只接收可训练参数。

DROID 的参数可以作为先验，不能直接复用为最终值：

```text
pi05_full_droid_finetune:
  global batch = 256
  peak lr = 5e-5
  warmup = 1000

pi05_droid_finetune:
  global batch = 32
```

推荐分阶段确定：

```text
Smoke test:
  batch size = 1/2/4/8
  lr = 1e-5、3e-5、5e-5
  steps = 100～500

小规模训练:
  global batch = 16 或 32
  peak lr = 3e-5 或 5e-5
  warmup = 500～1000
  steps = 2k～5k

正式训练:
  global batch = 32 起步
  peak lr = 5e-5 作为初始先验
  steps = 20k～30k
```

H20 上先测单卡 batch，再使用多卡 DDP 达到 global batch。当前 PyTorch trainer 不做 gradient accumulation；2 机 16 卡时 global batch=32 即每卡 batch=2。建议启用 bf16、gradient checkpointing 和 gradient clipping。

训练量计算：stationary Tianyi 训练集有 1,012,390 frames；global batch=32 时每 epoch 约 31,637 steps，因此 30,000 steps 覆盖约 0.948 epoch。若改为 batch=16，则只覆盖约 0.474 epoch。建议先以 batch=32、30k steps 作为约一轮 baseline，只有评测显示欠拟合时再延长到 60k。


PyTorch checkpoint 清理默认 `max_checkpoints=5`：每次新 checkpoint 原子落盘后，只保留 step 最大的 5 个数字目录，`tmp_*` 不参与清理。可通过 `--max-checkpoints=None` 关闭。

Smoke test 必须验证：

- 32D base 权重加载成功；
- transform 后 state/action 为 32D；
- action horizon 与 checkpoint/config 一致；
- norm/unnorm 和 absolute/delta round-trip；
- loss、gradient 和推理输出均为 finite；
- padding 维度不参与 loss；
- 反归一化动作满足 Isaac joint limit 和速度限制。

## 6. Tianyi Transform

已实现：

```text
openpi/src/openpi/policies/tianyi_policy.py
openpi/src/openpi/training/config.py::LeRobotTianyiDataConfig
openpi/src/openpi/training/config.py::pi05_tianyi_finetune
```

移动平台布局还交叉参考了已有实现：

```text
/iag_ad_01/ad/xiejinxia/code/test/RLinf/rlinf/data/datasets/dreamzero/data_transforms/tianyi_mobile.py
/iag_ad_01/ad/dushuyue/giga-world-policy/configs/giga_world_policy_0_5_tianyi_mobile_singleview_32d.py
/iag_ad_01/ad/dushuyue/giga-world-policy/TIANYI_MOBILE_32D_TRAINING.md
```

它们同样采用单顶视相机和 32D padding，但 OpenPI 适配沿用自己的 transform/normalization API，不直接复制 DreamZero/GWP 的 tokenizer 或 modality pipeline。

输入流程：

```text
LeRobot sample
  -> RepackTransform
  -> PromptFromLeRobotTask
  -> TianyiInputs
  -> DeltaActions（可选）
  -> Normalize
  -> PadStatesAndActions(32)
  -> ResizeImages(224, 224)
  -> TokenizePrompt(discrete_state_input=True)
```

输出流程：

```text
model action
  -> Unnormalize
  -> AbsoluteActions（仅 delta 训练时）
  -> TianyiOutputs
  -> stationary 16D 或 mobile 25D action
```

可以参考 DROID 的图像解析、prompt 和 transform 组织方式，但不能复用它的 8D action、DroidOutputs 截断、joint velocity 假设或 action 顺序。

另需核查 master/puppet 的 command-response lag。当前转换脚本按同一 frame index 组成 `state=puppet(t)`、`action=master(t)`，但 master 可能领先 puppet。应通过关节轨迹 cross-correlation 估计 lag，并对比：

```text
puppet(t) -> master(t)
puppet(t) -> master(t + lag)
puppet(t) -> puppet(t + 1)
```

## 7. 单头部相机

pi0.5 需要固定的三路图像 key，但允许缺失视角使用 padding mask：

```text
camera_top         -> base_0_rgb       + mask=True
zero image         -> left_wrist_0_rgb + mask=False
zero image         -> right_wrist_0_rgb+ mask=False
```

`mask=False` 会屏蔽对应图像 token，不会把黑图作为有效视角参与 attention。这是 OpenPI DROID/Libero 已有的处理方式，不是临时 workaround。

不应把同一张 top 图复制到三路，因为模型会把它们视作多个真实相机。

预期影响包括：

- 缺失手部接触和局部几何信息；
- 单路 top camera 与预训练多视角存在域差异；
- Tianyi 真实 top camera 与 Isaac `camera_head` 的分辨率、内参和视场不同。

建议比较：

```text
A. top only + wrist mask=False
B. A + camera dropout/遮挡增强
C. A + 冻结视觉 backbone，只训练 action expert/adapter
```

训练和推理必须保持相同的 camera layout 和 mask 语义。

## 8. Isaac 与云端评测

优先复用现有仓库，它已包含：

- Isaac Sim 5.1 TianYi2-BraInCo USD；
- 头部 RGB/depth 相机；
- 26D observation/action；
- 5 个任务；
- ZMQ task/start/obs/action/reset 协议；
- 本地 GUI/headless rollout；
- ACT/WALL 策略插件入口；
- MuJoCo proxy 和 USD-derived numerical IK。

建议新增：

```text
x-humanoid-vla-simulation-benchmark/tools/policies/pi05_tianyi_policy.py
```

云端链路：

```text
Isaac ZMQ
  -> Tianyi observation adapter
  -> OpenPI WebSocket client
  -> cloud pi0.5 policy server
  -> action chunk
  -> 16D-to-26D adapter（路线 A）
  -> safety limiter
  -> Isaac ZMQ action
```

安全层至少包含：

- action shape/order 检查；
- NaN/Inf 拒绝；
- joint position limit；
- 单步速度/加速度限制；
- episode reset 时清空 action queue；
- step_id 和延迟日志；
- 超时后保持当前姿态或安全 home pose。

当前 Tianyi LeRobot 为 15 fps，Isaac observation 约为 30 Hz。首版建议每个 15 Hz 动作保持两个仿真 observation，或对 chunk 做插值；随后比较 hold、interpolation 和 30 Hz 重采样训练。

### 评测任务风险

RoboMIND Tianyi 数据是 36 个 household manipulation tasks，而当前 Isaac benchmark 是：

```text
ind_task_01
ind_task_02
ind_task_03
lab_task_01
lab_task_03
```

两者不是同一任务集合。如果没有 benchmark 五个任务的示教数据，云端评测属于跨任务、跨场景和跨视角迁移，低成功率不能简单归因于模型或 transform bug。

正式提交前应确认：

1. 云端实际评测是否就是这五个任务；
2. 是否允许使用公开的 challenge 训练 HDF5；
3. 最终模型是只用 RoboMIND Tianyi，还是混合 challenge 26D 数据；
4. 手部 scalar-to-six 是否有官方映射；
5. 仿真器期望的推理频率、episode timeout 和 action chunk 执行策略。

## 9. 实施顺序

1. **已完成**：stationary/mobile schema、Tianyi transform、显式本地 `dataset_root`、逐样本 32D action loss mask 和 stationary-only 训练配置。
2. 完成 master/puppet lag 分析和 action replay 验证。
3. 确认使用 16D 部署映射还是标定后的 26D 数据。
4. **已完成**：使用 `scripts/create_tianyi_lerobot_overlay.py` 在 OpenPI 可写目录生成 stationary train/eval overlay；只读取原始目录，不向 `/iag_ad_01/ad/xiejinxia/data` 写入，并从 train overlay 的全量 `meta/stats.json` 生成 OpenPI norm stats。
5. **已完成 100 step extended smoke**：使用 `pi05_base_pytorch`、batch size 1、全量 Tianyi stats，完成前向/反向/optimizer 更新；当前不要加载 `tianyi-mobile`。
6. 在 `tianyi-eval` 上做 absolute/delta、horizon、camera mask 离线对比。
7. 增加 Isaac pi0.5 policy adapter 和 WebSocket/ZMQ bridge。
8. 完成静态协议、单关节、手部开合、reset、限位测试。
9. 本地 Isaac 五任务 rollout 后再提交云端评测。
10. 未来启用 mobile/mixed 前，修正 mobile prompts、分别计算 norm stats，并实现多 repo weighted sampler。

## 10. 当前主要阻塞项

| 优先级 | 问题 | 解决方式 |
|---|---|---|
| P0 | stationary train/eval 缺少 `episodes_stats.jsonl` | 在 OpenPI 可写目录生成 overlay；复制小型 `meta/`，从原始 parquet 生成 stats，并软链接原始 `data/`、`videos/`，不修改 source |
| P0 | 原始 16D 与 Isaac 26D 不一致 | 获取 BrainCo 手部标定，或使用 challenge 26D 示教 |
| P0 | RoboMIND 36 任务与 Isaac 5 任务不一致 | 确认评测规则，并决定是否混入 challenge 数据 |
| P1 | master/puppet 时延未知 | cross-correlation 和 replay 确认监督对齐 |
| P1 | checkpoint stats 不是 Tianyi 专用 | 当前已从 train overlay `meta/stats.json` 转换；原始数据更新后重新转换或重算 OpenPI stats |
| P1 | Paligemma tokenizer | 已缓存并通过实际 tokenization 验证；训练时设置 `OPENPI_DATA_HOME` 指向缓存目录 |
| P1 | mobile 两个任务的语言元数据错误 | 按原始任务目录确认 prompt 并重新生成任务元数据 |
| P1 | 15 Hz 训练与 30 Hz 仿真不一致 | hold/interpolation/重采样 ablation |
| P1 | 真实和仿真头部相机存在域差异 | 对齐裁剪、增强、相机参数和任务画面 |
| P2 | 当前 LeRobot loader 只支持单 repo | 混合训练前实现 weighted multi-repo loader |
| P2 | PyTorch trainer 不支持冻结配置 | 增加 requires_grad/optimizer 参数过滤 |

