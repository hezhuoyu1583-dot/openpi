# DreamZero DROID LeRobot -> pi0.5 finetune

本方案在不修改源数据的前提下，从本地 `pi05_base_pytorch` 对同一份
9:1 episode-level train split 做两个动作空间实验：

- `pi05_droid_lerobot_joint_velocity_finetune`：`action[21:28] + action[12:13]`
- `pi05_droid_lerobot_joint_position_finetune`：`action[14:21] + action[12:13]`，训练前对 7 个 arm 维度做 `DeltaActions`

原始字段来自 LeRobot v2.0：`observation.state=(14,)`、`action=(28,)`。
训练后的模型内部仍保持 pi0.5 要求的 `action_dim=32`，DROID 有效动作是前 8 维。

## 固定路径

```text
源数据：/iag_ad_01/ad/xiejinxia/data/DreamZero-DROID-Data
训练清单：/iag_ad_01/ad/dushuyue/data/dreamzero_droid_lerobot_split_9to1/train_episode_indices.txt
验证清单：/iag_ad_01/ad/dushuyue/data/dreamzero_droid_lerobot_split_9to1/test_episode_indices.txt
pi05 base：/iag_ad_01/ad/dushuyue/openpi/data/checkpoints/pi05_base_pytorch
norm stats：/iag_ad_01/ad/dushuyue/openpi/data/checkpoints/pi05_base/assets/droid
```

训练清单包含 51,997 episodes / 13,267,639 frames；验证清单包含 5,777
episodes / 1,480,878 frames。清单仅由 loader 读取，不会改写源数据的
`meta/info.json`、Parquet 或视频。

## 2 机 16 卡启动

下面沿用之前平台 launcher 的约定：平台在两台 worker 上执行同一条命令，脚本
从 `HOSTNAME=<job>-worker-0/1` 自动推导 rank，并通过共享数据盘发布 master IP。
两台机器必须以完全相同的绝对路径挂载数据、代码和 checkpoint。

如果两个实验同时运行，必须使用不同的通信端口、master 地址文件和 HF Arrow
缓存。更稳妥的做法是同时显式设置 `OPENPI_TORCH_MASTER_ADDR` 为对应实验
worker-0 的内网 IP，并在两台 worker 上使用同一个值。下例固定为：JV 使用端口
`29630`，JP 使用端口 `29631`；`<JV_WORKER0_IP>` 和 `<JP_WORKER0_IP>` 需要
替换为各自 worker-0 日志中的地址。

joint velocity（两台 worker 使用同一条命令）：

```bash
cd /iag_ad_01/ad/dushuyue/openpi
OPENPI_ROOT=/iag_ad_01/ad/dushuyue/openpi \
OPENPI_RUNTIME=/iag_ad_01/ad/dushuyue/openpi/.venv \
OPENPI_DATA_ROOT=/iag_ad_01/ad/dushuyue/openpi/data \
OPENPI_NUM_MACHINES=2 OPENPI_GPUS_PER_MACHINE=8 OPENPI_MAIN_PROCESS_PORT=29630 \
OPENPI_TORCH_MASTER_ADDR=<JV_WORKER0_IP> OPENPI_NCCL_SOCKET_IFNAME=eth1 \
OPENPI_TORCH_MASTER_FILE=/iag_ad_01/ad/dushuyue/openpi/data/.pi05_droid_jv_torch_master \
OPENPI_DATASET_REPO_ID=droid \
OPENPI_DATASET_ROOT=/iag_ad_01/ad/xiejinxia/data/DreamZero-DROID-Data \
OPENPI_EPISODE_INDICES_PATH=/iag_ad_01/ad/dushuyue/data/dreamzero_droid_lerobot_split_9to1/train_episode_indices.txt \
OPENPI_ACTION_SPACE=JOINT_VELOCITY \
PI05_BASE_PYTORCH_PATH=/iag_ad_01/ad/dushuyue/openpi/data/checkpoints/pi05_base_pytorch \
CHECKPOINT_BASE_DIR=/iag_ad_01/ad/dushuyue/openpi/data/checkpoints \
OPENPI_CONFIG_NAME=pi05_droid_lerobot_joint_velocity_finetune \
OPENPI_EXP_NAME=pi05_droid_lerobot_jv_2x8 \
OPENPI_BATCH_SIZE=256 OPENPI_NUM_TRAIN_STEPS=100000 \
OPENPI_SAVE_INTERVAL=5000 OPENPI_MAX_CHECKPOINTS=5 \
OPENPI_VIDEO_BACKEND=pyav OPENPI_NUM_WORKERS=0 \
HF_HOME=/iag_ad_01/ad/dushuyue/openpi/data/.hf_cache \
OPENPI_HF_DATASETS_CACHE=/iag_ad_01/ad/dushuyue/openpi/data/.hf_cache/droid_lerobot_v20_jv \
OPENPI_DATA_HOME=/iag_ad_01/ad/dushuyue/openpi/data/.openpi_cache \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline \
bash /iag_ad_01/ad/dushuyue/openpi/scripts/launch_platform_tianyi_pi05_torchrun.sh
```

worker 1 使用同一条命令；其 hostname 必须匹配 `*-worker-1`，无需手动设置 rank。

joint position（两台 worker 使用同一条命令）：

```bash
cd /iag_ad_01/ad/dushuyue/openpi
OPENPI_ROOT=/iag_ad_01/ad/dushuyue/openpi \
OPENPI_RUNTIME=/iag_ad_01/ad/dushuyue/openpi/.venv \
OPENPI_DATA_ROOT=/iag_ad_01/ad/dushuyue/openpi/data \
OPENPI_NUM_MACHINES=2 OPENPI_GPUS_PER_MACHINE=8 OPENPI_MAIN_PROCESS_PORT=29631 \
OPENPI_TORCH_MASTER_ADDR=<JP_WORKER0_IP> OPENPI_NCCL_SOCKET_IFNAME=eth1 \
OPENPI_TORCH_MASTER_FILE=/iag_ad_01/ad/dushuyue/openpi/data/.pi05_droid_jp_torch_master \
OPENPI_DATASET_REPO_ID=droid \
OPENPI_DATASET_ROOT=/iag_ad_01/ad/xiejinxia/data/DreamZero-DROID-Data \
OPENPI_EPISODE_INDICES_PATH=/iag_ad_01/ad/dushuyue/data/dreamzero_droid_lerobot_split_9to1/train_episode_indices.txt \
OPENPI_ACTION_SPACE=JOINT_POSITION \
PI05_BASE_PYTORCH_PATH=/iag_ad_01/ad/dushuyue/openpi/data/checkpoints/pi05_base_pytorch \
CHECKPOINT_BASE_DIR=/iag_ad_01/ad/dushuyue/openpi/data/checkpoints \
OPENPI_CONFIG_NAME=pi05_droid_lerobot_joint_position_finetune \
OPENPI_EXP_NAME=pi05_droid_lerobot_jp_2x8 \
OPENPI_BATCH_SIZE=256 OPENPI_NUM_TRAIN_STEPS=100000 \
OPENPI_SAVE_INTERVAL=5000 OPENPI_MAX_CHECKPOINTS=5 \
OPENPI_VIDEO_BACKEND=pyav OPENPI_NUM_WORKERS=0 \
HF_HOME=/iag_ad_01/ad/dushuyue/openpi/data/.hf_cache \
OPENPI_HF_DATASETS_CACHE=/iag_ad_01/ad/dushuyue/openpi/data/.hf_cache/droid_lerobot_v20_jp \
OPENPI_DATA_HOME=/iag_ad_01/ad/dushuyue/openpi/data/.openpi_cache \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline \
bash /iag_ad_01/ad/dushuyue/openpi/scripts/launch_platform_tianyi_pi05_torchrun.sh
```

worker 1 使用同一条命令；脚本自动推导 rank 1。

配置默认沿用 `pi05_full_droid_finetune` 的核心设置：100,000 steps、global
batch 256、warmup 1,000、peak LR `5e-5`、15 FPS action chunk horizon 16、
AdamW 和 DROID norm stats。DDP 下每卡 batch 为 16。若需要短 smoke run，可在
命令末尾临时追加 `--num-train-steps=10 --batch-size=16 --no-wandb-enabled`；
正式实验不要保留这些覆盖参数。

## 加速配置（推荐先做短跑验收）

当前训练集有 13,267,639 帧。原配置 `global batch=256`、`100,000 steps`
约等于 25.6M 个样本（约 1.93 个 epoch）。在 H20 96GB 上，建议先将每卡
batch 从 16 提到 32：

```text
global batch = 512       # 2 机 16 卡时每卡 batch=32
num_train_steps = 50,000 # 与原配置保持相同的 25.6M 样本量
save_interval = 2,500    # 与原配置保持相近的样本间隔
```

JV 长跑可使用以下覆盖参数（JP 只需替换配置名和实验名）：

```bash
OPENPI_BATCH_SIZE=512 OPENPI_NUM_TRAIN_STEPS=50000 \
OPENPI_SAVE_INTERVAL=2500 OPENPI_VIDEO_BACKEND=pyav OPENPI_NUM_WORKERS=0 \
OPENPI_CONFIG_NAME=pi05_droid_lerobot_joint_velocity_finetune \
OPENPI_EXP_NAME=pi05_droid_lerobot_jv_b512 \
bash /iag_ad_01/ad/dushuyue/openpi/scripts/launch_platform_tianyi_pi05_torchrun.sh
```

JP 实验替换为：

```text
OPENPI_CONFIG_NAME=pi05_droid_lerobot_joint_position_finetune
OPENPI_EXP_NAME=pi05_droid_lerobot_jp_b512
```

## 4 机 32 卡启动（推荐加速配置）

以下两条命令分别提交为两个独立作业。每个作业的 worker-0、worker-1、
worker-2、worker-3 都执行同一条命令；launcher 根据 hostname 的
`*-worker-N` 后缀自动设置 rank。无需填写随机分配的 worker-0 IP：worker-0
会自动发现 `eth1` 内网地址并写入该实验的共享 master 文件，其余 worker 自动读取。

joint velocity（global batch 1024，即每卡 batch 32）：

```bash
cd /iag_ad_01/ad/dushuyue/openpi
OPENPI_ROOT=/iag_ad_01/ad/dushuyue/openpi \
OPENPI_RUNTIME=/iag_ad_01/ad/dushuyue/openpi/.venv \
OPENPI_DATA_ROOT=/iag_ad_01/ad/dushuyue/openpi/data \
OPENPI_NUM_MACHINES=4 OPENPI_GPUS_PER_MACHINE=8 OPENPI_MAIN_PROCESS_PORT=29632 \
OPENPI_DATASET_REPO_ID=droid \
OPENPI_DATASET_ROOT=/iag_ad_01/ad/xiejinxia/data/DreamZero-DROID-Data \
OPENPI_EPISODE_INDICES_PATH=/iag_ad_01/ad/dushuyue/data/dreamzero_droid_lerobot_split_9to1/train_episode_indices.txt \
OPENPI_ACTION_SPACE=JOINT_VELOCITY \
PI05_BASE_PYTORCH_PATH=/iag_ad_01/ad/dushuyue/openpi/data/checkpoints/pi05_base_pytorch \
CHECKPOINT_BASE_DIR=/iag_ad_01/ad/dushuyue/openpi/data/checkpoints \
OPENPI_CONFIG_NAME=pi05_droid_lerobot_joint_velocity_finetune \
OPENPI_EXP_NAME=pi05_droid_lerobot_jv_4x8_b1024 \
OPENPI_BATCH_SIZE=1024 OPENPI_NUM_TRAIN_STEPS=25000 \
OPENPI_SAVE_INTERVAL=2500 OPENPI_MAX_CHECKPOINTS=5 \
OPENPI_WARMUP_STEPS=250 OPENPI_DECAY_STEPS=250000 \
OPENPI_VIDEO_BACKEND=pyav OPENPI_NUM_WORKERS=0 \
HF_HOME=/iag_ad_01/ad/dushuyue/openpi/data/.hf_cache/droid_jv_4x8 \
OPENPI_HF_DATASETS_CACHE=/iag_ad_01/ad/dushuyue/openpi/data/.hf_cache/droid_lerobot_v20_jv_4x8 \
OPENPI_DATA_HOME=/iag_ad_01/ad/dushuyue/openpi/data/.openpi_cache/droid_jv_4x8 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline \
bash /iag_ad_01/ad/dushuyue/openpi/scripts/launch_platform_tianyi_pi05_torchrun.sh
```

joint position（global batch 1024，即每卡 batch 32）：

```bash
cd /iag_ad_01/ad/dushuyue/openpi
OPENPI_ROOT=/iag_ad_01/ad/dushuyue/openpi \
OPENPI_RUNTIME=/iag_ad_01/ad/dushuyue/openpi/.venv \
OPENPI_DATA_ROOT=/iag_ad_01/ad/dushuyue/openpi/data \
OPENPI_NUM_MACHINES=4 OPENPI_GPUS_PER_MACHINE=8 OPENPI_MAIN_PROCESS_PORT=29633 \
OPENPI_DATASET_REPO_ID=droid \
OPENPI_DATASET_ROOT=/iag_ad_01/ad/xiejinxia/data/DreamZero-DROID-Data \
OPENPI_EPISODE_INDICES_PATH=/iag_ad_01/ad/dushuyue/data/dreamzero_droid_lerobot_split_9to1/train_episode_indices.txt \
OPENPI_ACTION_SPACE=JOINT_POSITION \
PI05_BASE_PYTORCH_PATH=/iag_ad_01/ad/dushuyue/openpi/data/checkpoints/pi05_base_pytorch \
CHECKPOINT_BASE_DIR=/iag_ad_01/ad/dushuyue/openpi/data/checkpoints \
OPENPI_CONFIG_NAME=pi05_droid_lerobot_joint_position_finetune \
OPENPI_EXP_NAME=pi05_droid_lerobot_jp_4x8_b1024 \
OPENPI_BATCH_SIZE=1024 OPENPI_NUM_TRAIN_STEPS=25000 \
OPENPI_SAVE_INTERVAL=2500 OPENPI_MAX_CHECKPOINTS=5 \
OPENPI_WARMUP_STEPS=250 OPENPI_DECAY_STEPS=250000 \
OPENPI_VIDEO_BACKEND=pyav OPENPI_NUM_WORKERS=0 \
HF_HOME=/iag_ad_01/ad/dushuyue/openpi/data/.hf_cache/droid_jp_4x8 \
OPENPI_HF_DATASETS_CACHE=/iag_ad_01/ad/dushuyue/openpi/data/.hf_cache/droid_lerobot_v20_jp_4x8 \
OPENPI_DATA_HOME=/iag_ad_01/ad/dushuyue/openpi/data/.openpi_cache/droid_jp_4x8 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline \
bash /iag_ad_01/ad/dushuyue/openpi/scripts/launch_platform_tianyi_pi05_torchrun.sh
```

正式长跑前建议先将两条命令的 `OPENPI_NUM_TRAIN_STEPS` 改为 `200` 做 smoke
run。当前云端命令已使用 `OPENPI_VIDEO_BACKEND=pyav` 和
`OPENPI_NUM_WORKERS=0`，不需要 `TORCHCODEC_FFMPEG_OK`；只有确认 worker
镜像具备 FFmpeg 动态库后，才切换到 TorchCodec。
若 global batch 1024 OOM，回退到 global batch 512、50,000 steps（每卡 batch
16），并将 warmup/decay 改为 500/500,000。

正式长跑前，先将 `OPENPI_NUM_TRAIN_STEPS=200`，在所有 worker 上观察
step 20--100 的稳定耗时、显存和数据 worker 是否退出。使用 PyAV 时不需要
`TORCHCODEC_FFMPEG_OK`；只要没有 DataLoader crash、死锁或 NCCL 等待即可。
若 batch 512 OOM，回退到 `global batch=384`、`num_train_steps=66667`（每卡
24）；若 TorchCodec 在云端无法加载，则使用 `OPENPI_VIDEO_BACKEND=pyav`
和 `OPENPI_NUM_WORKERS=0`。首次 Arrow cache 构建时间不计入稳定 step time。

该配置先保持原 peak LR `5e-5` 和 warmup 设置不变，避免同时改变优化器和
batch 导致 JV/JP 结果不可比较。只有在 batch 512 稳定且吞吐提升确认后，才
考虑单独测试线性放大学习率；不能用未经测试的高学习率替换正式结果。

## 时长估算

OpenPI 文档的参照是 8×H100、100k steps 约 2 天，不能直接套用当前 H20。
你已有日志显示 global batch 256 时稳定耗时约 18.46 秒/step，因此原配置
100k steps 约 **513 小时（21.4 天）**，还应预留 10--30% 的 IO/DDP 波动。
加速配置是 50k steps；若 batch 512 的稳定耗时为 25--35 秒/step，单个实验约
**347--486 小时（14.5--20.3 天）**。实际数值必须以 200-step smoke run 的
step 20--100 平均值为准：
`总小时 = steps × 平均 step 秒 / 3600`。

## 已验证内容

已验证真实 episode 0 的 Parquet 和三路视频可被 LeRobot v2.0 loader 打开；两种
动作配置均产生 `(16, 8)` 的有限值 action chunk。Python 单元测试覆盖动作切片、
episode manifest 的 JSON/TXT 读取、缺失/重复 episode 校验，以及 LeRobot v2.0
非连续 episode ID 的运行时索引兼容。已用实际 ID `[0, 57542]` 验证高 ID episode
可正常生成 16-step action chunk。全量训练尚未启动。
