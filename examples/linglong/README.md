# Linglong 0714 SFT：当前方案

统一 prompt：

> Use the right arm to pick up the three round parts from the table one by one and place them on the green conveyor belt.

配置名 `pi05_linglong_sft`，新目录 `openpi/data/linglong_0714_sft_15hz/`。旧30Hz数据保留，不再用于此配置。

| 项目 | 当前设置 |
|---|---|
| 完整连续episode | 289（排除12条，不是成功率筛选） |
| 总帧数 | 77,970 |
| 训练 / 验证 | 232 / 57条；64,016 / 13,954帧 |
| 频率 / horizon | 15Hz / 25步，约1.67秒 |
| 资源 / batch | 单机8卡；全局64，每卡8，无梯度累积 |
| steps / warmup | 5,000 / 100 |
| 学习率 | 峰值1e-5，5,000步衰减至1e-6 |
| 保存 | 每500步及最后一步，全部保留评测权重，最近2份保留optimizer |

## 数据规则

按原始episode时间顺序80/20划分后筛选，同源不跨划分。15Hz均匀网格，最近头部帧偏差<=20ms，各流<=80ms。任何网格无效、原始头部间隔>50ms或原始对齐偏差>80ms，整条排除。不裁头尾、不切段，也不代表已确认任务成功。`source_mapping.json`记录来源与行索引。

状态q、绝对动作q_exp；24D为左臂7、左夹爪、右臂7、右夹爪、腰4、头2、底盘2。内部32D，损失仅前22D有效。统计量由新训练集重算。图片仍引用源JPEG绝对路径，迁移须保持挂载或重建；加载时resize/pad到224。离线最近邻并非在线因果同步。

训练末尾不足窗口时重复最后动作，不跨episode；评测仅选完整25步窗口。drop_last使一个epoch为1000步、丢尾部16个起点，5000步约5遍，不保证每帧都使用。

## 保存及评测

### 训练指标自动落盘

专用入口由rank0自动追加记录到 `openpi/data/training_logs/linglong/<实验名>/`：

- `train.log`：Python logging日志与异常堆栈（不含其他rank全部stdout/stderr及进度条）。
- `metrics.jsonl` / `metrics.csv`：每20步的loss、learning_rate、grad_norm、区间耗时、UTC时间和run_id；断点恢复追加，不覆盖，重叠step通过run_id区分。
- `training_curves.png`：正常结束或可捕获异常退出后生成四项指标曲线，按run_id分开绘制。

指标解析现有训练器日志，因此保留其显示精度（loss四位小数），不是未经舍入的每步原始loss。loss是rank0当前日志区间的均值，不是8卡all-reduce平均或验证loss；step沿用训练日志零基索引，checkpoint使用已完成更新计数。首次日志区间较短，区间耗时不能直接当每步耗时。记录即时flush，但SIGKILL/机器断电不能保证最后一次写入或自动绘图；可手动运行 `openpi/.venv/bin/python openpi/examples/linglong/metrics.py openpi/data/training_logs/linglong/<实验名>` 重绘。训练loss须与训后固定留出集指标共同判断，不能用其下降证明真机任务收敛。

必须使用专用 `examples/linglong/train.py` 或shell入口。它仅在本进程安装保存hook，共享 `scripts/train_pytorch.py` 不变。直接运行共享入口会保留全部optimizer。

标准长跑在500、1000、…、5000保存10份模型、metadata、统计量及inference_config，最新两份额外保留optimizer.pt。只清理旧optimizer，不删除早期权重、不额外复制权重。保存成功后再清理，临时需多一份空间。最新两份支持现有resume；早期只用于推理。恢复行为沿用共享训练器，不承诺采样器/RNG逐位恢复。

按用户选择训完统一评测；不在训练内自动验证。evaluate.py固定留出样本和噪声seed，逐checkpoint报告左右臂/夹爪/腰/头MAE、保持当前位置基线、预测数组和耗时。关节单位rad、夹爪为原始开合单位，不是成功率。优先比较右臂/右夹爪改善，不按混合单位总loss排名。真实评测需等SFT权重生成。

## 命令（工作区根目录）

数据已生成，无需重复。重建使用新目录，同源未完成转换可加--resume。

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 \
openpi/.venv/bin/python openpi/examples/linglong/prepare_sft.py \
 --source linglong_rawdata_spec/processed_0714 --output openpi/data/linglong_0714_sft_15hz

OPENPI_DATA_HOME=/iag_ad_01/ad/dushuyue/cache/openpi \
HF_DATASETS_CACHE=/iag_ad_01/ad/dushuyue/openpi/data/.hf_cache/linglong_sft15 \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
openpi/.venv/bin/python openpi/examples/linglong/verify_sft.py

# 空闲8卡上先smoke，再正式训练（本次未执行）
bash openpi/examples/linglong/train_sft.sh linglong15_smoke --num-train-steps=2 --save-interval=1 --batch-size=8
bash openpi/examples/linglong/train_sft.sh linglong15_b64_5k
# 中断恢复
bash openpi/examples/linglong/train_sft.sh linglong15_b64_5k --resume

# 训后在空闲GPU运行，输出目录须不存在
OPENPI_DATA_HOME=/iag_ad_01/ad/dushuyue/cache/openpi \
HF_DATASETS_CACHE=/iag_ad_01/ad/dushuyue/openpi/data/.hf_cache/linglong_sft15 \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 JAX_PLATFORMS=cpu \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
openpi/.venv/bin/python openpi/examples/linglong/evaluate.py \
 --checkpoints openpi/data/checkpoints/pi05_linglong_sft/linglong15_b64_5k \
 --output openpi/data/eval/linglong15_b64_5k --samples 128 --seed 42
```

入口检查GPU占用，忙则退出，不停止服务。当前验证范围为真实数据加载、变换、loss维度掩码、checkpoint形状与保存策略。5项测试通过，其中用小型CPU模型调用真实保存器，确认早期权重保留、仅最新2份optimizer保留、权重和optimizer可重新加载，且非500倍数的最终步也正确保存。未进行完整pi0.5模型GPU训练、训练器端到端断点恢复或真实训后评测。未修改DROID/JV/JP训练逻辑。
