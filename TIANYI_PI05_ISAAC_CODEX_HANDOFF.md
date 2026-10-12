# TianYi + PI0.5 + Isaac 任务交接入口

这份文件供新机器上的 Codex 恢复上下文使用。新会话不应依赖旧聊天记录，先阅读本文件，再阅读下方的详细方案和脚本。

## 当前目标

先打通 TianYi2-BrainCo2 Isaac 最小仿真链路，暂不扩展 household 场景，也暂不接入正式 scorer：

```text
Isaac TianYi2-BrainCo2 -> ZMQ obs -> no-motion smoke client
-> ZMQ 26D current-state action -> Isaac position controller
```

通过后再把 smoke client 替换为 PI0.5 WebSocket bridge。

## 必读文件

1. `TIANYI_PI05_ISAAC_ADAPTATION_STATUS_AND_PLAN.md`：完整适配状态、26D/16D 映射、风险和分阶段计划。
2. `x-humanoid-vla-simulation-benchmark/docs/tianyi_isaac_smoke.md`：最小 Isaac/ZMQ smoke 说明。
3. `x-humanoid-vla-simulation-benchmark/tools/run_tianyi_isaac_smoke.sh`：一键 smoke 启动脚本。
4. `x-humanoid-vla-simulation-benchmark/tools/tianyi_isaac_smoke.py`：ZMQ handshake、obs/action/reset 客户端。
5. `x-humanoid-vla-simulation-benchmark/tools/policies/pi05_smoke_policy.py`：26D 当前状态回传策略。

## 已完成

- TianYi2-BrainCo2 USD、Isaac 任务、`camera_head` 和 26D ZMQ 协议已在 benchmark 仓库中。
- PI0.5 TianYi stationary 后训练 checkpoint 已存在：`openpi/data/checkpoints/pi05_tianyi_finetune/`。
- PI0.5 base checkpoint 已存在：`openpi/data/checkpoints/pi05_base_pytorch/`。
- TianYi 离线评测脚本和历史结果在 `openpi/scripts/eval_tianyi.py`、`openpi/data/eval/`。
- no-motion smoke policy/client/script 已加入 benchmark 仓库；已做 Python/shell/26D 顺序检查。

## 尚未完成（下一步）

1. 在带 RT Core 的 4090/RTX 节点执行 smoke：固定 `task=ind_task_01`、`loop=1`、`seed=0`、`timeout=30`。
2. 检查 `client.log` 出现 `handshake OK` 且 obs/actions 大于 0；检查 `benchmark.log` 无 renderer、None image、shape、NaN 错误。
3. 确认 `logs/task_videos/` 生成 episode 视频。
4. 通过 smoke 后实现 PI0.5 WebSocket bridge：26D Isaac obs -> 16D PI0.5 输入 -> 16D 输出 -> 标定的 26D BrainCo hand action。
5. scalar gripper 到 BrainCo hand6 的标定完成前，不做正式任务成功率结论。

## Smoke 启动

```bash
cd /iag_ad_01/ad/dushuyue/x-humanoid-vla-simulation-benchmark
export TIANYI_SMOKE_TASK=ind_task_01
export TIANYI_SMOKE_LOOP=1
export TIANYI_SMOKE_SEED=0
export TIANYI_SMOKE_TIMEOUT=30
export TIANYI_SMOKE_PYTHON=/iag_ad_01/ad/dushuyue/openpi/.venv/bin/python
bash tools/run_tianyi_isaac_smoke.sh
```

结果默认保存到：

```text
x-humanoid-vla-simulation-benchmark/logs/tianyi_isaac_smoke/<timestamp>/
x-humanoid-vla-simulation-benchmark/logs/task_videos/
```

## 重要边界

- 当前机器 H20 不适合现有 Isaac RTX 相机渲染；4090 节点用于真正 smoke。
- 当前 benchmark 的 5 个任务不是 TianYi household 36 任务；直接运行属于同本体跨任务 zero-shot。
- RoboTwin/LIBERO 等不同本体 benchmark 不作为主结果。
- HDF5/LeRobot 数据用于离线动作评测和任务语义，不自动提供 Isaac 物理场景。
- 不复制 `/root/.codex/auth.json`；新机器上的 Codex 需要单独 `codex login`。

## 给新 Codex 会话的首条提示词

```text
请先阅读 /iag_ad_01/ad/dushuyue/openpi/TIANYI_PI05_ISAAC_CODEX_HANDOFF.md，
以及其中列出的完整方案和 smoke 文档。检查当前 4090 节点的 Isaac Sim、GPU、
ZMQ 端口和 smoke 结果；若 smoke 未通过，先修复最小链路，不要扩展 household
场景或修改原始 TianYi 数据。smoke 通过后再继续实现 PI0.5 WebSocket bridge，
并保持 26D 顺序、日志和安全边界。
```
