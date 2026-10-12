# LIBERO 仿真重试结果不一致排查结论

## 问题

同一任务跑多次trial(如`num_trials_per_task=2`),结果不一致(比如trial 1失败、trial 2成功),
且两次生成的视频看起来很像,怀疑仿真存在异常随机性。

## 结论

**根因:每个trial使用的初始状态本身就不同,不是仿真bug或异常随机性。**

`examples/libero/main.py`中:

```python
initial_states = task_suite.get_task_init_states(task_id)   # shape (50, 92)
...
obs = env.set_init_state(initial_states[episode_idx])
```

`initial_states`是LIBERO benchmark为每个任务预录制的50条初始状态(来自`.pruned_init`文件),
按`episode_idx`顺序取值。trial 0和trial 1天生对应数组里第0条和第1条,不是同一个起点。

### 92维状态向量的具体含义

```
[0]      = time
[1:49]   = qpos (48维,机器人关节 + 场景内各物体位姿)
[49:92]  = qvel (43维,对应速度,静止状态下恒为0)
```

实测两个trial的qpos部分平均有18个维度不同,包括:

- 机械臂7个关节角(`robot0_joint1~7`)
- 场景内各物体的桌面(x,y)位置(碗、饼干盒、小碟、盘子等)
- 任务相关的机构状态(如本例中"从柜子抽屉里拿碗"任务,`wooden_cabinet_1_top_level`抽屉的开合程度)

也就是说,50条初始状态本质是LIBERO故意设计的50种不同难度/摆位的开局,用来测试模型泛化能力,
不是需要修的bug。这解释了为什么同一任务、同一模型,不同trial会有不同的成功率。

### 关于"两次视频看起来一样"

实测抽取视频首帧做像素级对比,均存在明显差异(mean_abs_diff约6-7,满量程255),
说明两次trial从第一帧起就是不同的初始画面,不是仿真产生了真正相同的结果。
视觉上"看起来像"是因为同一模型面对同一任务时,开局若干步的动作策略比较接近,
但严格来说从起点开始就已经是两条不同的轨迹。

补充:若两次trial**结果相同**(都成功或都失败),`main.py`保存视频时的文件名只由
`任务描述 + success/failure`拼成,不含episode编号,此时第二次trial会直接覆盖
第一次trial同名的视频文件,导致"看起来只有一份"的假象。此问题未修复,若需要保留
每个trial独立的视频,需要在文件名中加入`episode_idx`。

## 排查方法(供后续参考)

1. 在`main.py`临时加日志,打印每个episode取到的`initial_states[episode_idx]`对应的
   `env.set_init_state()`返回的obs中的`robot0_eef_pos`等字段,对比是否一致。
2. 用`ffprobe`看两个视频的帧数/时长是否一致(不一致说明episode执行长度不同,必然不是同一结果)。
3. 用`ffmpeg`抽取对应帧,逐像素比较(`numpy`对两帧图像求绝对差均值),验证画面内容是否真的相同。
4. 若要复现某个具体任务,先用`libero.benchmark.get_benchmark_dict()[suite]().get_task(i).language`
   遍历确认任务名对应的task_id,再针对该task_id单独跑,避免跑全量任务耗时。

## 其他连带修复

排查过程中发现并修复了一个和本问题无关的真实bug:
`src/openpi/training/config.py`中错误地把`roboarena_config`导入路径写成了不存在的
`ena_config`,导致任何策略加载(包括正常训练/推理)都会直接`ModuleNotFoundError`崩溃。
已改回正确的`roboarena_config`模块名,此修复保留。
