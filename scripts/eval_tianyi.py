#!/usr/bin/env python3
"""Offline teacher-forced evaluation for a PyTorch PI0.5 Tianyi checkpoint.

This measures predicted actions against held-out LeRobot demonstrations.  It is
not a physical success-rate evaluation; that still requires the Isaac/robot
environment adapter.  The source dataset is opened read-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import random
from collections import defaultdict

import cv2
import numpy as np
import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from openpi.policies import policy_config
from openpi.training import config as training_config
from openpi.shared import normalize as normalize_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="checkpoint directory containing model.safetensors")
    parser.add_argument("--eval-root", required=True, help="local Tianyi eval LeRobot root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="pi05_tianyi_finetune")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--selection-file",
        type=pathlib.Path,
        default=None,
        help="JSON selected_episodes.json from a previous run; overrides seed-based sampling",
    )
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=None,
        help="deterministically sample this many episodes per task (overrides the episode prefix)",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--video-every", type=int, default=0, help="export every Nth episode; 0 disables")
    parser.add_argument(
        "--comparison-video",
        action="store_true",
        help="write side-by-side GT observation and model-error video",
    )
    parser.add_argument("--model-label", default="PI0.5", help="label used in comparison videos")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--norm-stats-dir", type=pathlib.Path, default=None, help="explicit directory containing norm_stats.json")
    parser.add_argument("--video-backend", choices=("pyav", "torchcodec"), default="pyav")
    parser.add_argument("--seed", type=int, default=0, help="random seed for reproducible flow-matching noise")
    parser.add_argument(
        "--openpi-data-home",
        default=str(pathlib.Path(__file__).resolve().parents[1] / "data/.openpi_cache"),
        help="existing OpenPI asset cache containing the PaliGemma tokenizer",
    )
    return parser.parse_args()


def np_(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def episode_ids(
    root: pathlib.Path,
    limit: int | None,
    episodes_per_task: int | None,
    seed: int,
) -> tuple[list[int], dict[str, list[int]]]:
    with (root / "meta/episodes.jsonl").open() as stream:
        records = [json.loads(line) for line in stream]
    if episodes_per_task is None:
        values = [int(record["episode_index"]) for record in records]
        selected = values if limit is None else values[:limit]
        return selected, {"__prefix__": selected}
    if episodes_per_task <= 0:
        raise ValueError("--episodes-per-task must be positive")

    grouped: dict[str, list[int]] = {}
    for record in records:
        episode = int(record["episode_index"])
        tasks = record.get("tasks") or [record.get("task", "")]
        for task in tasks:
            grouped.setdefault(str(task), []).append(episode)

    selected_by_task: dict[str, list[int]] = {}
    for task in sorted(grouped):
        candidates = sorted(set(grouped[task]))
        task_seed = hashlib.sha256(f"{seed}\0{task}".encode("utf-8")).digest()
        rng = random.Random(int.from_bytes(task_seed[:8], "little"))
        count = min(episodes_per_task, len(candidates))
        selected_by_task[task] = sorted(rng.sample(candidates, count))

    selected = sorted({episode for values in selected_by_task.values() for episode in values})
    if limit is not None:
        selected = selected[:limit]
        selected_set = set(selected)
        selected_by_task = {
            task: [episode for episode in values if episode in selected_set]
            for task, values in selected_by_task.items()
        }
    return selected, selected_by_task


def write_video(
    path: pathlib.Path,
    frames: list[np.ndarray],
    errors: list[float],
    fps: int,
    *,
    comparison: bool = False,
    model_label: str = "PI0.5",
) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    first = np.asarray(frames[0])
    if first.ndim == 3 and first.shape[0] == 3:
        first = np.transpose(first, (1, 2, 0))
    height, width = first.shape[:2]
    output_width = width * 2 if comparison else width
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (output_width, height))
    for frame, error in zip(frames, errors, strict=True):
        image = np.asarray(frame)
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        if image.dtype != np.uint8:
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        gt = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        predicted = gt.copy()
        cv2.putText(gt, "GT observation", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 220, 0), 2)
        cv2.putText(predicted, f"{model_label} | action MAE={error:.6f}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 220, 0), 2)
        writer.write(np.concatenate((gt, predicted), axis=1) if comparison else predicted)
    writer.release()


def main() -> None:
    args = parse_args()
    checkpoint = pathlib.Path(args.checkpoint).resolve()
    eval_root = pathlib.Path(args.eval_root).resolve()
    output = pathlib.Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(checkpoint / "model.safetensors")
    if args.video_backend == "pyav":
        # PyAV is intentionally used in the rank process; worker subprocesses
        # can segfault on some torchvision/FFmpeg combinations.
        os.environ["OPENPI_VIDEO_BACKEND"] = "pyav"
    os.environ.setdefault("OPENPI_DATA_HOME", str(pathlib.Path(args.openpi_data_home).resolve()))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_config = training_config.get_config(args.config)
    eval_factory = train_config.data
    eval_data_config = eval_factory.create(train_config.assets_dirs, train_config.model)
    explicit_norm_stats = None
    norm_stats_source = "checkpoint"
    if args.norm_stats_dir is not None:
        explicit_norm_stats = normalize_stats.load(args.norm_stats_dir)
        norm_stats_source = str(args.norm_stats_dir.resolve())
    policy = policy_config.create_trained_policy(
        train_config,
        checkpoint,
        repack_transforms=eval_data_config.repack_transforms,
        norm_stats=explicit_norm_stats,
        pytorch_device=args.device,
    )

    meta = LeRobotDatasetMetadata("tianyi", root=eval_root)
    if args.selection_file is not None:
        selected_by_task = json.loads(args.selection_file.read_text())
        selected_episodes = sorted({int(ep) for values in selected_by_task.values() for ep in values})
    else:
        selected_episodes, selected_by_task = episode_ids(
            eval_root,
            args.max_episodes,
            args.episodes_per_task,
            args.seed,
        )
    (output / "selected_episodes.json").write_text(json.dumps(selected_by_task, indent=2) + "\n")
    selected_set = set(selected_episodes)
    dataset = LeRobotDataset(
        "tianyi",
        root=eval_root,
        # Keep the full dataset index because this LeRobot version uses the
        # original episode id in __getitem__ even when episodes are filtered.
        episodes=None,
        video_backend=args.video_backend,
        delta_timestamps={"action": [i / meta.fps for i in range(train_config.model.action_horizon)]},
    )
    selected_frame_indices: list[int] = []
    frame_offset = 0
    for episode in range(meta.total_episodes):
        length = int(meta.episodes[episode]["length"])
        if episode in selected_set:
            selected_frame_indices.extend(range(frame_offset, frame_offset + length))
        frame_offset += length
    print(
        f"TIANYI_EVAL_SELECTION seed={args.seed} tasks={len(selected_by_task)} "
        f"episodes={len(selected_episodes)} frames={len(selected_frame_indices)}",
        flush=True,
    )

    abs_sum = np.zeros(16, dtype=np.float64)
    sq_sum = np.zeros(16, dtype=np.float64)
    valid_count = np.zeros(16, dtype=np.int64)
    task_errors: dict[str, list[float]] = defaultdict(list)
    episode_rows: list[dict] = []
    frames = 0
    current_episode = None
    video_frames: list[np.ndarray] = []
    video_errors: list[float] = []

    def finish_video() -> None:
        if current_episode is not None and args.video_every and current_episode % args.video_every == 0:
            write_video(
                output / "videos" / f"episode_{current_episode:06d}.mp4",
                video_frames,
                video_errors,
                meta.fps,
                comparison=args.comparison_video,
                model_label=args.model_label,
            )

    for index in selected_frame_indices:
        if args.max_frames is not None and frames >= args.max_frames:
            break
        sample = dataset[index]
        ep = int(np_(sample["episode_index"]).reshape(-1)[0])
        if current_episode != ep:
            finish_video()
            current_episode = ep
            video_frames = []
            video_errors = []

        sample["prompt"] = sample.get("task", "")
        prediction = np_(policy.infer(sample)["actions"])[..., :16]
        target = np_(sample["action"])[..., :16]
        padding = np_(sample.get("action_is_pad", np.zeros(target.shape[:-1], dtype=bool))).astype(bool)
        if padding.ndim == 0:
            padding = np.zeros(target.shape[:-1], dtype=bool)
        difference = prediction - target
        valid = ~padding
        abs_sum += np.where(valid[..., None], np.abs(difference), 0).sum(axis=0)
        sq_sum += np.where(valid[..., None], difference**2, 0).sum(axis=0)
        valid_count += valid.sum(axis=0)
        frame_mae = float(np.abs(difference[valid]).mean()) if valid.any() else float("nan")
        task_errors[str(sample.get("task", ""))].append(frame_mae)
        if not episode_rows or episode_rows[-1]["episode_index"] != ep:
            episode_rows.append({"episode_index": ep, "task": str(sample.get("task", "")), "frames": 0, "mae_sum": 0.0})
        episode_rows[-1]["frames"] += 1
        episode_rows[-1]["mae_sum"] += frame_mae
        if args.video_every and ep % args.video_every == 0:
            video_frames.append(sample["observation.images.camera_top"])
            video_errors.append(frame_mae)
        frames += 1
    finish_video()

    mae = abs_sum / np.maximum(valid_count, 1)
    rmse = np.sqrt(sq_sum / np.maximum(valid_count, 1))
    metrics = {
        "checkpoint": str(checkpoint),
        "eval_root": str(eval_root),
        "norm_stats_source": norm_stats_source,
        "video_mode": "gt_side_by_side" if args.comparison_video else "annotated_observation",
        "model_label": args.model_label,
        "frames": frames,
        "episodes": len(episode_rows),
        "valid_action_dims": 16,
        "action_mae": float(mae.mean()),
        "action_rmse": float(rmse.mean()),
        "per_dim_mae": mae.tolist(),
        "per_dim_rmse": rmse.tolist(),
        "task_mae": {task: float(np.nanmean(values)) for task, values in task_errors.items()},
        "selection_seed": args.seed,
        "selected_episodes": selected_episodes,
        "selected_episodes_by_task": selected_by_task,
        "note": "Teacher-forced offline action prediction; not physical task success.",
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with (output / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("episode_index", "task", "frames", "mae"))
        writer.writeheader()
        for row in episode_rows:
            writer.writerow({"episode_index": row["episode_index"], "task": row["task"], "frames": row["frames"], "mae": row["mae_sum"] / max(row["frames"], 1)})
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
