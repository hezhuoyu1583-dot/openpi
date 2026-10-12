"""Offline open-loop evaluation for the official pi05_droid checkpoint.

The local LeRobot conversion contains 1,000 successful DROID episodes. This
script evaluates a deterministic episode-index split without treating that
subset as the official DROID test set. It compares the model's 15-step action
chunk with the recorded joint-velocity and gripper-position actions.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq
import tyro

from openpi.policies import policy_config
from openpi.training import config as training_config

log = logging.getLogger(__name__)

ACTION_HORIZON = 15
ACTION_DIM = 8
DROID_CONTROL_HZ = 15
DROID_MAX_JOINT_DELTA_RAD = 0.2
NONIDLE_EPS = 0.05


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_video(path: Path, expected_frames: int) -> list[np.ndarray]:
    reader = cv2.VideoCapture(str(path))
    if not reader.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    frames: list[np.ndarray] = []
    try:
        while len(frames) < expected_frames:
            ok, frame = reader.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        reader.release()
    if len(frames) != expected_frames:
        raise RuntimeError(f"video frame count mismatch for {path}: {len(frames)} != {expected_frames}")
    return frames


def _episode_paths(dataset_root: Path, episode_index: int) -> tuple[Path, Path, Path]:
    chunk = episode_index // 1000
    episode_name = f"episode_{episode_index:06d}"
    data_path = dataset_root / "data" / f"chunk-{chunk:03d}" / f"{episode_name}.parquet"
    exterior_path = (
        dataset_root
        / "videos"
        / f"chunk-{chunk:03d}"
        / "observation.images.exterior_image_1_left"
        / f"{episode_name}.mp4"
    )
    wrist_path = (
        dataset_root
        / "videos"
        / f"chunk-{chunk:03d}"
        / "observation.images.wrist_image_left"
        / f"{episode_name}.mp4"
    )
    return data_path, exterior_path, wrist_path


def _load_episode(dataset_root: Path, episode_index: int) -> dict[str, Any]:
    data_path, exterior_path, wrist_path = _episode_paths(dataset_root, episode_index)
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    table = pq.read_table(data_path, columns=["observation.state", "action"])
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    if state.ndim != 2 or state.shape[1] != 14:
        raise ValueError(f"episode {episode_index}: expected state (T,14), got {state.shape}")
    if action.ndim != 2 or action.shape[1] != 28:
        raise ValueError(f"episode {episode_index}: expected action (T,28), got {action.shape}")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError(f"episode {episode_index}: state/action contains NaN or Inf")
    if not exterior_path.is_file() or not wrist_path.is_file():
        raise FileNotFoundError(f"missing episode {episode_index} camera video")
    exterior = _read_video(exterior_path, len(state))
    wrist = _read_video(wrist_path, len(state))
    return {
        "episode_index": episode_index,
        "state": state,
        "action": action,
        "exterior": exterior,
        "wrist": wrist,
    }


def _future_targets(action: np.ndarray, timestep: int) -> np.ndarray:
    """Return 15 recorded DROID actions, repeating the terminal row at EOF."""
    indices = np.minimum(np.arange(timestep, timestep + ACTION_HORIZON), len(action) - 1)
    return np.concatenate([action[indices, 21:28], action[indices, 12:13]], axis=-1)


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    if predictions.shape != targets.shape or predictions.ndim != 3 or predictions.shape[-1] != ACTION_DIM:
        raise ValueError(f"expected equal (N,15,8) arrays, got {predictions.shape} and {targets.shape}")
    error = predictions - targets
    arm_error = error[..., :7]
    grip_error = error[..., 7]
    arm_abs = np.abs(arm_error)
    grip_abs = np.abs(grip_error)
    arm_pred = predictions[..., :7]
    arm_target = targets[..., :7]

    def _pearson(x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x).reshape(-1)
        y = np.asarray(y).reshape(-1)
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        denominator = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
        return float(np.dot(x_centered, y_centered) / denominator) if denominator > 1e-12 else 0.0

    def _cosine_rows(x: np.ndarray, y: np.ndarray) -> float:
        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1, y.shape[-1])
        denominator = np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1)
        valid = denominator > 1e-12
        if not np.any(valid):
            return 0.0
        return float(np.mean(np.sum(x[valid] * y[valid], axis=-1) / denominator[valid]))

    nonidle = np.max(np.abs(arm_target), axis=-1) > NONIDLE_EPS
    sign_match = np.sign(arm_pred) == np.sign(arm_target)
    sign_nonidle = sign_match[nonidle]
    result: dict[str, Any] = {
        "num_chunks": int(len(predictions)),
        # Raw DROID joint-velocity command units. These are dimensionless
        # normalized commands, not SI rad/s values.
        "arm_mae": float(arm_abs.mean()),
        "arm_rmse": float(np.sqrt(np.mean(arm_error**2))),
        "arm_mae_command": float(arm_abs.mean()),
        "arm_rmse_command": float(np.sqrt(np.mean(arm_error**2))),
        "arm_mae_rad_per_step": float(arm_abs.mean() * DROID_MAX_JOINT_DELTA_RAD),
        "arm_rmse_rad_per_step": float(np.sqrt(np.mean(arm_error**2)) * DROID_MAX_JOINT_DELTA_RAD),
        "arm_mae_rad_per_sec": float(arm_abs.mean() * DROID_MAX_JOINT_DELTA_RAD * DROID_CONTROL_HZ),
        "arm_rmse_rad_per_sec": float(np.sqrt(np.mean(arm_error**2)) * DROID_MAX_JOINT_DELTA_RAD * DROID_CONTROL_HZ),
        "gripper_mae": float(grip_abs.mean()),
        "overall_mae": float(np.abs(error).mean()),
        "overall_rmse": float(np.sqrt(np.mean(error**2))),
        "arm_sign_accuracy": float(sign_match.mean()),
        "arm_sign_accuracy_nonidle": float(sign_nonidle.mean()) if sign_nonidle.size else None,
        "arm_nonidle_fraction": float(nonidle.mean()),
        "arm_mae_nonidle": float(arm_abs[nonidle].mean()) if np.any(nonidle) else None,
        "arm_pearson_r": _pearson(arm_pred, arm_target),
        "arm_pearson_r_per_joint": [_pearson(arm_pred[..., j], arm_target[..., j]) for j in range(7)],
        "arm_cosine_similarity": _cosine_rows(arm_pred, arm_target),
        "horizon_arm_mae": arm_abs.mean(axis=(0, 2)).tolist(),
        "horizon_gripper_mae": grip_abs.mean(axis=0).tolist(),
        "per_joint_mae": arm_abs.mean(axis=(0, 1)).tolist(),
    }
    pred_open = predictions[..., 7] > 0.5
    target_open = targets[..., 7] > 0.5
    tp = int(np.logical_and(pred_open, target_open).sum())
    fp = int(np.logical_and(pred_open, ~target_open).sum())
    fn = int(np.logical_and(~pred_open, target_open).sum())
    result["gripper_f1"] = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 1.0
    result["gripper_accuracy"] = float((pred_open == target_open).mean())
    result["gripper_precision"] = float(tp / (tp + fp)) if (tp + fp) else 1.0
    result["gripper_recall"] = float(tp / (tp + fn)) if (tp + fn) else 1.0
    result["command_units"] = "normalized DROID joint command; 1.0 = 0.2 rad per 15 Hz step"
    return result


def evaluate(
    dataset_root: str = "/iag_ad_01/ad/dushuyue/dreamzero/data/droid_lerobot",
    checkpoint_dir: str = "/iag_ad_01/ad/dushuyue/openpi/data/checkpoints/pi05_droid",
    output_dir: str = "/iag_ad_01/ad/dushuyue/data/pi05_droid_open_loop",
    start_episode: int = 0,
    max_episodes: int = 100,
    max_frames_per_episode: int | None = None,
    stride: int = 1,
    seed: int = 0,
) -> None:
    if max_episodes <= 0 or stride <= 0:
        raise ValueError("max_episodes and stride must be positive")
    root = Path(dataset_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    episode_meta = {item["episode_index"]: item for item in _read_jsonl(root / "meta" / "episodes.jsonl")}
    episode_indices = sorted(i for i in episode_meta if i >= start_episode)[:max_episodes]
    if len(episode_indices) < max_episodes:
        log.warning("requested %d episodes but found %d in local subset", max_episodes, len(episode_indices))
    if not episode_indices:
        raise ValueError("no episodes selected")

    # Pi05 flow matching samples an initial noise tensor. Supplying a seeded
    # tensor makes an open-loop report reproducible across repeated runs; the
    # model's regular parameter/state path remains unchanged.
    noise_rng = np.random.default_rng(seed)
    log.info("loading pi05_droid from %s", checkpoint_dir)
    policy = policy_config.create_trained_policy(training_config.get_config("pi05_droid"), checkpoint_dir)
    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    episode_results: list[dict[str, Any]] = []
    infer_ms: list[float] = []
    started = time.monotonic()

    for ordinal, episode_index in enumerate(episode_indices, 1):
        episode = _load_episode(root, episode_index)
        action = episode["action"]
        frame_count = len(action) if max_frames_per_episode is None else min(len(action), max_frames_per_episode)
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for timestep in range(0, frame_count, stride):
            obs = {
                "observation/exterior_image_1_left": episode["exterior"][timestep],
                "observation/wrist_image_left": episode["wrist"][timestep],
                # LeRobot DROID state layout is cartesian(6), gripper(1), joints(7).
                "observation/joint_position": episode["state"][timestep, 7:14],
                "observation/gripper_position": episode["state"][timestep, 6:7],
                "prompt": episode_meta[episode_index]["tasks"][0],
            }
            infer_started = time.monotonic()
            # pi05_droid has a 32-dimensional internal action head, while its
            # DROID output transform publishes the first 8 dimensions.
            noise = noise_rng.standard_normal((ACTION_HORIZON, 32), dtype=np.float32)
            result = policy.infer(obs, noise=noise)
            infer_ms.append((time.monotonic() - infer_started) * 1000)
            pred = np.asarray(result["actions"], dtype=np.float32)
            if pred.shape != (ACTION_HORIZON, ACTION_DIM) or not np.isfinite(pred).all():
                raise RuntimeError(f"episode {episode_index} timestep {timestep}: invalid prediction {pred.shape}")
            predictions.append(pred)
            targets.append(_future_targets(action, timestep))
        pred_array = np.asarray(predictions)
        target_array = np.asarray(targets)
        metrics = compute_metrics(pred_array, target_array)
        metrics.update({"episode_index": episode_index, "task": episode_meta[episode_index]["tasks"][0]})
        episode_results.append(metrics)
        all_predictions.append(pred_array)
        all_targets.append(target_array)
        log.info("[%d/%d] episode=%d chunks=%d arm_mae=%.5f", ordinal, len(episode_indices), episode_index, len(pred_array), metrics["arm_mae"])

    prediction_array = np.concatenate(all_predictions)
    target_array = np.concatenate(all_targets)
    aggregate = compute_metrics(prediction_array, target_array)
    # A zero-action baseline is useful because DROID contains substantial idle
    # periods.  Report it in the same command units and on non-idle rows so a
    # low aggregate error cannot be mistaken for useful motion prediction.
    zero_metrics = compute_metrics(np.zeros_like(target_array), target_array)
    for key in (
        "arm_mae",
        "arm_rmse",
        "arm_mae_nonidle",
        "arm_sign_accuracy",
        "arm_sign_accuracy_nonidle",
        "gripper_mae",
        "overall_mae",
        "overall_rmse",
    ):
        aggregate[f"zero_baseline_{key}"] = zero_metrics.get(key)
    if aggregate.get("arm_mae_nonidle") is not None and zero_metrics.get("arm_mae_nonidle"):
        aggregate["arm_mae_nonidle_skill_vs_zero"] = float(
            1.0 - aggregate["arm_mae_nonidle"] / zero_metrics["arm_mae_nonidle"]
        )
    if zero_metrics.get("arm_mae"):
        aggregate["arm_mae_skill_vs_zero"] = float(
            1.0 - aggregate["arm_mae"] / zero_metrics["arm_mae"]
        )
    aggregate.update(
        {
            "dataset_root": str(root),
            "checkpoint_dir": str(checkpoint_dir),
            "episode_indices": episode_indices,
            "num_episodes": len(episode_indices),
            "stride": stride,
            "elapsed_sec": time.monotonic() - started,
            "infer_p50_ms": float(np.percentile(infer_ms, 50)),
            "infer_p95_ms": float(np.percentile(infer_ms, 95)),
            "noise_seed": seed,
            "noise_shape": [ACTION_HORIZON, 32],
            "target_action_slices": {"joint_velocity": [21, 28], "gripper_position": [12, 13]},
            "control_hz": DROID_CONTROL_HZ,
            "max_joint_delta_rad_per_step": DROID_MAX_JOINT_DELTA_RAD,
            "nonidle_threshold_command": NONIDLE_EPS,
            "episodes": episode_results,
            "note": "local successful LeRobot subset; not the official DROID test split",
        }
    )
    result_path = output / "results.json"
    result_path.write_text(json.dumps(aggregate, indent=2))
    log.info("wrote %s", result_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(evaluate)
