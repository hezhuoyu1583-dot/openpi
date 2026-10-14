"""Add state_eef and action_eef to a joint dataset that was exported earlier.

New exports write these columns in prepare_sft.py. This script is only for a
joint dataset that already exists. The pose math is the same function.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from prepare_sft import eef_for_rows, end_message, load_eef_modules, load_end_poses

SOURCE = Path("/iag_ad_01/ad/dushuyue/openpi/data/linglong_0923_1010_sft_train_15hz")
OUTPUT = Path("/iag_ad_01/ad/dushuyue/openpi/data/linglong_0923_1010_eef_sft_train_15hz")
RAW_ROOTS = {
    "linglong_0923_sft_train_15hz": Path("/iag_ad_01/ad/dushuyue/openpi/data/linglong_0923_processed/train"),
    "linglong_1010_sft_train_15hz": Path("/iag_ad_01/ad/dushuyue/openpi/data/linglong_1010_processed/train"),
}
LAYOUT = "left_xyz3_quat_wxyz4_gripper,right_xyz3_quat_wxyz4_gripper"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def alignment_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fixed_list(values):
    flat = pa.array(np.asarray(values, dtype=np.float32).reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, 16)


def episode_stat(values):
    values = np.asarray(values, dtype=np.float64)
    return {name: arr.tolist() for name, arr in {
        "min": values.min(0), "max": values.max(0), "mean": values.mean(0),
        "std": values.std(0), "count": np.array([len(values)]),
    }.items()}


def locate_episode(record, decoded_root):
    source_id = record["source_id"]
    if record.get("origin_name"):
        origin = RAW_ROOTS[record["origin_name"]]
        decoded = origin / "decoded" / source_id
        database = origin / "raw" / source_id / "bag" / "bag_0.db3"
    else:
        if decoded_root is None:
            raise ValueError(f"{source_id} has no origin_name; pass --decoded-root")
        decoded = decoded_root / source_id
        database = Path(record["archive_member"]).parent / "bag" / "bag_0.db3"
    if not database.is_file():
        raise FileNotFoundError(f"no bag for {source_id}: {database}")
    return decoded, database


def build(source=SOURCE, output=OUTPUT, decoded_root=None):
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    _parser, end_frames = load_eef_modules()
    mapping = json.loads((source / "source_mapping.json").read_text())
    info = json.loads((source / "meta/info.json").read_text())
    stats_rows = [json.loads(line) for line in (source / "meta/episodes_stats.jsonl").read_text().splitlines() if line.strip()]
    if len(stats_rows) != len(mapping):
        raise ValueError(f"episodes_stats rows {len(stats_rows)} != mapping {len(mapping)}")
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    state_parts = []
    action_parts = []
    position_errors = []
    orientation_errors = []
    for record in mapping:
        source_id = record["source_id"]
        decoded, database = locate_episode(record, decoded_root)
        alignment = alignment_rows(decoded / "alignment.jsonl")
        selected = [int(index) for index in record["source_row_indices"]]
        if len(selected) != record["length"] or any(index < 0 or index >= len(alignment) for index in selected):
            raise ValueError(f"alignment index mismatch for {source_id}")
        poses = load_end_poses(_parser, database)
        episode = record["episode_index"]
        table = pq.read_table(source / f"data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet")
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float64)
        if state.shape != (record["length"], 24) or action.shape != state.shape:
            raise ValueError(f"joint shape mismatch for episode {episode}")
        messages = [end_message(alignment[index], poses) for index in selected]
        state_eef, action_eef, pos_mm, ang_deg = eef_for_rows(state, action, messages, end_frames)
        position_errors.append(pos_mm)
        orientation_errors.append(ang_deg)
        dest = partial / f"data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet"
        dest.parent.mkdir(parents=True, exist_ok=True)
        written = table.append_column("state_eef", fixed_list(state_eef)).append_column("action_eef", fixed_list(action_eef))
        pq.write_table(written, dest)
        state_parts.append(state_eef)
        action_parts.append(action_eef)
        if stats_rows[episode]["episode_index"] != episode:
            raise ValueError(f"episodes_stats out of order at episode {episode}")
        stats_rows[episode]["stats"]["state_eef"] = episode_stat(state_eef)
        stats_rows[episode]["stats"]["action_eef"] = episode_stat(action_eef)
        print(f"episode {episode} {source_id} frames={record['length']}", flush=True)

    position_errors = np.concatenate(position_errors)
    orientation_errors = np.concatenate(orientation_errors)
    if np.median(position_errors) > 10.0:
        raise ValueError("recorded end position disagrees with end_frames by more than 10 mm median")
    if np.median(orientation_errors) > 5.0:
        raise ValueError("recorded end orientation disagrees with wrist_yaw end_frames by more than 5 deg median")
    combined = {"state_eef": np.concatenate(state_parts), "action_eef": np.concatenate(action_parts)}
    stats = {}
    for key, values in combined.items():
        stats[key] = {
            "mean": values.mean(0).tolist(),
            "std": values.std(0).tolist(),
            "q01": np.quantile(values, 0.01, axis=0).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).tolist(),
        }
    meta = partial / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    for name in ("episodes.jsonl", "tasks.jsonl"):
        shutil.copy2(source / "meta" / name, meta / name)
    (meta / "episodes_stats.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in stats_rows))
    for name in ("train_episodes.json", "validation_episodes.json", "test_episodes.json", "source_mapping.json"):
        if (source / name).is_file():
            shutil.copy2(source / name, partial / name)
    shutil.copytree(source / "assets", partial / "assets")
    feature = {"dtype": "float32", "shape": [16], "names": None}
    info["features"]["state_eef"] = feature
    info["features"]["action_eef"] = dict(feature)
    write_json(partial / "meta/info.json", info)
    write_json(partial / "assets/linglong_h/eef_norm_stats.json", {"norm_stats": stats})
    write_json(partial / "manifest.json", {
        "status": "complete",
        "source_dataset": str(source),
        "original_columns_preserved": True,
        "state_eef": {
            "shape": [16],
            "layout": LAYOUT,
            "position": "logged /end_pos xyz, metres, base frame",
            "orientation": "logged rx,ry,rz radians converted by R=Rz(yaw)@Ry(pitch)@Rx(roll), quaternion wxyz",
            "gripper": "observation.state indices 7 and 15",
        },
        "action_eef": {
            "shape": [16],
            "layout": LAYOUT,
            "source": "end_frames(action), wrist_yaw_link plus the wrist-frame TCP offset",
            "orientation": "end_frames wrist_yaw_link rotation, quaternion wxyz",
            "gripper": "action indices 7 and 15",
        },
        "quaternion_sign": "continuous independently per hand within each episode",
        "frames": int(combined["state_eef"].shape[0]),
        "recorded_vs_end_frames_on_state_mm": {
            "left_median": float(np.median(position_errors[:, 0])),
            "right_median": float(np.median(position_errors[:, 1])),
            "left_max": float(position_errors[:, 0].max()),
            "right_max": float(position_errors[:, 1].max()),
        },
        "recorded_vs_end_frames_orientation_deg": {
            "left_median": float(np.median(orientation_errors[:, 0])),
            "right_median": float(np.median(orientation_errors[:, 1])),
        },
    })
    partial.rename(output)
    print(f"COMPLETE {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--decoded-root", type=Path, default=None,
                        help="Decoded episode directory when source_mapping has no origin_name")
    args = parser.parse_args()
    build(args.source, args.output, args.decoded_root)
