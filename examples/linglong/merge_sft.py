"""Merge Linglong train sets into one LeRobot dataset.

Image files stay at their original absolute paths. Episode IDs and the global
frame index are rewritten, each source keeps its own task string, and quantile
normalization is recomputed on the combined training frames.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from quantile_outliers import filter_dataset

DEFAULT_SOURCES = [
    Path("/iag_ad_01/ad/dushuyue/openpi/data/linglong_0923_sft_train_15hz"),
    Path("/iag_ad_01/ad/dushuyue/openpi/data/linglong_1010_sft_train_15hz"),
]
DEFAULT_OUTPUT = Path("/iag_ad_01/ad/dushuyue/openpi/data/linglong_0923_1010_sft_train_15hz")
CAMERAS = ("head", "left", "right")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_column(table, name, values):
    array = pa.array(values, type=pa.int64())
    return table.set_column(table.schema.get_field_index(name), name, array)


def episode_task(record):
    tasks = record["tasks"]
    if len(tasks) != 1 or not isinstance(tasks[0], str) or not tasks[0].strip():
        raise ValueError(f"expected one task string, got {tasks!r}")
    return tasks[0]


def check_images(table, episode_index):
    for camera in CAMERAS:
        for item in table[f"observation.images.{camera}"].to_pylist():
            path = item.get("path") if isinstance(item, dict) else None
            if not path or item.get("bytes") is not None or not Path(path).is_file():
                raise FileNotFoundError(f"episode {episode_index} {camera} image is not an existing path reference: {item!r}")


def joint_layout(manifest):
    layout = manifest.get("layout")
    if layout:
        return layout
    if manifest.get("original_columns_preserved"):
        source_manifest = Path(manifest.get("source_dataset", "")) / "manifest.json"
        if source_manifest.is_file():
            return read_json(source_manifest).get("layout")
    raise ValueError("joint layout is missing from the source manifest")


def load_source(root, selection="train_episodes.json"):
    root = Path(root)
    info = read_json(root / "meta/info.json")
    manifest = read_json(root / "manifest.json")
    if info["fps"] != 15 or info["robot_type"] != "linglong_h":
        raise ValueError(f"{root} is not a 15 Hz linglong_h dataset")
    semantics = manifest.get("action_semantics")
    if semantics is None and manifest.get("original_columns_preserved"):
        source_manifest = Path(manifest.get("source_dataset", "")) / "manifest.json"
        if source_manifest.is_file():
            semantics = read_json(source_manifest).get("action_semantics")
    if semantics != "absolute q_exp":
        raise ValueError(f"{root} action semantics are not absolute q_exp")
    episodes = {row["episode_index"]: row for row in read_jsonl(root / "meta/episodes.jsonl")}
    mapping = {row["episode_index"]: row for row in read_json(root / "source_mapping.json")}
    selected = read_json(root / selection)
    if not selected:
        raise ValueError(f"{root} {selection} has no episodes")
    if len(set(selected)) != len(selected):
        raise ValueError(f"{root} train_episodes.json has duplicate ids")
    missing = [episode for episode in selected if episode not in episodes or episode not in mapping]
    if missing:
        raise ValueError(f"{root} train manifest references missing episodes: {missing[:10]}")
    return {"root": root, "info": info, "manifest": manifest, "episodes": episodes, "mapping": mapping, "selected": selected}


def merge(sources, output, check_image_files, *, role="train", norm_reference=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if role not in {"train", "test"}:
        raise ValueError(f"unknown role {role}")
    selection = "train_episodes.json" if role == "train" else "test_episodes.json"
    if role == "test":
        if norm_reference is None:
            raise ValueError("a test merge needs the training set norm_stats.json")
        for path in sources:
            if not (Path(path) / selection).is_file():
                selection = "validation_episodes.json"
                break
        reference = Path(norm_reference)
        copied = read_json(reference)["norm_stats"]
        eef_reference = reference.parent / "eef_norm_stats.json"
        if not eef_reference.is_file():
            raise ValueError(f"missing {eef_reference}")
        copied_eef = read_json(eef_reference)["norm_stats"]
    loaded = [load_source(path, selection) for path in sources]
    features = loaded[0]["info"]["features"]
    expects_eef = "state_eef" in features and "action_eef" in features
    for source in loaded[1:]:
        if source["info"]["features"] != features:
            raise ValueError(f"feature schema differs: {source['root']}")

    episodes = []
    stats_rows = []
    mapping = []
    tasks = []
    stat_parts = []
    offset = 0
    for source in loaded:
        for origin_episode in source["selected"]:
            record = source["episodes"][origin_episode]
            prompt = episode_task(record)
            if prompt not in tasks:
                tasks.append(prompt)
            task_index = tasks.index(prompt)
            path = source["root"] / f"data/chunk-{origin_episode // 1000:03d}/episode_{origin_episode:06d}.parquet"
            table = pq.read_table(path)
            length = record["length"]
            if table.num_rows != length or table.num_rows != source["mapping"][origin_episode]["length"]:
                raise ValueError(f"row count mismatch: {path}")
            state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
            action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
            if state.shape != (length, 24) or action.shape != (length, 24):
                raise ValueError(f"expected {(length, 24)} state and action in {path}")
            if not np.isfinite(state).all() or not np.isfinite(action).all():
                raise ValueError(f"non-finite state or action in {path}")
            timestamp = np.asarray(table["timestamp"].to_pylist(), dtype=np.float32)
            if not np.allclose(timestamp, np.arange(length, dtype=np.float32) / 15, atol=1e-6):
                raise ValueError(f"timestamps are not a 15 Hz grid in {path}")
            if check_image_files:
                check_images(table, origin_episode)
            episode_index = len(episodes)
            table = replace_column(table, "episode_index", [episode_index] * length)
            table = replace_column(table, "index", list(range(offset, offset + length)))
            table = replace_column(table, "task_index", [task_index] * length)
            dest = output / f"data/chunk-{episode_index // 1000:03d}/episode_{episode_index:06d}.parquet"
            dest.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, dest)
            episodes.append({"episode_index": episode_index, "tasks": [prompt], "length": length})
            episode_stats = {}
            for key, values in (("observation.state", state), ("action", action)):
                episode_stats[key] = {
                    "min": values.min(0).tolist(),
                    "max": values.max(0).tolist(),
                    "mean": values.mean(0).tolist(),
                    "std": values.std(0).tolist(),
                    "count": [length],
                }
            stats_rows.append({"episode_index": episode_index, "stats": episode_stats})
            origin = dict(source["mapping"][origin_episode])
            origin.update({
                "episode_index": episode_index,
                "origin_episode_index": origin_episode,
                "origin_root": str(source["root"]),
                "origin_name": source["root"].name,
                "task_index": task_index,
                "task": prompt,
                "split": "train" if role == "train" else "validation",
            })
            mapping.append(origin)
            stat_parts.append(np.concatenate([state, action], axis=1))
            if expects_eef and not {"state_eef", "action_eef"} <= set(table.column_names):
                raise ValueError(f"missing end-effector columns in {path}")
            if "state_eef" in table.column_names:
                state_eef = np.asarray(table["state_eef"].to_pylist(), dtype=np.float64)
                action_eef = np.asarray(table["action_eef"].to_pylist(), dtype=np.float64)
                if state_eef.shape != (length, 16) or action_eef.shape != (length, 16):
                    raise ValueError(f"expected {(length, 16)} state_eef and action_eef in {path}")
                if not np.isfinite(state_eef).all() or not np.isfinite(action_eef).all():
                    raise ValueError(f"non-finite end-effector values in {path}")
                for key, values in (("state_eef", state_eef), ("action_eef", action_eef)):
                    episode_stats[key] = {
                        "min": values.min(0).tolist(),
                        "max": values.max(0).tolist(),
                        "mean": values.mean(0).tolist(),
                        "std": values.std(0).tolist(),
                        "count": [length],
                    }
            offset += length
            print(f"{source['root'].name} episode {origin_episode} -> {episode_index} frames={offset}", flush=True)

    if not stat_parts:
        raise ValueError("no training frames")
    info = dict(loaded[0]["info"])
    info.update({
        "total_episodes": len(episodes),
        "total_frames": offset,
        "total_tasks": len(tasks),
        "total_videos": 0,
        "total_chunks": (len(episodes) + 999) // 1000,
        "splits": {"train": f"0:{len(episodes)}"} if role == "train" else {"test": f"0:{len(episodes)}"},
    })
    write_jsonl(output / "meta/episodes.jsonl", episodes)
    write_jsonl(output / "meta/episodes_stats.jsonl", stats_rows)
    write_jsonl(output / "meta/tasks.jsonl", [{"task_index": index, "task": prompt} for index, prompt in enumerate(tasks)])
    write_json(output / "meta/info.json", info)
    episode_ids = list(range(len(episodes)))
    write_json(output / "train_episodes.json", episode_ids if role == "train" else [])
    write_json(output / "validation_episodes.json", [] if role == "train" else episode_ids)
    if role == "test":
        write_json(output / "test_episodes.json", episode_ids)
    write_json(output / "source_mapping.json", mapping)
    if role == "train":
        outlier_report = filter_dataset(output, episode_ids)
    else:
        outlier_report = filter_dataset(output, [], reference_stats={**copied, **copied_eef})
    columns = outlier_report["train_columns"]

    def quantile_stats(values):
        values = np.asarray(values, dtype=np.float64)
        return {
            "mean": values.mean(0).tolist(),
            "std": values.std(0).tolist(),
            "q01": np.quantile(values, 0.01, axis=0).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).tolist(),
        }

    if role == "train":
        norm = {"state": quantile_stats(columns["observation.state"]), "actions": quantile_stats(columns["action"])}
        write_json(output / "assets/linglong_h/norm_stats.json", {"norm_stats": norm})
        if "state_eef" in columns:
            eef_norm = {
                "state_eef": quantile_stats(columns["state_eef"]),
                "action_eef": quantile_stats(columns["action_eef"]),
            }
            write_json(output / "assets/linglong_h/eef_norm_stats.json", {"norm_stats": eef_norm})
    else:
        write_json(output / "assets/linglong_h/norm_stats.json", {"norm_stats": copied})
        write_json(output / "assets/linglong_h/eef_norm_stats.json", {"norm_stats": copied_eef})
    per_source = []
    for source in loaded:
        per_source.append({
            "root": str(source["root"]),
            "episodes": len(source["selected"]),
            "frames": sum(source["episodes"][episode]["length"] for episode in source["selected"]),
            "manifest_sha256": sha256(source["root"] / "manifest.json"),
            "prompt": episode_task(source["episodes"][source["selected"][0]]),
        })
    write_json(output / "manifest.json", {
        "status": "complete",
        "layout": joint_layout(loaded[0]["manifest"]),
        "action_semantics": "absolute q_exp",
        "fps": 15,
        "sources": per_source,
        "tasks": tasks,
        "segments": len(episodes),
        "train_segments": len(episodes) if role == "train" else 0,
        "validation_segments": 0 if role == "train" else len(episodes),
        "frames": outlier_report["frames"],
        "train_stat_frames": int(len(columns["action"])) if role == "train" else 0,
        "quantile_outliers": {key: value for key, value in outlier_report.items() if key != "train_columns"},
        "image_storage": "absolute original JPEG references; sources must remain accessible",
        "norm": "q01/q99 recomputed on the concatenated training frames after obvious quantile outliers are removed" if role == "train" else "copied from the merged training set",
        "norm_reference": None if role == "train" else str(norm_reference),
        "holdout": "source validation and test directories were not inputs" if role == "train" else "merged from the source test episodes",
        "order": "sources are concatenated in CLI order; episode ids restart at 0",
    })
    from dataset_catalog import update_catalog
    update_catalog(output)
    print(f"COMPLETE episodes={len(episodes)} frames={outlier_report['frames']} tasks={len(tasks)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", default=DEFAULT_SOURCES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-image-check", action="store_true", help="Do not stat every referenced JPEG")
    parser.add_argument("--role", choices=("train", "test"), default="train")
    parser.add_argument("--norm-reference", type=Path, help="Training norm_stats.json copied into a test merge")
    args = parser.parse_args()
    if len(args.sources) < 2:
        raise SystemExit("pass at least two source datasets")
    merge(args.sources, args.output, check_image_files=not args.skip_image_check, role=args.role, norm_reference=args.norm_reference)


if __name__ == "__main__":
    main()
