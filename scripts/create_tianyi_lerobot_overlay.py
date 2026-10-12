#!/usr/bin/env python3
"""Create a writable LeRobot metadata overlay without modifying the source dataset."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow.parquet as pq

_STATE_KEY = "observation.state"
_ACTION_KEY = "action"
_REQUIRED_META = ("info.json", "episodes.jsonl", "tasks.jsonl", "stats.json", "modality.json")


def _feature_stats(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"Expected a non-empty 2-D array, got {values.shape}.")
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(values.shape[0])],
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def _read_column(path: Path, key: str) -> np.ndarray:
    return np.asarray(pq.read_table(path, columns=[key]).column(key).to_pylist(), dtype=np.float32)


def _episode_stats(source: Path, info: dict, episode_index: int) -> dict:
    chunk = episode_index // int(info["chunks_size"])
    relative_path = info["data_path"].format(episode_chunk=chunk, episode_index=episode_index)
    parquet_path = source / relative_path
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Missing source parquet for episode {episode_index}: {parquet_path}")
    return {
        "episode_index": episode_index,
        "stats": {
            _STATE_KEY: _feature_stats(_read_column(parquet_path, _STATE_KEY)),
            _ACTION_KEY: _feature_stats(_read_column(parquet_path, _ACTION_KEY)),
        },
    }


def create_overlay(source: Path, destination: Path, *, overwrite: bool = False) -> None:
    source = source.resolve()
    destination = destination.resolve()
    source_meta = source / "meta"
    if not source_meta.is_dir():
        raise FileNotFoundError(f"Source is missing metadata directory: {source_meta}")
    with (source_meta / "info.json").open(encoding="utf-8") as handle:
        info = json.load(handle)
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"Expected a LeRobot v2.1 source, got {info.get('codebase_version')!r}.")
    for name in _REQUIRED_META:
        if not (source_meta / name).is_file():
            raise FileNotFoundError(f"Source is missing required metadata file: {source_meta / name}")
    if destination.exists() or destination.is_symlink():
        if not overwrite:
            raise FileExistsError(f"{destination} exists; choose another overlay path or pass --overwrite.")
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()

    destination.mkdir(parents=True)
    destination_meta = destination / "meta"
    destination_meta.mkdir()
    for name in _REQUIRED_META:
        shutil.copy2(source_meta / name, destination_meta / name)

    for name in ("data", "videos"):
        source_tree = source / name
        if not source_tree.is_dir():
            raise FileNotFoundError(f"Source is missing required LeRobot tree: {source_tree}")
        (destination / name).symlink_to(source_tree, target_is_directory=True)

    episodes = [
        json.loads(line)
        for line in (source_meta / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stats_path = destination_meta / "episodes_stats.jsonl"
    with stats_path.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            row = _episode_stats(source, info, int(episode["episode_index"]))
            handle.write(json.dumps(row, ensure_ascii=False) + chr(10))

    manifest = {
        "source": str(source),
        "overlay": str(destination),
        "source_is_modified": False,
        "symlinked_trees": ["data", "videos"],
        "generated_files": ["meta/episodes_stats.jsonl"],
    }
    (destination / "overlay_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + chr(10), encoding="utf-8"
    )
    print(f"Created metadata overlay: {destination}")
    print(f"Generated {len(episodes)} episode stats rows: {stats_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="Existing read-only LeRobot v2.1 root.")
    parser.add_argument("--dst", type=Path, required=True, help="Writable overlay root.")
    parser.add_argument("--overwrite", action="store_true", help="Replace only the destination overlay.")
    args = parser.parse_args()
    create_overlay(args.src, args.dst, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
