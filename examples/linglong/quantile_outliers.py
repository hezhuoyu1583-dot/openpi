"""Drop frames that explode quantile normalization.

After each 1%/99% fit, a frame is an obvious outlier when any joint or
end-effector dimension normalizes beyond ±10. Ordinary tails of a dimension
that actually moves stay near ±1. The fit uses training frames only; the same
thresholds are applied to every frame, then the quantiles are refit until the
remaining training frames are inside the limit.
"""
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

OUTLIER_NORM_LIMIT = 10.0
MAX_ROUNDS = 8
MAX_DROP_FRACTION = 0.05
COLUMNS = ("observation.state", "action", "state_eef", "action_eef")


def _quantile_norm(values, q01, q99):
    return (values - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def keep_mask(columns, train_rows, *, limit=OUTLIER_NORM_LIMIT, max_rounds=MAX_ROUNDS, max_drop_fraction=MAX_DROP_FRACTION, reference=None):
    """Return the rows to keep and a short report. columns map a name to an (N, D) array."""
    if not columns:
        raise ValueError("no columns to check")
    count = len(train_rows)
    for name, values in columns.items():
        values = np.asarray(values)
        if values.ndim != 2 or len(values) != count:
            raise ValueError(f"{name} must have shape ({count}, D)")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} has non-finite values")
    train_rows = np.asarray(train_rows, dtype=bool)
    if int(train_rows.sum()) < 2:
        raise ValueError("need at least two training frames to fit quantiles")
    keep = np.ones(count, dtype=bool)
    examples = []
    rounds = 0
    dropped_by_column = {name: 0 for name in columns}
    while rounds < max_rounds:
        rounds += 1
        newly = np.zeros(count, dtype=bool)
        worst = []
        for name, values in columns.items():
            fit = train_rows & keep
            if reference is None:
                chosen = values[fit]
                q01 = np.quantile(chosen, 0.01, axis=0)
                q99 = np.quantile(chosen, 0.99, axis=0)
            else:
                q01 = np.asarray(reference[name]["q01"], dtype=np.float64)
                q99 = np.asarray(reference[name]["q99"], dtype=np.float64)
            norm = np.abs(_quantile_norm(values, q01, q99))
            bad = keep & (norm.max(axis=1) > limit)
            newly |= bad
            dropped_by_column[name] += int(bad.sum())
            if bad.any():
                score = norm[bad].max(axis=1)
                pick = np.flatnonzero(bad)[int(np.argmax(score))]
                dim = int(np.argmax(norm[pick]))
                worst.append((float(norm[pick, dim]), name, pick, dim, float(values[pick, dim])))
        if not newly.any():
            break
        already = int((~keep).sum())
        if already + int(newly.sum()) > max_drop_fraction * count:
            raise ValueError(
                f"quantile outlier filter would drop {already + int(newly.sum())} of {count} frames; "
                "refusing because that is larger than an obvious-outlier tail"
            )
        keep[newly] = False
        if reference is not None:
            for item in worst:
                examples.append({"abs_norm": item[0], "column": item[1], "row": int(item[2]), "dim": item[3], "value": item[4]})
            break
        for item in sorted(worst, reverse=True)[:5]:
            examples.append({"abs_norm": item[0], "column": item[1], "row": int(item[2]), "dim": item[3], "value": item[4]})
    else:
        raise ValueError(f"quantile outliers were still present after {max_rounds} refits")
    report = {
        "limit": limit,
        "rounds": rounds,
        "dropped_frames": int((~keep).sum()),
        "dropped_training_frames": int((train_rows & ~keep).sum()),
        "flagged_by_column": dropped_by_column,
        "examples": examples[:20],
    }
    return keep, report


def _load_json(path):
    return json.loads(Path(path).read_text())


def _dump_json(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2)
    if Path(path).is_file() and Path(path).read_text().endswith("\n"):
        text += "\n"
    else:
        text += "\n"
    Path(path).write_text(text)


def _replace(table, name, values, typ):
    index = table.schema.get_field_index(name)
    return table.set_column(index, table.schema.field(index), pa.array(values, type=typ))


def _episode_stats(table, keys):
    stats = {}
    count = table.num_rows
    for key in keys:
        values = np.asarray(table.column(key).to_pylist(), dtype=np.float64)
        stats[key] = {
            "min": values.min(0).tolist(),
            "max": values.max(0).tolist(),
            "mean": values.mean(0).tolist(),
            "std": values.std(0).tolist(),
            "count": [count],
        }
    return stats


def _write_table(table, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp, compression="snappy")
    tmp.replace(path)


def filter_dataset(root, train_episode_ids, *, reference_stats=None, limit=OUTLIER_NORM_LIMIT):
    """Delete obvious quantile outliers from a LeRobot dataset and renumber what remains."""
    root = Path(root)
    episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
    mapping = _load_json(root / "source_mapping.json")
    if len(mapping) != len(episodes):
        raise ValueError("source_mapping and episodes.jsonl have different lengths")
    info = _load_json(root / "meta/info.json")
    keys = [key for key in COLUMNS if key in info["features"]]
    tables = []
    columns = {key: [] for key in keys}
    train_rows = []
    spans = []
    train_ids = set(int(episode) for episode in train_episode_ids)
    for record, item in zip(episodes, mapping):
        episode = int(record["episode_index"])
        if episode != int(item["episode_index"]):
            raise ValueError(f"mapping episode {item['episode_index']} != {episode}")
        path = root / f"data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet"
        table = pq.read_table(path)
        if table.num_rows != record["length"] or len(item["source_row_indices"]) != table.num_rows:
            raise ValueError(f"{path} row count does not match metadata")
        tables.append(table)
        spans.append(table.num_rows)
        train_rows.extend([episode in train_ids] * table.num_rows)
        for key in keys:
            columns[key].append(np.asarray(table.column(key).to_pylist(), dtype=np.float64))
    stacked = {key: np.concatenate(parts) for key, parts in columns.items()}
    reference = None
    if reference_stats is not None:
        reference = {}
        for key in keys:
            stats_key = {"observation.state": "state", "action": "actions"}.get(key, key)
            if stats_key not in reference_stats:
                raise ValueError(f"reference stats are missing {stats_key}")
            reference[key] = reference_stats[stats_key]
    keep, report = keep_mask(stacked, np.asarray(train_rows), limit=limit, reference=reference)
    if report["dropped_frames"] == 0:
        report["frames"] = int(keep.sum())
        report["train_columns"] = {key: values[np.asarray(train_rows)] for key, values in stacked.items()}
        print("quantile outliers: none", flush=True)
        return report

    offset = 0
    cursor = 0
    new_episodes = []
    new_stats = []
    new_mapping = []
    removed_files = []
    for record, item, table, length in zip(episodes, mapping, tables, spans):
        local = keep[cursor:cursor + length]
        cursor += length
        chosen = np.flatnonzero(local)
        if len(chosen) == 0:
            raise ValueError(f"episode {record['episode_index']} would have no frames after outlier removal")
        kept = table.take(chosen)
        kept_length = kept.num_rows
        timestamps = (np.arange(kept_length, dtype=np.float64) / 15).astype(np.float32)
        kept = _replace(kept, "frame_index", np.arange(kept_length, dtype=np.int64), pa.int64())
        kept = _replace(kept, "timestamp", timestamps, pa.float32())
        kept = _replace(kept, "index", np.arange(offset, offset + kept_length, dtype=np.int64), pa.int64())
        path = root / f"data/chunk-{record['episode_index'] // 1000:03d}/episode_{record['episode_index']:06d}.parquet"
        _write_table(kept, path)
        updated = dict(record)
        updated["length"] = kept_length
        new_episodes.append(updated)
        new_stats.append({"episode_index": record["episode_index"], "stats": _episode_stats(kept, keys)})
        updated_item = dict(item)
        updated_item["source_row_indices"] = [item["source_row_indices"][int(row)] for row in chosen]
        updated_item["length"] = kept_length
        new_mapping.append(updated_item)
        offset += kept_length
    if cursor != len(keep):
        raise RuntimeError("frame cursor did not cover the dataset")
    for path in removed_files:
        path.unlink()
    (root / "meta/episodes.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in new_episodes))
    (root / "meta/episodes_stats.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in new_stats))
    _dump_json(root / "source_mapping.json", new_mapping)
    info["total_episodes"] = len(new_episodes)
    info["total_frames"] = offset
    info["total_chunks"] = (len(new_episodes) + 999) // 1000
    _dump_json(root / "meta/info.json", info)
    if removed_files:
        removed_ids = {int(path.stem.split("_")[-1]) for path in removed_files}
        for name in ("train_episodes.json", "validation_episodes.json"):
            path = root / name
            if path.is_file():
                kept_ids = [episode for episode in _load_json(path) if int(episode) not in removed_ids]
                _dump_json(path, kept_ids)
    surviving = np.asarray(train_rows) & keep
    report["frames"] = offset
    report["removed_episodes"] = len(removed_files)
    report["train_columns"] = {key: values[surviving] for key, values in stacked.items()}
    print(
        f"quantile outliers: dropped {report['dropped_frames']} frames "
        f"({report['dropped_training_frames']} training) in {report['rounds']} rounds",
        flush=True,
    )
    return report
