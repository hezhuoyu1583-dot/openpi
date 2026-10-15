import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from quantile_outliers import filter_dataset, keep_mask


def test_constant_joint_outlier_is_removed_and_motion_is_kept():
    state = np.zeros((100, 24))
    action = np.zeros((100, 24))
    action[:, 0] = np.linspace(-1, 1, 100)
    action[:, 21] = 0.17452
    action[40, 21] = 0.0
    columns = {"observation.state": state, "action": action}
    keep, report = keep_mask(columns, np.ones(100, dtype=bool))
    assert keep.sum() == 99
    assert not keep[40]
    assert report["dropped_frames"] == 1
    refit = action[keep]
    q01, q99 = np.quantile(refit, 0.01, axis=0), np.quantile(refit, 0.99, axis=0)
    norm = np.abs((refit - q01) / (q99 - q01 + 1e-6) * 2 - 1)
    assert norm.max() <= 10


def test_eef_far_from_tight_band_is_removed():
    action = np.zeros((200, 16))
    action[:, 0] = 0.30
    action[10, 0] = 0.42
    action[:, 3] = 1
    action[:, 11] = 1
    keep, _ = keep_mask({"action_eef": action}, np.ones(200, dtype=bool))
    assert not keep[10]
    assert keep.sum() == 199


def test_filter_rewrites_episode_and_stats(tmp_path):
    root = Path(tmp_path)
    count = 150
    action = np.full((count, 24), 0.2, dtype=np.float32)
    action[2, 21] = 0
    state = np.zeros((count, 24), dtype=np.float32)
    table = pa.table({
        "observation.state": pa.FixedSizeListArray.from_arrays(pa.array(state.reshape(-1), type=pa.float32()), 24),
        "action": pa.FixedSizeListArray.from_arrays(pa.array(action.reshape(-1), type=pa.float32()), 24),
        "timestamp": pa.array(np.arange(count, dtype=np.float32) / 15, type=pa.float32()),
        "frame_index": pa.array(np.arange(count), type=pa.int64()),
        "episode_index": pa.array(np.zeros(count, dtype=np.int64), type=pa.int64()),
        "index": pa.array(np.arange(count), type=pa.int64()),
    })
    dest = root / "data/chunk-000/episode_000000.parquet"
    dest.parent.mkdir(parents=True)
    pq.write_table(table, dest)
    (root / "meta").mkdir()
    (root / "meta/episodes.jsonl").write_text(json.dumps({"episode_index": 0, "tasks": ["task"], "length": count}) + "\n")
    (root / "meta/episodes_stats.jsonl").write_text("{}\n")
    (root / "meta/info.json").write_text(json.dumps({
        "total_episodes": 1, "total_frames": count, "total_chunks": 1,
        "features": {"observation.state": {}, "action": {}},
    }))
    (root / "source_mapping.json").write_text(json.dumps([
        {"episode_index": 0, "length": count, "source_row_indices": list(range(0, count * 2, 2))},
    ]))
    (root / "train_episodes.json").write_text("[0]\n")
    report = filter_dataset(root, [0])
    assert report["dropped_frames"] == 1
    assert report["frames"] == count - 1
    saved = pq.read_table(dest)
    assert saved.num_rows == count - 1
    saved_action = np.asarray(saved.column("action").to_pylist())
    assert not np.any(saved_action[:, 21] == 0)
    assert np.allclose(np.asarray(saved.column("timestamp").to_pylist()), np.arange(count - 1) / 15, atol=1e-5)
    info = json.loads((root / "meta/info.json").read_text())
    assert info["total_frames"] == count - 1
    mapping = json.loads((root / "source_mapping.json").read_text())
    assert mapping[0]["source_row_indices"] == [i * 2 for i in range(count) if i != 2]
    assert mapping[0]["length"] == count - 1
