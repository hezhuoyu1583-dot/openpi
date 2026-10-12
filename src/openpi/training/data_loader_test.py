import dataclasses
import types

import jax
import pytest
import torch

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_validate_local_lerobot_metadata_requires_v21_episode_stats(tmp_path):
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    (meta_dir / "info.json").write_text('{"codebase_version": "v2.1"}')

    with pytest.raises(FileNotFoundError, match="episodes_stats.jsonl"):
        _data_loader._validate_local_lerobot_metadata(str(tmp_path))


def test_validate_local_lerobot_metadata_accepts_complete_v21_metadata(tmp_path):
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    (meta_dir / "info.json").write_text('{"codebase_version": "v2.1"}')
    (meta_dir / "episodes_stats.jsonl").write_text("")

    _data_loader._validate_local_lerobot_metadata(str(tmp_path))


def test_validate_local_lerobot_metadata_accepts_default_cache():
    _data_loader._validate_local_lerobot_metadata(None)


def test_load_episode_indices_text_and_json(tmp_path):
    available = {0: {}, 2: {}}
    txt = tmp_path / "episodes.txt"
    txt.write_text("0\n2\n")
    assert _data_loader._load_episode_indices(str(txt), available) == [0, 2]
    js = tmp_path / "episodes.json"
    js.write_text('{"episode_indices": [2, 0]}')
    assert _data_loader._load_episode_indices(str(js), available) == [2, 0]


def test_load_episode_indices_rejects_missing_and_duplicate(tmp_path):
    available = {0: {}}
    missing = tmp_path / "missing.txt"
    missing.write_text("3\n")
    with pytest.raises(ValueError, match="absent"):
        _data_loader._load_episode_indices(str(missing), available)
    duplicate = tmp_path / "duplicate.txt"
    duplicate.write_text("0\n0\n")
    with pytest.raises(ValueError, match="duplicate"):
        _data_loader._load_episode_indices(str(duplicate), available)


def test_restore_sparse_episode_index():
    dataset = types.SimpleNamespace(
        episodes=[0, 4, 9],
        episode_data_index={
            "from": torch.tensor([0, 3, 8]),
            "to": torch.tensor([3, 8, 10]),
        },
    )

    _data_loader._restore_sparse_episode_index(dataset)

    assert len(dataset.episode_data_index["from"]) == 10
    assert dataset.episode_data_index["from"][[0, 4, 9]].tolist() == [0, 3, 8]
    assert dataset.episode_data_index["to"][[0, 4, 9]].tolist() == [3, 8, 10]
