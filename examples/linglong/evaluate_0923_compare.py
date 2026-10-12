"""Compare pi0.5 base and Linglong checkpoints on the held-out 0923 set."""
import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import jax
import torch
from openpi.models import model as model_lib

from openpi.policies import policy_config
from openpi.training import config, data_loader
from openpi.training import checkpoints as checkpoint_lib


def _group_metrics(pred, target):
    err = np.abs(pred - target)
    groups = {
        "left_arm": list(range(7)),
        "left_gripper": [7],
        "right_arm": list(range(8, 15)),
        "right_gripper": [15],
        "waist": list(range(16, 20)),
        "head": [20, 21],
    }
    return {name: float(err[..., idx].mean()) for name, idx in groups.items()}


def _episodes_windows(root, horizon):
    mapping = json.loads((root / "source_mapping.json").read_text())
    indices, offset = [], 0
    for rec in mapping:
        n = int(rec["length"])
        indices.extend(range(offset, offset + max(0, n - horizon + 1)))
        offset += n
    return indices


def _load_norm(path):
    return checkpoint_lib.load_norm_stats(path / "assets", "linglong_h")


def main(args):
    root = args.test_root.resolve()
    new_cfg = config.get_config("pi05_linglong_0923_sft")
    new_dc = new_cfg.data.create(new_cfg.assets_dirs, new_cfg.model)
    new_dc = dataclasses.replace(
        new_dc,
        repo_id="linglong_0923_test",
        dataset_root=str(root),
        episode_indices_path=str(root / "test_episodes.json"),
    )
    dataset = data_loader.create_torch_dataset(new_dc, new_cfg.model.action_horizon, new_cfg.model)
    all_indices = _episodes_windows(root, new_cfg.model.action_horizon)
    rng = np.random.default_rng(args.seed)
    if args.samples and args.samples < len(all_indices):
        indices = sorted(rng.choice(all_indices, args.samples, replace=False).tolist())
    else:
        indices = all_indices

    # All entries use the same 0923 held-out observations and targets.  Each
    # policy keeps the normalization statistics it was trained with.
    models = [("pi05_base", args.base, new_cfg, new_dc.norm_stats)]
    for step in (500, 1000, 1500, 2000):
        p = args.new_root / str(step)
        models.append((f"linglong0923_{step}", p, new_cfg, _load_norm(p)))
    old_cfg = config.get_config("pi05_linglong_sft")
    old_p = args.old_root / "5000"
    models.append(("linglong0714_5000", old_p, old_cfg, _load_norm(old_p)))

    args.output.mkdir(parents=True, exist_ok=False)
    summary = {"test_root": str(root), "samples": len(indices), "seed": args.seed, "models": {}}
    (args.output / "sample_indices.json").write_text(json.dumps(indices, indent=2))

    for name, checkpoint, cfg, norm in models:
        checkpoint = Path(checkpoint)
        policy = policy_config.create_trained_policy(
            cfg, checkpoint, norm_stats=norm, pytorch_device=args.device
        )
        pred, target, hold, timings = [], [], [], []
        noise_rng = np.random.default_rng(args.seed)
        for start_idx in range(0, len(indices), args.batch_size):
            samples = [dataset[idx] for idx in indices[start_idx:start_idx+args.batch_size]]
            inputs = [policy._input_transform({k: (v if k == 'prompt' else np.asarray(v))
                      for k,v in sample.items() if k=='prompt' or k.startswith('observation.')})
                      for sample in samples]
            batch = jax.tree.map(lambda *xs: torch.from_numpy(np.stack(xs)).to(args.device), *inputs)
            noise = torch.from_numpy(noise_rng.standard_normal(
                (len(samples), cfg.model.action_horizon, cfg.model.action_dim), dtype=np.float32)).to(args.device)
            torch.cuda.synchronize()
            start = time.perf_counter()
            actions = policy._sample_actions(args.device, model_lib.Observation.from_dict(batch), noise=noise).cpu().numpy()
            timings.append((time.perf_counter() - start)/len(samples))
            for sample, inp, action in zip(samples, inputs, actions):
                out = policy._output_transform({'state': inp['state'], 'actions': action})
                if not np.isfinite(out['actions']).all(): raise ValueError('Nonfinite prediction')
                pred.append(np.asarray(out['actions'], dtype=np.float32))
                target.append(np.asarray(sample['action'], dtype=np.float32))
                hold.append(np.tile(np.asarray(sample["observation.state"], dtype=np.float32),
                                (cfg.model.action_horizon, 1)))
            print(f"{name}: {len(pred)}/{len(indices)}", flush=True)
        pred, target, hold = np.asarray(pred), np.asarray(target), np.asarray(hold)
        report = {
            "checkpoint": str(checkpoint), "samples": len(indices), "seed": args.seed,
            "mae_by_group": _group_metrics(pred, target),
            "hold_baseline_mae": _group_metrics(hold, target),
            "mean_inference_ms": float(np.mean(timings) * 1000),
            "p95_inference_ms": float(np.percentile(timings, 95) * 1000),
            "not_task_success_rate": True,
            "batch_size": args.batch_size,
            "timing_note": "Amortized batched model time, not single-observation control latency",
        }
        (args.output / f"{name}.json").write_text(json.dumps(report, indent=2))
        np.savez_compressed(args.output / f"{name}.npz", predictions=pred, targets=target)
        summary["models"][name] = report
        print(name, json.dumps(report["mae_by_group"]), flush=True)
        del policy
        import gc
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test-root", type=Path, default=Path("data/linglong_0923_sft_test_15hz"))
    p.add_argument("--new-root", type=Path, default=Path("data/checkpoints/pi05_linglong_0923_sft/linglong0923_b64_2000"))
    p.add_argument("--old-root", type=Path, default=Path("data/checkpoints/pi05_linglong_sft/linglong15_b64_5k"))
    p.add_argument("--base", type=Path, default=Path("data/checkpoints/pi05_base_pytorch"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, default=0, help="0 evaluates all full 25-step windows")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=16)
    main(p.parse_args())
