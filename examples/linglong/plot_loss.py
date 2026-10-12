"""Plot the completed Linglong SFT loss with an understandable smoothing label."""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "data/training_logs/linglong/linglong15_b64_5k"
RUN_ID = "531e4dab90f64fc08e425211e16cfd67"


def main() -> None:
    rows = list(csv.DictReader((LOG_DIR / "metrics.csv").open()))
    rows = [row for row in rows if row["run_id"] == RUN_ID]
    steps = np.asarray([int(row["step"]) for row in rows])
    losses = np.asarray([float(row["loss"]) for row in rows])
    moving = np.asarray([losses[max(0, i - 9) : i + 1].mean() for i in range(len(losses))])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax in axes:
        ax.plot(steps, losses, color="#86a7d5", alpha=0.7, lw=1, label="Logged training loss")
        ax.plot(
            steps,
            moving,
            color="#174d8c",
            lw=2,
            label="200-step moving mean (10 logging points)",
        )
        for step in range(500, 5001, 500):
            ax.axvline(step, color="gray", alpha=0.18, lw=0.8)
        ax.set(xlabel="Logged step (zero-based)", ylabel="Flow-matching training loss")
        ax.grid(alpha=0.2)
    axes[0].set(title="Full training run", xlim=(0, 5000))
    axes[0].legend(fontsize=8)
    axes[1].set(title="Late training (zoom)", xlim=(1000, 5000), ylim=(0.035, 0.085))
    fig.suptitle("Linglong pi0.5 SFT | batch 64 | 5,000 updates")
    fig.text(
        0.5,
        0.01,
        "The moving mean is only a visualization; it does not change training or validation loss.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(LOG_DIR / "loss_completed_run.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
