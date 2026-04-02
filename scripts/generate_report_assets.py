"""Generate local chart assets for experiment documentation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate documentation chart assets.")
    parser.add_argument("--metrics-csv", type=str, default="logs/train_metrics.csv")
    parser.add_argument("--output-dir", type=str, default="docs/assets")
    parser.add_argument("--baseline-update", type=int, default=366)
    parser.add_argument("--baseline-eval", type=float, required=True)
    parser.add_argument("--final-update", type=int, default=500)
    parser.add_argument("--final-eval", type=float, required=True)
    return parser.parse_args()


def load_metrics(csv_path: Path) -> dict[str, list[float]]:
    """Load scalar training metrics from the canonical CSV file."""
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "update": [float(row["update"]) for row in rows],
        "total_loss": [float(row["total_loss"]) for row in rows],
        "value_loss": [float(row["value_loss"]) for row in rows],
        "entropy": [float(row["entropy"]) for row in rows],
    }


def plot_training_summary(metrics: dict[str, list[float]], output_path: Path) -> None:
    """Render the main training summary chart."""
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    series = (
        ("total_loss", "Total loss", "#f25f5c"),
        ("value_loss", "Value loss", "#247ba0"),
        ("entropy", "Entropy", "#70c1b3"),
    )
    for axis, (metric_name, label, color) in zip(axes, series, strict=True):
        axis.plot(metrics["update"], metrics[metric_name], color=color, linewidth=1.8)
        axis.axvspan(200.5, 366.5, color="#d9d9d9", alpha=0.35)
        axis.axvline(366, color="#7a7a7a", linestyle="--", linewidth=1.0)
        axis.axvline(500, color="#1f1f1f", linestyle=":", linewidth=1.0)
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0].set_title("Shared PPO training metrics with the interactive logging gap highlighted")
    axes[-1].set_xlabel("Update")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_eval_comparison(
    baseline_update: int,
    baseline_eval: float,
    final_update: int,
    final_eval: float,
    output_path: Path,
) -> None:
    """Render a milestone comparison chart for deterministic evaluation."""
    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    labels = [f"Update {baseline_update}", f"Update {final_update}"]
    values = [baseline_eval, final_eval]
    colors = ["#74c69d", "#2d6a4f"]
    axis.bar(labels, values, color=colors, width=0.58)
    axis.axhline(0.0, color="#3c3c3c", linewidth=1.0)
    axis.set_ylabel("Average deterministic return (5 episodes)")
    axis.set_title("Milestone evaluation comparison")
    for idx, value in enumerate(values):
        axis.text(idx, value + (0.25 if value >= 0 else -0.65), f"{value:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_update_coverage(metrics: dict[str, list[float]], output_path: Path) -> None:
    """Render a compact update coverage chart showing the missing interval."""
    fig, axis = plt.subplots(figsize=(13, 2.8))
    axis.scatter(metrics["update"], [1.0] * len(metrics["update"]), s=12, color="#2563eb")
    axis.axvspan(200.5, 366.5, color="#d9d9d9", alpha=0.45, label="Interactive-only gap")
    axis.set_yticks([])
    axis.set_xlabel("Logged update")
    axis.set_title("Canonical CSV coverage")
    axis.set_xlim(0, 505)
    axis.grid(alpha=0.2, axis="x")
    axis.legend(loc="upper center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    metrics_csv = Path(args.metrics_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(metrics_csv)
    plot_training_summary(metrics, output_dir / "training-summary.png")
    plot_eval_comparison(
        baseline_update=args.baseline_update,
        baseline_eval=args.baseline_eval,
        final_update=args.final_update,
        final_eval=args.final_eval,
        output_path=output_dir / "eval-comparison.png",
    )
    plot_update_coverage(metrics, output_dir / "update-coverage.png")


if __name__ == "__main__":
    main()
