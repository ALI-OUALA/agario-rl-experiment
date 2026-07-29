"""Build documentation charts from tracked training and evaluation results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


METRICS = ("total_loss", "value_loss", "entropy")
COLORS = {"baseline": "#2563eb", "human": "#ea580c"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate experiment charts.")
    parser.add_argument("--baseline-metrics", default="logs/train_metrics_recovered.csv")
    parser.add_argument("--human-metrics", default="logs/human_ready_v1_train_metrics.csv")
    parser.add_argument("--baseline-eval", default="logs/human_ready_baseline_500.json")
    parser.add_argument("--human-eval", default="logs/human_ready_v1_eval.json")
    parser.add_argument("--output-dir", default="docs/assets")
    return parser.parse_args()


def load_metrics(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No training rows found in {path}")
    required = {"update", *METRICS}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    return [{key: float(row[key]) for key in required} for row in rows]


def load_evaluation(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    required = {
        "win_rate",
        "mean_survival_steps",
        "mean_final_mass",
        "corner_time_fraction",
        "threat_avoidance_rate",
        "small_target_pressure_rate",
    }
    missing = required.difference(data)
    if missing:
        raise ValueError(f"{path} is missing metrics: {', '.join(sorted(missing))}")
    return {key: float(data[key]) for key in required}


def rolling_mean(values: list[float], window: int = 10) -> list[float]:
    result: list[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result.append(sum(values[start : index + 1]) / (index - start + 1))
    return result


def contiguous_runs(rows: list[dict[str, float]]) -> list[list[dict[str, float]]]:
    runs = [[rows[0]]]
    for row in rows[1:]:
        if row["update"] > runs[-1][-1]["update"] + 1:
            runs.append([])
        runs[-1].append(row)
    return runs


def missing_ranges(rows: list[dict[str, float]]) -> list[tuple[int, int]]:
    updates = sorted(int(row["update"]) for row in rows)
    return [
        (left + 1, right - 1)
        for left, right in zip(updates, updates[1:], strict=False)
        if right > left + 1
    ]


def plot_training(
    baseline: list[dict[str, float]],
    human: list[dict[str, float]],
    output: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    gaps = missing_ranges(baseline)
    for axis, metric in zip(axes, METRICS, strict=True):
        for start, end in gaps:
            axis.axvspan(start - 0.5, end + 0.5, color="#e5e7eb", alpha=0.75, zorder=0)
        for rows, label, color in (
            (baseline, "Baseline", COLORS["baseline"]),
            (human, "Human-ready v1", COLORS["human"]),
        ):
            for index, run in enumerate(contiguous_runs(rows)):
                axis.plot(
                    [row["update"] for row in run],
                    rolling_mean([row[metric] for row in run]),
                    label=label if index == 0 else None,
                    color=color,
                    linewidth=2,
                )
        axis.set_ylabel(metric.replace("_", " ").title())
        axis.grid(alpha=0.2)
    for start, end in gaps:
        axes[0].text(
            (start + end) / 2,
            0.5,
            f"No log data\nupdates {start}–{end}",
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="center",
            color="#4b5563",
            fontsize=10,
            fontweight="bold",
        )
    axes[0].set_title("PPO training diagnostics (10-update rolling mean)")
    axes[0].legend()
    axes[-1].set_xlabel("Update")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_evaluation(
    baseline: dict[str, float],
    human: dict[str, float],
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    comparisons = (
        ("Mean survival", "mean_survival_steps", "Steps", 1.0),
        ("Mean final mass", "mean_final_mass", "Mass", 1.0),
        ("Behavior rates", None, "Percent", 100.0),
        ("Win rate", "win_rate", "Percent", 100.0),
    )
    for axis, (title, key, ylabel, scale) in zip(axes.flat, comparisons, strict=True):
        if key is None:
            labels = ["Corner time", "Threat avoidance", "Target pressure"]
            keys = [
                "corner_time_fraction",
                "threat_avoidance_rate",
                "small_target_pressure_rate",
            ]
            x = range(len(labels))
            width = 0.36
            axis.bar(
                [value - width / 2 for value in x],
                [baseline[item] * scale for item in keys],
                width,
                label="Baseline",
                color=COLORS["baseline"],
            )
            axis.bar(
                [value + width / 2 for value in x],
                [human[item] * scale for item in keys],
                width,
                label="Human-ready v1",
                color=COLORS["human"],
            )
            axis.set_xticks(list(x), labels, rotation=12)
            axis.legend()
        else:
            values = [baseline[key] * scale, human[key] * scale]
            bars = axis.bar(
                ["Baseline", "Human-ready v1"],
                values,
                color=[COLORS["baseline"], COLORS["human"]],
                width=0.6,
            )
            axis.bar_label(bars, fmt="%.2f", padding=3)
            if key == "win_rate":
                axis.set_ylim(0, 100)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2, axis="y")
    fig.suptitle("20-episode human-readiness evaluation", fontsize=15)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_coverage(
    baseline: list[dict[str, float]],
    human: list[dict[str, float]],
    output: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(12, 3.2))
    gaps = missing_ranges(baseline)
    for index, (start, end) in enumerate(gaps):
        axis.axvspan(
            start - 0.5,
            end + 0.5,
            color="#e5e7eb",
            alpha=0.75,
            label="No baseline log data" if index == 0 else None,
        )
    for y, rows, label, color in (
        (1, baseline, "Baseline", COLORS["baseline"]),
        (0, human, "Human-ready v1", COLORS["human"]),
    ):
        axis.scatter([row["update"] for row in rows], [y] * len(rows), s=14, color=color)
    axis.set_yticks([0, 1], ["Human-ready v1", "Baseline"])
    axis.set_xlabel("Logged update")
    axis.set_title("Training log coverage (missing updates are blank)")
    axis.grid(alpha=0.2, axis="x")
    if gaps:
        axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    baseline_metrics = load_metrics(Path(args.baseline_metrics))
    human_metrics = load_metrics(Path(args.human_metrics))
    baseline_eval = load_evaluation(Path(args.baseline_eval))
    human_eval = load_evaluation(Path(args.human_eval))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_training(baseline_metrics, human_metrics, output_dir / "training-summary.png")
    plot_evaluation(baseline_eval, human_eval, output_dir / "eval-comparison.png")
    plot_coverage(baseline_metrics, human_metrics, output_dir / "update-coverage.png")


if __name__ == "__main__":
    main()
