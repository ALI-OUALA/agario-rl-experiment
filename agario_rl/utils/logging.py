"""Training metrics logging utilities."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping


class CsvLogger:
    """Append scalar metrics to a CSV file with auto header creation."""

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = self.file_path.exists() and self.file_path.stat().st_size > 0

    def log(self, row: Mapping[str, float | int]) -> None:
        row_dict = dict(row)
        with self.file_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row_dict.keys()))
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row_dict)


TRAIN_METRICS_FIELDS: tuple[str, ...] = (
    "update",
    "policy_loss",
    "value_loss",
    "entropy",
    "imitation_loss",
    "total_loss",
    "batch_size",
    "update_count",
    "rollout_seconds",
    "update_seconds",
    "transitions_per_second",
)


def build_training_metrics_row(
    update: int,
    metrics: Mapping[str, float | int],
) -> dict[str, float | int]:
    """Build a canonical CSV row for one PPO update."""
    return {
        "update": int(update),
        "policy_loss": float(metrics.get("policy_loss", 0.0)),
        "value_loss": float(metrics.get("value_loss", 0.0)),
        "entropy": float(metrics.get("entropy", 0.0)),
        "imitation_loss": float(metrics.get("imitation_loss", 0.0)),
        "total_loss": float(metrics.get("total_loss", 0.0)),
        "batch_size": float(metrics.get("batch_size", 0.0)),
        "update_count": float(metrics.get("update_count", float(update))),
        "rollout_seconds": float(metrics.get("rollout_seconds", 0.0)),
        "update_seconds": float(metrics.get("update_seconds", 0.0)),
        "transitions_per_second": float(metrics.get("transitions_per_second", 0.0)),
    }


class TrainingMetricsLogger:
    """Store canonical update-indexed training metrics.

    The logger rewrites the CSV in sorted update order so repeated runs can
    resume cleanly without duplicate headers or duplicate update rows.
    """

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._rows_by_update: dict[int, dict[str, str]] = {}
        self._max_update = 0
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.file_path.exists() or self.file_path.stat().st_size == 0:
            return

        with self.file_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "update" not in reader.fieldnames:
                return
            for row in reader:
                if not row:
                    continue
                raw_update = row.get("update")
                if raw_update in (None, ""):
                    continue
                update = int(float(raw_update))
                normalized = {
                    field: row.get(field, "")
                    for field in TRAIN_METRICS_FIELDS
                }
                normalized["update"] = str(update)
                if not normalized.get("update_count"):
                    normalized["update_count"] = str(float(update))
                self._rows_by_update[update] = normalized
                self._max_update = max(self._max_update, update)

        self._write_all()

    def _write_all(self) -> None:
        with self.file_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(TRAIN_METRICS_FIELDS))
            writer.writeheader()
            for update in sorted(self._rows_by_update):
                writer.writerow(self._rows_by_update[update])

    def max_update(self) -> int:
        return max(self._rows_by_update, default=0)

    def log(self, row: Mapping[str, float | int]) -> None:
        row_dict = {
            field: row[field]
            for field in TRAIN_METRICS_FIELDS
        }
        update = int(float(row_dict["update"]))
        serialized_row = {
            field: str(row_dict[field])
            for field in TRAIN_METRICS_FIELDS
        }
        serialized_row["update"] = str(update)
        self._rows_by_update[update] = serialized_row
        if update > self._max_update:
            self._max_update = update
            with self.file_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(TRAIN_METRICS_FIELDS))
                if handle.tell() == 0:
                    writer.writeheader()
                writer.writerow(serialized_row)
            return
        self._write_all()


def maybe_log_training_metrics(
    logger: TrainingMetricsLogger,
    metrics: Mapping[str, float | int] | None,
    last_logged_update: int,
) -> int:
    """Append a new update row only when trainer metrics advanced."""
    if not metrics:
        return last_logged_update
    update = int(float(metrics.get("update_count", 0.0)))
    if update <= last_logged_update:
        return last_logged_update
    logger.log(build_training_metrics_row(update=update, metrics=metrics))
    return update
