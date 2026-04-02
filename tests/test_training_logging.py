"""Tests for canonical training metrics logging."""

from __future__ import annotations

import csv

from agario_rl.utils.logging import (
    TrainingMetricsLogger,
    build_training_metrics_row,
    maybe_log_training_metrics,
)


def test_training_metrics_logger_normalizes_duplicate_updates(tmp_path) -> None:
    csv_path = tmp_path / "train_metrics.csv"
    csv_path.write_text(
        "\n".join(
            (
                "update,policy_loss,value_loss,entropy,imitation_loss,total_loss,batch_size,update_count",
                "1,0.1,0.2,0.3,0.4,0.5,6,1",
                "1,0.9,0.8,0.7,0.6,0.5,6,1",
                "2,1.1,1.2,1.3,1.4,1.5,6,2",
            )
        ),
        encoding="utf-8",
    )

    logger = TrainingMetricsLogger(csv_path)

    assert logger.max_update() == 2
    rows = list(csv.DictReader(csv_path.open("r", newline="", encoding="utf-8")))
    assert [int(row["update"]) for row in rows] == [1, 2]
    assert rows[0]["policy_loss"] == "0.9"


def test_maybe_log_training_metrics_only_appends_new_updates(tmp_path) -> None:
    logger = TrainingMetricsLogger(tmp_path / "train_metrics.csv")
    last_logged_update = 0

    metrics = {
        "policy_loss": 0.1,
        "value_loss": 0.2,
        "entropy": 0.3,
        "imitation_loss": 0.4,
        "total_loss": 0.5,
        "batch_size": 12.0,
        "update_count": 3.0,
    }
    last_logged_update = maybe_log_training_metrics(
        logger,
        metrics,
        last_logged_update,
    )
    last_logged_update = maybe_log_training_metrics(
        logger,
        metrics,
        last_logged_update,
    )
    logger.log(
        build_training_metrics_row(
            update=4,
            metrics={**metrics, "update_count": 4.0, "policy_loss": 0.7},
        )
    )

    assert last_logged_update == 3
    rows = list(
        csv.DictReader((tmp_path / "train_metrics.csv").open("r", newline="", encoding="utf-8"))
    )
    assert [int(row["update"]) for row in rows] == [3, 4]
