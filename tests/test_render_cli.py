"""CLI validation for the Raylib-only public runtime."""

from __future__ import annotations

import argparse

import pytest

from scripts import benchmark_perf, showcase, train


def test_benchmark_cli_has_no_backend_option(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["benchmark_perf.py", "--backend", "pyglet"])
    with pytest.raises(SystemExit):
        benchmark_perf.parse_args()


def test_showcase_cli_defaults_to_full_arena_demo(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["showcase.py"])

    args = showcase.parse_args()

    assert args.scenario_preset == "full_arena"
    assert args.checkpoint == "checkpoints/human_ready_v1/latest.pt"
    assert args.seconds == 45.0
    assert args.headless is False


def test_showcase_headless_smoke_runs_without_writing_checkpoint(tmp_path) -> None:
    summary = showcase.run_showcase(
        argparse.Namespace(
            config="config/default.yaml",
            checkpoint=str(tmp_path / "missing.pt"),
            seconds=0.01,
            seed=17,
            headless=True,
            width=320,
            height=240,
            scenario_preset="full_arena",
        )
    )

    assert summary["scenario"] == "full_arena"
    assert summary["agents"] >= 6
    assert summary["map_size"] >= 2000
    assert summary["frames"] >= 1


def test_train_cli_accepts_scenario_flags(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--updates",
            "1",
            "--scenario-preset",
            "agario_curriculum",
            "--continuing-respawn",
        ],
    )
    args = train.parse_args()
    assert args.scenario_preset == "agario_curriculum"
    assert args.continuing_respawn is True
