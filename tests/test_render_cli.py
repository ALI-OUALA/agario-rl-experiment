"""CLI validation for the Raylib-only public runtime."""

from __future__ import annotations

import pytest

from scripts import benchmark_perf, supervise


def test_supervise_cli_has_no_backend_option(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["supervise.py", "--render-backend", "raylib"])
    with pytest.raises(SystemExit):
        supervise.parse_args()


def test_benchmark_cli_has_no_backend_option(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["benchmark_perf.py", "--backend", "pyglet"])
    with pytest.raises(SystemExit):
        benchmark_perf.parse_args()


def test_supervise_cli_accepts_current_flags(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["supervise.py", "--load-checkpoint"])
    args = supervise.parse_args()
    assert args.load_checkpoint is True
