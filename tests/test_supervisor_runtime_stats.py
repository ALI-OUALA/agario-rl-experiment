"""Supervisor runtime counters tests."""

from __future__ import annotations

from agario_rl.supervisor.runtime_stats import RuntimeSessionStats


def test_runtime_stats_counts_winner_and_resets() -> None:
    stats = RuntimeSessionStats.create(["agent_0", "agent_1", "agent_2"])

    stats.record_infos({"__global__": {"winner": "agent_1"}})
    stats.record_infos({"__global__": {"winner": None}})
    stats.record_infos({"__global__": {"winner": "agent_1"}})

    assert stats.wins_by_agent["agent_1"] == 2
    assert stats.wins_by_agent["agent_0"] == 0

    stats.reset_wins()
    assert stats.wins_by_agent == {"agent_0": 0, "agent_1": 0, "agent_2": 0}


def test_runtime_stats_exports_extra_stats_keys() -> None:
    stats = RuntimeSessionStats.create(["agent_0", "agent_1"])
    stats.record_infos({"__global__": {"winner": "agent_0"}})
    payload = stats.to_extra_stats()
    assert payload["wins_agent_0"] == 1.0
    assert payload["wins_agent_1"] == 0.0
