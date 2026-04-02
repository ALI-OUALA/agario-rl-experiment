"""Runtime counters and chart history used by supervisor mode."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(slots=True)
class RuntimeSessionStats:
    """Tracks session-level counters for interactive supervision."""

    wins_by_agent: dict[str, int]
    chart_capacity: int
    _history: dict[str, deque[float]]

    @classmethod
    def create(cls, agent_ids: list[str], chart_capacity: int = 120) -> "RuntimeSessionStats":
        return cls(
            wins_by_agent={agent_id: 0 for agent_id in agent_ids},
            chart_capacity=chart_capacity,
            _history={
                name: deque(maxlen=chart_capacity)
                for name in (
                    "render_fps",
                    "frame_ms",
                    "reward_mean",
                    "total_loss",
                    "wins_total",
                    "update_count",
                )
            },
        )

    def reset_wins(self) -> None:
        for agent_id in list(self.wins_by_agent.keys()):
            self.wins_by_agent[agent_id] = 0
        self._append_history("wins_total", 0.0)

    def record_infos(self, infos: dict[str, dict[str, Any]] | None) -> None:
        if not infos:
            return
        winner = infos.get("__global__", {}).get("winner")
        if winner is None:
            return
        if winner in self.wins_by_agent:
            self.wins_by_agent[winner] += 1
            self._append_history("wins_total", float(sum(self.wins_by_agent.values())))

    def record_frame(
        self,
        metrics: Mapping[str, float] | None,
        infos: Mapping[str, Mapping[str, Any]] | None,
    ) -> None:
        """Capture lightweight telemetry for rolling charts."""
        payload = dict(metrics or {})
        self._append_history("render_fps", float(payload.get("render_fps", 0.0)))
        self._append_history("frame_ms", float(payload.get("frame_ms", 0.0)))
        self._append_history("total_loss", float(payload.get("total_loss", 0.0)))
        self._append_history("update_count", float(payload.get("update_count", 0.0)))

        reward_mean = 0.0
        agent_infos = [
            data for agent_id, data in (infos or {}).items() if str(agent_id).startswith("agent_")
        ]
        if agent_infos:
            reward_mean = sum(float(data.get("episode_return", 0.0)) for data in agent_infos) / len(agent_infos)
        self._append_history("reward_mean", reward_mean)
        self._append_history("wins_total", float(sum(self.wins_by_agent.values())))

    def chart_series(self, name: str) -> tuple[float, ...]:
        history = self._history.get(name)
        if history is None:
            return ()
        return tuple(history)

    def _append_history(self, name: str, value: float) -> None:
        if name not in self._history:
            self._history[name] = deque(maxlen=self.chart_capacity)
        self._history[name].append(float(value))

    def to_extra_stats(self) -> dict[str, float]:
        return {f"wins_{agent_id}": float(wins) for agent_id, wins in self.wins_by_agent.items()}
