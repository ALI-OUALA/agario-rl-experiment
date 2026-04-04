"""Human-readiness proxy metrics for trained agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from agario_rl.env.world import AgarioWorld


@dataclass(slots=True)
class HumanReadinessSummary:
    """Serializable summary of human-readiness proxy metrics."""

    episodes: int
    wins: int
    win_rate: float
    mean_survival_steps: float
    mean_final_mass: float
    corner_time_fraction: float
    threat_avoidance_rate: float
    small_target_pressure_rate: float


class HumanReadinessTracker:
    """Track proxy metrics that are more relevant to human matches."""

    def __init__(self, learner_id: str) -> None:
        self.learner_id = learner_id
        self.episode_count = 0
        self.win_count = 0
        self._survival_steps: list[int] = []
        self._final_masses: list[float] = []
        self._corner_steps = 0
        self._alive_steps = 0
        self._threat_events = 0
        self._threat_successes = 0
        self._pressure_events = 0
        self._pressure_successes = 0
        self._previous_distances: dict[str, float] = {}
        self._current_survival_steps = 0

    def observe(self, world: AgarioWorld, infos: dict[str, dict[str, Any]]) -> None:
        """Update per-step proxy metrics."""
        if self.learner_id not in world.agent_ids:
            return

        learner_cells = world.agents[self.learner_id]
        if learner_cells:
            self._current_survival_steps += 1
            self._alive_steps += 1

            masses = np.array([cell.mass for cell in learner_cells], dtype=np.float32)
            positions = np.stack([cell.position for cell in learner_cells], axis=0)
            center = (positions * masses[:, None]).sum(axis=0) / max(float(masses.sum()), 1e-6)
            learner_mass = float(masses.sum())

            corner_margin = world.map_size * 0.18
            in_corner_x = center[0] <= corner_margin or center[0] >= world.map_size - corner_margin
            in_corner_y = center[1] <= corner_margin or center[1] >= world.map_size - corner_margin
            if in_corner_x and in_corner_y:
                self._corner_steps += 1

            opponents: list[tuple[str, float, float]] = []
            for other_id in world.agent_ids:
                if other_id == self.learner_id or not world.agents[other_id]:
                    continue
                other_mass = float(sum(cell.mass for cell in world.agents[other_id]))
                other_positions = np.stack([cell.position for cell in world.agents[other_id]], axis=0)
                other_masses = np.array([cell.mass for cell in world.agents[other_id]], dtype=np.float32)
                other_center = (other_positions * other_masses[:, None]).sum(axis=0) / max(float(other_masses.sum()), 1e-6)
                distance = float(np.linalg.norm(other_center - center))
                opponents.append((other_id, distance, other_mass / max(learner_mass, 1e-6)))
            opponents.sort(key=lambda item: item[1])

            nearest_threat = next((item for item in opponents if item[2] >= 1.15 and item[1] < world.map_size * 0.22), None)
            if nearest_threat is not None:
                self._threat_events += 1
                previous_distance = self._previous_distances.get(nearest_threat[0])
                if previous_distance is not None and nearest_threat[1] > previous_distance:
                    self._threat_successes += 1

            nearest_target = next((item for item in opponents if item[2] <= 0.85 and item[1] < world.map_size * 0.22), None)
            if nearest_target is not None:
                self._pressure_events += 1
                previous_distance = self._previous_distances.get(nearest_target[0])
                if previous_distance is not None and nearest_target[1] < previous_distance:
                    self._pressure_successes += 1

            self._previous_distances = {agent_id: distance for agent_id, distance, _ in opponents}
        else:
            self._previous_distances = {}

        global_info = infos.get("__global__", {})
        if bool(global_info) and (
            int(global_info.get("step", 0)) >= world.config.max_steps
            or int(global_info.get("alive_count", len(world.alive_agents))) <= 1
        ):
            self.finish_episode(
                winner=global_info.get("winner"),
                final_mass=float(sum(cell.mass for cell in world.agents[self.learner_id])),
            )

    def finish_episode(self, winner: str | None, final_mass: float) -> None:
        """Close out the current episode."""
        self.episode_count += 1
        if winner == self.learner_id:
            self.win_count += 1
        self._survival_steps.append(self._current_survival_steps)
        self._final_masses.append(float(final_mass))
        self._current_survival_steps = 0
        self._previous_distances = {}

    def summary(self) -> HumanReadinessSummary:
        """Return the aggregated proxy metrics."""
        episodes = max(1, self.episode_count)
        threat_events = max(1, self._threat_events)
        pressure_events = max(1, self._pressure_events)
        alive_steps = max(1, self._alive_steps)
        return HumanReadinessSummary(
            episodes=self.episode_count,
            wins=self.win_count,
            win_rate=self.win_count / episodes,
            mean_survival_steps=float(np.mean(self._survival_steps)) if self._survival_steps else 0.0,
            mean_final_mass=float(np.mean(self._final_masses)) if self._final_masses else 0.0,
            corner_time_fraction=self._corner_steps / alive_steps,
            threat_avoidance_rate=self._threat_successes / threat_events,
            small_target_pressure_rate=self._pressure_successes / pressure_events,
        )
