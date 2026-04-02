"""World entities for the Agar.io-like simulation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class Cell:
    """A movable cell owned by an agent."""

    cell_id: int
    agent_id: str
    position: np.ndarray
    velocity: np.ndarray
    mass: float
    split_cooldown: int = 0
    merge_cooldown: int = 0
    eject_cooldown: int = 0

    def radius(self, radius_scale: float) -> float:
        return radius_scale * float(np.sqrt(max(self.mass, 0.01)))


@dataclass(slots=True)
class Pellet:
    """A static pellet that provides mass when eaten."""

    pellet_id: int
    position: np.ndarray
    mass: float = 1.0


@dataclass(slots=True)
class AgentSnapshot:
    """Per-agent runtime metrics used for rewards and HUD."""

    total_mass: float = 0.0
    alive: bool = True
    episode_return: float = 0.0
    eliminated_opponents: int = 0
    recent_direction_counts: list[int] = field(default_factory=lambda: [0] * 9)

    def record_direction(self, direction_idx: int) -> None:
        self.recent_direction_counts[direction_idx] += 1
