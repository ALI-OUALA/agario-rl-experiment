"""Scripted and checkpoint-based opponents for human-readiness training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from agario_rl import AgarioConfig
from agario_rl.env.world import AgarioWorld
from agario_rl.rl.ppo_shared import SharedPPOTrainer


class OpponentPolicy(Protocol):
    """Interface for non-trainable opponents."""

    name: str

    def action(
        self,
        *,
        world: AgarioWorld,
        observations: dict[str, np.ndarray],
        agent_id: str,
    ) -> np.ndarray:
        """Return one action for the given agent."""


def _agent_center(world: AgarioWorld, agent_id: str) -> np.ndarray:
    cells = world.agents[agent_id]
    if not cells:
        return np.array([world.map_size * 0.5, world.map_size * 0.5], dtype=np.float32)
    masses = np.array([cell.mass for cell in cells], dtype=np.float32)
    positions = np.stack([cell.position for cell in cells], axis=0)
    return (positions * masses[:, None]).sum(axis=0) / max(float(masses.sum()), 1e-6)


def _agent_mass(world: AgarioWorld, agent_id: str) -> float:
    return float(sum(cell.mass for cell in world.agents[agent_id]))


def _vector_action(direction: np.ndarray, split: bool = False) -> np.ndarray:
    action = np.zeros((3,), dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm > 1e-6:
        action[:2] = np.clip(direction / max(1.0, norm), -1.0, 1.0).astype(np.float32)
    action[2] = 1.0 if split else 0.0
    return action


def _nearest_pellet_direction(world: AgarioWorld, agent_id: str) -> np.ndarray:
    if not world.pellets:
        return np.zeros((2,), dtype=np.float32)
    center = _agent_center(world, agent_id)
    target = min(world.pellets, key=lambda pellet: float(np.sum((pellet.position - center) ** 2)))
    return target.position - center


def _nearest_opponents(
    world: AgarioWorld,
    agent_id: str,
) -> list[tuple[str, np.ndarray, float, float]]:
    center = _agent_center(world, agent_id)
    own_mass = _agent_mass(world, agent_id)
    opponents: list[tuple[str, np.ndarray, float, float]] = []
    for other_id in world.agent_ids:
        if other_id == agent_id or not world.agents[other_id]:
            continue
        other_center = _agent_center(world, other_id)
        delta = other_center - center
        distance = float(np.linalg.norm(delta))
        opponents.append((other_id, delta, distance, _agent_mass(world, other_id) / max(own_mass, 1e-6)))
    opponents.sort(key=lambda item: item[2])
    return opponents


@dataclass(slots=True)
class PelletForagerPolicy:
    """Baseline bot that mainly converts pellets into safe growth."""

    name: str = "pellet_forager"

    def action(
        self,
        *,
        world: AgarioWorld,
        observations: dict[str, np.ndarray],
        agent_id: str,
    ) -> np.ndarray:
        center = _agent_center(world, agent_id)
        nearest_larger = next((item for item in _nearest_opponents(world, agent_id) if item[3] >= 1.15), None)
        if nearest_larger is not None and nearest_larger[2] < world.map_size * 0.2:
            wall_escape = center - np.array([world.map_size * 0.5, world.map_size * 0.5], dtype=np.float32)
            return _vector_action(-nearest_larger[1] + 0.35 * wall_escape)
        return _vector_action(_nearest_pellet_direction(world, agent_id))


@dataclass(slots=True)
class ThreatAwareEvaderPolicy:
    """Bot that prioritizes survival and keeping distance from larger masses."""

    name: str = "threat_aware_evader"

    def action(
        self,
        *,
        world: AgarioWorld,
        observations: dict[str, np.ndarray],
        agent_id: str,
    ) -> np.ndarray:
        center = _agent_center(world, agent_id)
        larger = [item for item in _nearest_opponents(world, agent_id) if item[3] >= 1.15]
        if larger:
            nearest = larger[0]
            map_center_bias = center - np.array([world.map_size * 0.5, world.map_size * 0.5], dtype=np.float32)
            return _vector_action(-nearest[1] + 0.45 * map_center_bias)
        return _vector_action(_nearest_pellet_direction(world, agent_id))


@dataclass(slots=True)
class OpportunisticHunterPolicy:
    """Bot that pressures smaller targets but still respects immediate danger."""

    name: str = "opportunistic_hunter"

    def action(
        self,
        *,
        world: AgarioWorld,
        observations: dict[str, np.ndarray],
        agent_id: str,
    ) -> np.ndarray:
        opponents = _nearest_opponents(world, agent_id)
        larger = [item for item in opponents if item[3] >= 1.12]
        smaller = [item for item in opponents if item[3] <= 0.86]

        if larger and larger[0][2] < world.map_size * 0.18:
            return _vector_action(-larger[0][1])
        if smaller:
            chase = smaller[0]
            split = bool(chase[2] < world.map_size * 0.08 and chase[3] <= 0.55)
            return _vector_action(chase[1], split=split)
        return _vector_action(_nearest_pellet_direction(world, agent_id))


@dataclass(slots=True)
class CheckpointPolicy:
    """Frozen checkpoint opponent used as a stronger self-play anchor."""

    config: AgarioConfig
    checkpoint_path: Path
    device: str = "cpu"
    name: str = "checkpoint_500_anchor"
    trainer: SharedPPOTrainer = field(init=False)

    def __post_init__(self) -> None:
        self.trainer = SharedPPOTrainer(
            config=self.config,
            observation_dim=self.config.nearest_pellets * 3 + self.config.nearest_opponents * 4 + 12,
            device=self.device,
        )
        if not self.trainer.load(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint opponent not found: {self.checkpoint_path}")

    def action(
        self,
        *,
        world: AgarioWorld,
        observations: dict[str, np.ndarray],
        agent_id: str,
    ) -> np.ndarray:
        actions = self.trainer.predict_actions(
            observations,
            deterministic=True,
            agent_ids=[agent_id],
        )
        return actions[agent_id]


def build_default_opponent_pool(
    config: AgarioConfig,
    checkpoint_path: str | Path,
) -> list[OpponentPolicy]:
    """Build the default pool used for human-readiness training."""
    return [
        CheckpointPolicy(config=config, checkpoint_path=Path(checkpoint_path)),
        PelletForagerPolicy(),
        ThreatAwareEvaderPolicy(),
        OpportunisticHunterPolicy(),
    ]
