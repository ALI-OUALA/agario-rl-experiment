"""Scripted and checkpoint-based opponents for human-readiness training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import random
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


def observation_dim_from_config(config: AgarioConfig) -> int:
    """Return the environment observation size produced by the world config."""
    return AgarioWorld(config=config, seed=config.seed).observation_dim


def assign_opponents(
    pool: list[OpponentPolicy],
    opponent_agent_ids: list[str],
    rng: random.Random,
    *,
    deterministic: bool = False,
) -> dict[str, OpponentPolicy]:
    """Assign opponent policies, cycling when slots exceed the pool size."""
    if not opponent_agent_ids:
        return {}
    if not pool:
        raise ValueError("Opponent pool must contain at least one policy.")

    if deterministic:
        policies = [pool[index % len(pool)] for index in range(len(opponent_agent_ids))]
    elif len(opponent_agent_ids) <= len(pool):
        policies = rng.sample(pool, k=len(opponent_agent_ids))
    else:
        policies = [rng.choice(pool) for _ in opponent_agent_ids]

    return {
        agent_id: policy
        for agent_id, policy in zip(opponent_agent_ids, policies, strict=True)
    }


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


def _nearby_virus_avoidance(world: AgarioWorld, agent_id: str) -> np.ndarray:
    if not world.viruses:
        return np.zeros((2,), dtype=np.float32)
    center = _agent_center(world, agent_id)
    own_mass = _agent_mass(world, agent_id)
    avoidance = np.zeros((2,), dtype=np.float32)
    for virus in world.viruses:
        delta = center - virus.position
        distance = max(float(np.linalg.norm(delta)), 1e-6)
        dangerous = own_mass >= world.config.viruses.min_split_mass
        if dangerous and distance < world.map_size * 0.12:
            avoidance += (delta / distance).astype(np.float32) * (1.0 - distance / max(world.map_size * 0.12, 1e-6))
    return avoidance


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
class AgarObjectivePolicy:
    """Balanced bot for human-facing matches: survive, grow, then pressure."""

    name: str = "agar_objective"

    def action(
        self,
        *,
        world: AgarioWorld,
        observations: dict[str, np.ndarray],
        agent_id: str,
    ) -> np.ndarray:
        center = _agent_center(world, agent_id)
        opponents = _nearest_opponents(world, agent_id)
        own_mass = _agent_mass(world, agent_id)
        larger = [item for item in opponents if item[3] >= 1.12]
        smaller = [item for item in opponents if item[3] <= 0.82]
        map_center = np.array([world.map_size * 0.5, world.map_size * 0.5], dtype=np.float32)
        center_bias = map_center - center
        virus_avoidance = _nearby_virus_avoidance(world, agent_id)

        if larger and larger[0][2] < world.map_size * 0.16:
            flee = -larger[0][1] + 0.28 * center_bias + 0.85 * virus_avoidance
            return _vector_action(flee)

        if smaller:
            target_id, chase_delta, distance, mass_ratio = smaller[0]
            split_range = max(70.0, np.sqrt(max(own_mass, 1.0)) * world.config.physics.radius_scale * 4.8)
            split = bool(mass_ratio <= 0.55 and distance <= split_range and not larger)
            return _vector_action(chase_delta + 0.25 * virus_avoidance, split=split)

        forage = _nearest_pellet_direction(world, agent_id)
        return _vector_action(forage + 0.25 * center_bias + 0.65 * virus_avoidance)


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
            observation_dim=observation_dim_from_config(self.config),
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
    pool: list[OpponentPolicy] = [
        AgarObjectivePolicy(),
        PelletForagerPolicy(),
        ThreatAwareEvaderPolicy(),
        OpportunisticHunterPolicy(),
    ]
    try:
        pool.insert(0, CheckpointPolicy(config=config, checkpoint_path=Path(checkpoint_path)))
    except (FileNotFoundError, RuntimeError, ValueError):
        pass
    return pool
