"""Headless-ready session wrapper for one human and two trained agents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np

from agario_rl import AgarioConfig
from agario_rl.env.gym_env import AgarioMultiAgentEnv
from agario_rl.opponents import OpponentPolicy, assign_opponents, build_default_opponent_pool
from agario_rl.play.input import HumanControlInput, PlayerCommand, build_player_command


@dataclass(slots=True)
class PlayStepResult:
    """Return payload for one human-play mode step."""

    observations: dict[str, np.ndarray] | None
    rewards: dict[str, float]
    dones: dict[str, bool]
    infos: dict[str, dict[str, Any]]
    actions: dict[str, np.ndarray]
    player_command: PlayerCommand


class HumanVsBotsSession:
    """Session coordinator for a single human slot in the RL world."""

    def __init__(
        self,
        config: AgarioConfig,
        checkpoint_path: str | Path,
        player_index: int = 0,
        seed: int | None = None,
        enable_eject: bool = False,
        opponent_pool: list[OpponentPolicy] | None = None,
        deterministic_opponents: bool = False,
    ) -> None:
        if config.simulation.action_mode != "continuous":
            raise ValueError("Human play mode requires continuous action mode.")

        self.config = config
        self.player_index = int(player_index)
        self.seed = config.seed if seed is None else int(seed)
        self.rng = random.Random(self.seed)
        self.deterministic_opponents = bool(deterministic_opponents)
        self.env = AgarioMultiAgentEnv(config=config, enable_render=False)
        if not 0 <= self.player_index < len(self.env.agent_ids):
            raise ValueError(f"player_index must be in range [0, {len(self.env.agent_ids) - 1}]")

        self.player_agent_id = self.env.agent_ids[self.player_index]
        self.config.physics.enable_eject_mechanic = bool(enable_eject)
        checkpoint = Path(checkpoint_path)
        self.opponent_pool = (
            list(opponent_pool)
            if opponent_pool is not None
            else build_default_opponent_pool(config=config, checkpoint_path=checkpoint)
        )
        self.opponent_agent_ids = [
            agent_id for agent_id in self.env.agent_ids if agent_id != self.player_agent_id
        ]
        self.active_opponents = assign_opponents(
            self.opponent_pool,
            self.opponent_agent_ids,
            self.rng,
            deterministic=self.deterministic_opponents,
        )

        self.current_obs = self.env.reset(seed=self.seed)
        self.last_actions: dict[str, np.ndarray] = {
            agent_id: np.zeros((3,), dtype=np.float32)
            for agent_id in self.env.agent_ids
        }
        self.physics_dt = 1.0 / max(1, int(config.simulation.physics_hz))
        self.substeps = max(1, int(round(config.simulation.physics_hz / config.simulation.decision_hz)))

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        """Reset the session and return the new observation dict."""
        next_seed = self.seed if seed is None else int(seed)
        self.current_obs = self.env.reset(seed=next_seed)
        self.active_opponents = assign_opponents(
            self.opponent_pool,
            self.opponent_agent_ids,
            self.rng,
            deterministic=self.deterministic_opponents,
        )
        self.last_actions = {
            agent_id: np.zeros((3,), dtype=np.float32)
            for agent_id in self.env.agent_ids
        }
        return self.current_obs

    def step(self, control: HumanControlInput) -> PlayStepResult:
        """Advance the session with one human action and bot policy actions."""
        if self.current_obs is None:
            raise RuntimeError("Session must be reset before stepping.")

        player_command = build_player_command(control)
        actions = {
            agent_id: policy.action(
                world=self.env.world,
                observations=self.current_obs,
                agent_id=agent_id,
            )
            for agent_id, policy in self.active_opponents.items()
        }
        actions[self.player_agent_id] = player_command.action

        if self.config.physics.enable_eject_mechanic and player_command.eject_requested:
            self.env.world.eject_mass(self.player_agent_id, player_command.action[:2])

        accumulated_rewards = {agent_id: 0.0 for agent_id in self.env.agent_ids}
        observations: dict[str, np.ndarray] | None = self.current_obs
        dones: dict[str, bool] = {"__all__": False}
        infos: dict[str, dict[str, Any]] = {}
        for _ in range(self.substeps):
            observations, rewards, dones, infos = self.env.step(
                actions,
                dt=self.physics_dt,
                compute_observations=True,
            )
            for agent_id in self.env.agent_ids:
                accumulated_rewards[agent_id] += float(rewards.get(agent_id, 0.0))
            if dones.get("__all__", False):
                break

        self.current_obs = observations
        self.last_actions = {
            agent_id: np.asarray(action, dtype=np.float32).copy()
            for agent_id, action in actions.items()
        }
        return PlayStepResult(
            observations=observations,
            rewards=accumulated_rewards,
            dones=dones,
            infos=infos,
            actions=self.last_actions,
            player_command=player_command,
        )

    def player_alive(self) -> bool:
        """Return whether the human-controlled agent still has cells."""
        return bool(self.env.world.agents[self.player_agent_id])

    def player_center(self) -> np.ndarray:
        """Return the mass-weighted player center."""
        cells = self.env.world.agents[self.player_agent_id]
        if not cells:
            return np.array(
                [self.env.world.map_size * 0.5, self.env.world.map_size * 0.5],
                dtype=np.float32,
            )
        masses = np.array([cell.mass for cell in cells], dtype=np.float32)
        positions = np.stack([cell.position for cell in cells], axis=0)
        return (positions * masses[:, None]).sum(axis=0) / max(float(masses.sum()), 1e-6)

    def leaderboard(self) -> list[tuple[str, float]]:
        """Return agents sorted by current total mass."""
        return sorted(
            (
                (agent_id, float(sum(cell.mass for cell in self.env.world.agents[agent_id])))
                for agent_id in self.env.agent_ids
            ),
            key=lambda item: item[1],
            reverse=True,
        )

    def close(self) -> None:
        """Close underlying environment resources."""
        self.env.close()
