"""Observation compute-flag behavior tests."""

from __future__ import annotations

import numpy as np

from agario_rl import AgarioConfig
from agario_rl.env.gym_env import AgarioMultiAgentEnv
from agario_rl.env.world import AgarioWorld
from agario_rl.rl.ppo_shared import SharedPPOTrainer


def _zero_actions_for_ids(agent_ids: list[str]) -> dict[str, np.ndarray]:
    return {agent_id: np.array([0.0, 0.0, 0.0], dtype=np.float32) for agent_id in agent_ids}


def test_world_step_can_skip_observation_computation() -> None:
    config = AgarioConfig()
    config.simulation.action_mode = "continuous"
    world_skip = AgarioWorld(config=config, seed=13)
    world_full = AgarioWorld(config=config, seed=13)
    actions = _zero_actions_for_ids(world_skip.agent_ids)

    skipped = world_skip.step(actions, compute_observations=False)
    regular = world_full.step(actions, compute_observations=True)

    assert skipped.observations is None
    assert regular.observations is not None
    assert skipped.rewards == regular.rewards
    assert skipped.dones == regular.dones
    assert skipped.infos["__global__"]["step"] == regular.infos["__global__"]["step"]
    assert skipped.infos["__global__"]["alive_count"] == regular.infos["__global__"]["alive_count"]


def test_env_step_can_skip_observation_computation() -> None:
    config = AgarioConfig()
    config.simulation.action_mode = "continuous"
    env = AgarioMultiAgentEnv(config=config, enable_render=False)
    env.reset(seed=11)
    actions = _zero_actions_for_ids(env.agent_ids)
    obs, rewards, dones, infos = env.step(actions, compute_observations=False)
    env.close()

    assert obs is None
    assert set(rewards.keys()) == set(env.agent_ids)
    assert "__all__" in dones
    assert "__global__" in infos


class _StepSpyEnv:
    def __init__(self, config: AgarioConfig, obs_dim: int) -> None:
        self.config = config
        self.agent_ids = [f"agent_{idx}" for idx in range(config.num_agents)]
        self.observation_space = {"shape": (obs_dim,), "dtype": np.float32}
        self.step_compute_flags: list[bool] = []
        self._tick = 0

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        _ = seed
        self._tick = 0
        return {
            agent_id: np.zeros(self.observation_space["shape"], dtype=np.float32)
            for agent_id in self.agent_ids
        }

    def step(
        self,
        actions: dict[str, np.ndarray],
        dt: float | None = None,
        compute_observations: bool = True,
    ) -> tuple[
        dict[str, np.ndarray] | None,
        dict[str, float],
        dict[str, bool],
        dict[str, dict[str, float | bool | None]],
    ]:
        _ = actions
        _ = dt
        self._tick += 1
        self.step_compute_flags.append(bool(compute_observations))
        rewards = {agent_id: 0.0 for agent_id in self.agent_ids}
        dones = {agent_id: False for agent_id in self.agent_ids}
        dones["__all__"] = False
        infos = {
            agent_id: {"alive": True, "winner": None}
            for agent_id in self.agent_ids
        }
        if not compute_observations:
            return None, rewards, dones, infos
        obs = {
            agent_id: np.full(self.observation_space["shape"], float(self._tick), dtype=np.float32)
            for agent_id in self.agent_ids
        }
        return obs, rewards, dones, infos


def test_trainer_requests_observations_only_on_last_substep() -> None:
    config = AgarioConfig()
    config.simulation.action_mode = "continuous"
    config.num_agents = 3
    obs_dim = 12
    env = _StepSpyEnv(config=config, obs_dim=obs_dim)
    trainer = SharedPPOTrainer(config=config, observation_dim=obs_dim, device="cpu")
    trainer.force_sync_with_env(env, seed=7)

    trainer.step_decision(env=env, substeps=4, track_experience=False, deterministic=False)

    assert env.step_compute_flags == [False, False, False, True]
    assert trainer.current_obs is not None
    for agent_obs in trainer.current_obs.values():
        assert float(agent_obs[0]) == 4.0
