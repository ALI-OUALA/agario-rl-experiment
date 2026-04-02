"""Observation vector shape and stability tests."""

from __future__ import annotations

import numpy as np

from agario_rl import AgarioConfig
from agario_rl.env.gym_env import AgarioMultiAgentEnv


def test_observation_shapes_are_consistent() -> None:
    config = AgarioConfig()
    for mode in ("continuous", "discrete_9way"):
        config.simulation.action_mode = mode
        env = AgarioMultiAgentEnv(config=config, enable_render=False)
        obs = env.reset(seed=11)
        expected_shape = env.observation_space["shape"]
        for agent_id in env.agent_ids:
            assert obs[agent_id].shape == expected_shape

        for _ in range(20):
            if mode == "continuous":
                actions = {agent_id: np.array([0.2, -0.1, 0.0], dtype=np.float32) for agent_id in env.agent_ids}
            else:
                actions = {agent_id: np.array([3.0, 0.0], dtype=np.float32) for agent_id in env.agent_ids}
            obs, _, dones, _ = env.step(actions)
            for agent_id in env.agent_ids:
                assert obs[agent_id].shape == expected_shape
            if dones.get("__all__", False):
                obs = env.reset()
        env.close()
