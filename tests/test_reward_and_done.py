"""Reward and termination behavior tests."""

from __future__ import annotations

import numpy as np

from agario_rl import AgarioConfig
from agario_rl.env.gym_env import AgarioMultiAgentEnv
from agario_rl.env.world import AgarioWorld
from agario_rl.rl.ppo_shared import SharedPPOTrainer


def _random_actions(agent_ids: list[str]) -> dict[str, np.ndarray]:
    return {
        agent_id: np.array(
            [
                np.random.uniform(-1.0, 1.0),
                np.random.uniform(-1.0, 1.0),
                np.random.uniform(0.0, 1.0),
            ],
            dtype=np.float32,
        )
        for agent_id in agent_ids
    }


def test_reset_is_deterministic_with_same_seed() -> None:
    config = AgarioConfig()
    env = AgarioMultiAgentEnv(config=config, enable_render=False)
    obs_a = env.reset(seed=42)
    obs_b = env.reset(seed=42)
    for agent_id in env.agent_ids:
        assert np.allclose(obs_a[agent_id], obs_b[agent_id])
    env.close()


def test_episode_done_when_only_one_agent_alive() -> None:
    config = AgarioConfig()
    config.simulation.action_mode = "continuous"
    world = AgarioWorld(config=config, seed=10)

    survivor = world.agent_ids[0]
    for idx, agent_id in enumerate(world.agent_ids):
        if idx == 0:
            continue
        world.agents[agent_id] = []
        world.snapshots[agent_id].alive = False

    outcome = world.step({agent_id: np.array([0.0, 0.0, 0.0], dtype=np.float32) for agent_id in world.agent_ids})
    assert outcome.dones["__all__"]
    assert outcome.infos["__global__"]["winner"] == survivor


def test_rewards_remain_finite_for_200_steps() -> None:
    config = AgarioConfig()
    config.simulation.action_mode = "continuous"
    config.max_steps = 5000
    env = AgarioMultiAgentEnv(config=config, enable_render=False)
    env.reset(seed=3)
    for _ in range(200):
        _, rewards, dones, _ = env.step(_random_actions(env.agent_ids))
        assert all(np.isfinite(value) for value in rewards.values())
        if dones.get("__all__", False):
            env.reset()
    env.close()


def test_checkpoint_save_load_round_trip(tmp_path) -> None:
    config = AgarioConfig()
    config.rl.steps_per_update = 24
    env = AgarioMultiAgentEnv(config=config, enable_render=False)
    trainer = SharedPPOTrainer(config=config, observation_dim=env.observation_space["shape"][0], device="cpu")

    trainer.collect_rollout(env, target_transitions=24)
    trainer.update()
    checkpoint = tmp_path / "roundtrip.pt"
    trainer.save(checkpoint)

    fresh = SharedPPOTrainer(config=config, observation_dim=env.observation_space["shape"][0], device="cpu")
    loaded = fresh.load(checkpoint)
    assert loaded
    env.close()
