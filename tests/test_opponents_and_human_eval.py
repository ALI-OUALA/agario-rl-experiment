"""Tests for scripted opponents and human-readiness metrics."""

from __future__ import annotations

import numpy as np

from agario_rl import AgarioConfig
from agario_rl.env.gym_env import AgarioMultiAgentEnv
from agario_rl.human_eval import HumanReadinessTracker
from agario_rl.opponents import (
    OpportunisticHunterPolicy,
    PelletForagerPolicy,
    ThreatAwareEvaderPolicy,
)
from agario_rl.rl.ppo_shared import SharedPPOTrainer


def test_scripted_opponents_return_valid_continuous_actions() -> None:
    config = AgarioConfig()
    config.simulation.action_mode = "continuous"
    env = AgarioMultiAgentEnv(config=config, enable_render=False)
    observations = env.reset(seed=7)
    world = env.world

    for policy in (PelletForagerPolicy(), ThreatAwareEvaderPolicy(), OpportunisticHunterPolicy()):
        action = policy.action(world=world, observations=observations, agent_id=env.agent_ids[1])
        assert action.shape == (3,)
        assert np.all(action[:2] <= 1.0)
        assert np.all(action[:2] >= -1.0)
        assert float(action[2]) in (0.0, 1.0)
    env.close()


def test_human_readiness_tracker_reports_summary() -> None:
    tracker = HumanReadinessTracker(learner_id="agent_0")
    config = AgarioConfig()
    config.simulation.action_mode = "continuous"
    env = AgarioMultiAgentEnv(config=config, enable_render=False)
    env.reset(seed=9)

    infos = {
        "agent_0": {"alive": True},
        "__global__": {"step": config.max_steps, "alive_count": 1, "winner": "agent_0"},
    }
    tracker.observe(env.world, infos)
    summary = tracker.summary()
    assert summary.episodes == 1
    assert summary.wins == 1
    env.close()


def test_mixed_opponent_step_records_only_tracked_agent(tmp_path) -> None:
    config = AgarioConfig()
    config.simulation.action_mode = "continuous"
    env = AgarioMultiAgentEnv(config=config, enable_render=False)
    trainer = SharedPPOTrainer(config=config, observation_dim=env.observation_space["shape"][0], device="cpu")
    trainer.set_tracked_agent_ids([env.agent_ids[0]])
    trainer.force_sync_with_env(env, seed=13)

    overrides = {
        env.agent_ids[1]: np.array([0.0, 1.0, 0.0], dtype=np.float32),
        env.agent_ids[2]: np.array([0.0, -1.0, 0.0], dtype=np.float32),
    }
    trainer.step_decision(
        env=env,
        substeps=1,
        dt=1.0 / config.simulation.physics_hz,
        track_experience=True,
        deterministic=False,
        action_overrides=overrides,
        policy_agent_ids=[env.agent_ids[0]],
    )

    assert set(trainer.trajectories.keys()) == {env.agent_ids[0]}
    assert trainer.transitions_since_update == 1
    env.close()
