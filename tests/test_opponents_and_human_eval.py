"""Tests for scripted opponents and human-readiness metrics."""

from __future__ import annotations

import random

import numpy as np

from agario_rl import AgarioConfig
from agario_rl.env.gym_env import AgarioMultiAgentEnv
from agario_rl.human_eval import HumanReadinessTracker
from agario_rl.opponents import (
    AgarObjectivePolicy,
    CheckpointPolicy,
    OpportunisticHunterPolicy,
    PelletForagerPolicy,
    ThreatAwareEvaderPolicy,
    assign_opponents,
    build_default_opponent_pool,
    observation_dim_from_config,
)
from agario_rl.rl.ppo_shared import SharedPPOTrainer


def test_scripted_opponents_return_valid_continuous_actions() -> None:
    config = AgarioConfig()
    config.simulation.action_mode = "continuous"
    env = AgarioMultiAgentEnv(config=config, enable_render=False)
    observations = env.reset(seed=7)
    world = env.world

    for policy in (
        AgarObjectivePolicy(),
        PelletForagerPolicy(),
        ThreatAwareEvaderPolicy(),
        OpportunisticHunterPolicy(),
    ):
        action = policy.action(world=world, observations=observations, agent_id=env.agent_ids[1])
        assert action.shape == (3,)
        assert np.all(action[:2] <= 1.0)
        assert np.all(action[:2] >= -1.0)
        assert float(action[2]) in (0.0, 1.0)
    env.close()


def test_assign_opponents_allows_more_slots_than_policy_pool() -> None:
    config = AgarioConfig()
    config.simulation.action_mode = "continuous"
    pool = [PelletForagerPolicy(), ThreatAwareEvaderPolicy()]
    assignments = assign_opponents(
        pool,
        [f"agent_{idx}" for idx in range(1, 7)],
        rng=random.Random(7),
    )

    assert set(assignments) == {f"agent_{idx}" for idx in range(1, 7)}
    assert all(policy in pool for policy in assignments.values())


def test_default_opponent_pool_falls_back_when_checkpoint_missing(tmp_path) -> None:
    config = AgarioConfig()
    config.simulation.action_mode = "continuous"

    pool = build_default_opponent_pool(config, tmp_path / "missing.pt")

    assert [policy.name for policy in pool] == [
        "agar_objective",
        "pellet_forager",
        "threat_aware_evader",
        "opportunistic_hunter",
    ]


def test_agar_objective_policy_splits_on_close_weak_target() -> None:
    config = AgarioConfig()
    config.simulation.action_mode = "continuous"
    config.num_agents = 2
    world = AgarioMultiAgentEnv(config=config, enable_render=False)
    world.reset(seed=11)
    agent_id, target_id = world.agent_ids[:2]
    world.world.agents[agent_id][0].mass = 90.0
    world.world.agents[agent_id][0].position = np.array([240.0, 240.0], dtype=np.float32)
    world.world.agents[target_id][0].mass = 24.0
    world.world.agents[target_id][0].position = np.array([282.0, 240.0], dtype=np.float32)

    action = AgarObjectivePolicy().action(
        world=world.world,
        observations={},
        agent_id=agent_id,
    )

    assert action.shape == (3,)
    assert action[0] > 0.0
    assert action[2] == 1.0
    world.close()


def test_checkpoint_policy_uses_world_observation_dim_for_extended_features(tmp_path) -> None:
    config = AgarioConfig()
    config.simulation.action_mode = "continuous"
    config.observation_features.enabled = True
    config.observation_features.include_threats = True
    config.observation_features.include_viruses = True
    config.observation_features.include_eject_state = True
    env = AgarioMultiAgentEnv(config=config, enable_render=False)
    checkpoint_path = tmp_path / "extended.pt"
    trainer = SharedPPOTrainer(config=config, observation_dim=env.observation_space["shape"][0], device="cpu")
    trainer.save(checkpoint_path)

    policy = CheckpointPolicy(config=config, checkpoint_path=checkpoint_path, device="cpu")

    assert observation_dim_from_config(config) == env.observation_space["shape"][0]
    assert policy.trainer.observation_dim == env.observation_space["shape"][0]
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
