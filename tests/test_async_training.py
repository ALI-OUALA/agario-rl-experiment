"""Async training worker smoke tests."""

from __future__ import annotations

import time

from agario_rl import AgarioConfig
from agario_rl.env.gym_env import AgarioMultiAgentEnv
from agario_rl.rl.async_trainer import AsyncTrainerCoordinator
from agario_rl.rl.ppo_shared import SharedPPOTrainer


def test_async_worker_accepts_rollout_and_returns_weights() -> None:
    config = AgarioConfig()
    config.simulation.action_mode = "continuous"
    config.rl.steps_per_update = 12
    config.async_training.rollout_queue_size = 4
    config.async_training.max_pending_weight_updates = 2

    env = AgarioMultiAgentEnv(config=config, enable_render=False)
    trainer = SharedPPOTrainer(config=config, observation_dim=env.observation_space["shape"][0], device="cpu")
    trainer.force_sync_with_env(env, seed=7)
    trainer.collect_rollout(env, target_transitions=24)

    payload = trainer.prepare_update_job_payload()
    assert payload is not None

    coordinator = AsyncTrainerCoordinator(config=config, observation_dim=env.observation_space["shape"][0])
    coordinator.start()
    coordinator.sync_from_trainer(trainer)
    submitted = False
    deadline_submit = time.time() + 5.0
    while time.time() < deadline_submit:
        if coordinator.submit_update(payload):
            submitted = True
            break
        time.sleep(0.05)
    assert submitted

    updated = None
    deadline = time.time() + 12.0
    while time.time() < deadline:
        updated = coordinator.poll_updates(trainer)
        if updated is not None:
            break
        time.sleep(0.1)

    coordinator.shutdown()
    env.close()
    assert updated is not None
