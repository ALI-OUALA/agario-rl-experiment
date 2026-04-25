"""Smoke tests for scenario training without touching repo logs/checkpoints."""

from __future__ import annotations

import csv

import pytest

from agario_rl import AgarioConfig, apply_scenario_preset
from agario_rl.env.gym_env import AgarioMultiAgentEnv
from agario_rl.rl.ppo_shared import SharedPPOTrainer
from agario_rl.utils.logging import TrainingMetricsLogger, build_training_metrics_row


@pytest.mark.parametrize("preset", ["classic", "agario_curriculum", "full_arena"])
def test_short_training_update_writes_metrics_and_checkpoint(tmp_path, preset: str) -> None:
    config = apply_scenario_preset(AgarioConfig(), preset)
    if preset == "full_arena":
        config.map.start_size = 512
        config.map.max_size = 512
        config.map.pellets_per_10k_area = 8
        config.map.pellet_respawn_per_step = 2
    config.rl.steps_per_update = 6
    config.rl.minibatch_size = 3
    config.rl.ppo_epochs = 1
    config.rl.imitation_batch_size = 999
    config.simulation.continuing_respawn = preset != "classic"
    env = AgarioMultiAgentEnv(config=config, enable_render=False)
    trainer = SharedPPOTrainer(
        config=config,
        observation_dim=env.observation_space["shape"][0],
        device="cpu",
        inference_device="cpu",
    )

    trainer.collect_rollout(env, target_transitions=config.rl.steps_per_update)
    metrics = trainer.update()
    checkpoint_path = tmp_path / f"{preset}.pt"
    trainer.save(checkpoint_path)
    logger = TrainingMetricsLogger(tmp_path / f"{preset}.csv")
    logger.log(build_training_metrics_row(update=1, metrics=metrics))

    rows = list(csv.DictReader((tmp_path / f"{preset}.csv").open("r", newline="", encoding="utf-8")))
    assert checkpoint_path.exists()
    assert metrics["batch_size"] > 0
    assert rows[0]["update"] == "1"
    env.close()


def test_full_arena_preset_expands_to_large_readable_match() -> None:
    config = apply_scenario_preset(AgarioConfig(), "full_arena")

    assert config.num_agents >= 6
    assert config.map.start_size >= 2000
    assert config.map.max_size >= 2000
    assert config.map.pellets_per_10k_area == 2
    assert config.map.pellet_respawn_per_step == 5
    assert config.max_steps >= 3600
    assert config.simulation.continuing_respawn is True
    assert config.physics.enable_eject_mechanic is True
    assert config.nearest_opponents >= 5
