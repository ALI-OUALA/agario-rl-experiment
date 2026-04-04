"""Tests for the human-playable mode helpers."""

from __future__ import annotations

import numpy as np

from agario_rl import AgarioConfig
from agario_rl.env.gym_env import AgarioMultiAgentEnv
from agario_rl.play import HumanControlInput, HumanVsBotsSession, build_player_command
from agario_rl.rl.ppo_shared import SharedPPOTrainer


def test_build_player_command_maps_mouse_target_to_continuous_action() -> None:
    command = build_player_command(
        HumanControlInput(
            player_position=(10.0, 10.0),
            target_world=(13.0, 14.0),
            split_pressed=True,
            eject_pressed=True,
            alive=True,
        )
    )

    np.testing.assert_allclose(command.action[:2], np.array([0.6, 0.8], dtype=np.float32), atol=1e-3)
    assert float(command.action[2]) == 1.0
    assert command.eject_requested is True


def test_build_player_command_returns_no_action_when_dead() -> None:
    command = build_player_command(
        HumanControlInput(
            player_position=(10.0, 10.0),
            target_world=(40.0, 60.0),
            split_pressed=True,
            eject_pressed=True,
            alive=False,
        )
    )

    np.testing.assert_array_equal(command.action, np.zeros((3,), dtype=np.float32))
    assert command.eject_requested is False


def test_human_vs_bots_session_steps_without_breaking_policy_path(tmp_path) -> None:
    config = AgarioConfig()
    config.simulation.action_mode = "continuous"
    config.rl.steps_per_update = 6
    config.rl.minibatch_size = 6
    checkpoint_path = tmp_path / "latest.pt"

    env = AgarioMultiAgentEnv(config=config, enable_render=False)
    trainer = SharedPPOTrainer(config=config, observation_dim=env.observation_space["shape"][0], device="cpu")
    trainer.save(checkpoint_path)
    env.close()

    session = HumanVsBotsSession(config=config, checkpoint_path=checkpoint_path, player_index=0)
    player_center = session.player_center()
    result = session.step(
        HumanControlInput(
            player_position=(float(player_center[0]), float(player_center[1])),
            target_world=(float(player_center[0] + 30.0), float(player_center[1])),
            split_pressed=False,
            eject_pressed=True,
            alive=True,
        )
    )

    assert result.observations is not None
    assert len(result.actions) == 3
    assert result.actions[session.player_agent_id].shape == (3,)
    assert session.player_agent_id in result.infos
    session.close()


def test_human_vs_bots_session_disables_human_only_eject_by_default(tmp_path) -> None:
    config = AgarioConfig()
    config.simulation.action_mode = "continuous"
    checkpoint_path = tmp_path / "latest.pt"

    env = AgarioMultiAgentEnv(config=config, enable_render=False)
    trainer = SharedPPOTrainer(config=config, observation_dim=env.observation_space["shape"][0], device="cpu")
    trainer.save(checkpoint_path)
    env.close()

    session = HumanVsBotsSession(config=config, checkpoint_path=checkpoint_path, player_index=0)
    assert session.config.physics.enable_eject_mechanic is False
    session.close()
