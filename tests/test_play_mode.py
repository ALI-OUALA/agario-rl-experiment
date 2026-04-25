"""Tests for the human-playable mode helpers."""

from __future__ import annotations

import numpy as np

from agario_rl import AgarioConfig
from agario_rl.env.gym_env import AgarioMultiAgentEnv
from agario_rl.opponents import PelletForagerPolicy, ThreatAwareEvaderPolicy
from agario_rl.play import HumanControlInput, HumanVsBotsSession, build_player_command
from agario_rl.rl.ppo_shared import SharedPPOTrainer
from scripts import play


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


def test_human_vs_bots_session_uses_mixed_opponent_pool_by_default(tmp_path) -> None:
    config = AgarioConfig()
    config.simulation.action_mode = "continuous"
    checkpoint_path = tmp_path / "latest.pt"

    env = AgarioMultiAgentEnv(config=config, enable_render=False)
    trainer = SharedPPOTrainer(config=config, observation_dim=env.observation_space["shape"][0], device="cpu")
    trainer.save(checkpoint_path)
    env.close()

    session = HumanVsBotsSession(config=config, checkpoint_path=checkpoint_path, player_index=0)
    try:
        pool_names = {policy.name for policy in session.opponent_pool}

        assert {"checkpoint_500_anchor", "pellet_forager", "threat_aware_evader", "opportunistic_hunter"} <= pool_names
        assert set(session.active_opponents) == set(session.opponent_agent_ids)
    finally:
        session.close()


def test_human_vs_bots_session_can_assign_deterministic_debug_opponents(tmp_path) -> None:
    config = AgarioConfig()
    config.num_agents = 6
    config.simulation.action_mode = "continuous"
    pool = [PelletForagerPolicy(), ThreatAwareEvaderPolicy()]

    session = HumanVsBotsSession(
        config=config,
        checkpoint_path=tmp_path / "unused.pt",
        player_index=0,
        opponent_pool=pool,
        deterministic_opponents=True,
    )
    try:
        names = [session.active_opponents[agent_id].name for agent_id in session.opponent_agent_ids]

        assert names == [
            "pellet_forager",
            "threat_aware_evader",
            "pellet_forager",
            "threat_aware_evader",
            "pellet_forager",
        ]
    finally:
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


def test_play_cli_defaults_to_richer_readable_arena(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["play.py"])

    args = play.parse_args()

    assert args.scenario_preset == "full_arena"
    assert args.render_detail == "high"
    assert args.continuing_respawn is False


def test_screen_visibility_allows_small_margin() -> None:
    assert play._is_screen_visible(20.0, 20.0, 1600, 900)
    assert play._is_screen_visible(-10.0, 450.0, 1600, 900)
    assert not play._is_screen_visible(-50.0, 450.0, 1600, 900)


def test_smoothing_is_frame_rate_independent_directional() -> None:
    small_step = play._smooth_toward(0.0, 10.0, dt=1.0 / 120.0, response=6.0)
    large_step = play._smooth_toward(0.0, 10.0, dt=1.0 / 30.0, response=6.0)

    assert 0.0 < small_step < large_step < 10.0


def test_play_camera_target_includes_nearby_bot(tmp_path) -> None:
    config = AgarioConfig()
    config.simulation.action_mode = "continuous"
    checkpoint_path = tmp_path / "latest.pt"

    env = AgarioMultiAgentEnv(config=config, enable_render=False)
    trainer = SharedPPOTrainer(config=config, observation_dim=env.observation_space["shape"][0], device="cpu")
    trainer.save(checkpoint_path)
    env.close()

    session = HumanVsBotsSession(config=config, checkpoint_path=checkpoint_path, player_index=0)
    try:
        player_id = session.player_agent_id
        bot_id = next(agent_id for agent_id in session.env.agent_ids if agent_id != player_id)
        player_cell = session.env.world.agents[player_id][0]
        bot_cell = session.env.world.agents[bot_id][0]
        player_cell.position = np.array([100.0, 100.0], dtype=np.float32)
        bot_cell.position = np.array([240.0, 100.0], dtype=np.float32)

        camera, zoom = play._target_camera_and_zoom(session, width=1200, height=800)

        assert camera[0] > 100.0
        assert zoom > 0.0
    finally:
        session.close()
