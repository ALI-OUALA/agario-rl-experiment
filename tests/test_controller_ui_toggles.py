"""Supervisor semantic command tests."""

from __future__ import annotations

from pathlib import Path

from agario_rl import AgarioConfig
from agario_rl.env.gym_env import AgarioMultiAgentEnv
from agario_rl.rl.ppo_shared import SharedPPOTrainer
from agario_rl.supervisor.actions import SupervisorCommand
from agario_rl.supervisor.controller import SupervisorController


def test_overlay_and_grid_toggle_commands() -> None:
    config = AgarioConfig()
    controller = SupervisorController(config=config)
    controller.handle_commands(
        [
            SupervisorCommand("toggle_overlay_mode"),
            SupervisorCommand("toggle_grid"),
        ]
    )
    assert controller.overlay_mode == "full"
    assert controller.show_grid is False


def test_speed_commands_change_multiplier() -> None:
    config = AgarioConfig()
    controller = SupervisorController(config=config)
    before = controller.speed_multiplier
    controller.handle_commands([SupervisorCommand("speed_delta", 1.0)])
    assert controller.speed_multiplier > before

    faster = controller.speed_multiplier
    controller.handle_commands([SupervisorCommand("speed_delta", -1.0)])
    assert controller.speed_multiplier < faster


def test_train_toggle_and_focus_commands() -> None:
    config = AgarioConfig()
    controller = SupervisorController(config=config)
    initial = controller.auto_train_enabled
    controller.handle_commands(
        [
            SupervisorCommand("toggle_auto_train"),
            SupervisorCommand("focus_agent", 2),
        ]
    )
    assert controller.auto_train_enabled != initial
    assert controller.events.toggle_auto_train_requested
    assert controller.focus_agent_index == 2
    assert controller.events.focus_agent_index == 2


def test_apply_runtime_overrides_reports_checkpoint_status(tmp_path: Path) -> None:
    config = AgarioConfig()
    config.supervisor.checkpoint_path = str(tmp_path / "status-checkpoint.pt")
    controller = SupervisorController(config=config)
    env = AgarioMultiAgentEnv(config=config, enable_render=False)
    trainer = SharedPPOTrainer(config=config, observation_dim=env.observation_space["shape"][0], device="cpu")
    trainer.force_sync_with_env(env, seed=7)

    controller.handle_commands([SupervisorCommand("save_checkpoint")])
    controller.apply_runtime_overrides(config=config, trainer=trainer, env=env)
    assert "saved" in controller.status_message.lower()

    controller.handle_commands([SupervisorCommand("load_checkpoint")])
    controller.apply_runtime_overrides(config=config, trainer=trainer, env=env)
    assert "loaded" in controller.status_message.lower()
    env.close()
