"""Runtime supervisor state machine for semantic UI commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from agario_rl import AgarioConfig
from agario_rl.supervisor.actions import SupervisorCommand


@dataclass(slots=True)
class SupervisorEvents:
    """Transient commands applied by the supervisor loop."""

    quit_requested: bool = False
    reset_requested: bool = False
    step_tick_once: bool = False
    step_policy_once: bool = False
    save_requested: bool = False
    load_requested: bool = False
    map_increase_requested: bool = False
    map_decrease_requested: bool = False
    focus_agent_index: int | None = None
    toggle_auto_train_requested: bool = False


class SupervisorController:
    """Semantic state machine for runtime actions and status banners."""

    def __init__(self, config: AgarioConfig) -> None:
        self.config = config
        self.paused = False
        self.show_help = bool(config.render.show_help_by_default)
        self.show_grid = bool(config.render.grid_enabled_default)
        self.overlay_mode = str(config.render.overlay_mode_default)
        self.speed_multiplier = 1.0
        self.auto_curriculum = bool(config.curriculum.enabled)
        self.auto_train_enabled = bool(config.supervisor.auto_update_when_ready)
        self.focus_agent_index: int | None = None
        self.status_message = "Observer cockpit ready."
        self.status_level = "info"
        self.events = SupervisorEvents()

    def clear_transient_events(self) -> None:
        self.events = SupervisorEvents()

    def set_status(self, message: str, level: str = "info") -> None:
        self.status_message = str(message)
        self.status_level = str(level)

    def _decrease_speed(self) -> None:
        self.speed_multiplier = max(
            self.config.supervisor.min_speed_multiplier,
            self.speed_multiplier - self.config.supervisor.speed_step,
        )
        self.set_status(f"Speed set to x{self.speed_multiplier:.2f}.")

    def _increase_speed(self) -> None:
        self.speed_multiplier = min(
            self.config.supervisor.max_speed_multiplier,
            self.speed_multiplier + self.config.supervisor.speed_step,
        )
        self.set_status(f"Speed set to x{self.speed_multiplier:.2f}.")

    def _set_speed(self, value: float) -> None:
        self.speed_multiplier = min(
            self.config.supervisor.max_speed_multiplier,
            max(self.config.supervisor.min_speed_multiplier, float(value)),
        )
        self.set_status(f"Speed set to x{self.speed_multiplier:.2f}.")

    def _toggle_auto_train(self) -> None:
        self.auto_train_enabled = not self.auto_train_enabled
        self.events.toggle_auto_train_requested = True
        state = "ON" if self.auto_train_enabled else "OFF"
        self.set_status(f"Train More is {state}.")

    def handle_commands(self, commands: Sequence[SupervisorCommand]) -> None:
        """Apply semantic commands emitted by the renderer."""
        self.clear_transient_events()
        for command in commands:
            action = command.action
            value = command.value
            if action == "quit":
                self.events.quit_requested = True
            elif action == "toggle_pause":
                self.paused = not self.paused
                self.set_status("Simulation paused." if self.paused else "Simulation resumed.")
            elif action == "step_physics":
                self.events.step_tick_once = True
            elif action == "step_decision":
                self.events.step_policy_once = True
            elif action == "speed_delta":
                if float(value or 0.0) < 0.0:
                    self._decrease_speed()
                else:
                    self._increase_speed()
            elif action == "speed_set" and value is not None:
                self._set_speed(float(value))
            elif action == "toggle_auto_train":
                self._toggle_auto_train()
            elif action == "toggle_curriculum":
                self.auto_curriculum = not self.auto_curriculum
                state = "ON" if self.auto_curriculum else "OFF"
                self.set_status(f"Curriculum is {state}.")
            elif action == "reset_episode":
                self.events.reset_requested = True
            elif action == "save_checkpoint":
                self.events.save_requested = True
            elif action == "load_checkpoint":
                self.events.load_requested = True
            elif action == "map_scale":
                if float(value or 0.0) < 0.0:
                    self.events.map_decrease_requested = True
                else:
                    self.events.map_increase_requested = True
            elif action == "focus_agent":
                if value is None:
                    continue
                self.focus_agent_index = int(value)
                self.events.focus_agent_index = self.focus_agent_index
                self.set_status(f"Camera focused on agent {self.focus_agent_index + 1}.")
            elif action == "toggle_overlay_mode":
                self.overlay_mode = "full" if self.overlay_mode == "minimal" else "minimal"
                self.set_status(f"Cockpit mode set to {self.overlay_mode}.")
            elif action == "toggle_grid":
                self.show_grid = not self.show_grid
                self.set_status(f"Grid {'enabled' if self.show_grid else 'hidden'}.")
            elif action == "toggle_help":
                self.show_help = not self.show_help
                self.set_status(f"Help {'visible' if self.show_help else 'hidden'}.")

    def apply_runtime_overrides(self, config: AgarioConfig, trainer: Any, env: Any) -> None:
        """Apply stateful commands to the trainer and environment."""
        env.world.auto_curriculum = self.auto_curriculum
        if self.events.map_increase_requested:
            env.world.adjust_map_size(increase=True)
            self.set_status(f"Map size increased to {int(env.world.map_size)}.")
        if self.events.map_decrease_requested:
            env.world.adjust_map_size(increase=False)
            self.set_status(f"Map size decreased to {int(env.world.map_size)}.")
        if self.events.save_requested:
            checkpoint_path = Path(config.supervisor.checkpoint_path)
            trainer.save(checkpoint_path)
            self.set_status(f"Checkpoint saved to {checkpoint_path.as_posix()}.")
        if self.events.load_requested:
            checkpoint_path = Path(config.supervisor.checkpoint_path)
            if trainer.load(checkpoint_path):
                self.set_status(f"Checkpoint loaded from {checkpoint_path.as_posix()}.")
            else:
                self.set_status(f"Checkpoint missing at {checkpoint_path.as_posix()}.", level="warning")
        if self.events.focus_agent_index is not None:
            env.set_focus_agent(self.events.focus_agent_index)

