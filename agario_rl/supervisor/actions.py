"""Semantic supervisor actions emitted by interactive renderers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


SupervisorActionName = Literal[
    "quit",
    "toggle_pause",
    "step_physics",
    "step_decision",
    "speed_delta",
    "speed_set",
    "toggle_auto_train",
    "toggle_curriculum",
    "reset_episode",
    "save_checkpoint",
    "load_checkpoint",
    "map_scale",
    "focus_agent",
    "toggle_overlay_mode",
    "toggle_grid",
    "toggle_help",
]


@dataclass(frozen=True, slots=True)
class SupervisorCommand:
    """A typed runtime action emitted by the UI."""

    action: SupervisorActionName
    value: float | int | str | bool | None = None

