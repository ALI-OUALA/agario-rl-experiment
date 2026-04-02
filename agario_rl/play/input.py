"""Input adapters for human-controlled play mode."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class HumanControlInput:
    """Raw human control state for one simulation step."""

    player_position: tuple[float, float]
    target_world: tuple[float, float]
    split_pressed: bool = False
    eject_pressed: bool = False
    alive: bool = True


@dataclass(slots=True)
class PlayerCommand:
    """Resolved human command in the agent action format."""

    action: np.ndarray
    eject_requested: bool


def build_player_command(control: HumanControlInput) -> PlayerCommand:
    """Map world-space mouse aim to the existing continuous action space."""
    action = np.zeros((3,), dtype=np.float32)
    if not control.alive:
        return PlayerCommand(action=action, eject_requested=False)

    player_position = np.asarray(control.player_position, dtype=np.float32)
    target_world = np.asarray(control.target_world, dtype=np.float32)
    delta = target_world - player_position
    distance = float(np.linalg.norm(delta))
    if distance > 1e-6:
        steer = delta / max(1.0, distance)
        steer = np.clip(steer, -1.0, 1.0).astype(np.float32)
        action[0:2] = steer

    action[2] = 1.0 if control.split_pressed else 0.0
    return PlayerCommand(
        action=action,
        eject_requested=bool(control.eject_pressed),
    )
