"""Minimal multi-agent env wrapper around the world simulation."""

from __future__ import annotations

from typing import Any

import numpy as np

from agario_rl import AgarioConfig
from agario_rl.env.world import AgarioWorld
from agario_rl.rendering.backend import RendererBackend
from agario_rl.rendering.factory import create_renderer
from agario_rl.rendering.models import RenderFrame
from agario_rl.supervisor.actions import SupervisorCommand


class AgarioMultiAgentEnv:
    """Simple reset/step/render API for multi-agent RL."""

    def __init__(self, config: AgarioConfig, enable_render: bool = False) -> None:
        self.config = config
        self.world = AgarioWorld(config=config, seed=config.seed)
        self.agent_ids = list(self.world.agent_ids)
        self.enable_render = enable_render
        self._renderer: RendererBackend | None = None
        self.last_infos: dict[str, dict[str, Any]] = {}
        self.focus_agent_index: int | None = None

        if config.simulation.action_mode == "continuous":
            self.action_space = {
                "type": "Box",
                "shape": (3,),
                "low": np.array([-1.0, -1.0, 0.0], dtype=np.float32),
                "high": np.array([1.0, 1.0, 1.0], dtype=np.float32),
                "dtype": np.float32,
            }
        else:
            self.action_space = {
                "type": "MultiDiscrete",
                "nvec": np.array([9, 2], dtype=np.int64),
            }
        self.observation_space = {
            "shape": (self.world.observation_dim,),
            "dtype": np.float32,
        }

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        observations = self.world.reset(seed=seed)
        self.last_infos = {}
        return observations

    def step(
        self,
        actions: dict[str, np.ndarray],
        dt: float | None = None,
        compute_observations: bool = True,
    ) -> tuple[
        dict[str, np.ndarray] | None,
        dict[str, float],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        step_dt = dt if dt is not None else (1.0 / max(1, self.config.simulation.physics_hz))
        outcome = self.world.step(actions, dt=step_dt, compute_observations=compute_observations)
        self.last_infos = outcome.infos
        return outcome.observations, outcome.rewards, outcome.dones, outcome.infos

    def render(
        self,
        frame: RenderFrame | None = None,
        mode: str = "human",
    ) -> dict[str, float]:
        if mode != "human":
            raise ValueError(f"Unsupported render mode: {mode}")
        if not self.enable_render:
            return {}
        if self._renderer is None:
            self._renderer = create_renderer(config=self.config)
        if frame is None:
            raise ValueError("RenderFrame is required for interactive rendering.")
        return self._renderer.render(frame)

    def set_focus_agent(self, agent_index: int | None) -> None:
        self.focus_agent_index = agent_index

    def poll_commands(self) -> list[SupervisorCommand]:
        if self._renderer is None:
            return []
        return self._renderer.poll_commands()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
