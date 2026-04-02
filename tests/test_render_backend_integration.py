"""Backend integration smoke tests."""

from __future__ import annotations

import os

import numpy as np
import pytest

from agario_rl import AgarioConfig
from agario_rl.env.gym_env import AgarioMultiAgentEnv
from agario_rl.rendering.view_model import build_render_frame
from agario_rl.supervisor.controller import SupervisorController
from agario_rl.supervisor.runtime_stats import RuntimeSessionStats


def _idle_actions(agent_ids: list[str]) -> dict[str, np.ndarray]:
    return {agent_id: np.array([0.0, 0.0, 0.0], dtype=np.float32) for agent_id in agent_ids}


def test_env_render_with_raylib_backend_smoke() -> None:
    os.environ.setdefault("AGARIO_RL_HEADLESS_RENDER", "1")
    config = AgarioConfig()
    controller = SupervisorController(config=config)
    env = AgarioMultiAgentEnv(config=config, enable_render=True)
    runtime_stats = RuntimeSessionStats.create(env.agent_ids)
    try:
        env.reset(seed=7)
        env.step(_idle_actions(env.agent_ids))
        frame = build_render_frame(
            config=config,
            world=env.world,
            infos=env.last_infos,
            metrics={},
            controller=controller,
            runtime_stats=runtime_stats,
            interpolation_alpha=0.5,
            focus_agent_index=env.focus_agent_index,
        )
        try:
            out = env.render(frame=frame)
        except Exception as exc:  # pragma: no cover - platform-specific headless limitations
            pytest.skip(f"raylib runtime unavailable in this environment: {exc}")
        assert "frame_ms" in out
        assert "render_fps" in out
    finally:
        env.close()
