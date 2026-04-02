"""Render frame snapshot tests."""

from __future__ import annotations

import numpy as np

from agario_rl import AgarioConfig
from agario_rl.env.gym_env import AgarioMultiAgentEnv
from agario_rl.rendering.view_model import build_render_frame
from agario_rl.supervisor.controller import SupervisorController
from agario_rl.supervisor.runtime_stats import RuntimeSessionStats


def _idle_actions(agent_ids: list[str]) -> dict[str, np.ndarray]:
    return {agent_id: np.array([0.0, 0.0, 0.0], dtype=np.float32) for agent_id in agent_ids}


def test_build_render_frame_contains_cards_and_charts() -> None:
    config = AgarioConfig()
    env = AgarioMultiAgentEnv(config=config, enable_render=False)
    controller = SupervisorController(config=config)
    runtime_stats = RuntimeSessionStats.create(env.agent_ids)
    env.reset(seed=7)
    env.step(_idle_actions(env.agent_ids))
    runtime_stats.record_infos({"__global__": {"winner": "agent_1"}})
    runtime_stats.record_frame({"render_fps": 60.0, "total_loss": 1.2, "update_count": 4.0}, env.last_infos)

    frame = build_render_frame(
        config=config,
        world=env.world,
        infos=env.last_infos,
        metrics={"render_fps": 60.0, "total_loss": 1.2, "update_count": 4.0},
        controller=controller,
        runtime_stats=runtime_stats,
        interpolation_alpha=0.5,
        focus_agent_index=1,
    )

    assert frame.world.focus_agent_id == "agent_1"
    assert len(frame.agent_cards) == len(env.agent_ids)
    assert len(frame.session_cards) >= 4
    assert len(frame.training_cards) >= 4
    assert len(frame.charts) >= 4
    assert any(card.wins == 1 for card in frame.agent_cards if card.agent_id == "agent_1")
    env.close()
