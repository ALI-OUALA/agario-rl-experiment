"""Render frame pass-through tests."""

from __future__ import annotations

from agario_rl import AgarioConfig
from agario_rl.env.gym_env import AgarioMultiAgentEnv
from agario_rl.rendering.models import RenderFrame, StatusFrame, WorldFrame


class _StubRenderer:
    def __init__(self, config: AgarioConfig) -> None:
        self.config = config
        self.last_frame = None
        self.closed = False

    def render(self, frame):
        self.last_frame = frame
        return {"frame_ms": 1.0, "render_fps": 60.0}

    def poll_commands(self):
        return []

    def close(self) -> None:
        self.closed = True


def test_env_render_passes_render_frame(monkeypatch) -> None:
    monkeypatch.setattr("agario_rl.env.gym_env.create_renderer", lambda config: _StubRenderer(config))
    config = AgarioConfig()
    env = AgarioMultiAgentEnv(config=config, enable_render=True)
    frame = RenderFrame(
        title="test",
        world=WorldFrame(
            map_size=1.0,
            stage=0,
            step=0,
            alive_count=0,
            winner=None,
            focus_agent_id=None,
            cells=(),
            pellets=(),
        ),
        session_cards=(),
        training_cards=(),
        agent_cards=(),
        charts=(),
        controls=(),
        status=StatusFrame("ready", "info"),
        overlay_mode="minimal",
        show_grid=False,
        show_help=False,
        help_rows=(),
        interpolation_alpha=0.25,
    )
    out = env.render(frame=frame)
    assert out["render_fps"] == 60.0
    assert isinstance(env._renderer, _StubRenderer)
    assert env._renderer.last_frame is frame
    assert env.poll_commands() == []
    env.close()
    assert env._renderer is None
