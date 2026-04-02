"""Renderer factory tests."""

from __future__ import annotations

from agario_rl import AgarioConfig
from agario_rl.rendering.factory import create_renderer


def test_factory_builds_raylib_renderer(monkeypatch) -> None:
    class _StubRenderer:
        def __init__(self, config: AgarioConfig) -> None:
            self.config = config

    monkeypatch.setattr("agario_rl.rendering.raylib_backend.RaylibRenderer", _StubRenderer)
    config = AgarioConfig()
    out = create_renderer(config=config)
    assert isinstance(out, _StubRenderer)
