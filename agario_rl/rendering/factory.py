"""Renderer factory."""

from __future__ import annotations

from agario_rl import AgarioConfig
from agario_rl.rendering.backend import RendererBackend


def create_renderer(config: AgarioConfig) -> RendererBackend:
    """Build the single supported renderer."""
    from agario_rl.rendering.raylib_backend import RaylibRenderer

    return RaylibRenderer(config=config)
