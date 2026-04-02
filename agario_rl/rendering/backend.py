"""Shared renderer backend contract."""

from __future__ import annotations

from typing import Protocol

from agario_rl.rendering.models import RenderFrame
from agario_rl.supervisor.actions import SupervisorCommand


class RendererBackend(Protocol):
    """Renderer interface used by the environment wrapper."""

    def poll_commands(self) -> list[SupervisorCommand]:
        ...

    def render(
        self,
        frame: RenderFrame,
    ) -> dict[str, float]:
        ...

    def close(self) -> None:
        ...
