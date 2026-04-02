"""Immutable frame payloads consumed by interactive renderers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UiRect:
    """Backend-neutral rectangle used for layout and hit-testing."""

    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px <= self.right and self.y <= py <= self.bottom


@dataclass(frozen=True, slots=True)
class WorldCellFrame:
    """World-space cell snapshot for one render frame."""

    agent_id: str
    position: tuple[float, float]
    previous_position: tuple[float, float]
    mass: float
    radius: float
    is_focus: bool


@dataclass(frozen=True, slots=True)
class PelletFrame:
    """World-space pellet snapshot for one render frame."""

    position: tuple[float, float]
    mass: float


@dataclass(frozen=True, slots=True)
class AgentCardFrame:
    """Per-agent observer card."""

    agent_id: str
    display_name: str
    alive: bool
    total_mass: float
    episode_return: float
    eliminations: int
    wins: int
    focus: bool
    color: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class MetricCardFrame:
    """Single-value summary card."""

    title: str
    value: str
    accent: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class ChartFrame:
    """Small rolling chart."""

    title: str
    values: tuple[float, ...]
    accent: tuple[int, int, int]
    value_label: str


@dataclass(frozen=True, slots=True)
class ControlButtonFrame:
    """Interactive control button description."""

    action: str
    label: str
    active: bool = False
    accent: tuple[int, int, int] = (70, 118, 170)


@dataclass(frozen=True, slots=True)
class StatusFrame:
    """Short runtime status banner."""

    message: str
    level: str


@dataclass(frozen=True, slots=True)
class WorldFrame:
    """Observer-facing world snapshot."""

    map_size: float
    stage: int
    step: int
    alive_count: int
    winner: str | None
    focus_agent_id: str | None
    cells: tuple[WorldCellFrame, ...]
    pellets: tuple[PelletFrame, ...]


@dataclass(frozen=True, slots=True)
class RenderFrame:
    """Complete immutable payload consumed by a renderer."""

    title: str
    world: WorldFrame
    session_cards: tuple[MetricCardFrame, ...]
    training_cards: tuple[MetricCardFrame, ...]
    agent_cards: tuple[AgentCardFrame, ...]
    charts: tuple[ChartFrame, ...]
    controls: tuple[ControlButtonFrame, ...]
    status: StatusFrame
    overlay_mode: str
    show_grid: bool
    show_help: bool
    help_rows: tuple[str, ...]
    interpolation_alpha: float
