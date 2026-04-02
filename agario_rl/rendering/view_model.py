"""Build immutable render snapshots from world and supervisor state."""

from __future__ import annotations

from typing import Any

from agario_rl import AgarioConfig
from agario_rl.rendering.models import (
    AgentCardFrame,
    ChartFrame,
    ControlButtonFrame,
    MetricCardFrame,
    PelletFrame,
    RenderFrame,
    StatusFrame,
    WorldCellFrame,
    WorldFrame,
)
from agario_rl.supervisor.controller import SupervisorController
from agario_rl.supervisor.runtime_stats import RuntimeSessionStats


AGENT_COLORS: tuple[tuple[int, int, int], ...] = (
    (80, 252, 54),
    (36, 244, 255),
    (243, 31, 46),
    (255, 183, 77),
)


HELP_ROWS: tuple[str, ...] = (
    "Controls",
    "Space: pause/resume",
    "N: one physics tick",
    "Shift+N: one decision tick",
    "- / +: slower / faster",
    "T: toggle train more",
    "R: reset episode + wins",
    "C: toggle curriculum",
    "M / Shift+M: map bigger / smaller",
    "1/2/3: focus agent",
    "WASD or arrows: free camera pan",
    "Middle mouse drag: pan viewport",
    "0: return to agent follow",
    "Tab: compact / full cockpit",
    "G: toggle grid",
    "F11: toggle fullscreen",
    "P / L: save / load checkpoint",
    "F1: toggle help",
)


def _agent_color(index: int) -> tuple[int, int, int]:
    return AGENT_COLORS[index % len(AGENT_COLORS)]


def _series_label(values: tuple[float, ...], fmt: str = "{:.2f}") -> str:
    if not values:
        return "-"
    return fmt.format(values[-1])


def build_render_frame(
    *,
    config: AgarioConfig,
    world: Any,
    infos: dict[str, dict[str, Any]],
    metrics: dict[str, float],
    controller: SupervisorController,
    runtime_stats: RuntimeSessionStats,
    interpolation_alpha: float,
    focus_agent_index: int | None,
) -> RenderFrame:
    """Convert runtime state into an immutable render payload."""

    alive = world.alive_agents
    focus_agent_id = None
    if focus_agent_index is not None and 0 <= focus_agent_index < len(world.agent_ids):
        focus_agent_id = world.agent_ids[focus_agent_index]
    elif alive:
        focus_agent_id = alive[0]

    cells: list[WorldCellFrame] = []
    for agent_index, agent_id in enumerate(world.agent_ids):
        for cell in world.agents[agent_id]:
            prev = world.previous_cell_position(cell)
            cells.append(
                WorldCellFrame(
                    agent_id=agent_id,
                    position=(float(cell.position[0]), float(cell.position[1])),
                    previous_position=(float(prev[0]), float(prev[1])),
                    mass=float(cell.mass),
                    radius=float(cell.radius(config.physics.radius_scale)),
                    is_focus=agent_id == focus_agent_id,
                )
            )

    pellets = tuple(
        PelletFrame(position=(float(p.position[0]), float(p.position[1])), mass=float(p.mass))
        for p in world.pellets
    )

    world_frame = WorldFrame(
        map_size=float(world.map_size),
        stage=int(getattr(world, "stage", 0)),
        step=int(getattr(world, "step_count", 0)),
        alive_count=len(alive),
        winner=infos.get("__global__", {}).get("winner"),
        focus_agent_id=focus_agent_id,
        cells=tuple(cells),
        pellets=pellets,
    )

    session_cards = (
        MetricCardFrame("State", "Paused" if controller.paused else "Running", (77, 163, 116)),
        MetricCardFrame("Speed", f"x{controller.speed_multiplier:.2f}", (70, 118, 170)),
        MetricCardFrame("FPS", f"{metrics.get('render_fps', 0.0):.1f}", (255, 183, 77)),
        MetricCardFrame("Frame", f"{metrics.get('frame_ms', 0.0):.2f} ms", (157, 111, 193)),
        MetricCardFrame("Physics", f"{metrics.get('physics_steps_per_sec', 0.0):.1f}/s", (0, 157, 173)),
        MetricCardFrame("Queue", str(int(metrics.get("worker_queue", 0.0))), (236, 100, 75)),
        MetricCardFrame("Updates", str(int(metrics.get("update_count", 0.0))), (94, 148, 255)),
    )

    training_cards = (
        MetricCardFrame("Policy", f"{metrics.get('policy_loss', 0.0):.4f}", (255, 183, 77)),
        MetricCardFrame("Value", f"{metrics.get('value_loss', 0.0):.4f}", (94, 148, 255)),
        MetricCardFrame("Entropy", f"{metrics.get('entropy', 0.0):.4f}", (77, 163, 116)),
        MetricCardFrame("Imitation", f"{metrics.get('imitation_loss', 0.0):.4f}", (157, 111, 193)),
        MetricCardFrame("Sync Age", f"{metrics.get('policy_sync_age_steps', 0.0):.0f}", (236, 100, 75)),
    )

    agent_cards: list[AgentCardFrame] = []
    for agent_index, agent_id in enumerate(world.agent_ids):
        info = infos.get(agent_id, {})
        agent_cards.append(
            AgentCardFrame(
                agent_id=agent_id,
                display_name=agent_id.replace("_", " ").title(),
                alive=bool(info.get("alive", False)),
                total_mass=float(info.get("total_mass", 0.0)),
                episode_return=float(info.get("episode_return", 0.0)),
                eliminations=int(info.get("eliminations", 0)),
                wins=runtime_stats.wins_by_agent.get(agent_id, 0),
                focus=agent_id == focus_agent_id,
                color=_agent_color(agent_index),
            )
        )

    charts = (
        ChartFrame("FPS", runtime_stats.chart_series("render_fps"), (255, 183, 77), _series_label(runtime_stats.chart_series("render_fps"), "{:.1f}")),
        ChartFrame("Reward", runtime_stats.chart_series("reward_mean"), (77, 163, 116), _series_label(runtime_stats.chart_series("reward_mean"), "{:.2f}")),
        ChartFrame("Loss", runtime_stats.chart_series("total_loss"), (236, 100, 75), _series_label(runtime_stats.chart_series("total_loss"), "{:.3f}")),
        ChartFrame("Wins", runtime_stats.chart_series("wins_total"), (94, 148, 255), _series_label(runtime_stats.chart_series("wins_total"), "{:.0f}")),
        ChartFrame("Updates", runtime_stats.chart_series("update_count"), (157, 111, 193), _series_label(runtime_stats.chart_series("update_count"), "{:.0f}")),
    )

    controls = (
        ControlButtonFrame("toggle_pause", "Pause" if not controller.paused else "Resume", controller.paused),
        ControlButtonFrame("step_physics", "Step Tick"),
        ControlButtonFrame("step_decision", "Step Decision"),
        ControlButtonFrame("speed_down", "Slower"),
        ControlButtonFrame("speed_up", "Faster"),
        ControlButtonFrame("toggle_auto_train", f"Train More {'ON' if controller.auto_train_enabled else 'OFF'}", controller.auto_train_enabled),
        ControlButtonFrame("toggle_curriculum", f"Curriculum {'ON' if controller.auto_curriculum else 'OFF'}", controller.auto_curriculum),
        ControlButtonFrame("map_down", "Map -"),
        ControlButtonFrame("map_up", "Map +"),
        ControlButtonFrame("reset_episode", "Reset"),
        ControlButtonFrame("save_checkpoint", "Save"),
        ControlButtonFrame("load_checkpoint", "Load"),
        ControlButtonFrame("toggle_grid", "Grid", controller.show_grid),
        ControlButtonFrame("toggle_overlay_mode", "Cockpit", controller.overlay_mode == "full"),
        ControlButtonFrame("toggle_fullscreen", "Fullscreen"),
        ControlButtonFrame("toggle_help", "Help", controller.show_help),
    )

    return RenderFrame(
        title="Agario RL Observer Cockpit",
        world=world_frame,
        session_cards=session_cards,
        training_cards=training_cards,
        agent_cards=tuple(agent_cards),
        charts=charts,
        controls=controls,
        status=StatusFrame(controller.status_message, controller.status_level),
        overlay_mode=controller.overlay_mode,
        show_grid=controller.show_grid,
        show_help=controller.show_help,
        help_rows=HELP_ROWS,
        interpolation_alpha=float(interpolation_alpha),
    )
