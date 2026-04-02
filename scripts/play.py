"""Human-playable Agar.io session against trained agents."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agario_rl import load_config
from agario_rl.play import HumanControlInput, HumanVsBotsSession
from agario_rl.utils.seeding import set_global_seeds


PLAYER_LABELS = {
    "agent_0": "You",
    "agent_1": "Bot 1",
    "agent_2": "Bot 2",
}

AGENT_COLORS = {
    "agent_0": (102, 224, 94),
    "agent_1": (74, 198, 255),
    "agent_2": (255, 110, 126),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play against the trained Agar.io agents.")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--player-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    return parser.parse_args()


def _resolve_path(project_root: Path, raw_path: str | None, fallback: str) -> Path:
    candidate = Path(raw_path) if raw_path else Path(fallback)
    return candidate if candidate.is_absolute() else (project_root / candidate)


def _color(pr, rgb: tuple[int, int, int], alpha: int = 255):  # noqa: ANN001, ANN202
    return pr.Color(rgb[0], rgb[1], rgb[2], alpha)


def _world_to_screen(
    world: Iterable[float],
    camera: np.ndarray,
    zoom: float,
    width: int,
    height: int,
) -> tuple[float, float]:
    wx, wy = world
    return (
        (float(wx) - float(camera[0])) * zoom + width * 0.5,
        (float(wy) - float(camera[1])) * zoom + height * 0.5,
    )


def _screen_to_world(
    screen_x: float,
    screen_y: float,
    camera: np.ndarray,
    zoom: float,
    width: int,
    height: int,
) -> tuple[float, float]:
    return (
        float(camera[0]) + (screen_x - width * 0.5) / max(zoom, 1e-6),
        float(camera[1]) + (screen_y - height * 0.5) / max(zoom, 1e-6),
    )


def _clamp_camera(camera: np.ndarray, zoom: float, width: int, height: int, map_size: float) -> np.ndarray:
    half_x = width * 0.5 / max(zoom, 1e-6)
    half_y = height * 0.5 / max(zoom, 1e-6)
    next_camera = camera.copy()
    if half_x * 2.0 >= map_size:
        next_camera[0] = map_size * 0.5
    else:
        next_camera[0] = float(np.clip(next_camera[0], half_x, map_size - half_x))
    if half_y * 2.0 >= map_size:
        next_camera[1] = map_size * 0.5
    else:
        next_camera[1] = float(np.clip(next_camera[1], half_y, map_size - half_y))
    return next_camera


def _draw_grid(pr, camera: np.ndarray, zoom: float, width: int, height: int, map_size: float) -> None:
    left_world, top_world = _screen_to_world(0.0, 0.0, camera, zoom, width, height)
    right_world, bottom_world = _screen_to_world(float(width), float(height), camera, zoom, width, height)
    spacing = 32.0
    start_x = int(max(0.0, np.floor(left_world / spacing) * spacing))
    end_x = int(min(map_size, np.ceil(right_world / spacing) * spacing))
    start_y = int(max(0.0, np.floor(top_world / spacing) * spacing))
    end_y = int(min(map_size, np.ceil(bottom_world / spacing) * spacing))
    grid_color = _color(pr, (213, 227, 234), 170)

    for world_x in range(start_x, end_x + 1, int(spacing)):
        sx0, sy0 = _world_to_screen((world_x, 0.0), camera, zoom, width, height)
        sx1, sy1 = _world_to_screen((world_x, map_size), camera, zoom, width, height)
        pr.draw_line(int(sx0), int(sy0), int(sx1), int(sy1), grid_color)
    for world_y in range(start_y, end_y + 1, int(spacing)):
        sx0, sy0 = _world_to_screen((0.0, world_y), camera, zoom, width, height)
        sx1, sy1 = _world_to_screen((map_size, world_y), camera, zoom, width, height)
        pr.draw_line(int(sx0), int(sy0), int(sx1), int(sy1), grid_color)


def _draw_hud(pr, session: HumanVsBotsSession, width: int, height: int, winner: str | None, player_dead: bool) -> None:
    leaderboard = session.leaderboard()
    player_mass = next((mass for agent_id, mass in leaderboard if agent_id == session.player_agent_id), 0.0)
    player_cells = len(session.env.world.agents[session.player_agent_id])

    hud_bg = _color(pr, (255, 255, 255), 230)
    pr.draw_rectangle_rounded(pr.Rectangle(width - 300, 20, 260, 170), 0.18, 8, hud_bg)
    pr.draw_text("Leaderboard", width - 272, 32, 28, _color(pr, (35, 49, 58)))
    for index, (agent_id, mass) in enumerate(leaderboard[:5], start=1):
        label = PLAYER_LABELS.get(agent_id, agent_id.replace("_", " ").title())
        text = f"{index}. {label}"
        pr.draw_text(text, width - 272, 38 + index * 24, 20, _color(pr, (55, 71, 82)))
        pr.draw_text(f"{mass:.1f}", width - 86, 38 + index * 24, 20, _color(pr, AGENT_COLORS.get(agent_id, (80, 80, 80))))

    pr.draw_rectangle_rounded(pr.Rectangle(20, 20, 320, 136), 0.18, 8, hud_bg)
    pr.draw_text("Human mode", 42, 32, 28, _color(pr, (35, 49, 58)))
    pr.draw_text(f"Mass: {player_mass:.1f}", 42, 70, 22, _color(pr, (55, 71, 82)))
    pr.draw_text(f"Cells: {player_cells}", 42, 98, 22, _color(pr, (55, 71, 82)))
    pr.draw_text("Mouse aim | Space split | E eject | Enter restart", 42, 126, 18, _color(pr, (98, 115, 126)))

    status_text = None
    if winner is not None:
        status_text = f"Winner: {PLAYER_LABELS.get(winner, winner)}. Press Enter to restart."
    elif player_dead:
        status_text = "You were eaten. Press Enter to restart."
    if status_text:
        pr.draw_rectangle_rounded(
            pr.Rectangle(20, height - 88, 520, 54),
            0.2,
            8,
            _color(pr, (35, 49, 58), 220),
        )
        pr.draw_text(status_text, 38, height - 72, 22, _color(pr, (255, 255, 255)))


def main() -> None:
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    config.simulation.action_mode = "continuous"
    config.render.enabled = True
    seed = config.seed if args.seed is None else int(args.seed)
    set_global_seeds(seed)

    checkpoint_path = _resolve_path(
        project_root=PROJECT_ROOT,
        raw_path=args.checkpoint,
        fallback=config.supervisor.checkpoint_path,
    )

    session = HumanVsBotsSession(
        config=config,
        checkpoint_path=checkpoint_path,
        player_index=args.player_index,
        seed=seed,
        enable_eject=True,
    )

    try:
        import pyray as pr
    except Exception as exc:  # pragma: no cover - depends on local runtime
        session.close()
        raise RuntimeError(
            "Playable mode requires the `raylib` package. Install with `pip install raylib`."
        ) from exc

    pr.set_config_flags(int(getattr(pr, "FLAG_MSAA_4X_HINT", 0)))
    pr.init_window(args.width, args.height, "Agario RL - play mode")
    pr.set_target_fps(max(60, int(config.render.fps)))

    camera = session.player_center()
    zoom = 2.3
    last_result = None

    while not pr.window_should_close():
        width = int(pr.get_screen_width())
        height = int(pr.get_screen_height())

        if pr.is_key_pressed(pr.KEY_ENTER):
            session.reset(seed=seed)
            last_result = None

        target_center = session.player_center()
        if session.player_alive():
            target_mass = max(
                20.0,
                float(sum(cell.mass for cell in session.env.world.agents[session.player_agent_id])),
            )
            target_span = float(np.clip(220.0 + np.sqrt(target_mass) * 20.0, 200.0, session.env.world.map_size))
            target_zoom = float(np.clip(min(width, height) / max(target_span, 1.0), 0.35, 4.0))
        else:
            target_zoom = 0.55

        camera = 0.84 * camera + 0.16 * target_center
        zoom = 0.9 * zoom + 0.1 * target_zoom
        camera = _clamp_camera(camera, zoom, width, height, session.env.world.map_size)

        mouse = pr.get_mouse_position()
        mouse_world = _screen_to_world(float(mouse.x), float(mouse.y), camera, zoom, width, height)
        control = HumanControlInput(
            player_position=(float(target_center[0]), float(target_center[1])),
            target_world=mouse_world,
            split_pressed=pr.is_key_pressed(pr.KEY_SPACE),
            eject_pressed=pr.is_key_pressed(pr.KEY_E),
            alive=session.player_alive(),
        )
        round_over = bool(last_result and last_result.dones.get("__all__", False))
        if round_over or not session.player_alive():
            result = last_result
        else:
            result = session.step(control)
            last_result = result

        pr.begin_drawing()
        pr.clear_background(_color(pr, (240, 247, 250)))
        _draw_grid(pr, camera, zoom, width, height, session.env.world.map_size)

        for pellet in session.env.world.pellets:
            sx, sy = _world_to_screen(pellet.position, camera, zoom, width, height)
            pr.draw_circle(int(sx), int(sy), max(2.5, 2.2 * zoom), _color(pr, (142, 228, 124)))

        for agent_id in session.env.agent_ids:
            label = PLAYER_LABELS.get(agent_id, agent_id.replace("_", " ").title())
            rgb = AGENT_COLORS.get(agent_id, (90, 90, 90))
            cells = session.env.world.agents[agent_id]
            for cell in cells:
                sx, sy = _world_to_screen(cell.position, camera, zoom, width, height)
                radius = max(4.0, cell.radius(config.physics.radius_scale) * zoom)
                pr.draw_circle(int(sx), int(sy), radius + 4.0, _color(pr, rgb, 80))
                pr.draw_circle(int(sx), int(sy), radius, _color(pr, rgb))
                pr.draw_circle_lines(int(sx), int(sy), radius, _color(pr, (255, 255, 255), 190))
                pr.draw_text(
                    label,
                    int(sx - radius * 0.45),
                    int(sy - 8),
                    max(16, int(12 + radius * 0.12)),
                    _color(pr, (255, 255, 255)),
                )

        aim_x, aim_y = _world_to_screen(mouse_world, camera, zoom, width, height)
        pr.draw_circle_lines(int(aim_x), int(aim_y), 12.0, _color(pr, (35, 49, 58), 150))

        winner = result.infos.get("__global__", {}).get("winner") if result is not None else None
        _draw_hud(
            pr,
            session=session,
            width=width,
            height=height,
            winner=winner,
            player_dead=not session.player_alive(),
        )
        pr.end_drawing()

    session.close()
    pr.close_window()


if __name__ == "__main__":
    main()
