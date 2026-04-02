"""Raylib-based observer cockpit renderer."""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np

from agario_rl import AgarioConfig
from agario_rl.rendering.models import ChartFrame, RenderFrame, UiRect
from agario_rl.supervisor.actions import SupervisorCommand


class RaylibRenderer:
    """Interactive observer cockpit powered by Raylib."""

    _MIN_WORLD_WIDTH = 640.0
    _MIN_WINDOW_WIDTH = 1280
    _MIN_WINDOW_HEIGHT = 720
    _PANEL_SCALE_MIN = 0.92
    _PANEL_SCALE_MAX = 1.55
    _WORLD_SCALE_MIN = 0.92
    _WORLD_SCALE_MAX = 1.28

    def __init__(self, config: AgarioConfig) -> None:
        try:
            import pyray as pr
        except Exception as exc:  # pragma: no cover - depends on local runtime
            raise RuntimeError(
                "Raylib backend requested but the `raylib` package is not available. "
                "Install with `pip install raylib`."
            ) from exc

        self.pr = pr
        self.config = config
        self.base_width = max(self._MIN_WINDOW_WIDTH, int(config.render.window_width))
        self.base_height = max(self._MIN_WINDOW_HEIGHT, int(config.render.window_height))
        self.base_side_panel = max(420, int(config.render.side_panel_width))
        self.width = self.base_width
        self.height = self.base_height
        self.side_panel = self.base_side_panel
        self.windowed_width = self.base_width
        self.windowed_height = self.base_height
        self.camera_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.zoom = 1.0
        self.camera_follow_enabled = True
        self.last_frame_time = time.perf_counter()
        self.last_frame_ms = 0.0
        self.last_render_fps = 0.0
        self._fps_ema = 0.0
        self._closed = False
        self._fullscreen = False
        self._headless = bool(int(os.environ.get("AGARIO_RL_HEADLESS_RENDER", "0")))
        self._button_rects: dict[str, UiRect] = {}
        self._focus_rects: dict[int, UiRect] = {}

        flags = int(getattr(pr, "FLAG_MSAA_4X_HINT", 0))
        if self._headless:
            flags |= int(getattr(pr, "FLAG_WINDOW_HIDDEN", 0))
        if bool(config.render.window_resizable):
            flags |= int(getattr(pr, "FLAG_WINDOW_RESIZABLE", 0))
        if flags:
            pr.set_config_flags(flags)
        pr.init_window(self.width, self.height, "Agario RL Observer Cockpit")
        if hasattr(pr, "set_window_min_size"):
            pr.set_window_min_size(self._MIN_WINDOW_WIDTH, self._MIN_WINDOW_HEIGHT)
        if bool(config.render.start_fullscreen) and not self._headless:
            self._set_fullscreen(True)
        self._sync_window_state(force=True)
        pr.set_target_fps(max(1, int(config.render.fps)))

    def poll_commands(self) -> list[SupervisorCommand]:
        """Read keyboard and mouse actions for the current frame."""
        if self._closed:
            return []
        pr = self.pr
        commands: list[SupervisorCommand] = []
        if pr.window_should_close():
            commands.append(SupervisorCommand("quit"))

        if pr.is_key_pressed(pr.KEY_F11):
            self._toggle_fullscreen()

        shift = pr.is_key_down(pr.KEY_LEFT_SHIFT) or pr.is_key_down(pr.KEY_RIGHT_SHIFT)
        keymap: tuple[tuple[int, SupervisorCommand], ...] = (
            (pr.KEY_SPACE, SupervisorCommand("toggle_pause")),
            (pr.KEY_N, SupervisorCommand("step_decision" if shift else "step_physics")),
            (pr.KEY_MINUS, SupervisorCommand("speed_delta", -1.0)),
            (pr.KEY_EQUAL, SupervisorCommand("speed_delta", 1.0)),
            (pr.KEY_T, SupervisorCommand("toggle_auto_train")),
            (pr.KEY_C, SupervisorCommand("toggle_curriculum")),
            (pr.KEY_R, SupervisorCommand("reset_episode")),
            (pr.KEY_M, SupervisorCommand("map_scale", -1.0 if shift else 1.0)),
            (pr.KEY_P, SupervisorCommand("save_checkpoint")),
            (pr.KEY_L, SupervisorCommand("load_checkpoint")),
            (pr.KEY_TAB, SupervisorCommand("toggle_overlay_mode")),
            (pr.KEY_G, SupervisorCommand("toggle_grid")),
            (pr.KEY_F1, SupervisorCommand("toggle_help")),
            (pr.KEY_ONE, SupervisorCommand("focus_agent", 0)),
            (pr.KEY_TWO, SupervisorCommand("focus_agent", 1)),
            (pr.KEY_THREE, SupervisorCommand("focus_agent", 2)),
        )
        for key, command in keymap:
            if pr.is_key_pressed(key):
                if command.action == "focus_agent":
                    self.camera_follow_enabled = True
                commands.append(command)

        if pr.is_mouse_button_pressed(pr.MOUSE_BUTTON_LEFT):
            mouse = pr.get_mouse_position()
            mx = float(mouse.x)
            my = float(mouse.y)
            for action, rect in self._button_rects.items():
                if rect.contains(mx, my):
                    commands.extend(self._commands_for_button(action))
                    break
            else:
                for agent_index, rect in self._focus_rects.items():
                    if rect.contains(mx, my):
                        self.camera_follow_enabled = True
                        commands.append(SupervisorCommand("focus_agent", agent_index))
                        break
        return commands

    def render(self, frame: RenderFrame) -> dict[str, float]:
        """Draw the cockpit for one frame."""
        if self._closed:
            return {"frame_ms": self.last_frame_ms, "render_fps": self.last_render_fps}
        pr = self.pr
        self._sync_window_state()

        world_rect, panel_rect, panel_scale, world_scale = self._layout(frame.overlay_mode)
        self._button_rects = {}
        self._focus_rects = {}
        self._handle_camera_input(frame, world_rect)

        pr.begin_drawing()
        pr.clear_background(self._color(245, 249, 252))
        self._update_camera(frame, world_rect)
        self._draw_world(frame, world_rect, world_scale)
        self._draw_side_panel(frame, panel_rect, panel_scale)
        self._draw_status_banner(frame, world_rect, world_scale)
        pr.end_drawing()

        now = time.perf_counter()
        self.last_frame_ms = (now - self.last_frame_time) * 1000.0
        self.last_frame_time = now
        current_fps = 1000.0 / max(self.last_frame_ms, 1e-6)
        self._fps_ema = current_fps if self._fps_ema <= 0.0 else (0.88 * self._fps_ema + 0.12 * current_fps)
        self.last_render_fps = self._fps_ema
        return {"frame_ms": self.last_frame_ms, "render_fps": self.last_render_fps}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.pr.close_window()

    def _sync_window_state(self, force: bool = False) -> None:
        fullscreen = bool(self.pr.is_window_fullscreen()) if hasattr(self.pr, "is_window_fullscreen") else self._fullscreen
        width = int(self.pr.get_screen_width()) if hasattr(self.pr, "get_screen_width") else self.width
        height = int(self.pr.get_screen_height()) if hasattr(self.pr, "get_screen_height") else self.height
        width = max(1, width)
        height = max(1, height)
        self._fullscreen = fullscreen
        if force or width != self.width or height != self.height:
            self.width = width
            self.height = height
        if not fullscreen and (force or width != self.windowed_width or height != self.windowed_height):
            self.windowed_width = width
            self.windowed_height = height

    def _toggle_fullscreen(self) -> None:
        self._set_fullscreen(not self._fullscreen)

    def _set_fullscreen(self, enabled: bool) -> None:
        if enabled == self._fullscreen:
            return
        if enabled:
            self._sync_window_state(force=True)
            self.windowed_width = self.width
            self.windowed_height = self.height
            if all(
                hasattr(self.pr, name)
                for name in ("get_current_monitor", "get_monitor_width", "get_monitor_height", "set_window_size")
            ):
                monitor = self.pr.get_current_monitor()
                self.pr.set_window_size(
                    int(self.pr.get_monitor_width(monitor)),
                    int(self.pr.get_monitor_height(monitor)),
                )
            self.pr.toggle_fullscreen()
        else:
            self.pr.toggle_fullscreen()
            if hasattr(self.pr, "set_window_size"):
                self.pr.set_window_size(int(self.windowed_width), int(self.windowed_height))
            if hasattr(self.pr, "restore_window"):
                self.pr.restore_window()
        self._fullscreen = enabled
        self._sync_window_state(force=True)

    def _layout(self, overlay_mode: str) -> tuple[UiRect, UiRect, float, float]:
        panel_width = self._panel_width_for_window(float(self.width), overlay_mode)
        self.side_panel = int(panel_width)
        world_rect = UiRect(0.0, 0.0, float(self.width - panel_width), float(self.height))
        panel_rect = UiRect(world_rect.right, 0.0, float(panel_width), float(self.height))
        return world_rect, panel_rect, self._panel_scale(panel_rect), self._world_scale(world_rect)

    def _panel_width_for_window(self, window_width: float, overlay_mode: str) -> float:
        base_ratio = float(np.clip(self.base_side_panel / max(self.base_width, 1), 0.26, 0.38))
        mode_factor = 1.0 if overlay_mode == "full" else 0.86
        min_width = 420.0 if overlay_mode == "full" else 360.0
        max_width = min(window_width * 0.48, window_width - self._MIN_WORLD_WIDTH)
        preferred = window_width * base_ratio * mode_factor
        return float(np.clip(preferred, min_width, max(min_width, max_width)))

    def _panel_scale(self, panel_rect: UiRect) -> float:
        width_scale = panel_rect.width / 680.0
        height_scale = panel_rect.height / 1080.0
        return float(np.clip(min(width_scale, height_scale), self._PANEL_SCALE_MIN, self._PANEL_SCALE_MAX))

    def _world_scale(self, world_rect: UiRect) -> float:
        width_scale = world_rect.width / 1600.0
        height_scale = world_rect.height / 960.0
        return float(np.clip(min(width_scale, height_scale), self._WORLD_SCALE_MIN, self._WORLD_SCALE_MAX))

    def _commands_for_button(self, action: str) -> list[SupervisorCommand]:
        if action == "toggle_fullscreen":
            self._toggle_fullscreen()
            return []
        if action == "speed_down":
            return [SupervisorCommand("speed_delta", -1.0)]
        if action == "speed_up":
            return [SupervisorCommand("speed_delta", 1.0)]
        if action == "map_down":
            return [SupervisorCommand("map_scale", -1.0)]
        if action == "map_up":
            return [SupervisorCommand("map_scale", 1.0)]
        return [SupervisorCommand(action)]

    def _handle_camera_input(self, frame: RenderFrame, world_rect: UiRect) -> None:
        dt = max(self.last_frame_ms, 1000.0 / max(1, int(self.config.render.fps))) / 1000.0
        visible_span = max(160.0, max(world_rect.width, world_rect.height) / max(self.zoom, 1e-6))
        pan_step = visible_span * 0.9 * dt

        horizontal = 0.0
        vertical = 0.0
        if self.pr.is_key_down(self.pr.KEY_LEFT) or self.pr.is_key_down(self.pr.KEY_A):
            horizontal -= 1.0
        if self.pr.is_key_down(self.pr.KEY_RIGHT) or self.pr.is_key_down(self.pr.KEY_D):
            horizontal += 1.0
        if self.pr.is_key_down(self.pr.KEY_UP) or self.pr.is_key_down(self.pr.KEY_W):
            vertical -= 1.0
        if self.pr.is_key_down(self.pr.KEY_DOWN) or self.pr.is_key_down(self.pr.KEY_S):
            vertical += 1.0

        moved = False
        if horizontal or vertical:
            self.camera_follow_enabled = False
            self.camera_pos[0] += float(horizontal * pan_step)
            self.camera_pos[1] += float(vertical * pan_step)
            moved = True

        mouse = self.pr.get_mouse_position()
        if self.pr.is_mouse_button_down(self.pr.MOUSE_BUTTON_MIDDLE) and world_rect.contains(float(mouse.x), float(mouse.y)):
            delta = self.pr.get_mouse_delta()
            self.camera_follow_enabled = False
            self.camera_pos[0] -= float(delta.x) / max(self.zoom, 1e-6)
            self.camera_pos[1] -= float(delta.y) / max(self.zoom, 1e-6)
            moved = True

        if self.pr.is_key_pressed(self.pr.KEY_ZERO):
            self.camera_follow_enabled = True
            moved = True

        if moved:
            self._clamp_camera(frame.world.map_size, world_rect)

    def _update_camera(self, frame: RenderFrame, world_rect: UiRect) -> None:
        focus_cells = [cell for cell in frame.world.cells if cell.agent_id == frame.world.focus_agent_id]
        if focus_cells:
            center = np.mean(np.array([cell.position for cell in focus_cells], dtype=np.float32), axis=0)
            mass = sum(cell.mass for cell in focus_cells)
        else:
            center = np.array([frame.world.map_size * 0.5, frame.world.map_size * 0.5], dtype=np.float32)
            mass = 25.0

        target_world_span = float(np.clip(180.0 + np.sqrt(max(mass, 1.0)) * 22.0, 180.0, frame.world.map_size))
        target_zoom = min(world_rect.width, world_rect.height) / max(target_world_span, 1.0)
        target_zoom = float(np.clip(target_zoom, 0.25, 4.0))

        if not self.camera_follow_enabled:
            self.zoom = (1.0 - float(np.clip(self.config.simulation.zoom_smoothness, 0.02, 1.0))) * self.zoom + (
                float(np.clip(self.config.simulation.zoom_smoothness, 0.02, 1.0)) * target_zoom
            )
            self._clamp_camera(frame.world.map_size, world_rect)
            return

        cam_alpha = float(np.clip(self.config.simulation.camera_smoothness, 0.02, 1.0))
        zoom_alpha = float(np.clip(self.config.simulation.zoom_smoothness, 0.02, 1.0))
        self.camera_pos = (1.0 - cam_alpha) * self.camera_pos + cam_alpha * center
        self.zoom = (1.0 - zoom_alpha) * self.zoom + zoom_alpha * target_zoom
        self._clamp_camera(frame.world.map_size, world_rect)

    def _clamp_camera(self, map_size: float, world_rect: UiRect) -> None:
        half_span_x = world_rect.width * 0.5 / max(self.zoom, 1e-6)
        half_span_y = world_rect.height * 0.5 / max(self.zoom, 1e-6)
        if half_span_x * 2.0 >= map_size:
            self.camera_pos[0] = map_size * 0.5
        else:
            self.camera_pos[0] = float(np.clip(self.camera_pos[0], half_span_x, map_size - half_span_x))
        if half_span_y * 2.0 >= map_size:
            self.camera_pos[1] = map_size * 0.5
        else:
            self.camera_pos[1] = float(np.clip(self.camera_pos[1], half_span_y, map_size - half_span_y))

    def _draw_world(self, frame: RenderFrame, world_rect: UiRect, scale: float) -> None:
        pr = self.pr
        pr.draw_rectangle_rec(self._rect(world_rect), self._color(236, 247, 250))
        if frame.show_grid:
            self._draw_grid(frame, world_rect)
        for pellet in frame.world.pellets:
            px, py = self._world_to_screen(pellet.position, world_rect)
            pr.draw_circle(int(px), int(py), 2.0, self._color(129, 219, 111))

        for cell in frame.world.cells:
            ix = cell.previous_position[0] + (cell.position[0] - cell.previous_position[0]) * frame.interpolation_alpha
            iy = cell.previous_position[1] + (cell.position[1] - cell.previous_position[1]) * frame.interpolation_alpha
            sx, sy = self._world_to_screen((ix, iy), world_rect)
            radius = max(3.0, cell.radius * self.zoom)
            color = self._agent_color(cell.agent_id)
            pr.draw_circle(int(sx), int(sy), radius, color)
            pr.draw_circle_lines(int(sx), int(sy), radius, self._color(255, 255, 255))
            if cell.is_focus:
                pr.draw_circle_lines(int(sx), int(sy), radius + 5.0, self._color(255, 255, 255, 120))

        chip_padding = 16.0 * scale
        chip_height = 42.0 * scale
        chip_width = 220.0 * scale
        chip = UiRect(
            world_rect.x + chip_padding,
            world_rect.bottom - (chip_padding + chip_height),
            chip_width,
            chip_height,
        )
        focus_name = frame.world.focus_agent_id or "none"
        camera_mode = "follow" if self.camera_follow_enabled else "free"
        self._draw_panel_box(chip, self._color(255, 255, 255, 220), self._color(203, 218, 232))
        self._draw_text(
            f"Camera: {camera_mode} | target {focus_name}",
            chip.x + 12.0 * scale,
            chip.y + 11.0 * scale,
            self._font(16, scale),
            self._color(38, 62, 86),
        )

    def _draw_grid(self, frame: RenderFrame, world_rect: UiRect) -> None:
        pr = self.pr
        spacing = float(max(10, self.config.render.grid_spacing))
        half_w = world_rect.width * 0.5 / max(self.zoom, 1e-6)
        half_h = world_rect.height * 0.5 / max(self.zoom, 1e-6)
        min_x = self.camera_pos[0] - half_w
        max_x = self.camera_pos[0] + half_w
        min_y = self.camera_pos[1] - half_h
        max_y = self.camera_pos[1] + half_h

        x = int(np.floor(min_x / spacing) * spacing)
        while x <= max_x:
            sx, _ = self._world_to_screen((float(x), 0.0), world_rect)
            pr.draw_line_v(
                self.pr.Vector2(sx, world_rect.y),
                self.pr.Vector2(sx, world_rect.bottom),
                self._color(225, 236, 241),
            )
            x += int(spacing)

        y = int(np.floor(min_y / spacing) * spacing)
        while y <= max_y:
            _, sy = self._world_to_screen((0.0, float(y)), world_rect)
            pr.draw_line_v(
                self.pr.Vector2(world_rect.x, sy),
                self.pr.Vector2(world_rect.right, sy),
                self._color(225, 236, 241),
            )
            y += int(spacing)

        tl = self._world_to_screen((0.0, 0.0), world_rect)
        br = self._world_to_screen((frame.world.map_size, frame.world.map_size), world_rect)
        left = min(tl[0], br[0])
        top = min(tl[1], br[1])
        width = max(1.0, abs(br[0] - tl[0]))
        height = max(1.0, abs(br[1] - tl[1]))
        pr.draw_rectangle_lines_ex(self._rect(UiRect(left, top, width, height)), 2.0, self._color(190, 206, 220))

    def _draw_side_panel(self, frame: RenderFrame, panel_rect: UiRect, scale: float) -> None:
        inset = 18.0 * scale
        section_gap = 12.0 * scale
        self._draw_panel_box(panel_rect, self._color(251, 252, 254), self._color(225, 232, 239))
        self._draw_text(
            frame.title,
            panel_rect.x + inset,
            panel_rect.y + 16.0 * scale,
            self._font(24, scale),
            self._color(31, 53, 78),
        )

        y = panel_rect.y + 62.0 * scale
        width = panel_rect.width - inset * 2.0
        y = self._draw_metric_grid(panel_rect.x + inset, y, width, frame.session_cards, cols=2, scale=scale)
        y += section_gap
        y = self._draw_metric_grid(panel_rect.x + inset, y, width, frame.training_cards, cols=2, scale=scale)
        y += section_gap
        y = self._draw_controls(panel_rect.x + inset, y, width, frame, scale)
        y += section_gap
        y = self._draw_agent_cards(panel_rect.x + inset, y, width, frame, scale)
        y += section_gap
        y = self._draw_charts(panel_rect.x + inset, y, width, panel_rect.bottom - y - inset, frame, scale)
        if frame.show_help:
            help_y = min(y + 8.0 * scale, panel_rect.bottom - 230.0 * scale)
            self._draw_help_overlay(panel_rect.x + inset, help_y, width, frame, scale)

    def _draw_status_banner(self, frame: RenderFrame, world_rect: UiRect, scale: float) -> None:
        status = frame.status
        accent = {
            "info": self._color(42, 88, 130),
            "warning": self._color(182, 120, 18),
            "error": self._color(182, 65, 54),
        }.get(status.level, self._color(42, 88, 130))
        inset = 16.0 * scale
        banner_height = 44.0 * scale
        banner = UiRect(
            world_rect.x + inset,
            world_rect.y + inset,
            min(520.0 * scale, world_rect.width - inset * 2.0),
            banner_height,
        )
        self._draw_panel_box(banner, self._color(255, 255, 255, 228), self._color(205, 219, 232))
        self.pr.draw_rectangle_rec(self._rect(UiRect(banner.x, banner.y, 5.0 * scale, banner.height)), accent)
        self._draw_text(
            status.message,
            banner.x + 16.0 * scale,
            banner.y + 13.0 * scale,
            self._font(18, scale),
            self._color(40, 60, 82),
        )

    def _draw_metric_grid(self, x: float, y: float, width: float, cards: tuple[Any, ...], cols: int, scale: float) -> float:
        gap = 10.0 * scale
        card_width = (width - gap * (cols - 1)) / max(1, cols)
        card_height = 64.0 * scale
        for idx, card in enumerate(cards):
            col = idx % cols
            row = idx // cols
            rect = UiRect(x + col * (card_width + gap), y + row * (card_height + gap), card_width, card_height)
            self._draw_panel_box(rect, self._color(255, 255, 255, 230), self._color(220, 230, 238))
            accent_height = max(4.0, 6.0 * scale)
            self.pr.draw_rectangle_rec(
                self._rect(UiRect(rect.x, rect.bottom - accent_height, rect.width, accent_height)),
                self._color(*card.accent),
            )
            self._draw_text(card.title, rect.x + 12.0 * scale, rect.y + 10.0 * scale, self._font(16, scale), self._color(85, 102, 122))
            self._draw_text(card.value, rect.x + 12.0 * scale, rect.y + 31.0 * scale, self._font(21, scale), self._color(28, 49, 73))
        rows = (len(cards) + cols - 1) // cols
        return y + rows * (card_height + gap)

    def _draw_controls(self, x: float, y: float, width: float, frame: RenderFrame, scale: float) -> float:
        cols = 3
        gap = 8.0 * scale
        button_width = (width - gap * (cols - 1)) / cols
        button_height = 40.0 * scale
        self._draw_text("Control Surface", x, y - 24.0 * scale, self._font(20, scale), self._color(35, 60, 85))
        for idx, button in enumerate(frame.controls):
            col = idx % cols
            row = idx // cols
            rect = UiRect(x + col * (button_width + gap), y + row * (button_height + gap), button_width, button_height)
            self._button_rects[button.action] = rect
            fill = self._color(*button.accent, 210 if button.active else 135)
            border = self._color(*button.accent)
            self._draw_panel_box(rect, fill, border)
            self._draw_text(button.label, rect.x + 12.0 * scale, rect.y + 11.0 * scale, self._font(15, scale), self._color(19, 35, 52))
        rows = (len(frame.controls) + cols - 1) // cols
        return y + rows * (button_height + gap)

    def _draw_agent_cards(self, x: float, y: float, width: float, frame: RenderFrame, scale: float) -> float:
        self._draw_text("Agent Observer Cards", x, y - 24.0 * scale, self._font(20, scale), self._color(35, 60, 85))
        card_height = 84.0 * scale
        gap = 9.0 * scale
        for idx, agent in enumerate(frame.agent_cards):
            rect = UiRect(x, y + idx * (card_height + gap), width, card_height)
            self._focus_rects[idx] = rect
            fill = self._color(255, 255, 255, 235)
            border = self._color(*agent.color)
            self._draw_panel_box(rect, fill, border, highlight=agent.focus)
            self.pr.draw_rectangle_rec(self._rect(UiRect(rect.x, rect.y, 6.0 * scale, rect.height)), self._color(*agent.color))
            self._draw_text(agent.display_name, rect.x + 15.0 * scale, rect.y + 11.0 * scale, self._font(19, scale), self._color(27, 48, 72))
            status = "alive" if agent.alive else "dead"
            self._draw_text(
                f"{status} | wins {agent.wins} | KOs {agent.eliminations}",
                rect.x + 15.0 * scale,
                rect.y + 39.0 * scale,
                self._font(16, scale),
                self._color(77, 99, 120),
            )
            self._draw_text(
                f"mass {agent.total_mass:.1f} | return {agent.episode_return:.2f}",
                rect.x + 15.0 * scale,
                rect.y + 61.0 * scale,
                self._font(16, scale),
                self._color(77, 99, 120),
            )
        return y + len(frame.agent_cards) * (card_height + gap)

    def _draw_charts(self, x: float, y: float, width: float, height: float, frame: RenderFrame, scale: float) -> float:
        self._draw_text("Live Telemetry", x, y - 24.0 * scale, self._font(20, scale), self._color(35, 60, 85))
        chart_height = 102.0 * scale
        gap = 10.0 * scale
        available = max(chart_height, height)
        rows = max(1, int(available // (chart_height + gap)))
        for idx, chart in enumerate(frame.charts[:rows]):
            rect = UiRect(x, y + idx * (chart_height + gap), width, chart_height)
            self._draw_chart(rect, chart, scale)
        return y + min(rows, len(frame.charts)) * (chart_height + gap)

    def _draw_chart(self, rect: UiRect, chart: ChartFrame, scale: float) -> None:
        self._draw_panel_box(rect, self._color(255, 255, 255, 230), self._color(220, 230, 238))
        self._draw_text(chart.title, rect.x + 12.0 * scale, rect.y + 11.0 * scale, self._font(18, scale), self._color(48, 68, 92))
        self._draw_text(
            chart.value_label,
            rect.right - 86.0 * scale,
            rect.y + 11.0 * scale,
            self._font(18, scale),
            self._color(48, 68, 92),
        )
        plot = UiRect(rect.x + 12.0 * scale, rect.y + 38.0 * scale, rect.width - 24.0 * scale, rect.height - 50.0 * scale)
        values = chart.values
        if len(values) < 2:
            self.pr.draw_rectangle_rec(self._rect(plot), self._color(247, 250, 252))
            return
        low = min(values)
        high = max(values)
        span = max(high - low, 1e-6)
        self.pr.draw_rectangle_rec(self._rect(plot), self._color(247, 250, 252))
        for idx in range(1, len(values)):
            x0 = plot.x + (idx - 1) * (plot.width / max(1, len(values) - 1))
            x1 = plot.x + idx * (plot.width / max(1, len(values) - 1))
            y0 = plot.bottom - ((values[idx - 1] - low) / span) * plot.height
            y1 = plot.bottom - ((values[idx] - low) / span) * plot.height
            self.pr.draw_line_v(self.pr.Vector2(x0, y0), self.pr.Vector2(x1, y1), self._color(*chart.accent))

    def _draw_help_overlay(self, x: float, y: float, width: float, frame: RenderFrame, scale: float) -> None:
        row_height = 18.0 * scale
        panel = UiRect(x, y, width, min(250.0 * scale, 26.0 * scale + len(frame.help_rows) * row_height))
        self._draw_panel_box(panel, self._color(20, 30, 42, 224), self._color(81, 125, 165))
        for idx, row in enumerate(frame.help_rows):
            color = self._color(255, 255, 255) if idx == 0 else self._color(207, 221, 235)
            self._draw_text(
                row,
                panel.x + 12.0 * scale,
                panel.y + 10.0 * scale + idx * row_height,
                self._font(15, scale),
                color,
            )

    def _world_to_screen(self, position: tuple[float, float], world_rect: UiRect) -> tuple[float, float]:
        sx = (position[0] - float(self.camera_pos[0])) * self.zoom + world_rect.width * 0.5 + world_rect.x
        sy = (position[1] - float(self.camera_pos[1])) * self.zoom + world_rect.height * 0.5 + world_rect.y
        return sx, sy

    def _draw_panel_box(self, rect: UiRect, fill: Any, border: Any, highlight: bool = False) -> None:
        self.pr.draw_rectangle_rounded(self._rect(rect), 0.18, 10, fill)
        self.pr.draw_rectangle_lines_ex(self._rect(rect), 1.5 if highlight else 1.0, border)

    def _font(self, base_size: int, scale: float) -> int:
        return max(12, int(round(base_size * scale)))

    def _draw_text(self, text: str, x: float, y: float, size: int, color: Any) -> None:
        self.pr.draw_text(text, int(x), int(y), int(size), color)

    def _rect(self, rect: UiRect) -> Any:
        return self.pr.Rectangle(rect.x, rect.y, rect.width, rect.height)

    def _color(self, r: int, g: int, b: int, a: int = 255) -> Any:
        return self.pr.Color(int(r), int(g), int(b), int(a))

    def _agent_color(self, agent_id: str) -> Any:
        digits = "".join(ch for ch in agent_id if ch.isdigit())
        index = int(digits or "0")
        colors = (
            (80, 252, 54),
            (36, 244, 255),
            (243, 31, 46),
            (255, 183, 77),
        )
        color = colors[index % len(colors)]
        return self._color(*color)
