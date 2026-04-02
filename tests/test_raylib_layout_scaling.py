"""Raylib layout, camera, and fullscreen behavior tests."""

from __future__ import annotations

from agario_rl import AgarioConfig
from agario_rl.rendering.models import RenderFrame, StatusFrame, UiRect, WorldFrame
from agario_rl.rendering.raylib_backend import RaylibRenderer
from agario_rl.supervisor.actions import SupervisorCommand


def _renderer_stub() -> RaylibRenderer:
    renderer = RaylibRenderer.__new__(RaylibRenderer)
    renderer.config = AgarioConfig()
    renderer.base_width = 1920
    renderer.base_height = 1080
    renderer.base_side_panel = 620
    renderer.width = 1920
    renderer.height = 1080
    renderer.side_panel = 620
    renderer.windowed_width = 1600
    renderer.windowed_height = 900
    renderer._fullscreen = False
    renderer._closed = False
    renderer.last_frame_ms = 16.0
    renderer.zoom = 1.0
    renderer.camera_pos = [500.0, 500.0]
    renderer.camera_follow_enabled = True
    renderer._button_rects = {}
    renderer._focus_rects = {}
    return renderer


class _StubPyray:
    def __init__(self) -> None:
        self.fullscreen = False
        self.width = 1600
        self.height = 900
        self.size_calls: list[tuple[int, int]] = []
        self.restore_calls = 0
        self._pressed: set[int] = set()
        self._down: set[int] = set()
        self._mouse_down: set[int] = set()
        self.KEY_LEFT = 1
        self.KEY_RIGHT = 2
        self.KEY_UP = 3
        self.KEY_DOWN = 4
        self.KEY_A = 5
        self.KEY_D = 6
        self.KEY_W = 7
        self.KEY_S = 8
        self.KEY_ZERO = 9
        self.KEY_F11 = 10
        self.KEY_LEFT_SHIFT = 11
        self.KEY_RIGHT_SHIFT = 12
        self.KEY_SPACE = 13
        self.KEY_N = 14
        self.KEY_MINUS = 15
        self.KEY_EQUAL = 16
        self.KEY_T = 17
        self.KEY_C = 18
        self.KEY_R = 19
        self.KEY_M = 20
        self.KEY_P = 21
        self.KEY_L = 22
        self.KEY_TAB = 23
        self.KEY_G = 24
        self.KEY_F1 = 25
        self.KEY_ONE = 26
        self.KEY_TWO = 27
        self.KEY_THREE = 28
        self.MOUSE_BUTTON_LEFT = 29
        self.MOUSE_BUTTON_MIDDLE = 30

    def is_window_fullscreen(self) -> bool:
        return self.fullscreen

    def get_screen_width(self) -> int:
        return self.width

    def get_screen_height(self) -> int:
        return self.height

    def toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen

    def set_window_size(self, width: int, height: int) -> None:
        self.width = int(width)
        self.height = int(height)
        self.size_calls.append((self.width, self.height))

    def restore_window(self) -> None:
        self.restore_calls += 1

    def get_current_monitor(self) -> int:
        return 0

    def get_monitor_width(self, _monitor: int) -> int:
        return 2560

    def get_monitor_height(self, _monitor: int) -> int:
        return 1440

    def is_key_pressed(self, key: int) -> bool:
        return key in self._pressed

    def is_key_down(self, key: int) -> bool:
        return key in self._down

    def is_mouse_button_down(self, button: int) -> bool:
        return button in self._mouse_down

    def get_mouse_position(self):  # noqa: ANN202
        class _Mouse:
            x = 200.0
            y = 200.0

        return _Mouse()

    def get_mouse_delta(self):  # noqa: ANN202
        class _MouseDelta:
            x = 0.0
            y = 0.0

        return _MouseDelta()

    def is_mouse_button_pressed(self, _button: int) -> bool:
        return False

    def window_should_close(self) -> bool:
        return False


def _frame_stub() -> RenderFrame:
    return RenderFrame(
        title="test",
        world=WorldFrame(
            map_size=1000.0,
            stage=0,
            step=0,
            alive_count=1,
            winner=None,
            focus_agent_id="agent_0",
            cells=(),
            pellets=(),
        ),
        session_cards=(),
        training_cards=(),
        agent_cards=(),
        charts=(),
        controls=(),
        status=StatusFrame("ready", "info"),
        overlay_mode="full",
        show_grid=False,
        show_help=False,
        help_rows=(),
        interpolation_alpha=1.0,
    )


def test_panel_width_grows_with_window_size() -> None:
    renderer = _renderer_stub()

    compact_width = renderer._panel_width_for_window(1600.0, "full")
    expanded_width = renderer._panel_width_for_window(3200.0, "full")
    minimal_width = renderer._panel_width_for_window(3200.0, "minimal")

    assert expanded_width > compact_width
    assert minimal_width < expanded_width


def test_panel_scale_grows_with_available_panel_size() -> None:
    renderer = _renderer_stub()

    small_panel = UiRect(0.0, 0.0, 620.0, 900.0)
    large_panel = UiRect(0.0, 0.0, 1240.0, 1440.0)

    assert renderer._panel_scale(large_panel) > renderer._panel_scale(small_panel)


def test_fullscreen_toggle_preserves_windowed_size() -> None:
    renderer = _renderer_stub()
    renderer.pr = _StubPyray()

    renderer._set_fullscreen(True)
    assert renderer._fullscreen is True
    assert renderer.pr.size_calls[0] == (2560, 1440)

    renderer._set_fullscreen(False)
    assert renderer._fullscreen is False
    assert renderer.pr.size_calls[-1] == (1600, 900)
    assert renderer.pr.restore_calls == 1


def test_fullscreen_button_action_toggles_without_supervisor_command() -> None:
    renderer = _renderer_stub()
    renderer.pr = _StubPyray()

    commands = renderer._commands_for_button("toggle_fullscreen")

    assert commands == []
    assert renderer._fullscreen is True


def test_camera_pan_switches_to_free_camera() -> None:
    renderer = _renderer_stub()
    renderer.pr = _StubPyray()
    renderer.pr._down.add(renderer.pr.KEY_D)
    frame = _frame_stub()
    world_rect = UiRect(0.0, 0.0, 700.0, 500.0)

    renderer._handle_camera_input(frame, world_rect)

    assert renderer.camera_follow_enabled is False
    assert renderer.camera_pos[0] > 500.0


def test_camera_reset_shortcut_returns_to_follow_mode() -> None:
    renderer = _renderer_stub()
    renderer.pr = _StubPyray()
    renderer.camera_follow_enabled = False
    renderer.pr._pressed.add(renderer.pr.KEY_ZERO)
    frame = _frame_stub()
    world_rect = UiRect(0.0, 0.0, 700.0, 500.0)

    renderer._handle_camera_input(frame, world_rect)

    assert renderer.camera_follow_enabled is True


def test_focus_command_reenables_follow_camera() -> None:
    renderer = _renderer_stub()
    renderer.pr = _StubPyray()
    renderer.camera_follow_enabled = False
    renderer.pr._pressed.add(renderer.pr.KEY_ONE)

    commands = renderer.poll_commands()

    assert renderer.camera_follow_enabled is True
    assert commands == [SupervisorCommand("focus_agent", 0)]
