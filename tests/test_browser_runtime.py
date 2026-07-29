from __future__ import annotations

from pathlib import Path

import numpy as np

from agario_rl.web.frames import build_browser_frame
from agario_rl.web.runtime import BrowserGameSession, BrowserInput
from agario_rl.web.server import create_app
from scripts import benchmark_perf, run_game, train


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_browser_input_normalizes_and_consumes_split() -> None:
    browser_input = BrowserInput(steer_x=4.0, steer_y=3.0, split=True)

    action = browser_input.as_action()

    np.testing.assert_allclose(action[:2], np.array([0.8, 0.6], dtype=np.float32), atol=1e-6)
    assert float(action[2]) == 1.0
    assert browser_input.split is False


def test_browser_play_mode_ejects_mass_in_steer_direction() -> None:
    session = BrowserGameSession(
        project_root=PROJECT_ROOT,
        mode="play",
        checkpoint_path="checkpoints/does-not-exist.pt",
        seed=19,
    )
    try:
        session.apply_client_message(
            {"type": "input", "steer": {"x": 1.0, "y": 0.0}, "eject": True}
        )
        session.step()
        ejected = session.env.world.ejected_masses
    finally:
        session.close()

    assert len(ejected) == 1
    assert ejected[0].velocity[0] > 0.0


def test_browser_input_rejects_invalid_steer_values() -> None:
    browser_input = BrowserInput(steer_x=0.25, steer_y=-0.25)

    browser_input.update_steer("nan", 1.0)
    browser_input.update_steer(2.0, -3.0)

    assert browser_input.steer_x == 1.0
    assert browser_input.steer_y == -1.0


def test_browser_session_frame_schema_exposes_training_state() -> None:
    session = BrowserGameSession(
        project_root=PROJECT_ROOT,
        mode="showcase",
        checkpoint_path="checkpoints/does-not-exist.pt",
        seed=23,
    )
    try:
        frame = session.step()
    finally:
        session.close()

    assert frame["type"] == "frame"
    assert frame["mode"] == "showcase"
    assert frame["mapSize"] >= 2000
    assert len(frame["agents"]) >= 6
    assert len(frame["pellets"]["x"]) > 0
    assert "leaderboard" in frame
    assert "policySource" in frame["training"]
    assert "humanReadiness" in frame["training"]
    assert {"splitSafety", "unsafeSplits", "usefulSplits", "finalMassLeader"} <= set(
        frame["training"]["humanReadiness"]
    )


def test_browser_play_mode_ingests_human_input() -> None:
    session = BrowserGameSession(
        project_root=PROJECT_ROOT,
        mode="play",
        checkpoint_path="checkpoints/does-not-exist.pt",
        seed=31,
    )
    try:
        session.apply_client_message({"type": "input", "steer": {"x": 1.0, "y": 0.0}, "split": True})
        frame = session.step()
    finally:
        session.close()

    assert frame["playerId"] == "agent_0"
    assert frame["agents"][0]["name"] == "You"
    assert frame["agents"][0]["split"]["attempts"] >= 0


def test_browser_frame_includes_split_safety_payload_from_infos() -> None:
    session = BrowserGameSession(
        project_root=PROJECT_ROOT,
        mode="showcase",
        checkpoint_path="checkpoints/does-not-exist.pt",
        seed=41,
    )
    try:
        agent_id = session.env.agent_ids[0]
        frame = build_browser_frame(
            world=session.env.world,
            infos={
                agent_id: {
                    "reward_breakdown": {"useful_split_bonus": 0.12, "unsafe_split_penalty": -0.35},
                    "split_attempts": 2,
                    "successful_splits": 1,
                    "unsafe_splits": 1,
                    "useful_splits": 1,
                }
            },
            mode="training-view",
            tick=1,
            player_id=None,
            policy_source="test",
            checkpoint="checkpoints/test.pt",
            metrics={"update": 7.0},
            fps=30.0,
        )
    finally:
        session.close()

    first_agent = frame["agents"][0]
    assert first_agent["rewardBreakdown"]["useful_split_bonus"] == 0.12
    assert first_agent["rewardBreakdown"]["unsafe_split_penalty"] == -0.35
    assert first_agent["split"]["attempts"] == 2
    assert frame["training"]["updateCount"] == 7


def test_fastapi_app_exposes_browser_routes() -> None:
    app = create_app()
    route_paths = {route.path for route in app.routes}

    assert "/api/health" in route_paths
    assert "/api/modes" in route_paths
    assert "/ws" in route_paths


def test_run_game_cli_defaults_to_browser_runtime(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["run_game.py"])

    args = run_game.parse_args()

    assert args.mode == "showcase"
    assert args.api_port == 8765
    assert args.web_port == 5173


def test_benchmark_cli_removes_render_mode(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["benchmark_perf.py", "--mode", "render"])

    try:
        benchmark_perf.parse_args()
    except SystemExit:
        pass
    else:
        raise AssertionError("render mode should not be a benchmark option")


def test_train_cli_accepts_human_ready_scenario_flags(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--updates",
            "1",
            "--scenario-preset",
            "agario_curriculum",
            "--continuing-respawn",
        ],
    )
    args = train.parse_args()
    assert args.scenario_preset == "agario_curriculum"
    assert args.continuing_respawn is True
