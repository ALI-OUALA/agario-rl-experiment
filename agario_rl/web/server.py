"""FastAPI app for the browser game runtime."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from agario_rl.web.runtime import BrowserGameSession


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if sys.platform == "win32":
    # Windows' default scheduler timer quantum (~15.6ms) means asyncio.sleep()
    # calls in our ~33ms (30Hz) frame pacing loop frequently miss their tick
    # and wait for the next one, capping real throughput around 20fps
    # regardless of how fast the simulation itself runs. Raising the system
    # timer resolution to 1ms for this process's lifetime fixes that; this is
    # the standard technique used by latency-sensitive Windows apps (games,
    # audio) and has no effect on non-Windows platforms.
    import ctypes

    ctypes.windll.winmm.timeBeginPeriod(1)


def create_app() -> FastAPI:
    """Create the FastAPI app used by `scripts/run_game.py`."""
    app = FastAPI(title="Agario RL Browser Runtime")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/modes")
    async def modes() -> dict[str, list[str]]:
        return {"modes": ["showcase", "play", "training-view"]}

    @app.websocket("/ws")
    async def websocket_game(
        websocket: WebSocket,
        mode: str = Query(default="showcase"),
        checkpoint: str = Query(default="checkpoints/human_ready_v1/latest.pt"),
    ) -> None:
        await websocket.accept()
        session = await asyncio.to_thread(
            BrowserGameSession,
            project_root=PROJECT_ROOT,
            mode=mode,
            checkpoint_path=checkpoint,
        )
        disconnected = asyncio.Event()

        async def receive_loop() -> None:
            try:
                while True:
                    message: dict[str, Any] = await websocket.receive_json()
                    session.apply_client_message(message)
            except (WebSocketDisconnect, RuntimeError):
                disconnected.set()

        receiver = asyncio.create_task(receive_loop())
        loop = asyncio.get_running_loop()
        frame_interval = 1.0 / 30.0
        next_frame_at = loop.time()
        try:
            while not disconnected.is_set():
                await websocket.send_json(session.step())
                next_frame_at += frame_interval
                delay = next_frame_at - loop.time()
                if delay < -frame_interval:
                    next_frame_at = loop.time()
                    delay = 0.0
                if delay > 0.0:
                    await asyncio.sleep(delay)
        except (WebSocketDisconnect, RuntimeError):
            disconnected.set()
        finally:
            receiver.cancel()
            with suppress(asyncio.CancelledError, WebSocketDisconnect, RuntimeError):
                await receiver
            session.close()

    return app
