"""FastAPI app for the browser game runtime."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from agario_rl.web.runtime import BrowserGameSession


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
        session = BrowserGameSession(
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
            except WebSocketDisconnect:
                disconnected.set()

        receiver = asyncio.create_task(receive_loop())
        try:
            while not disconnected.is_set():
                frame_started_at = asyncio.get_running_loop().time()
                await websocket.send_json(session.step())
                elapsed = asyncio.get_running_loop().time() - frame_started_at
                await asyncio.sleep(max(0.0, (1.0 / 30.0) - elapsed))
        except WebSocketDisconnect:
            disconnected.set()
        finally:
            receiver.cancel()
            session.close()

    return app
