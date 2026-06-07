"""Run the browser-based Agar.io RL game."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import webbrowser


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = PROJECT_ROOT / "web"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the browser game runtime.")
    parser.add_argument("--mode", choices=["showcase", "play", "training-view"], default="showcase")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--api-port", type=int, default=8765)
    parser.add_argument("--web-port", type=int, default=5173)
    parser.add_argument("--checkpoint", default="checkpoints/human_ready_v1/latest.pt")
    parser.add_argument("--no-open", action="store_true", help="Do not open the browser automatically.")
    parser.add_argument("--skip-npm-install", action="store_true", help="Do not auto-install web dependencies.")
    return parser.parse_args()


def _npm_command() -> str:
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    if npm is None:
        raise RuntimeError("npm is required for the browser frontend. Install Node.js and retry.")
    return npm


def _maybe_install_web_dependencies(args: argparse.Namespace) -> None:
    if args.skip_npm_install or (WEB_ROOT / "node_modules").exists():
        return
    subprocess.run([_npm_command(), "install"], cwd=WEB_ROOT, check=True)


def _spawn_api(args: argparse.Namespace) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "agario_rl.web.server:create_app",
            "--factory",
            "--host",
            args.host,
            "--port",
            str(args.api_port),
        ],
        cwd=PROJECT_ROOT,
    )


def _spawn_web(args: argparse.Namespace) -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    env["VITE_AGARIO_API_PORT"] = str(args.api_port)
    env["VITE_AGARIO_DEFAULT_MODE"] = args.mode
    env["VITE_AGARIO_DEFAULT_CHECKPOINT"] = args.checkpoint
    return subprocess.Popen(
        [
            _npm_command(),
            "run",
            "dev",
            "--",
            "--host",
            args.host,
            "--port",
            str(args.web_port),
        ],
        cwd=WEB_ROOT,
        env=env,
    )


def main() -> None:
    args = parse_args()
    _maybe_install_web_dependencies(args)
    api_process = _spawn_api(args)
    web_process = _spawn_web(args)
    url = f"http://{args.host}:{args.web_port}/?mode={args.mode}"
    print(f"Agario RL browser game: {url}")
    print(f"API WebSocket: ws://{args.host}:{args.api_port}/ws?mode={args.mode}")
    if not args.no_open:
        time.sleep(1.0)
        webbrowser.open(url)
    try:
        while True:
            api_code = api_process.poll()
            web_code = web_process.poll()
            if api_code is not None:
                raise SystemExit(f"API server exited with code {api_code}.")
            if web_code is not None:
                raise SystemExit(f"Web server exited with code {web_code}.")
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for process in (web_process, api_process):
            if process.poll() is None:
                process.terminate()
        for process in (web_process, api_process):
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    main()
