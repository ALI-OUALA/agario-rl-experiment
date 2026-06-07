"""Authoritative Python runtime used by the browser game."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import random
import time
from typing import Any

import numpy as np

from agario_rl import AgarioConfig, apply_scenario_preset, load_config
from agario_rl.env.gym_env import AgarioMultiAgentEnv
from agario_rl.opponents import OpponentPolicy, assign_opponents, build_default_opponent_pool
from agario_rl.utils.seeding import set_global_seeds
from agario_rl.web.frames import build_browser_frame, latest_training_metrics


@dataclass(slots=True)
class BrowserInput:
    """Latest human input received from the browser."""

    steer_x: float = 0.0
    steer_y: float = 0.0
    split: bool = False

    def as_action(self) -> np.ndarray:
        steer = np.array([self.steer_x, self.steer_y], dtype=np.float32)
        norm = float(np.linalg.norm(steer))
        if norm > 1.0:
            steer = steer / norm
        steer = np.clip(steer, -1.0, 1.0)
        action = np.array(
            [
                float(steer[0]),
                float(steer[1]),
                1.0 if self.split else 0.0,
            ],
            dtype=np.float32,
        )
        self.split = False
        return action

    def update_steer(self, steer_x: object, steer_y: object) -> None:
        """Store finite, clamped browser steer values."""
        try:
            x = float(steer_x)
            y = float(steer_y)
        except (TypeError, ValueError):
            return
        if not math.isfinite(x) or not math.isfinite(y):
            return
        self.steer_x = max(-1.0, min(1.0, x))
        self.steer_y = max(-1.0, min(1.0, y))


class BrowserGameSession:
    """One live browser game session backed by AgarioWorld."""

    def __init__(
        self,
        *,
        project_root: Path,
        mode: str,
        config_path: str = "config/default.yaml",
        checkpoint_path: str = "checkpoints/human_ready_v1/latest.pt",
        seed: int | None = None,
    ) -> None:
        if mode not in {"showcase", "play", "training-view"}:
            raise ValueError(f"Unsupported browser game mode: {mode}")
        self.project_root = Path(project_root)
        self.mode = mode
        self.checkpoint_path = checkpoint_path
        self.config: AgarioConfig = load_config(self.project_root / config_path)
        apply_scenario_preset(self.config, "full_arena")
        self.config.map.pellets_per_10k_area = 1
        self.config.map.pellet_respawn_per_step = 3
        self.config.simulation.physics_hz = 30
        self.config.simulation.decision_hz = 30
        self.config.simulation.max_substeps_per_frame = 1
        if seed is not None:
            self.config.seed = int(seed)
        set_global_seeds(self.config.seed)

        self.env = AgarioMultiAgentEnv(config=self.config, enable_render=False)
        self.observations = self.env.reset(seed=self.config.seed)
        self.rng = random.Random(self.config.seed)
        self.player_id = self.env.agent_ids[0] if mode == "play" else None
        self.input = BrowserInput()
        self.tick = 0
        self.started_at = time.perf_counter()
        self.last_frame_at = self.started_at
        self.physics_dt = 1.0 / max(1, int(self.config.simulation.physics_hz))
        self.substeps = max(1, int(round(self.config.simulation.physics_hz / self.config.simulation.decision_hz)))
        self.metrics_csv = self.project_root / self.config.logging.train_metrics_csv
        self._metrics_cache: dict[str, float] = {}
        self._metrics_cache_at = 0.0
        self.opponent_pool = build_default_opponent_pool(
            self.config,
            self._resolve_path(checkpoint_path),
        )
        self.requires_policy_observations = any(
            policy.name.startswith("checkpoint") for policy in self.opponent_pool
        )
        self.opponent_action_interval = 1 if self.requires_policy_observations else 3
        self._opponent_actions: dict[str, np.ndarray] = {}
        self.opponent_agent_ids = [
            agent_id for agent_id in self.env.agent_ids if agent_id != self.player_id
        ]
        self.active_opponents = self._assign_opponents()

    def _resolve_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        return path if path.is_absolute() else self.project_root / path

    def _assign_opponents(self) -> dict[str, OpponentPolicy]:
        return assign_opponents(self.opponent_pool, self.opponent_agent_ids, self.rng)

    @property
    def policy_source(self) -> str:
        names = [policy.name for policy in self.opponent_pool]
        if any(name.startswith("checkpoint") for name in names):
            return "checkpoint + scripted safety opponents"
        return "scripted safety opponents"

    def apply_client_message(self, message: dict[str, Any]) -> None:
        """Apply a WebSocket message from the browser."""
        kind = str(message.get("type", "input"))
        if kind == "reset":
            self.reset()
            return
        if kind != "input":
            return
        steer = message.get("steer", {})
        if isinstance(steer, dict):
            self.input.update_steer(steer.get("x", 0.0), steer.get("y", 0.0))
        self.input.split = self.input.split or bool(message.get("split", False))

    def reset(self) -> None:
        self.observations = self.env.reset(seed=self.config.seed + self.tick + 1)
        self.active_opponents = self._assign_opponents()
        self._opponent_actions = {}
        self.input = BrowserInput()

    def _opponent_step_actions(self) -> dict[str, np.ndarray]:
        should_refresh = (
            not self._opponent_actions
            or self.tick % self.opponent_action_interval == 0
        )
        if should_refresh:
            self._opponent_actions = {
                agent_id: policy.action(
                    world=self.env.world,
                    observations=self.observations,
                    agent_id=agent_id,
                )
                for agent_id, policy in self.active_opponents.items()
            }
        return {
            agent_id: action.copy()
            for agent_id, action in self._opponent_actions.items()
        }

    def _latest_metrics(self) -> dict[str, float]:
        now = time.perf_counter()
        if now - self._metrics_cache_at >= 1.0:
            self._metrics_cache = latest_training_metrics(self.metrics_csv)
            self._metrics_cache_at = now
        return self._metrics_cache

    def step(self) -> dict[str, Any]:
        """Advance one browser frame and return a serialized frame."""
        actions = self._opponent_step_actions()
        if self.player_id is not None:
            actions[self.player_id] = self.input.as_action()

        latest_infos = self.env.last_infos
        for _ in range(self.substeps):
            observations, _rewards, dones, latest_infos = self.env.step(
                actions,
                dt=self.physics_dt,
                compute_observations=self.requires_policy_observations,
            )
            if observations is not None:
                self.observations = observations
            if dones.get("__all__", False):
                self.reset()
                latest_infos = self.env.last_infos
                break

        now = time.perf_counter()
        fps = 1.0 / max(now - self.last_frame_at, 1e-6)
        self.last_frame_at = now
        self.tick += 1
        return build_browser_frame(
            world=self.env.world,
            infos=latest_infos,
            mode=self.mode,
            tick=self.tick,
            player_id=self.player_id,
            policy_source=self.policy_source,
            checkpoint=self.checkpoint_path,
            metrics=self._latest_metrics(),
            fps=fps,
        )

    def close(self) -> None:
        self.env.close()
