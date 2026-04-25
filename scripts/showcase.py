"""Run a no-save showcase of the large Agar.io-style experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agario_rl import apply_scenario_preset, load_config
from agario_rl.env.gym_env import AgarioMultiAgentEnv
from agario_rl.opponents import assign_opponents, build_default_opponent_pool
from agario_rl.rendering.view_model import build_render_frame
from agario_rl.supervisor.controller import SupervisorController
from agario_rl.supervisor.runtime_stats import RuntimeSessionStats
from agario_rl.utils.seeding import set_global_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show the experiment as a polished, no-save Agar.io-style demo."
    )
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/human_ready_v1/latest.pt")
    parser.add_argument("--seconds", type=float, default=45.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--scenario-preset", choices=["full_arena", "agario_curriculum"], default="full_arena")
    return parser.parse_args()


def _resolve_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def run_showcase(args: argparse.Namespace) -> dict[str, float | int | str]:
    config = load_config(PROJECT_ROOT / args.config)
    apply_scenario_preset(config, args.scenario_preset)
    config.render.window_width = int(args.width)
    config.render.window_height = int(args.height)
    config.render.show_help_by_default = False
    if args.seed is not None:
        config.seed = int(args.seed)
    set_global_seeds(config.seed)

    env = AgarioMultiAgentEnv(config=config, enable_render=not args.headless)
    controller = SupervisorController(config=config)
    runtime_stats = RuntimeSessionStats.create(env.agent_ids)
    rng = random.Random(config.seed)
    opponent_pool = build_default_opponent_pool(config, _resolve_path(args.checkpoint))
    active_opponents = assign_opponents(opponent_pool, env.agent_ids, rng)

    physics_dt = 1.0 / max(1, int(config.simulation.physics_hz))
    decision_dt = 1.0 / max(1, int(config.simulation.decision_hz))
    substeps = max(1, int(round(config.simulation.physics_hz / config.simulation.decision_hz)))
    observations = env.reset(seed=config.seed)
    latest_infos = env.last_infos
    latest_actions = {agent_id: active_opponents[agent_id].action(world=env.world, observations=observations, agent_id=agent_id) for agent_id in env.agent_ids}
    frames = 0
    decisions = 0
    start = time.perf_counter()
    next_decision_at = start

    try:
        while time.perf_counter() - start < float(args.seconds):
            now = time.perf_counter()
            if now >= next_decision_at:
                latest_actions = {
                    agent_id: policy.action(world=env.world, observations=observations, agent_id=agent_id)
                    for agent_id, policy in active_opponents.items()
                }
                for _ in range(substeps):
                    observations, _rewards, dones, latest_infos = env.step(
                        latest_actions,
                        dt=physics_dt,
                        compute_observations=True,
                    )
                    if dones.get("__all__", False):
                        observations = env.reset(seed=config.seed + decisions + 1)
                        active_opponents = assign_opponents(opponent_pool, env.agent_ids, rng)
                        break
                decisions += 1
                next_decision_at = now + decision_dt

            frame_stats = {"render_fps": 0.0, "frame_ms": 0.0, "update_count": 0.0, "total_loss": 0.0}
            frame = build_render_frame(
                config=config,
                world=env.world,
                infos=latest_infos,
                metrics=frame_stats,
                controller=controller,
                runtime_stats=runtime_stats,
                interpolation_alpha=1.0,
                focus_agent_index=0,
                last_actions=latest_actions,
            )
            if not args.headless:
                frame_stats = env.render(frame=frame)
                runtime_stats.record_frame(frame_stats, latest_infos)
                if env.poll_commands():
                    break
            else:
                runtime_stats.record_frame(frame_stats, latest_infos)
            frames += 1
            if args.headless:
                time.sleep(0.001)
    finally:
        env.close()

    masses = {
        agent_id: float(sum(cell.mass for cell in env.world.agents[agent_id]))
        for agent_id in env.agent_ids
    }
    leader = max(masses, key=masses.get)
    return {
        "scenario": args.scenario_preset,
        "agents": len(env.agent_ids),
        "map_size": int(env.world.map_size),
        "frames": frames,
        "decisions": decisions,
        "leader": leader,
        "leader_mass": round(masses[leader], 2),
    }


def main() -> None:
    summary = run_showcase(parse_args())
    print(summary)


if __name__ == "__main__":
    main()
