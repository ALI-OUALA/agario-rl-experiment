"""Lightweight performance benchmark for environment stepping."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agario_rl import load_config
from agario_rl.env.gym_env import AgarioMultiAgentEnv
from agario_rl.rl.ppo_shared import SharedPPOTrainer
from agario_rl.utils.device import synchronize_torch_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark env stepping and PPO training performance.")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--mode", type=str, choices=["step", "train"], default="step")
    parser.add_argument("--steps", type=int, default=400, help="Number of env steps for mode=step.")
    parser.add_argument("--updates", type=int, default=5, help="Number of PPO updates for mode=train.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda", "xpu"], default="auto")
    parser.add_argument("--inference-device", type=str, choices=["auto", "cpu", "cuda", "xpu"], default=None)
    return parser.parse_args()


def _idle_actions(cfg, agent_ids: list[str]) -> dict[str, np.ndarray]:
    if cfg.simulation.action_mode == "continuous":
        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    else:
        action = np.array([0.0, 0.0], dtype=np.float32)
    return {agent_id: action.copy() for agent_id in agent_ids}


def run_step_benchmark(args: argparse.Namespace) -> None:
    cfg = load_config(PROJECT_ROOT / args.config)
    env = AgarioMultiAgentEnv(cfg, enable_render=False)
    env.reset(seed=args.seed)
    actions = _idle_actions(cfg, env.agent_ids)

    start = time.perf_counter()
    for _ in range(args.steps):
        env.step(actions)
    elapsed = time.perf_counter() - start

    avg_ms = elapsed * 1000.0 / args.steps
    print(f"mode=step steps={args.steps} elapsed_s={elapsed:.4f} avg_step_ms={avg_ms:.4f}")
    env.close()


def run_train_benchmark(args: argparse.Namespace) -> None:
    cfg = load_config(PROJECT_ROOT / args.config)
    env = AgarioMultiAgentEnv(cfg, enable_render=False)
    trainer = SharedPPOTrainer(
        cfg,
        observation_dim=env.observation_space["shape"][0],
        device=args.device,
        inference_device=args.inference_device,
    )
    trainer.force_sync_with_env(env, seed=args.seed)

    rollout_times: list[float] = []
    update_times: list[float] = []
    for _ in range(args.updates):
        rollout_start = time.perf_counter()
        trainer.collect_rollout(env, target_transitions=cfg.rl.steps_per_update)
        rollout_times.append(time.perf_counter() - rollout_start)

        synchronize_torch_device(trainer.device)
        update_start = time.perf_counter()
        trainer.update()
        synchronize_torch_device(trainer.device)
        update_times.append(time.perf_counter() - update_start)

    print(
        "mode=train "
        f"device={trainer.device} inference_device={trainer.inference_device} updates={args.updates} "
        f"rollout_mean_s={float(np.mean(np.array(rollout_times, dtype=np.float32))):.4f} "
        f"update_mean_s={float(np.mean(np.array(update_times, dtype=np.float32))):.4f}"
    )
    env.close()


def main() -> None:
    args = parse_args()
    if args.mode == "step":
        run_step_benchmark(args)
        return
    run_train_benchmark(args)


if __name__ == "__main__":
    main()
