"""Evaluate a checkpoint with human-readiness proxy metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agario_rl import load_config
from agario_rl.env.gym_env import AgarioMultiAgentEnv
from agario_rl.human_eval import HumanReadinessTracker
from agario_rl.opponents import build_default_opponent_pool
from agario_rl.rl.ppo_shared import SharedPPOTrainer
from agario_rl.utils.seeding import set_global_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint with human-readiness proxy metrics.")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--opponent-checkpoint", type=str, default="checkpoints/checkpoint_00500.pt")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--learner-agent", type=str, default="agent_0")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda", "xpu"], default="auto")
    parser.add_argument("--inference-device", type=str, choices=["auto", "cpu", "cuda", "xpu"], default=None)
    return parser.parse_args()


def _resolve_path(project_root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    return candidate if candidate.is_absolute() else (project_root / candidate)


def main() -> None:
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    config.simulation.action_mode = "continuous"
    if args.seed is not None:
        config.seed = int(args.seed)
    set_global_seeds(config.seed)

    env = AgarioMultiAgentEnv(config=config, enable_render=False)
    learner_agent_id = args.learner_agent
    trainer = SharedPPOTrainer(
        config=config,
        observation_dim=env.observation_space["shape"][0],
        device=args.device,
        inference_device=args.inference_device,
    )
    checkpoint = _resolve_path(PROJECT_ROOT, args.checkpoint)
    if not trainer.load(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    opponent_pool = build_default_opponent_pool(
        config=config,
        checkpoint_path=_resolve_path(PROJECT_ROOT, args.opponent_checkpoint),
    )
    rng = random.Random(config.seed)
    tracker = HumanReadinessTracker(learner_id=learner_agent_id)
    opponent_agent_ids = [agent_id for agent_id in env.agent_ids if agent_id != learner_agent_id]

    for episode_idx in range(args.episodes):
        observations = trainer.force_sync_with_env(env, seed=config.seed + episode_idx)
        active_policies = rng.sample(opponent_pool, k=len(opponent_agent_ids))
        active_opponents = {
            agent_id: policy
            for agent_id, policy in zip(opponent_agent_ids, active_policies, strict=True)
        }

        done = False
        while not done:
            actions = trainer.predict_actions(
                observations,
                deterministic=True,
                agent_ids=[learner_agent_id],
            )
            for agent_id, policy in active_opponents.items():
                actions[agent_id] = policy.action(
                    world=env.world,
                    observations=observations,
                    agent_id=agent_id,
                )
            observations, rewards, dones, infos = env.step(actions)
            tracker.observe(env.world, infos)
            done = bool(dones.get("__all__", False))

    summary = tracker.summary()
    payload = {
        "checkpoint": str(checkpoint),
        "device": str(trainer.device),
        "inference_device": str(trainer.inference_device),
        "episodes": summary.episodes,
        "wins": summary.wins,
        "win_rate": summary.win_rate,
        "mean_survival_steps": summary.mean_survival_steps,
        "mean_final_mass": summary.mean_final_mass,
        "corner_time_fraction": summary.corner_time_fraction,
        "threat_avoidance_rate": summary.threat_avoidance_rate,
        "small_target_pressure_rate": summary.small_target_pressure_rate,
    }
    print(json.dumps(payload, indent=2))

    if args.output:
        output_path = _resolve_path(PROJECT_ROOT, args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    env.close()


if __name__ == "__main__":
    main()
