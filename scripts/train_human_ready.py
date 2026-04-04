"""Train a fresh learner against a stronger mixed opponent pool."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agario_rl import load_config
from agario_rl.env.gym_env import AgarioMultiAgentEnv
from agario_rl.opponents import OpponentPolicy, build_default_opponent_pool
from agario_rl.rl.ppo_shared import SharedPPOTrainer
from agario_rl.utils.logging import TrainingMetricsLogger, build_training_metrics_row
from agario_rl.utils.seeding import set_global_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train from scratch against scripted and frozen opponents.")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--opponent-checkpoint", type=str, default="checkpoints/checkpoint_00500.pt")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/human_ready_v1")
    parser.add_argument("--metrics-csv", type=str, default="logs/human_ready_v1_train_metrics.csv")
    parser.add_argument("--learner-agent", type=str, default="agent_0")
    return parser.parse_args()


def _resolve_path(project_root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    return candidate if candidate.is_absolute() else (project_root / candidate)


def _episode_done(infos: dict[str, dict[str, object]], max_steps: int) -> bool:
    global_info = infos.get("__global__", {})
    if not global_info:
        return False
    return int(global_info.get("step", 0)) >= max_steps or int(global_info.get("alive_count", 99)) <= 1


def _sample_opponents(
    pool: list[OpponentPolicy],
    opponent_agent_ids: list[str],
    rng: random.Random,
) -> dict[str, OpponentPolicy]:
    sampled = rng.sample(pool, k=len(opponent_agent_ids))
    return {
        agent_id: policy
        for agent_id, policy in zip(opponent_agent_ids, sampled, strict=True)
    }


def main() -> None:
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    config.simulation.action_mode = "continuous"
    if args.seed is not None:
        config.seed = int(args.seed)
    set_global_seeds(config.seed)

    env = AgarioMultiAgentEnv(config=config, enable_render=False)
    if args.learner_agent not in env.agent_ids:
        raise ValueError(f"Unknown learner agent id: {args.learner_agent}")
    learner_agent_id = args.learner_agent
    opponent_agent_ids = [agent_id for agent_id in env.agent_ids if agent_id != learner_agent_id]

    trainer = SharedPPOTrainer(config=config, observation_dim=env.observation_space["shape"][0])
    trainer.set_tracked_agent_ids([learner_agent_id])
    trainer.force_sync_with_env(env, seed=config.seed)

    opponent_checkpoint = _resolve_path(PROJECT_ROOT, args.opponent_checkpoint)
    opponent_pool = build_default_opponent_pool(config=config, checkpoint_path=opponent_checkpoint)
    rng = random.Random(config.seed)
    active_opponents = _sample_opponents(opponent_pool, opponent_agent_ids, rng)

    checkpoint_dir = _resolve_path(PROJECT_ROOT, args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_checkpoint = checkpoint_dir / "latest.pt"
    metrics_logger = TrainingMetricsLogger(_resolve_path(PROJECT_ROOT, args.metrics_csv))

    physics_dt = 1.0 / max(1, int(config.simulation.physics_hz))
    substeps = max(1, int(round(config.simulation.physics_hz / config.simulation.decision_hz)))

    for update_idx in range(1, args.updates + 1):
        while trainer.transitions_since_update < config.rl.steps_per_update:
            assert trainer.current_obs is not None
            action_overrides = {
                agent_id: policy.action(
                    world=env.world,
                    observations=trainer.current_obs,
                    agent_id=agent_id,
                )
                for agent_id, policy in active_opponents.items()
            }
            infos = trainer.step_decision(
                env=env,
                substeps=substeps,
                dt=physics_dt,
                track_experience=True,
                deterministic=False,
                action_overrides=action_overrides,
                policy_agent_ids=[learner_agent_id],
            )
            if _episode_done(infos, config.max_steps):
                active_opponents = _sample_opponents(opponent_pool, opponent_agent_ids, rng)

        metrics = trainer.update()
        metrics_logger.log(build_training_metrics_row(update=update_idx, metrics=metrics))
        print(
            f"[update {update_idx}] policy={metrics['policy_loss']:.4f} "
            f"value={metrics['value_loss']:.4f} entropy={metrics['entropy']:.4f} "
            f"imitation={metrics['imitation_loss']:.4f}"
        )
        if update_idx % config.logging.checkpoint_every_updates == 0:
            trainer.save(checkpoint_dir / f"checkpoint_{update_idx:05d}.pt")

    trainer.save(latest_checkpoint)
    env.close()
    print(f"Human-ready training complete. Latest checkpoint: {latest_checkpoint}")


if __name__ == "__main__":
    main()
