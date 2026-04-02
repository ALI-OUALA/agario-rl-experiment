"""Headless training entrypoint for the minimal Agar.io RL lab."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agario_rl import load_config
from agario_rl.env.gym_env import AgarioMultiAgentEnv
from agario_rl.rl.ppo_shared import SharedPPOTrainer
from agario_rl.utils.logging import TrainingMetricsLogger, build_training_metrics_row
from agario_rl.utils.seeding import set_global_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train shared PPO agents in the Agar.io RL world.")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--updates", type=int, default=60)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--action-mode", type=str, choices=["continuous", "discrete_9way"], default=None)
    return parser.parse_args()


def _resolve_project_path(project_root: Path, raw_path: str | None, fallback: str) -> Path:
    candidate = Path(raw_path) if raw_path else Path(fallback)
    if candidate.is_absolute():
        return candidate
    return project_root / candidate


def main() -> None:
    args = parse_args()
    project_root = PROJECT_ROOT
    config_path = project_root / args.config
    config = load_config(config_path)
    if args.seed is not None:
        config.seed = int(args.seed)
    if args.action_mode is not None:
        config.simulation.action_mode = args.action_mode
    set_global_seeds(config.seed)

    env = AgarioMultiAgentEnv(config=config, enable_render=False)
    trainer = SharedPPOTrainer(config=config, observation_dim=env.observation_space["shape"][0])

    logger = TrainingMetricsLogger(project_root / config.logging.train_metrics_csv)
    checkpoint_dir = project_root / args.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_checkpoint_path = _resolve_project_path(
        project_root=project_root,
        raw_path=args.checkpoint,
        fallback=config.supervisor.checkpoint_path,
    )

    starting_update = 0
    if args.resume:
        loaded = trainer.load(latest_checkpoint_path)
        if not loaded:
            raise FileNotFoundError(
                f"Resume requested but checkpoint was not found at {latest_checkpoint_path}."
            )
        starting_update = trainer.update_count
        print(f"Resuming from update {starting_update} using {latest_checkpoint_path}.")

    if args.updates < starting_update:
        raise ValueError(
            f"Target updates ({args.updates}) must be >= current checkpoint update "
            f"count ({starting_update})."
        )

    for update_idx in range(starting_update + 1, args.updates + 1):
        trainer.collect_rollout(env, target_transitions=config.rl.steps_per_update)
        metrics = trainer.update()
        logger.log(build_training_metrics_row(update=update_idx, metrics=metrics))

        if update_idx % config.logging.print_every_updates == 0:
            print(
                f"[update {update_idx}] policy={metrics['policy_loss']:.4f} "
                f"value={metrics['value_loss']:.4f} entropy={metrics['entropy']:.4f} "
                f"imitation={metrics['imitation_loss']:.4f}"
            )

        if update_idx % config.logging.checkpoint_every_updates == 0:
            trainer.save(checkpoint_dir / f"checkpoint_{update_idx:05d}.pt")

    trainer.save(latest_checkpoint_path)
    default_latest_path = project_root / config.supervisor.checkpoint_path
    if latest_checkpoint_path != default_latest_path:
        trainer.save(default_latest_path)
    env.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
