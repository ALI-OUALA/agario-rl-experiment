"""Evaluation script for saved shared-policy checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agario_rl import load_config
from agario_rl.env.gym_env import AgarioMultiAgentEnv
from agario_rl.rl.ppo_shared import SharedPPOTrainer
from agario_rl.utils.seeding import set_global_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--action-mode", type=str, choices=["continuous", "discrete_9way"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = PROJECT_ROOT
    config = load_config(project_root / args.config)
    if args.seed is not None:
        config.seed = int(args.seed)
    if args.action_mode is not None:
        config.simulation.action_mode = args.action_mode
    set_global_seeds(config.seed)

    env = AgarioMultiAgentEnv(config=config, enable_render=args.render)
    trainer = SharedPPOTrainer(config=config, observation_dim=env.observation_space["shape"][0])

    checkpoint = Path(args.checkpoint) if args.checkpoint else (project_root / config.supervisor.checkpoint_path)
    loaded = trainer.load(checkpoint)
    print(f"Checkpoint loaded: {loaded} ({checkpoint})")

    episode_scores = []
    for episode_idx in range(args.episodes):
        obs = trainer.force_sync_with_env(env, seed=config.seed + episode_idx)
        done = False
        score = {agent_id: 0.0 for agent_id in env.agent_ids}
        while not done:
            actions = trainer.predict_actions(obs, deterministic=args.deterministic)
            obs, rewards, dones, infos = env.step(actions)
            for agent_id, reward in rewards.items():
                score[agent_id] += reward
            done = bool(dones.get("__all__", False))
            if args.render:
                env.render(extra_stats={"episode": episode_idx + 1}, show_help=False)
        meanscore = float(np.mean(list(score.values())))
        episode_scores.append(meanscore)
        winner = infos.get("__global__", {}).get("winner")
        print(f"[episode {episode_idx + 1}] winner={winner} mean_return={meanscore:.3f}")

    print(f"Average return over {args.episodes} episodes: {float(np.mean(episode_scores)):.3f}")
    env.close()


if __name__ == "__main__":
    main()
