"""Tests for resume-aware training milestones."""

from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys

import torch
import yaml

from agario_rl import load_config
from agario_rl.env.gym_env import AgarioMultiAgentEnv
from agario_rl.rl.ppo_shared import SharedPPOTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_train_script_resumes_to_target_update_and_logs_global_counts(tmp_path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    latest_checkpoint = checkpoint_dir / "latest.pt"
    metrics_csv = tmp_path / "train_metrics.csv"
    config_path = tmp_path / "resume_config.yaml"

    raw_config = yaml.safe_load((PROJECT_ROOT / "config/default.yaml").read_text(encoding="utf-8"))
    raw_config["rl"]["steps_per_update"] = 6
    raw_config["rl"]["minibatch_size"] = 6
    raw_config["rl"]["ppo_epochs"] = 1
    raw_config["logging"]["train_metrics_csv"] = str(metrics_csv)
    raw_config["logging"]["checkpoint_every_updates"] = 10
    raw_config["logging"]["print_every_updates"] = 1000
    raw_config["supervisor"]["checkpoint_path"] = str(latest_checkpoint)
    raw_config["curriculum"]["enabled"] = False
    config_path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")

    config = load_config(config_path)
    env = AgarioMultiAgentEnv(config=config, enable_render=False)
    trainer = SharedPPOTrainer(config=config, observation_dim=env.observation_space["shape"][0], device="cpu")
    trainer.update_count = 366
    trainer.last_metrics["update_count"] = 366.0
    trainer.save(latest_checkpoint)
    env.close()

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/train.py"),
            "--config",
            str(config_path),
            "--updates",
            "370",
            "--resume",
            "--checkpoint",
            str(latest_checkpoint),
            "--checkpoint-dir",
            str(checkpoint_dir),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = torch.load(latest_checkpoint, map_location="cpu")
    assert int(payload["update_count"]) == 370
    assert (checkpoint_dir / "checkpoint_00370.pt").exists()

    rows = list(csv.DictReader(metrics_csv.open("r", newline="", encoding="utf-8")))
    assert [int(row["update"]) for row in rows] == [367, 368, 369, 370]
