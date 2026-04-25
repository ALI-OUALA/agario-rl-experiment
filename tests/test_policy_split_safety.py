"""Policy-side split gating tests."""

from __future__ import annotations

import torch

from agario_rl import AgarioConfig
from agario_rl.rl.ppo_shared import SharedPPOTrainer


def test_policy_biases_against_unready_split_action() -> None:
    config = AgarioConfig()
    config.observation_features.enabled = True
    config.observation_features.include_eject_state = True
    config.rl.split_logit_bias = -1.0
    config.rl.unready_split_logit_penalty = 5.0
    trainer = SharedPPOTrainer(config=config, observation_dim=12, device="cpu")
    outputs = {"ability_logits": torch.zeros((2, 2), dtype=torch.float32)}
    obs = torch.zeros((2, 12), dtype=torch.float32)
    obs[1, -2] = 1.0

    logits = trainer._ability_logits(outputs, obs)

    assert logits[0, 1].item() == -6.0
    assert logits[1, 1].item() == -1.0
    assert logits[0, 0].item() == 0.0
