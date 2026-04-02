"""Policy/value networks for shared-parameter multi-agent PPO."""

from __future__ import annotations

import torch
from torch import nn


class ActorCriticNetwork(nn.Module):
    """Shared actor-critic with discrete and continuous action heads."""

    def __init__(self, observation_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.direction_head = nn.Linear(hidden_dim, 9)
        self.ability_head = nn.Linear(hidden_dim, 2)
        self.steer_mean_head = nn.Linear(hidden_dim, 2)
        self.steer_log_std = nn.Parameter(torch.full((2,), -0.5))
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(obs)
        value = self.value_head(features).squeeze(-1)
        return {
            "direction_logits": self.direction_head(features),
            "ability_logits": self.ability_head(features),
            "steer_mean": self.steer_mean_head(features),
            "steer_log_std": self.steer_log_std.expand(features.shape[0], 2),
            "value": value,
        }
