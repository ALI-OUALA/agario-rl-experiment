"""Rollout data structures and GAE helper functions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(slots=True)
class Transition:
    obs: np.ndarray
    action: np.ndarray
    logprob: float
    value: float
    reward: float
    done: bool


@dataclass(slots=True)
class RolloutSample:
    obs: np.ndarray
    action: np.ndarray
    logprob: float
    value: float
    advantage: float
    ret: float


@dataclass(slots=True)
class RolloutBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    old_logprob: torch.Tensor
    old_value: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    @classmethod
    def from_samples(cls, samples: list[RolloutSample], device: torch.device) -> "RolloutBatch":
        obs = torch.tensor(np.stack([s.obs for s in samples], axis=0), dtype=torch.float32, device=device)
        actions = torch.tensor(np.stack([s.action for s in samples], axis=0), dtype=torch.float32, device=device)
        old_logprob = torch.tensor([s.logprob for s in samples], dtype=torch.float32, device=device)
        old_value = torch.tensor([s.value for s in samples], dtype=torch.float32, device=device)
        advantages = torch.tensor([s.advantage for s in samples], dtype=torch.float32, device=device)
        returns = torch.tensor([s.ret for s in samples], dtype=torch.float32, device=device)
        return cls(
            obs=obs,
            actions=actions,
            old_logprob=old_logprob,
            old_value=old_value,
            advantages=advantages,
            returns=returns,
        )


def compute_gae(
    transitions: list[Transition],
    bootstrap_value: float,
    gamma: float,
    gae_lambda: float,
) -> list[RolloutSample]:
    """Compute generalized advantage estimates for one trajectory."""
    if not transitions:
        return []

    samples: list[RolloutSample] = []
    gae = 0.0
    next_value = bootstrap_value
    for transition in reversed(transitions):
        non_terminal = 0.0 if transition.done else 1.0
        delta = transition.reward + gamma * next_value * non_terminal - transition.value
        gae = delta + gamma * gae_lambda * non_terminal * gae
        ret = gae + transition.value
        samples.append(
            RolloutSample(
                obs=transition.obs,
                action=transition.action,
                logprob=transition.logprob,
                value=transition.value,
                advantage=gae,
                ret=ret,
            )
        )
        next_value = transition.value
    samples.reverse()
    return samples
