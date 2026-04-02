"""Peer imitation buffer and imitation loss behavior tests."""

from __future__ import annotations

import numpy as np
import torch

from agario_rl import AgarioConfig
from agario_rl.rl.peer_imitation import PeerImitationBuffer
from agario_rl.rl.ppo_shared import SharedPPOTrainer


def test_best_agent_trajectory_is_inserted() -> None:
    buffer = PeerImitationBuffer(capacity=50, seed=1)
    episode_data = {
        "agent_0": [(np.zeros(5, dtype=np.float32), np.array([0.0, 0.0, 0.0], dtype=np.float32)) for _ in range(2)],
        "agent_1": [(np.ones(5, dtype=np.float32), np.array([0.5, -0.2, 1.0], dtype=np.float32)) for _ in range(4)],
    }
    scores = {"agent_0": 1.0, "agent_1": 2.0}
    buffer.add_episode(episode_data, scores)
    assert len(buffer) == 4

    sample = buffer.sample(batch_size=3)
    assert sample["obs"].shape == (3, 5)
    assert sample["actions"].shape == (3, 3)


def test_imitation_loss_decreases_on_fixed_demo_batch() -> None:
    config = AgarioConfig()
    config.simulation.action_mode = "continuous"
    trainer = SharedPPOTrainer(config=config, observation_dim=6, device="cpu")
    trainer.agent_ids = ["agent_0", "agent_1", "agent_2"]

    obs = torch.randn(256, 6)
    actions = torch.zeros(256, 3, dtype=torch.float32)
    actions[:, 0] = 0.4
    actions[:, 1] = -0.3
    actions[:, 2] = 1.0

    with torch.no_grad():
        initial_loss = float(trainer.compute_imitation_loss(obs, actions).item())

    optimizer = torch.optim.Adam(trainer.policy.parameters(), lr=2e-3)
    for _ in range(80):
        loss = trainer.compute_imitation_loss(obs, actions)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = float(trainer.compute_imitation_loss(obs, actions).item())

    assert final_loss < initial_loss
