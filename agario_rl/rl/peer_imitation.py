"""Peer imitation replay buffer for cross-agent behavior cloning loss."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


class PeerImitationBuffer:
    """Stores transitions from the best-performing peer trajectory."""

    def __init__(self, capacity: int, seed: int = 0) -> None:
        self.capacity = capacity
        self.storage: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.storage)

    def add_episode(
        self,
        episode_data: dict[str, list[tuple[np.ndarray, np.ndarray]]],
        score_by_agent: dict[str, float],
    ) -> None:
        """Insert transitions from the highest-scoring agent in an episode."""
        if not episode_data or not score_by_agent:
            return
        best_agent = max(score_by_agent.keys(), key=lambda agent_id: score_by_agent[agent_id])
        for obs, action in episode_data.get(best_agent, []):
            self.storage.append((obs.astype(np.float32), action.astype(np.float32)))

    def sample(self, batch_size: int) -> dict[str, Any]:
        if len(self.storage) == 0:
            raise ValueError("Cannot sample from an empty imitation buffer.")
        indices = self.rng.integers(0, len(self.storage), size=batch_size)
        obs = np.stack([self.storage[i][0] for i in indices], axis=0).astype(np.float32)
        actions = np.stack([self.storage[i][1] for i in indices], axis=0).astype(np.float32)
        return {"obs": obs, "actions": actions}
