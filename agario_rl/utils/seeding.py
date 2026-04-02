"""Random seed helpers for reproducible experiments."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_global_seeds(seed: int) -> None:
    """Set seeds for Python, NumPy, and Torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_rng(seed: int | None) -> np.random.Generator:
    """Create a NumPy generator for deterministic local randomness."""
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(int(seed))
