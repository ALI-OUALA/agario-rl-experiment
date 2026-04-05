"""Torch device selection helpers."""

from __future__ import annotations

from typing import Any

import torch


def resolve_torch_device(requested: str | None = None) -> str:
    """Resolve the requested torch device with Intel XPU support."""
    choice = (requested or "auto").strip().lower()
    if choice == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        return "cpu"
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return choice
    if choice == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("Intel XPU was requested but is not available.")
        return choice
    if choice == "cpu":
        return choice
    raise ValueError(f"Unsupported torch device request: {requested}")


def synchronize_torch_device(device: str | torch.device | None) -> None:
    """Synchronize the current accelerator before timing-sensitive reads."""
    if device is None:
        return
    device_type = torch.device(device).type
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device_type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.synchronize()


def device_summary() -> dict[str, Any]:
    """Return a small capability summary for logging and docs."""
    return {
        "cuda_available": bool(torch.cuda.is_available()),
        "xpu_available": bool(hasattr(torch, "xpu") and torch.xpu.is_available()),
        "torch_version": str(torch.__version__),
    }
