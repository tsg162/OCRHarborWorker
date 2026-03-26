from __future__ import annotations

import torch


def detect_device(requested: str | None = None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_dtype(device: torch.device, requested: torch.dtype | None = None) -> torch.dtype:
    if requested is not None:
        return requested
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32
