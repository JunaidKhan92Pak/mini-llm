"""Small runtime helpers for device selection and reproducible experiments."""

import random

import torch

from mini_llm.config import DevicePreference


def resolve_device(preference: DevicePreference = "auto") -> torch.device:
    """Resolve a configured preference without assuming that CUDA is available."""

    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device")
    if preference not in ("cpu", "cuda"):
        raise ValueError(f"unsupported device preference: {preference!r}")
    return torch.device(preference)


def seed_everything(seed: int, *, deterministic_algorithms: bool = False) -> None:
    """Seed Python and PyTorch RNGs for repeatable experiments."""

    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms)

