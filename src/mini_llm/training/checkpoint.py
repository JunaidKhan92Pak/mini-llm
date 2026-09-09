"""Versioned checkpoint persistence for model, optimizer, and training state."""

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from mini_llm.config import TrainingConfig
from mini_llm.model import JeePeeTee

CHECKPOINT_VERSION = 1


def save_checkpoint(
    path: str | Path,
    *,
    model: JeePeeTee,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    tokens_processed: int,
    training_config: TrainingConfig,
) -> None:
    """Atomically save the state required to continue a baseline run."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_config": asdict(model.config),
        "training_config": asdict(training_config),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "global_step": global_step,
        "tokens_processed": tokens_processed,
        "torch_rng_state": torch.get_rng_state(),
    }
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_checkpoint(
    path: str | Path,
    *,
    model: JeePeeTee,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[int, int]:
    """Restore a compatible checkpoint and return step/token counters."""

    payload: Any = torch.load(path, map_location=device, weights_only=True)
    required_keys = {
        "checkpoint_version",
        "model_config",
        "training_config",
        "model_state",
        "optimizer_state",
        "global_step",
        "tokens_processed",
        "torch_rng_state",
    }
    if not isinstance(payload, dict) or not required_keys.issubset(payload):
        raise ValueError("checkpoint is missing required fields")
    if payload["checkpoint_version"] != CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported checkpoint version: {payload['checkpoint_version']!r}"
        )
    if payload["model_config"] != asdict(model.config):
        raise ValueError("checkpoint model configuration does not match the current model")
    if not isinstance(payload["global_step"], int) or payload["global_step"] < 0:
        raise ValueError("checkpoint global_step must be a non-negative integer")
    if not isinstance(payload["tokens_processed"], int) or payload["tokens_processed"] < 0:
        raise ValueError("checkpoint tokens_processed must be a non-negative integer")

    model.load_state_dict(payload["model_state"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    return payload["global_step"], payload["tokens_processed"]
