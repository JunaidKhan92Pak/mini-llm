"""A minimal single-device baseline trainer for JeePeeTee."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch

from mini_llm.config import RuntimeConfig, TrainingConfig
from mini_llm.model import JeePeeTee, causal_language_model_loss
from mini_llm.runtime import resolve_device, seed_everything
from mini_llm.training.checkpoint import load_checkpoint, save_checkpoint

BatchLoader = Iterable[tuple[torch.Tensor, torch.Tensor]]


@dataclass(slots=True)
class TrainingState:
    """Counters that continue across checkpoints."""

    global_step: int = 0
    tokens_processed: int = 0


@dataclass(frozen=True, slots=True)
class TrainingMetric:
    """One objective measurement at a known training step."""

    global_step: int
    tokens_processed: int
    train_loss: float | None
    validation_loss: float


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Final counters and ordered metrics from one trainer invocation."""

    state: TrainingState
    metrics: tuple[TrainingMetric, ...]


def evaluate_model(
    model: JeePeeTee,
    validation_loader: BatchLoader,
    *,
    device: torch.device,
) -> float:
    """Return token-weighted mean validation loss without changing caller mode."""

    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for inputs, targets in validation_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            loss = causal_language_model_loss(model(inputs), targets)
            token_count = targets.numel()
            total_loss += loss.item() * token_count
            total_tokens += token_count
    model.train(was_training)
    if total_tokens == 0:
        raise ValueError("validation loader produced no target tokens")
    return total_loss / total_tokens


def train_model(
    model: JeePeeTee,
    train_loader: BatchLoader,
    validation_loader: BatchLoader,
    training_config: TrainingConfig,
    runtime_config: RuntimeConfig | None = None,
    *,
    checkpoint_path: str | Path | None = None,
    resume_from: str | Path | None = None,
) -> TrainingResult:
    """Train to ``max_steps`` and optionally save or resume a versioned checkpoint."""

    runtime_config = runtime_config or RuntimeConfig()
    seed_everything(
        runtime_config.seed,
        deterministic_algorithms=runtime_config.deterministic_algorithms,
    )
    device = resolve_device(runtime_config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    state = TrainingState()
    if resume_from is not None:
        state.global_step, state.tokens_processed = load_checkpoint(
            resume_from,
            model=model,
            optimizer=optimizer,
            device=device,
        )
    if state.global_step > training_config.max_steps:
        raise ValueError(
            f"checkpoint step {state.global_step} exceeds max_steps {training_config.max_steps}"
        )

    metrics = [
        TrainingMetric(
            global_step=state.global_step,
            tokens_processed=state.tokens_processed,
            train_loss=None,
            validation_loss=evaluate_model(model, validation_loader, device=device),
        )
    ]
    train_iterator = iter(train_loader)

    while state.global_step < training_config.max_steps:
        try:
            inputs, targets = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            try:
                inputs, targets = next(train_iterator)
            except StopIteration as error:
                raise ValueError("training loader produced no batches") from error

        inputs = inputs.to(device)
        targets = targets.to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = causal_language_model_loss(model(inputs), targets)
        loss.backward()
        if training_config.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=training_config.gradient_clip_norm,
            )
        optimizer.step()

        state.global_step += 1
        state.tokens_processed += targets.numel()
        should_evaluate = (
            state.global_step % training_config.evaluation_interval == 0
            or state.global_step == training_config.max_steps
        )
        if should_evaluate:
            metrics.append(
                TrainingMetric(
                    global_step=state.global_step,
                    tokens_processed=state.tokens_processed,
                    train_loss=loss.item(),
                    validation_loss=evaluate_model(model, validation_loader, device=device),
                )
            )
            if checkpoint_path is not None:
                save_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    global_step=state.global_step,
                    tokens_processed=state.tokens_processed,
                    training_config=training_config,
                )

    return TrainingResult(state=state, metrics=tuple(metrics))
