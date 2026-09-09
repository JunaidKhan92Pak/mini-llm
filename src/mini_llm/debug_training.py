"""A deterministic one-batch overfit check for the first JeePeeTee decoder."""

from dataclasses import dataclass

import torch

from mini_llm.config import DebugOverfitConfig, ModelConfig, RuntimeConfig
from mini_llm.model.language_model import (
    JeePeeTee,
    causal_language_model_loss,
    prepare_next_token_batch,
)
from mini_llm.runtime import resolve_device, seed_everything


@dataclass(frozen=True, slots=True)
class DebugOverfitResult:
    """Metrics proving whether one tiny token batch was learned."""

    initial_loss: float
    final_loss: float
    steps: int
    parameter_count: int
    device: str


def run_tiny_overfit(
    model_config: ModelConfig,
    token_batch: torch.Tensor,
    training_config: DebugOverfitConfig | None = None,
    runtime_config: RuntimeConfig | None = None,
) -> DebugOverfitResult:
    """Train JeePeeTee repeatedly on one batch as a correctness gate."""

    training_config = training_config or DebugOverfitConfig()
    runtime_config = runtime_config or RuntimeConfig()
    seed_everything(
        runtime_config.seed,
        deterministic_algorithms=runtime_config.deterministic_algorithms,
    )
    device = resolve_device(runtime_config.device)
    token_batch = token_batch.to(device)
    inputs, targets = prepare_next_token_batch(token_batch)
    if inputs.shape[1] > model_config.context_length:
        raise ValueError(
            f"input length {inputs.shape[1]} exceeds context length {model_config.context_length}"
        )
    if token_batch.min() < 0 or token_batch.max() >= model_config.vocab_size:
        raise ValueError("token batch contains IDs outside the configured vocabulary")

    model = JeePeeTee(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    model.eval()
    with torch.no_grad():
        initial_loss = causal_language_model_loss(model(inputs), targets).item()

    model.train()
    for _ in range(training_config.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = causal_language_model_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_loss = causal_language_model_loss(model(inputs), targets).item()

    return DebugOverfitResult(
        initial_loss=initial_loss,
        final_loss=final_loss,
        steps=training_config.steps,
        parameter_count=model.parameter_count(),
        device=str(device),
    )
