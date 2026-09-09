"""Deterministic tiny-text overfitting gate for the complete JeePeeTee model."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mini_llm.config import ModelConfig, RuntimeConfig, TrainingConfig
from mini_llm.model import JeePeeTee, causal_language_model_loss
from mini_llm.runtime import resolve_device, seed_everything
from mini_llm.tokenizer import CharacterTokenizer
from mini_llm.training import NextTokenDataset, load_checkpoint, save_checkpoint

DEFAULT_TINY_TEXT = "JeePeeTee can learn tiny patterns.\n" * 8


@dataclass(frozen=True, slots=True)
class TinyOverfitMetric:
    """Loss and teacher-forced next-token accuracy at one training step."""

    step: int
    loss: float
    accuracy: float


@dataclass(frozen=True, slots=True)
class TinyOverfitResult:
    """Evidence produced by a deterministic tiny-text memorization run."""

    initial_loss: float
    final_loss: float
    initial_accuracy: float
    final_accuracy: float
    steps: int
    parameter_count: int
    parameters_updated: bool
    checkpoint_consistent: bool | None
    example_target: str
    example_prediction: str
    metrics: tuple[TinyOverfitMetric, ...]


def tiny_model_config(vocab_size: int) -> ModelConfig:
    """Return the CPU-friendly architecture used by the Phase 9 gate."""

    return ModelConfig(
        vocab_size=vocab_size,
        context_length=16,
        embedding_dim=32,
        num_layers=2,
        num_heads=4,
        feed_forward_dim=64,
        dropout=0.0,
    )


def tiny_training_config() -> TrainingConfig:
    """Return deterministic optimizer settings for intentional memorization."""

    return TrainingConfig(
        batch_size=32,
        learning_rate=0.01,
        weight_decay=0.0,
        max_steps=100,
        evaluation_interval=20,
        gradient_clip_norm=1.0,
    )


def _evaluate(
    model: JeePeeTee,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = causal_language_model_loss(logits, targets)
            token_count = targets.numel()
            total_loss += loss.item() * token_count
            total_correct += (logits.argmax(dim=-1) == targets).sum().item()
            total_tokens += token_count
    if total_tokens == 0:
        raise ValueError("tiny overfit loader produced no target tokens")
    return total_loss / total_tokens, total_correct / total_tokens


def run_tiny_text_overfit(
    text: str = DEFAULT_TINY_TEXT,
    *,
    model_config: ModelConfig | None = None,
    training_config: TrainingConfig | None = None,
    runtime_config: RuntimeConfig | None = None,
    checkpoint_path: str | Path | None = None,
    log: Callable[[str], None] | None = None,
) -> TinyOverfitResult:
    """Intentionally memorize a tiny local corpus and return measurable evidence."""

    tokenizer = CharacterTokenizer.train([text])
    model_config = model_config or tiny_model_config(tokenizer.vocab_size)
    training_config = training_config or tiny_training_config()
    runtime_config = runtime_config or RuntimeConfig(
        seed=42,
        device="cpu",
        deterministic_algorithms=True,
    )
    tokenizer.validate_vocab_size(model_config.vocab_size)

    seed_everything(
        runtime_config.seed,
        deterministic_algorithms=runtime_config.deterministic_algorithms,
    )
    device = resolve_device(runtime_config.device)
    token_ids = torch.tensor(
        tokenizer.encode(text, add_bos=True, add_eos=True),
        dtype=torch.long,
    )
    dataset = NextTokenDataset(token_ids, model_config.context_length, stride=1)
    training_loader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(runtime_config.seed),
    )
    evaluation_loader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    model = JeePeeTee(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    initial_parameters = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    initial_loss, initial_accuracy = _evaluate(model, evaluation_loader, device=device)
    metrics = [TinyOverfitMetric(0, initial_loss, initial_accuracy)]
    if log is not None:
        log(f"step=0 loss={initial_loss:.6f} accuracy={initial_accuracy:.2%}")

    train_iterator = iter(training_loader)
    tokens_processed = 0
    for step in range(1, training_config.max_steps + 1):
        try:
            inputs, targets = next(train_iterator)
        except StopIteration:
            train_iterator = iter(training_loader)
            inputs, targets = next(train_iterator)

        inputs = inputs.to(device)
        targets = targets.to(device)
        tokens_processed += targets.numel()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = causal_language_model_loss(model(inputs), targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss at step {step}")
        loss.backward()
        gradients = [
            parameter.grad for parameter in model.parameters() if parameter.grad is not None
        ]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError(f"missing or non-finite gradients at step {step}")
        if training_config.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=training_config.gradient_clip_norm,
            )
        optimizer.step()

        if (
            step % training_config.evaluation_interval == 0
            or step == training_config.max_steps
        ):
            evaluation_loss, accuracy = _evaluate(model, evaluation_loader, device=device)
            if not torch.isfinite(torch.tensor(evaluation_loss)):
                raise RuntimeError(f"non-finite evaluation loss at step {step}")
            metric = TinyOverfitMetric(step, evaluation_loss, accuracy)
            metrics.append(metric)
            if log is not None:
                log(
                    f"step={step} loss={evaluation_loss:.6f} "
                    f"accuracy={accuracy:.2%}"
                )

    final_loss = metrics[-1].loss
    final_accuracy = metrics[-1].accuracy
    parameters_updated = any(
        not torch.equal(parameter.detach(), initial_parameters[name])
        for name, parameter in model.named_parameters()
    )

    example_inputs, example_targets = dataset[0]
    model.eval()
    with torch.no_grad():
        example_logits = model(example_inputs.unsqueeze(0).to(device))
    prediction_ids = example_logits.argmax(dim=-1).squeeze(0).cpu().tolist()
    example_target = tokenizer.decode(example_targets.tolist(), skip_special_tokens=True)
    example_prediction = tokenizer.decode(prediction_ids, skip_special_tokens=True)

    checkpoint_consistent: bool | None = None
    if checkpoint_path is not None:
        save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            global_step=training_config.max_steps,
            tokens_processed=tokens_processed,
            training_config=training_config,
        )
        restored_model = JeePeeTee(model_config).to(device)
        load_checkpoint(
            checkpoint_path,
            model=restored_model,
            optimizer=None,
            device=device,
        )
        restored_model.eval()
        with torch.no_grad():
            restored_logits = restored_model(example_inputs.unsqueeze(0).to(device))
        checkpoint_consistent = torch.equal(restored_logits, example_logits)

    return TinyOverfitResult(
        initial_loss=initial_loss,
        final_loss=final_loss,
        initial_accuracy=initial_accuracy,
        final_accuracy=final_accuracy,
        steps=training_config.max_steps,
        parameter_count=model.parameter_count(),
        parameters_updated=parameters_updated,
        checkpoint_consistent=checkpoint_consistent,
        example_target=example_target,
        example_prediction=example_prediction,
        metrics=tuple(metrics),
    )


def main() -> None:
    """Run the default Phase 9 gate and print its compact learning trace."""

    result = run_tiny_text_overfit(log=print)
    print(f"target:     {result.example_target!r}")
    print(f"prediction: {result.example_prediction!r}")


if __name__ == "__main__":
    main()
