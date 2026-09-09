"""CPU-friendly, resumable Phase 12 pretraining for JeePeeTee."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from mini_llm.bpe_tokenizer import BPETokenizer
from mini_llm.config import ModelConfig, RuntimeConfig
from mini_llm.generation import GenerationConfig, generate
from mini_llm.model import JeePeeTee, causal_language_model_loss
from mini_llm.pretraining.pipeline import FixedTokenSequenceDataset
from mini_llm.runtime import resolve_device, seed_everything

PHASE12_CHECKPOINT_VERSION = 2
DEFAULT_PROMPTS = (
    "Python is",
    "Artificial intelligence is",
    "A computer program",
    "The capital of France",
)


@dataclass(frozen=True, slots=True)
class PretrainingRunConfig:
    """Bounded optimizer, evaluation, and persistence settings."""

    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 300
    warmup_steps: int = 20
    minimum_learning_rate_ratio: float = 0.1
    evaluation_interval: int = 50
    evaluation_batches: int = 10
    checkpoint_interval: int = 100
    generation_interval: int = 100
    generation_max_new_tokens: int = 24
    gradient_clip_norm: float = 1.0

    def __post_init__(self) -> None:
        positive = {
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "max_steps": self.max_steps,
            "evaluation_interval": self.evaluation_interval,
            "evaluation_batches": self.evaluation_batches,
            "checkpoint_interval": self.checkpoint_interval,
            "generation_interval": self.generation_interval,
            "generation_max_new_tokens": self.generation_max_new_tokens,
            "gradient_clip_norm": self.gradient_clip_norm,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not 0 <= self.warmup_steps <= self.max_steps:
            raise ValueError("warmup_steps must be between zero and max_steps")
        if not 0 < self.minimum_learning_rate_ratio <= 1:
            raise ValueError("minimum_learning_rate_ratio must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class PretrainingMetric:
    global_step: int
    tokens_processed: int
    training_loss: float
    validation_loss: float
    perplexity: float
    learning_rate: float
    gradient_norm: float | None


@dataclass(frozen=True, slots=True)
class GenerationSample:
    global_step: int
    prompt: str
    text: str


@dataclass(frozen=True, slots=True)
class PretrainingResult:
    global_step: int
    tokens_processed: int
    metrics: tuple[PretrainingMetric, ...]
    samples: tuple[GenerationSample, ...]
    checkpoints: tuple[Path, ...]


def phase12_model_config(vocab_size: int, *, context_length: int = 64) -> ModelConfig:
    """Return the first real-training architecture (under one million parameters)."""

    return ModelConfig(
        vocab_size=vocab_size,
        context_length=context_length,
        embedding_dim=96,
        num_layers=4,
        num_heads=4,
        feed_forward_dim=384,
        dropout=0.1,
    )


def _schedule_multiplier(step: int, config: PretrainingRunConfig) -> float:
    if config.warmup_steps and step < config.warmup_steps:
        return (step + 1) / config.warmup_steps
    decay_steps = max(1, config.max_steps - config.warmup_steps)
    progress = min(1.0, (step - config.warmup_steps) / decay_steps)
    return 1.0 - progress * (1.0 - config.minimum_learning_rate_ratio)


def _step_batch(
    sequences: torch.Tensor,
    *,
    step: int,
    batch_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch deterministically from the step number for exact resume."""

    generator = torch.Generator().manual_seed(seed + step)
    indices = torch.randint(0, sequences.shape[0], (batch_size,), generator=generator)
    rows = sequences[indices]
    return rows[:, :-1], rows[:, 1:]


def evaluate_sequences(
    model: JeePeeTee,
    sequences: torch.Tensor,
    *,
    batch_size: int,
    max_batches: int,
    device: torch.device,
) -> float:
    """Evaluate a fixed leading slice and restore the caller's model mode."""

    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for start in range(0, min(len(sequences), batch_size * max_batches), batch_size):
            rows = sequences[start : start + batch_size].to(device)
            inputs, targets = rows[:, :-1], rows[:, 1:]
            loss = causal_language_model_loss(model(inputs), targets)
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()
    model.train(was_training)
    if total_tokens == 0:
        raise ValueError("evaluation sequences contain no target tokens")
    result = total_loss / total_tokens
    if not math.isfinite(result):
        raise FloatingPointError("evaluation produced a non-finite loss")
    return result


def generate_greedy(
    model: JeePeeTee,
    tokenizer: BPETokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    """Generate a controlled argmax sample for before/after comparisons."""

    return generate(
        model,
        tokenizer,
        prompt,
        GenerationConfig(max_new_tokens=max_new_tokens, strategy="greedy"),
        device=device,
    ).text


def _tokenizer_metadata(tokenizer: BPETokenizer, tokenizer_path: Path) -> dict[str, Any]:
    serialized_config = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    return {
        "type": "byte-level-bpe",
        "vocab_size": tokenizer.vocab_size,
        "special_token_ids": tokenizer.special_token_ids,
        "sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "config": serialized_config,
    }


def save_phase12_checkpoint(
    path: Path,
    *,
    model: JeePeeTee,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    global_step: int,
    tokens_processed: int,
    run_config: PretrainingRunConfig,
    runtime_config: RuntimeConfig,
    tokenizer_metadata: dict[str, Any],
    dataset_metadata: dict[str, Any],
) -> None:
    """Atomically persist every state required to reproduce and resume a run."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "checkpoint_version": PHASE12_CHECKPOINT_VERSION,
            "model_config": asdict(model.config),
            "run_config": asdict(run_config),
            "training_metadata": {
                "optimizer": "AdamW",
                "objective": "causal_next_token_prediction",
                "target_alignment": "inputs=row[:-1], targets=row[1:]",
                "runtime_config": asdict(runtime_config),
            },
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "global_step": global_step,
            "tokens_processed": tokens_processed,
            "tokenizer_metadata": tokenizer_metadata,
            "dataset_metadata": dataset_metadata,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        temporary,
    )
    temporary.replace(path)


def load_phase12_checkpoint(
    path: Path,
    *,
    model: JeePeeTee,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    run_config: PretrainingRunConfig,
    runtime_config: RuntimeConfig,
    tokenizer_metadata: dict[str, Any],
) -> tuple[int, int, dict[str, Any]]:
    """Load a strictly compatible Phase 12 checkpoint."""

    payload: Any = torch.load(path, map_location=device, weights_only=True)
    required = {
        "checkpoint_version", "model_config", "run_config", "training_metadata", "model_state",
        "optimizer_state", "scheduler_state", "global_step", "tokens_processed",
        "tokenizer_metadata", "dataset_metadata", "torch_rng_state", "cuda_rng_states",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("Phase 12 checkpoint is missing required fields")
    if payload["checkpoint_version"] != PHASE12_CHECKPOINT_VERSION:
        raise ValueError("unsupported Phase 12 checkpoint version")
    if payload["model_config"] != asdict(model.config):
        raise ValueError("checkpoint model configuration does not match")
    if payload["run_config"] != asdict(run_config):
        raise ValueError("checkpoint run configuration does not match")
    if payload["training_metadata"]["runtime_config"] != asdict(runtime_config):
        raise ValueError("checkpoint runtime configuration does not match")
    if payload["tokenizer_metadata"] != tokenizer_metadata:
        raise ValueError("checkpoint tokenizer metadata does not match")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    if device.type == "cuda" and payload["cuda_rng_states"]:
        torch.cuda.set_rng_state_all(payload["cuda_rng_states"])
    return payload["global_step"], payload["tokens_processed"], payload["dataset_metadata"]


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def train_phase12(
    model: JeePeeTee,
    tokenizer: BPETokenizer,
    train_sequences: torch.Tensor,
    validation_sequences: torch.Tensor,
    run_config: PretrainingRunConfig,
    runtime_config: RuntimeConfig,
    *,
    output_dir: Path,
    tokenizer_path: Path,
    dataset_metadata: dict[str, Any],
    resume_from: Path | None = None,
    stop_after_step: int | None = None,
    prompts: tuple[str, ...] = DEFAULT_PROMPTS,
) -> PretrainingResult:
    """Run finite, evaluated causal pretraining with periodic resumable checkpoints."""

    seed_everything(
        runtime_config.seed,
        deterministic_algorithms=runtime_config.deterministic_algorithms,
    )
    device = resolve_device(runtime_config.device)
    model.to(device)
    tokenizer.validate_vocab_size(model.config.vocab_size)
    for sequences in (train_sequences, validation_sequences):
        FixedTokenSequenceDataset(sequences, context_length=model.config.context_length)
        if sequences.min() < 0 or sequences.max() >= tokenizer.vocab_size:
            raise ValueError("dataset contains token IDs outside the tokenizer vocabulary")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=run_config.learning_rate, weight_decay=run_config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _schedule_multiplier(step, run_config)
    )
    tokenizer_info = _tokenizer_metadata(tokenizer, tokenizer_path)
    global_step = 0
    tokens_processed = 0
    if resume_from is not None:
        global_step, tokens_processed, saved_dataset_metadata = load_phase12_checkpoint(
            resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            run_config=run_config,
            runtime_config=runtime_config,
            tokenizer_metadata=tokenizer_info,
        )
        if saved_dataset_metadata != dataset_metadata:
            raise ValueError("checkpoint dataset metadata does not match")

    limit = (
        run_config.max_steps
        if stop_after_step is None
        else min(stop_after_step, run_config.max_steps)
    )
    if not global_step <= limit:
        raise ValueError("checkpoint step exceeds the requested stopping step")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "metrics.jsonl"
    manifest_path = output_dir / "run-manifest.json"
    manifest = {
        "model_config": asdict(model.config),
        "parameter_count": model.parameter_count(),
        "run_config": asdict(run_config),
        "runtime_config": asdict(runtime_config),
        "tokenizer_metadata": tokenizer_info,
        "dataset_metadata": dataset_metadata,
    }
    if not manifest_path.exists():
        temporary_manifest = manifest_path.with_suffix(".json.tmp")
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_manifest.replace(manifest_path)
    metrics: list[PretrainingMetric] = []
    samples: list[GenerationSample] = []
    checkpoints: list[Path] = []

    def record_metrics(gradient_norm: float | None) -> None:
        train_loss = evaluate_sequences(
            model, train_sequences, batch_size=run_config.batch_size,
            max_batches=run_config.evaluation_batches, device=device,
        )
        validation_loss = evaluate_sequences(
            model, validation_sequences, batch_size=run_config.batch_size,
            max_batches=run_config.evaluation_batches, device=device,
        )
        metric = PretrainingMetric(
            global_step=global_step,
            tokens_processed=tokens_processed,
            training_loss=train_loss,
            validation_loss=validation_loss,
            perplexity=math.exp(validation_loss),
            learning_rate=optimizer.param_groups[0]["lr"],
            gradient_norm=gradient_norm,
        )
        metrics.append(metric)
        _append_jsonl(log_path, {"event": "metrics", **asdict(metric)})

    def record_samples() -> None:
        for prompt in prompts:
            sample = GenerationSample(
                global_step=global_step,
                prompt=prompt,
                text=generate_greedy(
                    model, tokenizer, prompt,
                    max_new_tokens=run_config.generation_max_new_tokens, device=device,
                ),
            )
            samples.append(sample)
            _append_jsonl(log_path, {"event": "generation", **asdict(sample)})

    if global_step == 0:
        record_metrics(None)
        record_samples()

    latest_gradient_norm: float | None = None
    while global_step < limit:
        inputs, targets = _step_batch(
            train_sequences,
            step=global_step,
            batch_size=run_config.batch_size,
            seed=runtime_config.seed,
        )
        inputs, targets = inputs.to(device), targets.to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = causal_language_model_loss(model(inputs), targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at step {global_step}")
        loss.backward()
        if any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise FloatingPointError(f"non-finite gradient at step {global_step}")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), run_config.gradient_clip_norm
        )
        latest_gradient_norm = float(gradient_norm)
        if not math.isfinite(latest_gradient_norm):
            raise FloatingPointError(f"non-finite gradient norm at step {global_step}")
        optimizer.step()
        scheduler.step()
        global_step += 1
        tokens_processed += targets.numel()

        if global_step % run_config.evaluation_interval == 0 or global_step == limit:
            record_metrics(latest_gradient_norm)
        if global_step % run_config.generation_interval == 0 or global_step == limit:
            record_samples()
        if (
            global_step % run_config.checkpoint_interval == 0
            or global_step == limit
        ):
            checkpoint = output_dir / f"checkpoint-step-{global_step:06d}.pt"
            save_phase12_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=global_step,
                tokens_processed=tokens_processed,
                run_config=run_config,
                runtime_config=runtime_config,
                tokenizer_metadata=tokenizer_info,
                dataset_metadata=dataset_metadata,
            )
            checkpoints.append(checkpoint)

    return PretrainingResult(
        global_step=global_step,
        tokens_processed=tokens_processed,
        metrics=tuple(metrics),
        samples=tuple(samples),
        checkpoints=tuple(checkpoints),
    )
