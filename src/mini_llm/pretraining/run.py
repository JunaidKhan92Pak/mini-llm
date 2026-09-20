"""Command-line entry point for the bounded Phase 12 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from mini_llm.bpe_tokenizer import BPETokenizer
from mini_llm.config import (
    RuntimeConfig,
    fifteen_million_model_config,
    five_million_model_config,
)
from mini_llm.evaluation import FIXED_PROMPT_SUITE
from mini_llm.model import JeePeeTee
from mini_llm.pretraining.pipeline import load_token_sequences
from mini_llm.pretraining.real_training import (
    PretrainingRunConfig,
    phase12_model_config,
    train_phase12,
)
from mini_llm.runtime import seed_everything


def initialize_weights(model, checkpoint_path: Path, tokenizer_path: Path) -> dict:
    """Warm-start weights only; a changed corpus gets a fresh optimizer/schedule."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("model_config") != asdict(model.config):
        raise ValueError("initial checkpoint architecture does not match selected preset")
    if (
        payload.get("tokenizer_metadata", {}).get("sha256")
        != hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    ):
        raise ValueError("initial checkpoint tokenizer fingerprint does not match")
    state = payload.get("model_state")
    if not isinstance(state, dict) or any(not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("initial checkpoint contains invalid model weights")
    model.load_state_dict(state, strict=True)
    return {
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "checkpoint_global_step": payload["global_step"],
        "checkpoint_tokens_processed": payload.get("tokens_processed", 0),
        "mode": "weights_only_fresh_optimizer_and_scheduler",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run resumable JeePeeTee pretraining")
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--initialize-from", type=Path, help="Weights-only warm start for a new corpus"
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--minimum-learning-rate-ratio", type=float, default=0.1)
    parser.add_argument("--evaluation-interval", type=int, default=50)
    parser.add_argument("--evaluation-batches", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--generation-interval", type=int, default=100)
    parser.add_argument("--generation-max-new-tokens", type=int, default=24)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--model-preset", choices=("mini", "five-million", "fifteen-million"), default="mini"
    )
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)

    metadata = json.loads((args.data_dir / "metadata.json").read_text(encoding="utf-8"))
    context_length = int(metadata["context_length"])
    tokenizer_path = args.data_dir / "tokenizer.json"
    tokenizer = BPETokenizer.load(tokenizer_path)
    train_sequences = load_token_sequences(
        args.data_dir / "tokenized" / "train.pt", context_length=context_length
    )
    validation_sequences = load_token_sequences(
        args.data_dir / "tokenized" / "validation.pt", context_length=context_length
    )
    runtime_config = RuntimeConfig(seed=args.seed, device=args.device)
    run_config = PretrainingRunConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        minimum_learning_rate_ratio=args.minimum_learning_rate_ratio,
        evaluation_interval=args.evaluation_interval,
        evaluation_batches=args.evaluation_batches,
        checkpoint_interval=args.checkpoint_interval,
        generation_interval=args.generation_interval,
        generation_max_new_tokens=args.generation_max_new_tokens,
        gradient_clip_norm=args.gradient_clip_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    seed_everything(args.seed)
    presets = {
        "mini": phase12_model_config,
        "five-million": five_million_model_config,
        "fifteen-million": fifteen_million_model_config,
    }
    model_config = presets[args.model_preset](tokenizer.vocab_size, context_length=context_length)
    model = JeePeeTee(model_config)
    if args.initialize_from:
        metadata = {
            **metadata,
            "initialization": initialize_weights(model, args.initialize_from, tokenizer_path),
        }
        print(
            f"Warm-started from {args.initialize_from}; optimizer/schedule start fresh.", flush=True
        )
    result = train_phase12(
        model,
        tokenizer,
        train_sequences,
        validation_sequences,
        run_config,
        runtime_config,
        output_dir=args.output_dir,
        tokenizer_path=tokenizer_path,
        dataset_metadata=metadata,
        resume_from=args.resume_from,
        stop_after_step=args.stop_after_step,
        prompts=tuple(
            metadata.get("evaluation_prompts", [item.prompt for item in FIXED_PROMPT_SUITE])
        ),
    )
    print(
        json.dumps(
            {
                "parameters": model.parameter_count(),
                "global_step": result.global_step,
                "tokens_processed": result.tokens_processed,
                "metrics": [asdict(metric) for metric in result.metrics],
                "samples": [asdict(sample) for sample in result.samples],
                "checkpoints": [str(path) for path in result.checkpoints],
                "best_checkpoint": str(result.best_checkpoint) if result.best_checkpoint else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
