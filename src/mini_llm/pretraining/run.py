"""Command-line entry point for the bounded Phase 12 experiment."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from mini_llm.bpe_tokenizer import BPETokenizer
from mini_llm.config import RuntimeConfig, five_million_model_config
from mini_llm.evaluation import FIXED_PROMPT_SUITE
from mini_llm.model import JeePeeTee
from mini_llm.pretraining.pipeline import load_token_sequences
from mini_llm.pretraining.real_training import (
    PretrainingRunConfig,
    phase12_model_config,
    train_phase12,
)
from mini_llm.runtime import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Run resumable JeePeeTee pretraining")
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--resume-from", type=Path)
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
    parser.add_argument("--model-preset", choices=("mini", "five-million"), default="mini")
    args = parser.parse_args()

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
    model_config = (
        five_million_model_config(tokenizer.vocab_size, context_length=context_length)
        if args.model_preset == "five-million"
        else phase12_model_config(tokenizer.vocab_size, context_length=context_length)
    )
    model = JeePeeTee(model_config)
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
        prompts=tuple(item.prompt for item in FIXED_PROMPT_SUITE),
    )
    print(json.dumps({
        "parameters": model.parameter_count(),
        "global_step": result.global_step,
        "tokens_processed": result.tokens_processed,
        "metrics": [asdict(metric) for metric in result.metrics],
        "samples": [asdict(sample) for sample in result.samples],
        "checkpoints": [str(path) for path in result.checkpoints],
        "best_checkpoint": str(result.best_checkpoint) if result.best_checkpoint else None,
    }, indent=2))


if __name__ == "__main__":
    main()
