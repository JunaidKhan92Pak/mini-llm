"""Command-line entry point for the bounded Phase 12 experiment."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from mini_llm.bpe_tokenizer import BPETokenizer
from mini_llm.config import RuntimeConfig
from mini_llm.model import JeePeeTee
from mini_llm.pretraining.pipeline import load_token_sequences
from mini_llm.pretraining.real_training import (
    PretrainingRunConfig,
    phase12_model_config,
    train_phase12,
)
from mini_llm.runtime import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Run JeePeeTee Phase 12 pretraining")
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--evaluation-interval", type=int, default=50)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--generation-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
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
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        evaluation_interval=args.evaluation_interval,
        checkpoint_interval=args.checkpoint_interval,
        generation_interval=args.generation_interval,
    )
    seed_everything(args.seed)
    model = JeePeeTee(
        phase12_model_config(tokenizer.vocab_size, context_length=context_length)
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
    )
    print(json.dumps({
        "parameters": model.parameter_count(),
        "global_step": result.global_step,
        "tokens_processed": result.tokens_processed,
        "metrics": [asdict(metric) for metric in result.metrics],
        "samples": [asdict(sample) for sample in result.samples],
        "checkpoints": [str(path) for path in result.checkpoints],
    }, indent=2))


if __name__ == "__main__":
    main()
