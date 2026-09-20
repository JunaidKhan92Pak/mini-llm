"""Interactive local inference for a trained JeePeeTee checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from mini_llm.bpe_tokenizer import BPETokenizer
from mini_llm.evaluation import load_model_for_evaluation
from mini_llm.generation import GenerationConfig, generate

DEFAULT_CHECKPOINT = Path("checkpoints/tinystories-25k-v1/best-validation.pt")
DEFAULT_TOKENIZER = Path("data/processed/tinystories-25k-v1/tokenizer.json")


def _device_from_name(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test JeePeeTee interactively")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--strategy", choices=("greedy", "sample"), default="sample")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = _device_from_name(args.device)
    tokenizer = BPETokenizer.load(args.tokenizer)
    model, payload = load_model_for_evaluation(args.checkpoint, tokenizer, device=device)
    expected_hash = payload.get("tokenizer_metadata", {}).get("sha256")
    actual_hash = hashlib.sha256(args.tokenizer.read_bytes()).hexdigest()
    if expected_hash is not None and expected_hash != actual_hash:
        raise ValueError("tokenizer file fingerprint does not match checkpoint")

    config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        strategy=args.strategy,
        temperature=args.temperature,
        top_k=args.top_k if args.strategy == "sample" else None,
        top_p=args.top_p if args.strategy == "sample" else None,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )
    print(
        f"Checkpoint: {args.checkpoint.resolve()}\n"
        f"JeePeeTee loaded on {device} | step={payload.get('global_step')} | "
        f"parameters={model.parameter_count():,}"
    )
    print("Completion mode: this pretrained checkpoint continues text; it is not a QA assistant.")
    print("Enter a prompt. Each prompt is independent. Type /quit to exit.\n")
    while True:
        try:
            prompt = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.lower() in {"/quit", "quit", "exit"}:
            break
        if not prompt:
            continue
        prefix = prompt
        if len(tokenizer.encode(prefix, add_bos=True)) + config.max_new_tokens > (
            model.config.context_length
        ):
            print("Prompt too long for this context and output budget; nothing was truncated.\n")
            continue
        result = generate(model, tokenizer, prefix, config, device=device)
        completion = tokenizer.decode(result.generated_token_ids, skip_special_tokens=True)
        print(f"JeePeeTee> {completion.strip()}\n")


if __name__ == "__main__":
    main()
