"""Fixed, comparable evaluation suite for pretrained JeePeeTee checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from mini_llm.bpe_tokenizer import BPETokenizer
from mini_llm.config import ModelConfig
from mini_llm.generation import GenerationConfig, generate
from mini_llm.model import JeePeeTee


@dataclass(frozen=True, slots=True)
class EvaluationPrompt:
    category: str
    prompt: str


FIXED_PROMPT_SUITE = (
    EvaluationPrompt("english_completion", "The weather today is"),
    EvaluationPrompt("general_knowledge", "The capital of France is"),
    EvaluationPrompt("computer_science", "A computer stores information"),
    EvaluationPrompt("python", "In Python, a function is defined using"),
    EvaluationPrompt("javascript", "In JavaScript, a variable can be declared with"),
    EvaluationPrompt("code_completion", "def add(a, b):\n    return"),
    EvaluationPrompt("technical_explanation", "A database is"),
)


def load_model_for_evaluation(
    checkpoint_path: Path,
    tokenizer: BPETokenizer,
    *,
    device: torch.device,
) -> tuple[JeePeeTee, dict[str, Any]]:
    """Load and validate a Phase 12 model without restoring training-only state."""

    payload: Any = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if (
        not isinstance(payload, dict)
        or "model_config" not in payload
        or "model_state" not in payload
    ):
        raise ValueError("checkpoint does not contain a model configuration and state")
    tokenizer_metadata = payload.get("tokenizer_metadata", {})
    if tokenizer_metadata.get("vocab_size") != tokenizer.vocab_size:
        raise ValueError("checkpoint tokenizer vocabulary does not match loaded tokenizer")
    model = JeePeeTee(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    return model, payload


def evaluate_prompt_suite(
    model: JeePeeTee,
    tokenizer: BPETokenizer,
    config: GenerationConfig,
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Generate every fixed prompt and attach basic repetition diagnostics."""

    results = []
    for item in FIXED_PROMPT_SUITE:
        prompt_ids = tokenizer.encode(item.prompt, add_bos=True)
        context = prompt_ids[-model.config.context_length :]
        with torch.no_grad():
            logits = model(torch.tensor([context], dtype=torch.long, device=device))[0, -1]
            probabilities = torch.softmax(logits, dim=-1)
            log_probabilities = torch.log_softmax(logits, dim=-1)
            entropy = float(-(probabilities * log_probabilities).sum().item())
            top_probabilities, top_ids = probabilities.topk(5)
        generated = generate(model, tokenizer, item.prompt, config, device=device)
        ids = generated.generated_token_ids
        unique_ratio = len(set(ids)) / len(ids) if ids else 1.0
        results.append({
            "category": item.category,
            "prompt": item.prompt,
            "text": generated.text,
            "generated_token_ids": list(ids),
            "finish_reason": generated.finish_reason,
            "unique_token_ratio": unique_ratio,
            "prompt_token_count": len(prompt_ids),
            "prompt_was_truncated": len(prompt_ids) > model.config.context_length,
            "logits_are_finite": bool(torch.isfinite(logits).all()),
            "logit_range": [float(logits.min()), float(logits.max())],
            "next_token_entropy": entropy,
            "top_next_tokens": [
                {
                    "id": token_id,
                    "token": tokenizer.vocabulary[token_id],
                    "probability": float(probability),
                }
                for token_id, probability in zip(
                    top_ids.tolist(), top_probabilities.tolist(), strict=True
                )
            ],
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a JeePeeTee checkpoint")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    tokenizer = BPETokenizer.load(args.tokenizer)
    model, payload = load_model_for_evaluation(args.checkpoint, tokenizer, device=device)
    tokenizer_hash = hashlib.sha256(args.tokenizer.read_bytes()).hexdigest()
    if payload["tokenizer_metadata"].get("sha256") != tokenizer_hash:
        raise ValueError("tokenizer file fingerprint does not match checkpoint")

    configs = {
        "greedy": GenerationConfig(max_new_tokens=32, strategy="greedy"),
        "controlled_sample": GenerationConfig(
            max_new_tokens=32,
            strategy="sample",
            temperature=0.8,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.15,
            seed=42,
        ),
    }
    report = {
        "checkpoint": str(args.checkpoint),
        "global_step": payload.get("global_step"),
        "tokens_processed": payload.get("tokens_processed"),
        "parameter_count": model.parameter_count(),
        "model_config": asdict(model.config),
        "generation": {
            name: {
                "config": asdict(config),
                "outputs": evaluate_prompt_suite(
                    model, tokenizer, config, device=device
                ),
            }
            for name, config in configs.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
