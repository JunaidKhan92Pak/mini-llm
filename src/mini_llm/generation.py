"""Autoregressive generation utilities kept separate from model architecture."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from mini_llm.bpe_tokenizer import BPETokenizer
from mini_llm.model import JeePeeTee

DecodingStrategy = Literal["greedy", "sample"]
FinishReason = Literal["eos", "max_new_tokens"]


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Controlled decoding settings for evaluation and inference."""

    max_new_tokens: int = 32
    strategy: DecodingStrategy = "greedy"
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float = 1.0
    seed: int = 42
    suppress_control_tokens: bool = True

    def __post_init__(self) -> None:
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if self.strategy not in ("greedy", "sample"):
            raise ValueError(f"unsupported decoding strategy: {self.strategy!r}")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive or None")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1] or None")
        if not math.isfinite(self.repetition_penalty) or self.repetition_penalty < 1:
            raise ValueError("repetition_penalty must be finite and at least 1.0")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Decoded output plus auditable stopping and token information."""

    text: str
    prompt_token_count: int
    generated_token_ids: tuple[int, ...]
    finish_reason: FinishReason


def _apply_repetition_penalty(
    logits: torch.Tensor,
    token_ids: list[int],
    penalty: float,
) -> torch.Tensor:
    if penalty == 1.0:
        return logits
    adjusted = logits.clone()
    seen = torch.tensor(sorted(set(token_ids)), device=logits.device)
    values = adjusted[seen]
    adjusted[seen] = torch.where(values < 0, values * penalty, values / penalty)
    return adjusted


def _filter_sampling_logits(
    logits: torch.Tensor,
    *,
    top_k: int | None,
    top_p: float | None,
) -> torch.Tensor:
    filtered = logits.clone()
    if top_k is not None:
        cutoff = torch.topk(filtered, min(top_k, filtered.numel())).values[-1]
        filtered[filtered < cutoff] = -torch.inf
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        probabilities = torch.softmax(sorted_logits, dim=-1)
        remove = probabilities.cumsum(dim=-1) - probabilities >= top_p
        sorted_logits[remove] = -torch.inf
        filtered = torch.full_like(filtered, -torch.inf)
        filtered.scatter_(0, sorted_indices, sorted_logits)
    return filtered


def generate(
    model: JeePeeTee,
    tokenizer: BPETokenizer,
    prompt: str,
    config: GenerationConfig | None = None,
    *,
    device: torch.device | None = None,
) -> GenerationResult:
    """Generate tokens autoregressively with safe context cropping and EOS stopping."""

    config = config or GenerationConfig()
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    tokenizer.validate_vocab_size(model.config.vocab_size)
    if device is None:
        device = next(model.parameters()).device
    token_ids = tokenizer.encode(prompt, add_bos=True)
    prompt_token_count = len(token_ids)
    generated: list[int] = []
    was_training = model.training
    model.eval()
    generator = torch.Generator(device=device).manual_seed(config.seed)
    finish_reason: FinishReason = "max_new_tokens"
    try:
        with torch.no_grad():
            for _ in range(config.max_new_tokens):
                context = token_ids[-model.config.context_length :]
                inputs = torch.tensor([context], dtype=torch.long, device=device)
                logits = model(inputs)[0, -1]
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("model produced non-finite generation logits")
                logits = _apply_repetition_penalty(
                    logits, token_ids, config.repetition_penalty
                )
                if config.suppress_control_tokens:
                    for token_id in (
                        tokenizer.pad_token_id,
                        tokenizer.bos_token_id,
                        tokenizer.unk_token_id,
                    ):
                        logits[token_id] = -torch.inf
                if config.strategy == "greedy":
                    next_id = int(logits.argmax().item())
                else:
                    filtered = _filter_sampling_logits(
                        logits / config.temperature,
                        top_k=config.top_k,
                        top_p=config.top_p,
                    )
                    probabilities = torch.softmax(filtered, dim=-1)
                    next_id = int(torch.multinomial(probabilities, 1, generator=generator).item())
                token_ids.append(next_id)
                generated.append(next_id)
                if next_id == tokenizer.eos_token_id:
                    finish_reason = "eos"
                    break
    finally:
        model.train(was_training)

    if any(not 0 <= token_id < tokenizer.vocab_size for token_id in generated):
        raise RuntimeError("generation produced an invalid token ID")
    return GenerationResult(
        text=tokenizer.decode(token_ids, skip_special_tokens=True),
        prompt_token_count=prompt_token_count,
        generated_token_ids=tuple(generated),
        finish_reason=finish_reason,
    )
