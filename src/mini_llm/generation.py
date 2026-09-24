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
    trace_top_n: int = 0

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
        if self.trace_top_n < 0:
            raise ValueError("trace_top_n must be non-negative")


@dataclass(frozen=True, slots=True)
class TokenCandidate:
    token_id: int
    token: str
    probability: float
    raw_logit: float | None = None


@dataclass(frozen=True, slots=True)
class AttentionFocus:
    position: int
    token_id: int
    token: str
    probability: float


@dataclass(frozen=True, slots=True)
class GenerationStep:
    index: int
    context_token_count: int
    selected_token_id: int
    selected_token: str
    selected_probability: float
    candidates: tuple[TokenCandidate, ...]
    embedding_preview: tuple[float, ...] = ()
    embedding_norm: float | None = None
    layer_norms: tuple[float, ...] = ()
    attention_focus: tuple[AttentionFocus, ...] = ()
    projection_input_norm: float | None = None
    context_token_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Decoded output plus auditable stopping and token information."""

    text: str
    prompt_token_count: int
    generated_token_ids: tuple[int, ...]
    finish_reason: FinishReason
    steps: tuple[GenerationStep, ...] = ()


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
    steps: list[GenerationStep] = []
    was_training = model.training
    model.eval()
    generator = torch.Generator(device=device).manual_seed(config.seed)
    finish_reason: FinishReason = "max_new_tokens"
    embedding_preview: list[float] = []
    embedding_norm: list[float] = []
    layer_norms: list[float] = []
    attention_focus: list[tuple[int, float]] = []
    projection_input_norm: list[float] = []
    trace_handles = []
    if config.trace_top_n and hasattr(model, "embeddings") and hasattr(model, "blocks"):

        def record_embedding(
            _module: torch.nn.Module, _inputs: tuple, output: torch.Tensor
        ) -> None:
            vector = output[0, -1].detach().float()
            embedding_preview.extend(float(value) for value in vector[:8].tolist())
            embedding_norm.append(float(torch.linalg.vector_norm(vector)))

        def record_layer(_module: torch.nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
            layer_norms.append(float(torch.linalg.vector_norm(output[0, -1].detach().float())))

        trace_handles.append(model.embeddings.register_forward_hook(record_embedding))
        trace_handles.extend(block.register_forward_hook(record_layer) for block in model.blocks)
        if model.blocks:

            def record_attention(module: torch.nn.Module, inputs: tuple) -> None:
                normalized = inputs[0][0].detach()
                length = normalized.shape[0]
                projected = module.qkv_projection(normalized).reshape(
                    length, 3, module.num_heads, module.head_dim
                )
                query = projected[-1, 0, 0]
                keys = projected[:, 1, 0]
                scores = keys @ query / math.sqrt(module.head_dim)
                weights = torch.softmax(scores, dim=0)
                values, positions = weights.topk(min(5, length))
                attention_focus.extend(
                    (int(position), float(value))
                    for position, value in zip(positions, values, strict=True)
                )

            trace_handles.append(
                model.blocks[-1].attention.register_forward_pre_hook(record_attention)
            )
        if hasattr(model, "final_norm"):

            def record_projection(
                _module: torch.nn.Module, _inputs: tuple, output: torch.Tensor
            ) -> None:
                projection_input_norm.append(
                    float(torch.linalg.vector_norm(output[0, -1].detach().float()))
                )

            trace_handles.append(model.final_norm.register_forward_hook(record_projection))
    try:
        with torch.no_grad():
            for _ in range(config.max_new_tokens):
                embedding_preview.clear()
                embedding_norm.clear()
                layer_norms.clear()
                attention_focus.clear()
                projection_input_norm.clear()
                context = token_ids[-model.config.context_length :]
                inputs = torch.tensor([context], dtype=torch.long, device=device)
                logits = model(inputs)[0, -1]
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("model produced non-finite generation logits")
                raw_logits = logits.clone() if config.trace_top_n else None
                logits = _apply_repetition_penalty(logits, token_ids, config.repetition_penalty)
                if config.suppress_control_tokens:
                    for token_id in (
                        tokenizer.pad_token_id,
                        tokenizer.bos_token_id,
                        tokenizer.unk_token_id,
                    ):
                        logits[token_id] = -torch.inf
                if config.strategy == "greedy":
                    probabilities = torch.softmax(logits, dim=-1)
                    next_id = int(logits.argmax().item())
                else:
                    filtered = _filter_sampling_logits(
                        logits / config.temperature,
                        top_k=config.top_k,
                        top_p=config.top_p,
                    )
                    probabilities = torch.softmax(filtered, dim=-1)
                    next_id = int(torch.multinomial(probabilities, 1, generator=generator).item())
                if config.trace_top_n:
                    count = min(config.trace_top_n, tokenizer.vocab_size)
                    top_probabilities, top_ids = probabilities.topk(count)
                    candidates = tuple(
                        TokenCandidate(
                            token_id=token_id,
                            token=tokenizer.decode([token_id], skip_special_tokens=True),
                            probability=float(probability),
                            raw_logit=float(raw_logits[token_id]),
                        )
                        for token_id, probability in zip(
                            top_ids.tolist(), top_probabilities.tolist(), strict=True
                        )
                    )
                    steps.append(
                        GenerationStep(
                            index=len(generated),
                            context_token_count=len(context),
                            selected_token_id=next_id,
                            selected_token=tokenizer.decode([next_id], skip_special_tokens=True),
                            selected_probability=float(probabilities[next_id]),
                            candidates=candidates,
                            embedding_preview=tuple(embedding_preview),
                            embedding_norm=embedding_norm[0] if embedding_norm else None,
                            layer_norms=tuple(layer_norms),
                            attention_focus=tuple(
                                AttentionFocus(
                                    position=position,
                                    token_id=context[position],
                                    token=tokenizer.decode(
                                        [context[position]], skip_special_tokens=True
                                    ),
                                    probability=probability,
                                )
                                for position, probability in attention_focus
                            ),
                            projection_input_norm=(
                                projection_input_norm[0] if projection_input_norm else None
                            ),
                            context_token_ids=tuple(context),
                        )
                    )
                token_ids.append(next_id)
                generated.append(next_id)
                if next_id == tokenizer.eos_token_id:
                    finish_reason = "eos"
                    break
    finally:
        for handle in trace_handles:
            handle.remove()
        model.train(was_training)

    if any(not 0 <= token_id < tokenizer.vocab_size for token_id in generated):
        raise RuntimeError("generation produced an invalid token ID")
    return GenerationResult(
        text=tokenizer.decode(token_ids, skip_special_tokens=True),
        prompt_token_count=prompt_token_count,
        generated_token_ids=tuple(generated),
        finish_reason=finish_reason,
        steps=tuple(steps),
    )
