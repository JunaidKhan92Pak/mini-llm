"""Causal scaled dot-product and multi-head self-attention."""

import math

import torch
from torch import nn
from torch.nn import functional as functional

from mini_llm.config import ModelConfig


def make_causal_mask(sequence_length: int, *, device: torch.device) -> torch.Tensor:
    """Return a boolean mask where each position can see only itself and the past."""

    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}")
    return torch.ones(
        (sequence_length, sequence_length),
        dtype=torch.bool,
        device=device,
    ).tril()


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    is_causal: bool = True,
    dropout_p: float = 0.0,
    training: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute attention values and probabilities from explicit Q, K, and V tensors.

    Expected shapes are ``(..., query_sequence, head_dim)`` for query and
    ``(..., key_sequence, head_dim)`` for key/value. The returned attention output has the
    query shape, while probabilities have shape ``(..., query_sequence, key_sequence)``.
    """

    if query.ndim < 2 or key.ndim < 2 or value.ndim < 2:
        raise ValueError("query, key, and value must each have at least two dimensions")
    if query.shape[:-2] != key.shape[:-2] or key.shape[:-2] != value.shape[:-2]:
        raise ValueError("query, key, and value batch/head dimensions must match")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key head dimensions must match")
    if key.shape[-2] != value.shape[-2]:
        raise ValueError("key and value sequence lengths must match")
    if value.shape[-1] != query.shape[-1]:
        raise ValueError("value and query head dimensions must match")
    if query.shape[-1] <= 0:
        raise ValueError("head dimension must be positive")
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError(f"dropout_p must be in [0.0, 1.0), got {dropout_p}")

    scores = query @ key.transpose(-2, -1)
    scores = scores / math.sqrt(query.shape[-1])

    if is_causal:
        if query.shape[-2] != key.shape[-2]:
            raise ValueError("causal attention requires equal query and key sequence lengths")
        mask = make_causal_mask(query.shape[-2], device=query.device)
        scores = scores.masked_fill(~mask, float("-inf"))

    probabilities = torch.softmax(scores, dim=-1)
    probabilities = functional.dropout(probabilities, p=dropout_p, training=training)
    return probabilities @ value, probabilities


class MultiHeadSelfAttention(nn.Module):
    """Apply causal self-attention in parallel heads and merge their outputs.

    Input and output shape: ``(batch, sequence, embedding_dim)``.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads
        self.context_length = config.context_length
        self.dropout_probability = config.dropout

        self.qkv_projection = nn.Linear(config.embedding_dim, 3 * config.embedding_dim)
        self.output_projection = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.output_dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(
                "attention inputs must have shape (batch, sequence, embedding_dim), "
                f"got {tuple(inputs.shape)}"
            )
        batch_size, sequence_length, embedding_dim = inputs.shape
        if embedding_dim != self.embedding_dim:
            raise ValueError(
                f"expected embedding dimension {self.embedding_dim}, got {embedding_dim}"
            )
        if sequence_length > self.context_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds context length {self.context_length}"
            )

        query_key_value = self.qkv_projection(inputs)
        query_key_value = query_key_value.reshape(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        query, key, value = query_key_value.unbind(dim=0)

        attended, _ = scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            dropout_p=self.dropout_probability,
            training=self.training,
        )
        merged = attended.transpose(1, 2).contiguous().reshape(
            batch_size,
            sequence_length,
            self.embedding_dim,
        )
        return self.output_dropout(self.output_projection(merged))
