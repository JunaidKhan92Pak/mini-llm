"""Feed-forward and pre-normalized Transformer blocks."""

import torch
from torch import nn

from mini_llm.config import ModelConfig
from mini_llm.model.attention import MultiHeadSelfAttention


class FeedForward(nn.Module):
    """Apply a position-wise two-layer MLP without mixing sequence positions."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.embedding_dim, config.feed_forward_dim),
            nn.GELU(),
            nn.Linear(config.feed_forward_dim, config.embedding_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class TransformerBlock(nn.Module):
    """Apply pre-norm causal attention and MLP sublayers with residual paths."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.embedding_dim)
        self.attention = MultiHeadSelfAttention(config)
        self.feed_forward_norm = nn.LayerNorm(config.embedding_dim)
        self.feed_forward = FeedForward(config)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs + self.attention(self.attention_norm(inputs))
        return hidden + self.feed_forward(self.feed_forward_norm(hidden))
