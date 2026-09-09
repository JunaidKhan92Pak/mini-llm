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
        self.embedding_dim = config.embedding_dim
        self.attention_norm = nn.LayerNorm(config.embedding_dim)
        self.attention = MultiHeadSelfAttention(config)
        self.feed_forward_norm = nn.LayerNorm(config.embedding_dim)
        self.feed_forward = FeedForward(config)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(
                "TransformerBlock inputs must have shape (batch, sequence, embedding_dim), "
                f"got {tuple(inputs.shape)}"
            )
        if inputs.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"expected embedding dimension {self.embedding_dim}, got {inputs.shape[-1]}"
            )

        attention_output = self.attention(self.attention_norm(inputs))
        hidden = self._add_residual(inputs, attention_output, sublayer="attention")
        feed_forward_output = self.feed_forward(self.feed_forward_norm(hidden))
        return self._add_residual(hidden, feed_forward_output, sublayer="feed-forward")

    @staticmethod
    def _add_residual(
        inputs: torch.Tensor,
        update: torch.Tensor,
        *,
        sublayer: str,
    ) -> torch.Tensor:
        if update.shape != inputs.shape:
            raise RuntimeError(
                f"{sublayer} output shape {tuple(update.shape)} cannot be added to "
                f"residual shape {tuple(inputs.shape)}"
            )
        return inputs + update

    def parameter_breakdown(self) -> dict[str, int]:
        """Return trainable parameter counts for the block's major components."""

        normalization = sum(
            parameter.numel()
            for module in (self.attention_norm, self.feed_forward_norm)
            for parameter in module.parameters()
            if parameter.requires_grad
        )
        attention = sum(
            parameter.numel()
            for parameter in self.attention.parameters()
            if parameter.requires_grad
        )
        feed_forward = sum(
            parameter.numel()
            for parameter in self.feed_forward.parameters()
            if parameter.requires_grad
        )
        return {
            "normalization": normalization,
            "attention": attention,
            "feed_forward": feed_forward,
        }

    def parameter_count(self) -> int:
        """Return the block's total number of trainable scalar parameters."""

        return sum(self.parameter_breakdown().values())
