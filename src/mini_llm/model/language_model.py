"""The first complete decoder-only JeePeeTee language model."""

import torch
from torch import nn
from torch.nn import functional as functional

from mini_llm.config import ModelConfig
from mini_llm.model.blocks import TransformerBlock
from mini_llm.model.embeddings import TokenPositionEmbedding


def prepare_next_token_batch(token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift token sequences into aligned inputs and next-token targets.

    Input shape: ``(batch, sequence + 1)``.
    Returned shapes: two tensors of ``(batch, sequence)``.
    """

    if token_ids.ndim != 2:
        raise ValueError(
            f"token_ids must have shape (batch, sequence), got {tuple(token_ids.shape)}"
        )
    if token_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"token_ids must use an integer dtype, got {token_ids.dtype}")
    if token_ids.shape[1] < 2:
        raise ValueError("next-token prediction requires sequences containing at least two tokens")
    return token_ids[:, :-1], token_ids[:, 1:]


def causal_language_model_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute cross-entropy over aligned ``(batch, sequence)`` next-token targets."""

    if logits.ndim != 3:
        raise ValueError(
            "logits must have shape (batch, sequence, vocabulary), "
            f"got {tuple(logits.shape)}"
        )
    if targets.ndim != 2:
        raise ValueError(
            f"targets must have shape (batch, sequence), got {tuple(targets.shape)}"
        )
    if logits.shape[:2] != targets.shape:
        raise ValueError(
            "logit and target batch/sequence dimensions must match, got "
            f"{tuple(logits.shape[:2])} and {tuple(targets.shape)}"
        )
    if targets.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"targets must use an integer dtype, got {targets.dtype}")

    valid_targets = targets[targets != ignore_index]
    if valid_targets.numel() == 0:
        raise ValueError("at least one target must not be ignored")
    if valid_targets.min() < 0 or valid_targets.max() >= logits.shape[-1]:
        raise ValueError("targets contain token IDs outside the logits vocabulary")

    return functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1).long(),
        ignore_index=ignore_index,
    )


class JeePeeTee(nn.Module):
    """A configurable decoder-only Transformer that returns vocabulary logits.

    Input shape: ``(batch, sequence)`` token IDs.
    Output shape: ``(batch, sequence, vocab_size)`` logits.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = TokenPositionEmbedding(config)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.num_layers)
        )
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.language_model_head = nn.Linear(
            config.embedding_dim,
            config.vocab_size,
            bias=False,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        self._validate_token_ids(token_ids)
        hidden = self.embeddings(token_ids)
        for block in self.blocks:
            hidden = block(hidden)
        return self.language_model_head(self.final_norm(hidden))

    def _validate_token_ids(self, token_ids: torch.Tensor) -> None:
        if token_ids.ndim != 2:
            raise ValueError(
                f"token_ids must have shape (batch, sequence), got {tuple(token_ids.shape)}"
            )
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"token_ids must use an integer dtype, got {token_ids.dtype}")
        if token_ids.shape[0] == 0:
            raise ValueError("token_ids batch must not be empty")
        if token_ids.shape[1] == 0:
            raise ValueError("token_ids sequence must not be empty")
        if token_ids.shape[1] > self.config.context_length:
            raise ValueError(
                f"sequence length {token_ids.shape[1]} exceeds context length "
                f"{self.config.context_length}"
            )
        if token_ids.min() < 0 or token_ids.max() >= self.config.vocab_size:
            raise ValueError(
                f"token IDs must be within [0, {self.config.vocab_size - 1}]"
            )

    def parameter_breakdown(self) -> dict[str, int]:
        """Return trainable parameter counts for the model's major components."""

        modules = {
            "embeddings": self.embeddings,
            "transformer_blocks": self.blocks,
            "final_normalization": self.final_norm,
            "language_model_head": self.language_model_head,
        }
        return {
            name: sum(
                parameter.numel()
                for parameter in module.parameters()
                if parameter.requires_grad
            )
            for name, module in modules.items()
        }

    def parameter_counts(self) -> dict[str, int]:
        """Return total and trainable scalar parameter counts."""

        return {
            "total": sum(parameter.numel() for parameter in self.parameters()),
            "trainable": sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
        }

    def parameter_count(self) -> int:
        """Return the trainable count retained for backwards compatibility."""

        return self.parameter_counts()["trainable"]
