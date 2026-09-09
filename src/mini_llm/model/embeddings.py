"""Token and learned positional embeddings for a future decoder model."""

import torch
from torch import nn

from mini_llm.config import ModelConfig


class TokenPositionEmbedding(nn.Module):
    """Combine token identity and learned absolute position information.

    Input shape: ``(batch, sequence)`` integer token IDs.
    Output shape: ``(batch, sequence, embedding_dim)``.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.context_length = config.context_length
        self.embedding_dim = config.embedding_dim
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError(
                "token_ids must have shape (batch, sequence), "
                f"got {tuple(token_ids.shape)}"
            )
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"token_ids must use an integer dtype, got {token_ids.dtype}")

        sequence_length = token_ids.shape[1]
        if sequence_length == 0:
            raise ValueError("token_ids sequence must not be empty")
        if sequence_length > self.context_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds context length {self.context_length}"
            )

        positions = torch.arange(sequence_length, device=token_ids.device)
        token_vectors = self.token_embedding(token_ids)
        position_vectors = self.position_embedding(positions)
        return self.dropout(token_vectors + position_vectors.unsqueeze(0))
