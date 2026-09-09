"""Leakage-aware token-stream splitting and next-token window batching."""

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset

from mini_llm.config import DataConfig, ModelConfig, TrainingConfig


class NextTokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Create aligned input/target windows from one contiguous token stream."""

    def __init__(self, token_ids: torch.Tensor, context_length: int, *, stride: int = 1) -> None:
        if token_ids.ndim != 1:
            raise ValueError(f"token_ids must be one-dimensional, got {tuple(token_ids.shape)}")
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"token_ids must use an integer dtype, got {token_ids.dtype}")
        if context_length <= 0:
            raise ValueError(f"context_length must be positive, got {context_length}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        if token_ids.numel() < context_length + 1:
            raise ValueError(
                f"token stream needs at least {context_length + 1} tokens, "
                f"got {token_ids.numel()}"
            )

        self.token_ids = token_ids.detach().clone()
        self.context_length = context_length
        self.starts = tuple(range(0, token_ids.numel() - context_length, stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[index]
        stop = start + self.context_length
        return self.token_ids[start:stop], self.token_ids[start + 1 : stop + 1]


def split_token_stream(
    token_ids: torch.Tensor,
    *,
    validation_fraction: float,
    context_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split contiguously before windowing so no window crosses train/validation data."""

    if token_ids.ndim != 1:
        raise ValueError(f"token_ids must be one-dimensional, got {tuple(token_ids.shape)}")
    if token_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"token_ids must use an integer dtype, got {token_ids.dtype}")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0.0 and 1.0")
    if context_length <= 0:
        raise ValueError(f"context_length must be positive, got {context_length}")
    minimum_split_size = context_length + 1
    if token_ids.numel() < 2 * minimum_split_size:
        raise ValueError(
            "token stream is too short for independent train and validation windows: "
            f"need at least {2 * minimum_split_size}, got {token_ids.numel()}"
        )

    validation_size = max(minimum_split_size, round(token_ids.numel() * validation_fraction))
    validation_size = min(validation_size, token_ids.numel() - minimum_split_size)
    split_index = token_ids.numel() - validation_size
    return token_ids[:split_index].clone(), token_ids[split_index:].clone()


@dataclass(frozen=True, slots=True)
class DataLoaders:
    """Training and validation loaders plus their isolated token streams."""

    train: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    validation: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    train_tokens: torch.Tensor
    validation_tokens: torch.Tensor


def build_dataloaders(
    token_ids: torch.Tensor,
    model_config: ModelConfig,
    data_config: DataConfig,
    training_config: TrainingConfig,
    *,
    seed: int,
) -> DataLoaders:
    """Build deterministic single-process loaders from one local token stream."""

    if token_ids.numel() > 0 and (
        token_ids.min() < 0 or token_ids.max() >= model_config.vocab_size
    ):
        raise ValueError("token stream contains IDs outside the configured vocabulary")
    train_tokens, validation_tokens = split_token_stream(
        token_ids,
        validation_fraction=data_config.validation_fraction,
        context_length=model_config.context_length,
    )
    train_dataset = NextTokenDataset(
        train_tokens,
        model_config.context_length,
        stride=data_config.stride,
    )
    validation_dataset = NextTokenDataset(
        validation_tokens,
        model_config.context_length,
        stride=data_config.stride,
    )
    if len(train_dataset) < training_config.batch_size:
        raise ValueError(
            "training dataset has fewer windows than batch_size: "
            f"{len(train_dataset)} < {training_config.batch_size}"
        )

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    return DataLoaders(
        train=train_loader,
        validation=validation_loader,
        train_tokens=train_tokens,
        validation_tokens=validation_tokens,
    )
