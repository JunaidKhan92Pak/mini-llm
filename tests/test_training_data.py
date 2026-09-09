import pytest
import torch

from mini_llm.config import DataConfig, ModelConfig, TrainingConfig
from mini_llm.training import NextTokenDataset, build_dataloaders, split_token_stream


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        context_length=4,
        embedding_dim=16,
        num_layers=1,
        num_heads=4,
        feed_forward_dim=32,
        dropout=0.0,
    )


def test_next_token_windows_are_shifted_exactly() -> None:
    dataset = NextTokenDataset(torch.arange(8), context_length=4, stride=2)

    assert len(dataset) == 2
    first_inputs, first_targets = dataset[0]
    second_inputs, second_targets = dataset[1]
    assert torch.equal(first_inputs, torch.tensor([0, 1, 2, 3]))
    assert torch.equal(first_targets, torch.tensor([1, 2, 3, 4]))
    assert torch.equal(second_inputs, torch.tensor([2, 3, 4, 5]))
    assert torch.equal(second_targets, torch.tensor([3, 4, 5, 6]))


def test_stream_is_split_before_windows_without_boundary_overlap() -> None:
    tokens = torch.arange(20)

    train_tokens, validation_tokens = split_token_stream(
        tokens,
        validation_fraction=0.25,
        context_length=4,
    )

    assert torch.equal(torch.cat((train_tokens, validation_tokens)), tokens)
    assert train_tokens[-1] + 1 == validation_tokens[0]
    assert set(train_tokens.tolist()).isdisjoint(validation_tokens.tolist())


def test_dataloader_shuffle_is_reproducible(model_config: ModelConfig) -> None:
    tokens = torch.arange(40) % model_config.vocab_size
    data_config = DataConfig(validation_fraction=0.25, stride=1)
    training_config = TrainingConfig(batch_size=4)

    first = build_dataloaders(tokens, model_config, data_config, training_config, seed=7)
    second = build_dataloaders(tokens, model_config, data_config, training_config, seed=7)
    first_batch = next(iter(first.train))
    second_batch = next(iter(second.train))

    assert torch.equal(first_batch[0], second_batch[0])
    assert torch.equal(first_batch[1], second_batch[1])
    assert torch.equal(first.train_tokens, second.train_tokens)
    assert torch.equal(first.validation_tokens, second.validation_tokens)


def test_data_pipeline_rejects_short_or_invalid_streams(model_config: ModelConfig) -> None:
    with pytest.raises(ValueError, match="too short"):
        split_token_stream(
            torch.arange(9),
            validation_fraction=0.2,
            context_length=model_config.context_length,
        )
    with pytest.raises(ValueError, match="at least"):
        NextTokenDataset(torch.arange(4), context_length=4)
    with pytest.raises(TypeError, match="integer dtype"):
        NextTokenDataset(torch.arange(8, dtype=torch.float32), context_length=4)
    with pytest.raises(ValueError, match="context_length"):
        split_token_stream(
            torch.arange(10),
            validation_fraction=0.2,
            context_length=0,
        )
    with pytest.raises(ValueError, match="outside the configured vocabulary"):
        build_dataloaders(
            torch.arange(40),
            model_config,
            DataConfig(validation_fraction=0.2),
            TrainingConfig(batch_size=2),
            seed=42,
        )
