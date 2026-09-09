import pytest
import torch

from mini_llm.config import ModelConfig
from mini_llm.model import TokenPositionEmbedding
from mini_llm.runtime import seed_everything


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=16,
        context_length=8,
        embedding_dim=12,
        num_layers=2,
        num_heads=3,
        feed_forward_dim=48,
        dropout=0.0,
    )


def test_token_position_embedding_has_expected_shape_and_device(
    model_config: ModelConfig,
) -> None:
    module = TokenPositionEmbedding(model_config)
    token_ids = torch.tensor([[1, 2, 3], [3, 2, 1]], dtype=torch.long)

    output = module(token_ids)

    assert output.shape == (2, 3, model_config.embedding_dim)
    assert output.device == token_ids.device


def test_positions_change_repeated_token_representations(model_config: ModelConfig) -> None:
    seed_everything(42)
    module = TokenPositionEmbedding(model_config)
    repeated_token_ids = torch.full((1, 3), 5, dtype=torch.long)

    output = module(repeated_token_ids)

    assert not torch.equal(output[:, 0], output[:, 1])


def test_embedding_backward_reaches_token_and_position_tables(model_config: ModelConfig) -> None:
    module = TokenPositionEmbedding(model_config)
    token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    module(token_ids).sum().backward()

    assert module.token_embedding.weight.grad is not None
    assert module.position_embedding.weight.grad is not None
    assert torch.count_nonzero(module.token_embedding.weight.grad) > 0
    assert torch.count_nonzero(module.position_embedding.weight.grad) > 0


def test_embedding_rejects_invalid_shape_dtype_and_context(model_config: ModelConfig) -> None:
    module = TokenPositionEmbedding(model_config)

    with pytest.raises(ValueError, match="shape"):
        module(torch.ones(3, dtype=torch.long))
    with pytest.raises(TypeError, match="integer dtype"):
        module(torch.ones((1, 3), dtype=torch.float32))
    with pytest.raises(ValueError, match="exceeds context length"):
        module(torch.ones((1, model_config.context_length + 1), dtype=torch.long))
