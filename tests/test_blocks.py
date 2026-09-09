import pytest
import torch

from mini_llm.config import ModelConfig
from mini_llm.model import FeedForward, TransformerBlock


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=8,
        context_length=6,
        embedding_dim=16,
        num_layers=2,
        num_heads=4,
        feed_forward_dim=32,
        dropout=0.0,
    )


def test_feed_forward_preserves_shape_and_supports_backward(
    model_config: ModelConfig,
) -> None:
    module = FeedForward(model_config)
    inputs = torch.randn(2, 4, model_config.embedding_dim, requires_grad=True)

    output = module(inputs)
    output.sum().backward()

    assert output.shape == inputs.shape
    assert inputs.grad is not None
    assert all(parameter.grad is not None for parameter in module.parameters())


def test_transformer_block_preserves_shape(model_config: ModelConfig) -> None:
    block = TransformerBlock(model_config)
    inputs = torch.randn(2, 4, model_config.embedding_dim)

    assert block(inputs).shape == inputs.shape


def test_zeroed_sublayers_prove_residual_identity_path(model_config: ModelConfig) -> None:
    block = TransformerBlock(model_config)
    for parameter in block.parameters():
        torch.nn.init.zeros_(parameter)
    inputs = torch.randn(2, 4, model_config.embedding_dim)

    output = block(inputs)

    assert torch.equal(output, inputs)
