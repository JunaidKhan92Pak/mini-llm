import pytest
import torch
from torch import nn

from mini_llm.config import ModelConfig, development_model_config
from mini_llm.model import FeedForward, TokenPositionEmbedding, TransformerBlock
from mini_llm.runtime import seed_everything


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


@pytest.mark.parametrize(("batch_size", "sequence_length"), [(1, 1), (2, 4), (3, 6)])
def test_transformer_block_preserves_shape_and_finite_outputs(
    model_config: ModelConfig,
    batch_size: int,
    sequence_length: int,
) -> None:
    block = TransformerBlock(model_config)
    inputs = torch.randn(batch_size, sequence_length, model_config.embedding_dim)

    output = block(inputs)

    assert output.shape == inputs.shape
    assert torch.isfinite(output).all()


def test_transformer_block_backward_reaches_all_parameters(model_config: ModelConfig) -> None:
    block = TransformerBlock(model_config)
    inputs = torch.randn(2, 4, model_config.embedding_dim, requires_grad=True)

    block(inputs).square().mean().backward()

    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert all(parameter.grad is not None for parameter in block.parameters())
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in block.parameters()
        if parameter.grad is not None
    )


def test_zeroed_sublayers_prove_residual_identity_path(model_config: ModelConfig) -> None:
    block = TransformerBlock(model_config)
    for parameter in block.parameters():
        nn.init.zeros_(parameter)
    inputs = torch.randn(2, 4, model_config.embedding_dim)

    output = block(inputs)

    assert torch.equal(output, inputs)


def test_residual_addition_rejects_wrong_sublayer_shape(model_config: ModelConfig) -> None:
    class WrongShape(nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs[..., :1]

    block = TransformerBlock(model_config)
    block.attention = WrongShape()

    with pytest.raises(RuntimeError, match="cannot be added to residual shape"):
        block(torch.randn(2, 4, model_config.embedding_dim))


def test_complete_block_remains_causal(model_config: ModelConfig) -> None:
    seed_everything(42)
    block = TransformerBlock(model_config).eval()
    original = torch.randn(1, 4, model_config.embedding_dim)
    changed_future = original.clone()
    changed_future[:, 2:] = 1000.0

    original_output = block(original)
    changed_output = block(changed_future)

    assert torch.equal(original_output[:, :2], changed_output[:, :2])
    assert not torch.equal(original_output[:, 2:], changed_output[:, 2:])


def test_dropout_changes_train_outputs_but_eval_is_deterministic() -> None:
    config = ModelConfig(
        vocab_size=8,
        context_length=6,
        embedding_dim=16,
        num_layers=1,
        num_heads=4,
        feed_forward_dim=32,
        dropout=0.5,
    )
    block = TransformerBlock(config)
    inputs = torch.randn(2, 4, config.embedding_dim)

    block.train()
    seed_everything(1)
    first_train_output = block(inputs)
    seed_everything(2)
    second_train_output = block(inputs)
    block.eval()
    first_eval_output = block(inputs)
    second_eval_output = block(inputs)

    assert not torch.equal(first_train_output, second_train_output)
    assert torch.equal(first_eval_output, second_eval_output)


def test_debug_embeddings_to_transformer_block_integration() -> None:
    config = development_model_config(vocab_size=20)
    embeddings = TokenPositionEmbedding(config)
    block = TransformerBlock(config)
    token_ids = torch.tensor([[1, 4, 5, 2], [1, 6, 7, 2]])

    embedded = embeddings(token_ids)
    output = block(embedded)

    assert embedded.shape == (2, 4, config.embedding_dim)
    assert output.shape == embedded.shape
    assert output.device == token_ids.device
    assert torch.isfinite(output).all()


def test_block_parameter_breakdown_matches_debug_configuration() -> None:
    block = TransformerBlock(development_model_config(vocab_size=20))

    breakdown = block.parameter_breakdown()

    assert breakdown == {
        "normalization": 256,
        "attention": 16_640,
        "feed_forward": 33_088,
    }
    assert block.parameter_count() == 49_984
    assert block.parameter_count() == sum(
        parameter.numel() for parameter in block.parameters() if parameter.requires_grad
    )


def test_block_rejects_invalid_inputs_and_configuration(model_config: ModelConfig) -> None:
    block = TransformerBlock(model_config)

    with pytest.raises(ValueError, match="shape"):
        block(torch.randn(2, model_config.embedding_dim))
    with pytest.raises(ValueError, match="expected embedding dimension"):
        block(torch.randn(2, 4, model_config.embedding_dim + 1))
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(
            vocab_size=8,
            context_length=6,
            embedding_dim=15,
            num_layers=1,
            num_heads=4,
            feed_forward_dim=32,
        )
