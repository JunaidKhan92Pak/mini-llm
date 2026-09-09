import pytest
import torch

from mini_llm.config import ModelConfig
from mini_llm.model import (
    MultiHeadSelfAttention,
    make_causal_mask,
    scaled_dot_product_attention,
)
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


def test_causal_mask_has_exact_lower_triangular_pattern() -> None:
    mask = make_causal_mask(4, device=torch.device("cpu"))
    expected = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )

    assert mask.dtype == torch.bool
    assert torch.equal(mask, expected)


def test_scaled_attention_probabilities_are_normalized_and_future_is_zero() -> None:
    query = torch.eye(3).reshape(1, 1, 3, 3)
    key = query.clone()
    value = torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]]])

    output, probabilities = scaled_dot_product_attention(query, key, value)

    assert output.shape == query.shape
    assert probabilities.shape == (1, 1, 3, 3)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones((1, 1, 3)))
    assert torch.count_nonzero(probabilities.triu(diagonal=1)) == 0


def test_multi_head_attention_preserves_shape_and_supports_backward(
    model_config: ModelConfig,
) -> None:
    attention = MultiHeadSelfAttention(model_config)
    inputs = torch.randn(2, 5, model_config.embedding_dim, requires_grad=True)

    output = attention(inputs)
    output.square().mean().backward()

    assert output.shape == inputs.shape
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert all(parameter.grad is not None for parameter in attention.parameters())


def test_future_input_cannot_change_past_attention_outputs(model_config: ModelConfig) -> None:
    seed_everything(42)
    attention = MultiHeadSelfAttention(model_config).eval()
    original = torch.randn(1, 4, model_config.embedding_dim)
    changed_future = original.clone()
    changed_future[:, -1] = 1000.0

    original_output = attention(original)
    changed_output = attention(changed_future)

    assert torch.equal(original_output[:, :-1], changed_output[:, :-1])
    assert not torch.equal(original_output[:, -1], changed_output[:, -1])


def test_attention_rejects_bad_shapes_and_excess_context(model_config: ModelConfig) -> None:
    attention = MultiHeadSelfAttention(model_config)

    with pytest.raises(ValueError, match="shape"):
        attention(torch.randn(2, model_config.embedding_dim))
    with pytest.raises(ValueError, match="expected embedding dimension"):
        attention(torch.randn(2, 3, model_config.embedding_dim + 1))
    with pytest.raises(ValueError, match="exceeds context length"):
        attention(
            torch.randn(1, model_config.context_length + 1, model_config.embedding_dim)
        )


def test_scaled_attention_rejects_incompatible_shapes() -> None:
    query = torch.randn(1, 2, 3, 4)
    key = torch.randn(1, 2, 3, 5)
    value = torch.randn(1, 2, 3, 4)

    with pytest.raises(ValueError, match="query and key head dimensions"):
        scaled_dot_product_attention(query, key, value)

    with pytest.raises(ValueError, match="equal query and key sequence lengths"):
        scaled_dot_product_attention(
            torch.randn(1, 2, 2, 4),
            torch.randn(1, 2, 3, 4),
            torch.randn(1, 2, 3, 4),
        )


def test_causal_mask_rejects_empty_sequence() -> None:
    with pytest.raises(ValueError, match="positive"):
        make_causal_mask(0, device=torch.device("cpu"))
