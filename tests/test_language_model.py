import pytest
import torch
from torch.nn import functional as functional

from mini_llm.config import DebugOverfitConfig, ModelConfig, RuntimeConfig
from mini_llm.debug_training import run_tiny_overfit
from mini_llm.model import (
    JeePeeTee,
    causal_language_model_loss,
    prepare_next_token_batch,
)
from mini_llm.runtime import seed_everything


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=8,
        context_length=4,
        embedding_dim=16,
        num_layers=1,
        num_heads=4,
        feed_forward_dim=32,
        dropout=0.0,
    )


def test_next_token_shift_is_exact() -> None:
    token_ids = torch.tensor([[1, 4, 5, 2], [1, 6, 7, 2]])

    inputs, targets = prepare_next_token_batch(token_ids)

    assert torch.equal(inputs, torch.tensor([[1, 4, 5], [1, 6, 7]]))
    assert torch.equal(targets, torch.tensor([[4, 5, 2], [6, 7, 2]]))


def test_model_returns_vocabulary_logits_and_full_backward(
    model_config: ModelConfig,
) -> None:
    model = JeePeeTee(model_config)
    token_ids = torch.tensor([[1, 4, 5, 2], [1, 6, 7, 2]])
    inputs, targets = prepare_next_token_batch(token_ids)

    logits = model(inputs)
    loss = causal_language_model_loss(logits, targets)
    loss.backward()

    assert logits.shape == (2, 3, model_config.vocab_size)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def test_language_model_loss_matches_direct_cross_entropy_and_ignores_padding() -> None:
    logits = torch.tensor(
        [[[3.0, 1.0, 0.0], [0.0, 1.0, 3.0], [1.0, 2.0, 0.0]]]
    )
    targets = torch.tensor([[0, 2, -100]])

    actual = causal_language_model_loss(logits, targets)
    expected = functional.cross_entropy(logits[:, :2].reshape(-1, 3), targets[:, :2].reshape(-1))

    assert torch.allclose(actual, expected)
    assert torch.allclose(
        causal_language_model_loss(logits, targets.to(torch.int32)),
        expected,
    )


def test_model_logits_are_causal(model_config: ModelConfig) -> None:
    seed_everything(42)
    model = JeePeeTee(model_config).eval()
    original = torch.tensor([[1, 4, 5, 6]])
    changed_future = torch.tensor([[1, 4, 5, 7]])

    original_logits = model(original)
    changed_logits = model(changed_future)

    assert torch.equal(original_logits[:, :-1], changed_logits[:, :-1])
    assert not torch.equal(original_logits[:, -1], changed_logits[:, -1])


def test_tiny_decoder_deterministically_overfits_one_batch(model_config: ModelConfig) -> None:
    token_batch = torch.tensor([[1, 4, 5, 6, 2], [1, 4, 5, 6, 2]])
    training = DebugOverfitConfig(steps=100, learning_rate=0.02)
    runtime = RuntimeConfig(seed=42, device="cpu", deterministic_algorithms=True)

    first = run_tiny_overfit(model_config, token_batch, training, runtime)
    second = run_tiny_overfit(model_config, token_batch, training, runtime)

    assert first == second
    assert first.final_loss < first.initial_loss * 0.01
    assert first.final_loss < 0.02
    assert first.parameter_count > 0
    assert first.device == "cpu"


def test_objective_and_debug_training_reject_invalid_inputs(
    model_config: ModelConfig,
) -> None:
    with pytest.raises(ValueError, match="at least two tokens"):
        prepare_next_token_batch(torch.tensor([[1]]))
    with pytest.raises(ValueError, match="batch/sequence dimensions"):
        causal_language_model_loss(torch.randn(1, 3, 8), torch.ones((1, 2), dtype=torch.long))
    with pytest.raises(ValueError, match="at least one target"):
        causal_language_model_loss(
            torch.randn(1, 2, 8),
            torch.full((1, 2), -100, dtype=torch.long),
        )
    with pytest.raises(ValueError, match="outside the configured vocabulary"):
        run_tiny_overfit(model_config, torch.tensor([[1, 4, 8]]))
