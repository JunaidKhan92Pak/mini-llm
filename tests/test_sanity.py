import pytest
import torch
from torch import nn

from mini_llm.config import RuntimeConfig, SanityTrainingConfig
from mini_llm.runtime import resolve_device, seed_everything
from mini_llm.sanity import (
    TinyRegressionModel,
    make_tiny_regression_problem,
    run_training_sanity,
)


def test_tiny_problem_has_expected_tensor_values_shapes_and_device() -> None:
    device = torch.device("cpu")

    inputs, targets = make_tiny_regression_problem(device)

    assert inputs.shape == (9, 1)
    assert targets.shape == (9, 1)
    assert inputs.dtype == torch.float32
    assert targets.device == device
    assert torch.equal(targets, 2.0 * inputs + 1.0)


def test_forward_pass_and_loss_are_scalar_and_finite() -> None:
    seed_everything(42)
    model = TinyRegressionModel()
    inputs, targets = make_tiny_regression_problem(torch.device("cpu"))

    predictions = model(inputs)
    loss = nn.functional.mse_loss(predictions, targets)

    assert predictions.shape == targets.shape
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_backward_pass_produces_finite_nonzero_gradients() -> None:
    seed_everything(42)
    model = TinyRegressionModel()
    inputs, targets = make_tiny_regression_problem(torch.device("cpu"))

    nn.functional.mse_loss(model(inputs), targets).backward()

    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients if gradient is not None)


def test_optimizer_step_updates_parameters() -> None:
    seed_everything(42)
    model = TinyRegressionModel()
    inputs, targets = make_tiny_regression_problem(torch.device("cpu"))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    optimizer.zero_grad(set_to_none=True)
    nn.functional.mse_loss(model(inputs), targets).backward()
    optimizer.step()

    assert all(
        not torch.equal(old_value, new_value)
        for old_value, new_value in zip(before, model.parameters(), strict=True)
    )


def test_training_sanity_is_reproducible_and_learns_tiny_problem() -> None:
    training = SanityTrainingConfig(steps=100, learning_rate=0.1)
    runtime = RuntimeConfig(seed=42, device="cpu", deterministic_algorithms=True)

    first = run_training_sanity(training, runtime)
    second = run_training_sanity(training, runtime)

    assert first == second
    assert first.final_loss < first.initial_loss * 0.001
    assert first.learned_weight == pytest.approx(2.0, abs=0.01)
    assert first.learned_bias == pytest.approx(1.0, abs=0.01)
    assert first.device == "cpu"


def test_training_sanity_uses_resolved_auto_device() -> None:
    result = run_training_sanity(
        SanityTrainingConfig(steps=1),
        RuntimeConfig(device="auto"),
    )

    assert result.device == str(resolve_device("auto"))
