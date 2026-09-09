"""Deterministic PyTorch training sanity check used before model development."""

from dataclasses import dataclass

import torch
from torch import nn

from mini_llm.config import RuntimeConfig, SanityTrainingConfig
from mini_llm.runtime import resolve_device, seed_everything


class TinyRegressionModel(nn.Module):
    """One linear layer used only to verify PyTorch learning mechanics."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


@dataclass(frozen=True, slots=True)
class SanityTrainingResult:
    """Observable results from a completed sanity-training run."""

    initial_loss: float
    final_loss: float
    learned_weight: float
    learned_bias: float
    steps: int
    device: str


def make_tiny_regression_problem(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Create the fixed relation ``y = 2x + 1`` directly on the target device."""

    inputs = torch.linspace(-1.0, 1.0, steps=9, device=device).unsqueeze(-1)
    targets = 2.0 * inputs + 1.0
    return inputs, targets


def run_training_sanity(
    training: SanityTrainingConfig | None = None,
    runtime: RuntimeConfig | None = None,
) -> SanityTrainingResult:
    """Train a one-layer model and return evidence that gradient descent worked."""

    training = training or SanityTrainingConfig()
    runtime = runtime or RuntimeConfig()
    seed_everything(
        runtime.seed,
        deterministic_algorithms=runtime.deterministic_algorithms,
    )
    device = resolve_device(runtime.device)

    model = TinyRegressionModel().to(device)
    inputs, targets = make_tiny_regression_problem(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=training.learning_rate)

    model.train()
    with torch.no_grad():
        initial_loss = loss_function(model(inputs), targets).item()

    for _ in range(training.steps):
        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        loss = loss_function(predictions, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_loss = loss_function(model(inputs), targets).item()

    return SanityTrainingResult(
        initial_loss=initial_loss,
        final_loss=final_loss,
        learned_weight=model.linear.weight.detach().item(),
        learned_bias=model.linear.bias.detach().item(),
        steps=training.steps,
        device=str(device),
    )


def main() -> None:
    """Run the sanity check from the command line."""

    result = run_training_sanity()
    print(f"device: {result.device}")
    print(f"initial loss: {result.initial_loss:.6f}")
    print(f"final loss: {result.final_loss:.6f}")
    print(f"learned relation: y = {result.learned_weight:.4f}x + {result.learned_bias:.4f}")


if __name__ == "__main__":
    main()
