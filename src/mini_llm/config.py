"""Validated configuration objects for Mini LLM experiments."""

from dataclasses import dataclass
from typing import Literal

DevicePreference = Literal["auto", "cpu", "cuda"]


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Architecture dimensions shared by future model components."""

    vocab_size: int
    context_length: int
    embedding_dim: int
    num_layers: int
    num_heads: int
    feed_forward_dim: int
    dropout: float = 0.0

    def __post_init__(self) -> None:
        positive_fields = {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "embedding_dim": self.embedding_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "feed_forward_dim": self.feed_forward_dim,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads, got "
                f"{self.embedding_dim} and {self.num_heads}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}")


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Runtime choices that affect device selection and reproducibility."""

    seed: int = 42
    device: DevicePreference = "auto"
    deterministic_algorithms: bool = False

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError(f"device must be 'auto', 'cpu', or 'cuda', got {self.device!r}")


@dataclass(frozen=True, slots=True)
class SanityTrainingConfig:
    """Settings for the tiny deterministic PyTorch learning check."""

    steps: int = 100
    learning_rate: float = 0.1

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError(f"steps must be positive, got {self.steps}")
        if self.learning_rate <= 0.0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )


@dataclass(frozen=True, slots=True)
class DebugOverfitConfig:
    """Settings for the one-batch language-model overfit gate."""

    steps: int = 100
    learning_rate: float = 0.02
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError(f"steps must be positive, got {self.steps}")
        if self.learning_rate <= 0.0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")


def development_model_config(vocab_size: int) -> ModelConfig:
    """Return the deliberately small CPU-friendly configuration for early validation."""

    return ModelConfig(
        vocab_size=vocab_size,
        context_length=64,
        embedding_dim=64,
        num_layers=2,
        num_heads=4,
        feed_forward_dim=256,
        dropout=0.0,
    )
