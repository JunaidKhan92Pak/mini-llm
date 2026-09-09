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


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Settings for splitting a token stream into local training windows."""

    validation_fraction: float = 0.1
    stride: int = 1

    def __post_init__(self) -> None:
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError(
                "validation_fraction must be between 0.0 and 1.0, got "
                f"{self.validation_fraction}"
            )
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Baseline single-device training settings."""

    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_steps: int = 100
    evaluation_interval: int = 10
    gradient_clip_norm: float | None = 1.0

    def __post_init__(self) -> None:
        positive_fields = {
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "max_steps": self.max_steps,
            "evaluation_interval": self.evaluation_interval,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0.0:
            raise ValueError(
                "gradient_clip_norm must be positive or None, got "
                f"{self.gradient_clip_norm}"
            )


def development_model_config(vocab_size: int) -> ModelConfig:
    """Return the deliberately small CPU-friendly configuration for early validation."""

    return debug_model_config(vocab_size)


def debug_model_config(vocab_size: int, *, context_length: int = 64) -> ModelConfig:
    """Return the smallest maintained architecture preset."""

    return ModelConfig(
        vocab_size=vocab_size,
        context_length=context_length,
        embedding_dim=64,
        num_layers=2,
        num_heads=4,
        feed_forward_dim=256,
        dropout=0.0,
    )


def mini_model_config(vocab_size: int, *, context_length: int = 64) -> ModelConfig:
    """Return the Phase 12 architecture preset near one million parameters."""

    return ModelConfig(
        vocab_size=vocab_size,
        context_length=context_length,
        embedding_dim=96,
        num_layers=4,
        num_heads=4,
        feed_forward_dim=384,
        dropout=0.1,
    )


def five_million_model_config(
    vocab_size: int,
    *,
    context_length: int = 128,
) -> ModelConfig:
    """Return the hardware-friendly Phase 14 preset near five million parameters."""

    return ModelConfig(
        vocab_size=vocab_size,
        context_length=context_length,
        embedding_dim=256,
        num_layers=5,
        num_heads=8,
        feed_forward_dim=1024,
        dropout=0.1,
    )
