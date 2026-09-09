import pytest

from mini_llm.config import (
    DataConfig,
    DebugOverfitConfig,
    ModelConfig,
    RuntimeConfig,
    SanityTrainingConfig,
    TrainingConfig,
    development_model_config,
)


def test_development_config_is_small_and_attention_compatible() -> None:
    config = development_model_config(vocab_size=128)

    assert config.vocab_size == 128
    assert config.context_length == 64
    assert config.embedding_dim % config.num_heads == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("vocab_size", 0),
        ("context_length", 0),
        ("embedding_dim", 0),
        ("num_layers", 0),
        ("num_heads", 0),
        ("feed_forward_dim", 0),
    ],
)
def test_model_config_rejects_non_positive_dimensions(field: str, value: int) -> None:
    values = {
        "vocab_size": 128,
        "context_length": 64,
        "embedding_dim": 64,
        "num_layers": 2,
        "num_heads": 4,
        "feed_forward_dim": 256,
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        ModelConfig(**values)


def test_model_config_rejects_incompatible_attention_dimensions() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(
            vocab_size=128,
            context_length=64,
            embedding_dim=63,
            num_layers=2,
            num_heads=4,
            feed_forward_dim=256,
        )


@pytest.mark.parametrize("dropout", [-0.1, 1.0])
def test_model_config_rejects_invalid_dropout(dropout: float) -> None:
    with pytest.raises(ValueError, match="dropout"):
        ModelConfig(
            vocab_size=128,
            context_length=64,
            embedding_dim=64,
            num_layers=2,
            num_heads=4,
            feed_forward_dim=256,
            dropout=dropout,
        )


def test_runtime_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="seed"):
        RuntimeConfig(seed=-1)
    with pytest.raises(ValueError, match="device"):
        RuntimeConfig(device="tpu")  # type: ignore[arg-type]


def test_sanity_training_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="steps"):
        SanityTrainingConfig(steps=0)
    with pytest.raises(ValueError, match="learning_rate"):
        SanityTrainingConfig(learning_rate=0.0)


def test_debug_overfit_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="steps"):
        DebugOverfitConfig(steps=0)
    with pytest.raises(ValueError, match="learning_rate"):
        DebugOverfitConfig(learning_rate=0.0)
    with pytest.raises(ValueError, match="weight_decay"):
        DebugOverfitConfig(weight_decay=-0.1)


def test_data_and_training_configs_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="validation_fraction"):
        DataConfig(validation_fraction=1.0)
    with pytest.raises(ValueError, match="stride"):
        DataConfig(stride=0)
    with pytest.raises(ValueError, match="batch_size"):
        TrainingConfig(batch_size=0)
    with pytest.raises(ValueError, match="gradient_clip_norm"):
        TrainingConfig(gradient_clip_norm=0.0)
