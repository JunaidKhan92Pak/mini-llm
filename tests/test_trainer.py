from pathlib import Path

import pytest
import torch

from mini_llm.config import (
    DataConfig,
    ModelConfig,
    RuntimeConfig,
    TrainingConfig,
)
from mini_llm.model import JeePeeTee
from mini_llm.tokenizer import CharacterTokenizer
from mini_llm.training import (
    DataLoaders,
    build_dataloaders,
    evaluate_model,
    load_checkpoint,
    train_model,
)


@pytest.fixture
def training_components() -> tuple[
    ModelConfig,
    TrainingConfig,
    DataLoaders,
]:
    text = "JeePeeTee learns tiny text. " * 12
    tokenizer = CharacterTokenizer.train([text])
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=8,
        embedding_dim=16,
        num_layers=1,
        num_heads=4,
        feed_forward_dim=32,
        dropout=0.0,
    )
    training_config = TrainingConfig(
        batch_size=4,
        learning_rate=0.02,
        weight_decay=0.0,
        max_steps=20,
        evaluation_interval=10,
        gradient_clip_norm=1.0,
    )
    loaders = build_dataloaders(
        torch.tensor(tokenizer.encode(text)),
        model_config,
        DataConfig(validation_fraction=0.2, stride=2),
        training_config,
        seed=42,
    )
    return model_config, training_config, loaders


def test_baseline_trainer_reduces_validation_loss_and_tracks_metrics(
    training_components: tuple[ModelConfig, TrainingConfig, DataLoaders],
) -> None:
    model_config, training_config, loaders = training_components
    model = JeePeeTee(model_config)

    result = train_model(
        model,
        loaders.train,
        loaders.validation,
        training_config,
        RuntimeConfig(seed=42, device="cpu", deterministic_algorithms=True),
    )

    assert result.state.global_step == training_config.max_steps
    assert result.state.tokens_processed > 0
    assert [metric.global_step for metric in result.metrics] == [0, 10, 20]
    assert result.metrics[-1].validation_loss < result.metrics[0].validation_loss
    assert result.metrics[-1].train_loss is not None


def test_checkpoint_round_trip_and_resume(
    training_components: tuple[ModelConfig, TrainingConfig, DataLoaders],
    tmp_path: Path,
) -> None:
    model_config, training_config, loaders = training_components
    checkpoint_path = tmp_path / "checkpoint.pt"
    model = JeePeeTee(model_config)
    runtime = RuntimeConfig(seed=42, device="cpu", deterministic_algorithms=True)

    first_result = train_model(
        model,
        loaders.train,
        loaders.validation,
        training_config,
        runtime,
        checkpoint_path=checkpoint_path,
    )
    fixed_inputs = next(iter(loaders.validation))[0]
    model.eval()
    with torch.no_grad():
        expected_logits = model(fixed_inputs)

    restored_model = JeePeeTee(model_config)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.02)
    step, tokens_processed = load_checkpoint(
        checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
        device=torch.device("cpu"),
    )
    restored_model.eval()
    with torch.no_grad():
        restored_logits = restored_model(fixed_inputs)

    assert step == first_result.state.global_step
    assert tokens_processed == first_result.state.tokens_processed
    assert torch.equal(restored_logits, expected_logits)
    assert restored_optimizer.state

    incompatible_config = ModelConfig(
        vocab_size=model_config.vocab_size,
        context_length=model_config.context_length,
        embedding_dim=20,
        num_layers=1,
        num_heads=4,
        feed_forward_dim=40,
    )
    with pytest.raises(ValueError, match="configuration does not match"):
        load_checkpoint(
            checkpoint_path,
            model=JeePeeTee(incompatible_config),
            optimizer=None,
            device=torch.device("cpu"),
        )

    resumed_config = TrainingConfig(
        batch_size=training_config.batch_size,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        max_steps=25,
        evaluation_interval=5,
    )
    resumed_result = train_model(
        JeePeeTee(model_config),
        loaders.train,
        loaders.validation,
        resumed_config,
        runtime,
        resume_from=checkpoint_path,
    )

    assert resumed_result.state.global_step == 25
    assert resumed_result.state.tokens_processed > tokens_processed
    assert resumed_result.metrics[0].global_step == step


def test_evaluation_restores_model_mode_and_rejects_empty_loader(
    training_components: tuple[ModelConfig, TrainingConfig, DataLoaders],
) -> None:
    model_config, _, loaders = training_components
    model = JeePeeTee(model_config).train()

    loss = evaluate_model(
        model,
        loaders.validation,
        device=torch.device("cpu"),
    )

    assert loss > 0.0
    assert model.training
    with pytest.raises(ValueError, match="no target tokens"):
        evaluate_model(model, [], device=torch.device("cpu"))
