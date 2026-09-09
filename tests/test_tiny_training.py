from pathlib import Path

from mini_llm.config import RuntimeConfig, TrainingConfig
from mini_llm.tiny_training import (
    DEFAULT_TINY_TEXT,
    run_tiny_text_overfit,
    tiny_model_config,
)
from mini_llm.tokenizer import CharacterTokenizer


def test_one_optimizer_step_updates_complete_model_parameters() -> None:
    tokenizer = CharacterTokenizer.train([DEFAULT_TINY_TEXT])
    result = run_tiny_text_overfit(
        model_config=tiny_model_config(tokenizer.vocab_size),
        training_config=TrainingConfig(
            batch_size=4,
            learning_rate=0.01,
            weight_decay=0.0,
            max_steps=1,
            evaluation_interval=1,
            gradient_clip_norm=1.0,
        ),
        runtime_config=RuntimeConfig(
            seed=42,
            device="cpu",
            deterministic_algorithms=True,
        ),
    )

    assert result.parameters_updated
    assert result.final_loss < result.initial_loss


def test_tiny_text_is_deterministically_memorized_and_checkpointed(
    tmp_path: Path,
) -> None:
    runtime = RuntimeConfig(seed=42, device="cpu", deterministic_algorithms=True)
    checkpoint_path = tmp_path / "tiny-overfit.pt"

    first = run_tiny_text_overfit(
        runtime_config=runtime,
        checkpoint_path=checkpoint_path,
    )
    second = run_tiny_text_overfit(runtime_config=runtime)

    assert first.initial_loss == second.initial_loss
    assert first.final_loss == second.final_loss
    assert first.final_accuracy == second.final_accuracy
    assert first.example_prediction == second.example_prediction
    assert first.parameters_updated
    assert first.checkpoint_consistent
    assert first.final_loss < first.initial_loss * 0.1
    assert first.final_accuracy >= 0.95
    assert first.example_prediction == first.example_target
