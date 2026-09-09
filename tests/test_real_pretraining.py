from pathlib import Path

import pytest
import torch

from mini_llm.bpe_tokenizer import BPETokenizer
from mini_llm.config import ModelConfig, RuntimeConfig
from mini_llm.model import JeePeeTee
from mini_llm.pretraining.real_training import (
    PretrainingRunConfig,
    phase12_model_config,
    train_phase12,
)
from mini_llm.runtime import seed_everything


def _artifacts(tmp_path: Path) -> tuple[BPETokenizer, Path, torch.Tensor]:
    tokenizer = BPETokenizer.train(
        ["Python is useful. A computer program follows instructions."], vocab_size=300
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    encoded = tokenizer.encode(
        "Python is useful. A computer program follows instructions.", add_eos=True
    )
    values = (encoded * 20)[: 8 * 9]
    return tokenizer, tokenizer_path, torch.tensor(values).view(8, 9)


def _model(vocab_size: int, *, context_length: int = 8) -> JeePeeTee:
    return JeePeeTee(ModelConfig(vocab_size, context_length, 16, 1, 4, 32, 0.1))


def _run_config() -> PretrainingRunConfig:
    return PretrainingRunConfig(
        batch_size=2,
        learning_rate=1e-3,
        max_steps=4,
        warmup_steps=1,
        evaluation_interval=2,
        evaluation_batches=1,
        checkpoint_interval=2,
        generation_interval=4,
        generation_max_new_tokens=2,
    )


def test_phase12_model_stays_below_one_million_parameters() -> None:
    model = JeePeeTee(phase12_model_config(2048))
    assert 500_000 < model.parameter_count() < 1_000_000


def test_training_updates_parameters_logs_and_saves_complete_checkpoint(
    tmp_path: Path,
) -> None:
    tokenizer, tokenizer_path, sequences = _artifacts(tmp_path)
    seed_everything(11)
    model = _model(tokenizer.vocab_size)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    result = train_phase12(
        model, tokenizer, sequences, sequences[:4], _run_config(), RuntimeConfig(11, "cpu"),
        output_dir=tmp_path / "run", tokenizer_path=tokenizer_path,
        dataset_metadata={"name": "test"}, prompts=("Python is",),
    )
    assert result.global_step == 4
    assert result.tokens_processed == 4 * 2 * 8
    assert any(not torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert all(torch.isfinite(torch.tensor(metric.validation_loss)) for metric in result.metrics)
    assert (tmp_path / "run" / "metrics.jsonl").is_file()
    payload = torch.load(result.checkpoints[-1], weights_only=True)
    assert payload["scheduler_state"]
    assert payload["tokenizer_metadata"]["vocab_size"] == tokenizer.vocab_size
    assert payload["tokenizer_metadata"]["config"]["type"] == "byte-level-bpe"
    assert payload["training_metadata"]["runtime_config"]["seed"] == 11
    assert payload["training_metadata"]["optimizer"] == "AdamW"
    assert payload["training_metadata"]["device_type"] == "cpu"
    assert payload["training_metadata"]["torch_version"] == str(torch.__version__)
    assert payload["dataset_metadata"] == {"name": "test"}
    assert (tmp_path / "run" / "run-manifest.json").is_file()


def test_interrupted_training_resumes_exactly(tmp_path: Path) -> None:
    tokenizer, tokenizer_path, sequences = _artifacts(tmp_path)
    config = _run_config()
    runtime = RuntimeConfig(23, "cpu")

    seed_everything(23)
    uninterrupted = _model(tokenizer.vocab_size)
    full = train_phase12(
        uninterrupted, tokenizer, sequences, sequences[:4], config, runtime,
        output_dir=tmp_path / "full", tokenizer_path=tokenizer_path,
        dataset_metadata={"name": "test"}, prompts=(),
    )

    seed_everything(23)
    interrupted = _model(tokenizer.vocab_size)
    first = train_phase12(
        interrupted, tokenizer, sequences, sequences[:4], config, runtime,
        output_dir=tmp_path / "resume", tokenizer_path=tokenizer_path,
        dataset_metadata={"name": "test"}, stop_after_step=2, prompts=(),
    )
    resumed = _model(tokenizer.vocab_size)
    second = train_phase12(
        resumed, tokenizer, sequences, sequences[:4], config, runtime,
        output_dir=tmp_path / "resume", tokenizer_path=tokenizer_path,
        dataset_metadata={"name": "test"}, resume_from=first.checkpoints[-1], prompts=(),
    )
    assert full.global_step == second.global_step == 4
    assert full.tokens_processed == second.tokens_processed
    for name, value in uninterrupted.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[name]), name


def test_resume_rejects_changed_tokenizer(tmp_path: Path) -> None:
    tokenizer, tokenizer_path, sequences = _artifacts(tmp_path)
    model = _model(tokenizer.vocab_size)
    first = train_phase12(
        model, tokenizer, sequences, sequences[:4], _run_config(), RuntimeConfig(7, "cpu"),
        output_dir=tmp_path / "run", tokenizer_path=tokenizer_path,
        dataset_metadata={}, stop_after_step=2, prompts=(),
    )
    tokenizer_path.write_text(
        tokenizer_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="tokenizer metadata"):
        train_phase12(
            _model(tokenizer.vocab_size), tokenizer, sequences, sequences[:4], _run_config(),
            RuntimeConfig(7, "cpu"), output_dir=tmp_path / "run2",
            tokenizer_path=tokenizer_path, dataset_metadata={},
            resume_from=first.checkpoints[-1], prompts=(),
        )


def test_trainer_accepts_dataset_context_shorter_than_model_maximum(
    tmp_path: Path,
) -> None:
    tokenizer, tokenizer_path, sequences = _artifacts(tmp_path)
    config = PretrainingRunConfig(
        batch_size=2,
        max_steps=1,
        warmup_steps=0,
        evaluation_interval=1,
        evaluation_batches=1,
        checkpoint_interval=1,
        generation_interval=1,
    )
    result = train_phase12(
        _model(tokenizer.vocab_size, context_length=16),
        tokenizer,
        sequences,
        sequences[:4],
        config,
        RuntimeConfig(14, "cpu"),
        output_dir=tmp_path / "run",
        tokenizer_path=tokenizer_path,
        dataset_metadata={},
        prompts=(),
    )
    assert result.global_step == 1


def test_gradient_accumulation_tracks_tokens_and_saves_best_checkpoint(
    tmp_path: Path,
) -> None:
    tokenizer, tokenizer_path, sequences = _artifacts(tmp_path)
    config = PretrainingRunConfig(
        batch_size=2,
        gradient_accumulation_steps=2,
        max_steps=2,
        warmup_steps=1,
        evaluation_interval=1,
        evaluation_batches=1,
        checkpoint_interval=2,
        generation_interval=2,
    )
    result = train_phase12(
        _model(tokenizer.vocab_size), tokenizer, sequences, sequences[:4], config,
        RuntimeConfig(31, "cpu"), output_dir=tmp_path / "run",
        tokenizer_path=tokenizer_path, dataset_metadata={"name": "accumulation"}, prompts=(),
    )

    assert result.tokens_processed == 2 * 2 * 2 * 8
    assert result.best_checkpoint == tmp_path / "run" / "best-validation.pt"
    assert result.best_checkpoint.is_file()
    assert all(metric.tokens_per_second > 0 for metric in result.metrics[1:])
    payload = torch.load(result.best_checkpoint, weights_only=True)
    assert payload["run_config"]["gradient_accumulation_steps"] == 2


def test_existing_output_directory_rejects_an_incompatible_run(tmp_path: Path) -> None:
    tokenizer, tokenizer_path, sequences = _artifacts(tmp_path)
    output_dir = tmp_path / "run"
    train_phase12(
        _model(tokenizer.vocab_size), tokenizer, sequences, sequences[:4], _run_config(),
        RuntimeConfig(41, "cpu"), output_dir=output_dir, tokenizer_path=tokenizer_path,
        dataset_metadata={"name": "first"}, stop_after_step=1, prompts=(),
    )

    with pytest.raises(ValueError, match="incompatible training run"):
        train_phase12(
            _model(tokenizer.vocab_size), tokenizer, sequences, sequences[:4], _run_config(),
            RuntimeConfig(41, "cpu"), output_dir=output_dir, tokenizer_path=tokenizer_path,
            dataset_metadata={"name": "different"}, stop_after_step=1, prompts=(),
        )
