import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from mini_llm.bpe_tokenizer import BPETokenizer
from mini_llm.config import five_million_model_config
from mini_llm.model import JeePeeTee, causal_language_model_loss
from mini_llm.pretraining.balanced_pipeline import (
    BalancedCorpusConfig,
    BalancedSourceConfig,
    format_record,
    passes_quality_filters,
    phase15_sources,
    prepare_balanced_corpus,
)
from mini_llm.pretraining.pipeline import FixedTokenSequenceDataset

CATEGORIES = (
    ("general", "general_educational", 0.40, 40),
    ("code", "programming_code", 0.30, 30),
    ("general_qa", "general_qa", 0.15, 15),
    ("technical_qa", "technical_qa", 0.10, 10),
    ("writing", "instruction_writing", 0.05, 5),
)


def _sources() -> tuple[BalancedSourceConfig, ...]:
    return tuple(
        BalancedSourceConfig(
            key=key,
            dataset_name=f"test/{key}",
            subset=None,
            split="train",
            text_fields=("text",),
            category=category,  # type: ignore[arg-type]
            language="en",
            mixture_weight=weight,
            max_examples=count + 3,
            source_url=f"https://example.test/{key}",
            license="test-license",
            license_status="documented",
            license_notes="Deterministic test fixture.",
        )
        for key, category, weight, count in CATEGORIES
    )


def _write_sources(tmp_path: Path) -> tuple[dict[str, Path], list[str]]:
    files = {}
    corpus = []
    for key, category, _, count in CATEGORIES:
        path = tmp_path / f"{key}.jsonl"
        records = [
            {"text": f"{category} educational sample number {index} with useful clear content."}
            for index in range(count)
        ]
        corpus.extend(record["text"] for record in records)
        lines = [json.dumps(record) for record in records]
        if key == "general":
            lines.extend([json.dumps(records[0]), json.dumps({"text": "tiny"}), "not-json"])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        files[key] = path
    return files, corpus


def _config(sources: tuple[BalancedSourceConfig, ...], output: Path) -> BalancedCorpusConfig:
    return BalancedCorpusConfig(
        sources=sources,
        output_dir=output,
        target_tokens=1_000_000,
        context_length=8,
        validation_fraction=0.2,
        min_characters=20,
        max_characters=1_000,
        seed=15,
    )


def test_phase15_catalog_has_requested_category_ratios_and_licenses() -> None:
    sources = phase15_sources()
    category_weights: dict[str, float] = {}
    for source in sources:
        category_weights[source.category] = (
            category_weights.get(source.category, 0.0) + source.mixture_weight
        )
    assert category_weights == pytest.approx({
        "general_educational": 0.40,
        "programming_code": 0.30,
        "general_qa": 0.15,
        "technical_qa": 0.10,
        "instruction_writing": 0.05,
    })
    assert all(source.license and source.license_notes for source in sources)
    assert all(
        source.license_status == "review_required"
        for source in sources
        if source.category == "programming_code"
    )


def test_qa_chat_formatting_and_quality_filters(tmp_path: Path) -> None:
    sources = _sources()
    qa = replace(
        sources[2],
        text_fields=("instruction", "context", "response"),
        record_style="instruction_response",
    )
    assert format_record(
        {"instruction": "What is RAM?", "context": "Computers", "response": "Memory."}, qa
    ) == "Question: What is RAM?\nContext: Computers\nAnswer: Memory."
    config = _config(sources, tmp_path)
    assert not passes_quality_filters("short", sources[0], config)
    assert not passes_quality_filters("same\nsame\nsame\nsame\nsame", sources[0], config)


def test_balanced_processing_is_deterministic_cached_and_model_compatible(
    tmp_path: Path,
) -> None:
    sources = _sources()
    files, corpus = _write_sources(tmp_path)
    tokenizer = BPETokenizer.train(corpus, vocab_size=320, min_frequency=1)
    config = _config(sources, tmp_path / "processed")
    first = prepare_balanced_corpus(config, tokenizer, files)
    train_mtime = (config.output_dir / "tokenized" / "train.pt").stat().st_mtime_ns
    cached = prepare_balanced_corpus(config, tokenizer, files)

    assert torch.equal(first.train_sequences, cached.train_sequences)
    assert torch.equal(first.validation_sequences, cached.validation_sequences)
    assert (config.output_dir / "tokenized" / "train.pt").stat().st_mtime_ns == train_mtime
    assert (config.output_dir / "tokenizer.json").is_file()
    assert first.metadata.duplicate_rejections == 1
    assert first.metadata.rejected_examples == 3
    assert first.metadata.train_validation_hash_overlap == 0
    assert first.train_sequences.shape[1] == config.context_length + 1
    assert first.validation_sequences.shape[1] == config.context_length + 1
    assert first.train_sequences.min() >= 0
    assert first.train_sequences.max() < tokenizer.vocab_size
    expected_percentages = {
        category: weight * 100 for _, category, weight, _ in CATEGORIES
    }
    for category, expected in expected_percentages.items():
        assert first.metadata.category_percentages[category] == pytest.approx(
            expected, abs=6.0
        )

    dataset = FixedTokenSequenceDataset(
        first.train_sequences, context_length=config.context_length
    )
    inputs, targets = next(iter(DataLoader(dataset, batch_size=2)))
    model = JeePeeTee(
        five_million_model_config(tokenizer.vocab_size, context_length=128)
    )
    logits = model(inputs)
    loss = causal_language_model_loss(logits, targets)
    assert logits.shape == (2, config.context_length, tokenizer.vocab_size)
    assert torch.isfinite(loss)
