import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from mini_llm.bpe_tokenizer import BPETokenizer
from mini_llm.config import ModelConfig
from mini_llm.model import JeePeeTee, causal_language_model_loss
from mini_llm.pretraining import (
    CleanDocument,
    DatasetSourceConfig,
    FixedTokenSequenceDataset,
    PretrainingDataConfig,
    build_jsonl_record_loader,
    development_pretraining_config,
    extract_text,
    load_token_sequences,
    mix_documents,
    normalize_text,
    prepare_pretraining_data,
    record_matches_filters,
    split_documents,
)

GENERAL_TEXTS = (
    "The solar system contains planets that orbit a star called the Sun.",
    "Photosynthesis converts light energy into chemical energy in plants.",
)
CODE_TEXTS = (
    "def add(left, right):\n    return left + right\n",
    "const square = (value) => value * value;\n",
    "SELECT name FROM students WHERE active = TRUE;\n",
)


def _source(
    name: str,
    *,
    field: str,
    weight: float,
    max_examples: int,
    content_kind: str,
    language: str,
    required_values: tuple[tuple[str, tuple[str, ...]], ...] = (),
) -> DatasetSourceConfig:
    return DatasetSourceConfig(
        name=name,
        subset=None,
        split="train",
        text_fields=(field,),
        max_examples=max_examples,
        mixture_weight=weight,
        source_url=f"https://example.test/{name}",
        license="test-license",
        license_status="documented",
        license_notes="Offline deterministic fixture.",
        content_kind=content_kind,  # type: ignore[arg-type]
        language=language,
        required_values=required_values,
    )


@pytest.fixture(scope="module")
def sources() -> tuple[DatasetSourceConfig, DatasetSourceConfig]:
    return (
        _source(
            "educational",
            field="text",
            weight=0.6,
            max_examples=6,
            content_kind="general",
            language="en",
            required_values=(("language", ("en",)),),
        ),
        _source(
            "code",
            field="content",
            weight=0.4,
            max_examples=4,
            content_kind="code",
            language="mixed-code",
        ),
    )


@pytest.fixture(scope="module")
def records(
    sources: tuple[DatasetSourceConfig, DatasetSourceConfig],
) -> dict[str, list[dict[str, object]]]:
    general, code = sources
    return {
        general.key: [
            {"text": GENERAL_TEXTS[0], "language": "en", "id": 1},
            {"text": GENERAL_TEXTS[1], "language": "en", "id": 2},
            {"text": GENERAL_TEXTS[0], "language": "en", "id": 3},
            {"text": "   \n", "language": "en", "id": 4},
            {"text": "Este texto no pasa el filtro.", "language": "es", "id": 5},
            {"text": None, "language": "en", "id": 6},
        ],
        code.key: [
            {"content": CODE_TEXTS[0], "lang": "Python"},
            {"content": CODE_TEXTS[1], "lang": "JavaScript"},
            {"content": CODE_TEXTS[2], "lang": "SQL"},
            {"content": "", "lang": "Python"},
        ],
    }


@pytest.fixture(scope="module")
def bpe_tokenizer() -> BPETokenizer:
    corpus = [*GENERAL_TEXTS, *CODE_TEXTS] * 4
    return BPETokenizer.train(corpus, vocab_size=280, min_frequency=1)


def test_text_extraction_cleaning_and_record_filtering() -> None:
    record = {"title": "  A title  ", "text": "line one  \r\nline two\x00\n\n\n\nend"}

    extracted = extract_text(record, ("title", "text", "missing"))

    assert extracted is not None
    assert normalize_text(extracted) == "A title\n\nline one\nline two\n\n\nend"
    assert extract_text({"text": None}, ("text",)) is None
    assert record_matches_filters({"language": "en"}, (("language", ("en",)),))
    assert not record_matches_filters({"language": "es"}, (("language", ("en",)),))


def test_weighted_mixing_and_split_are_deterministic() -> None:
    documents = {
        "general": [
            CleanDocument(f"general {index}", "general", "general", "en")
            for index in range(10)
        ],
        "code": [CleanDocument(f"code {index}", "code", "code", "python") for index in range(10)],
    }
    weights = {"general": 0.75, "code": 0.25}

    first = mix_documents(documents, weights, max_examples=8, seed=7)
    second = mix_documents(documents, weights, max_examples=8, seed=7)
    train, validation = split_documents(first, validation_fraction=0.25, seed=7)
    repeated_train, repeated_validation = split_documents(
        second,
        validation_fraction=0.25,
        seed=7,
    )

    assert first == second
    assert sum(document.source_key == "general" for document in first) == 6
    assert sum(document.source_key == "code" for document in first) == 2
    assert train == repeated_train
    assert validation == repeated_validation
    assert len(train) == 6
    assert len(validation) == 2


def test_prepare_pipeline_filters_deduplicates_saves_and_reproduces(
    tmp_path: Path,
    sources: tuple[DatasetSourceConfig, DatasetSourceConfig],
    records: dict[str, list[dict[str, object]]],
    bpe_tokenizer: BPETokenizer,
) -> None:
    def loader(source: DatasetSourceConfig) -> list[dict[str, object]]:
        return records[source.key]

    config = PretrainingDataConfig(
        sources=sources,
        context_length=8,
        validation_fraction=0.25,
        seed=42,
        output_dir=tmp_path / "first",
        cache_dir=None,
        deduplicate=True,
        min_characters=10,
        max_characters=1_000,
        max_mixed_examples=4,
    )
    first = prepare_pretraining_data(config, bpe_tokenizer, record_loader=loader)
    repeated_config = PretrainingDataConfig(
        sources=sources,
        context_length=8,
        validation_fraction=0.25,
        seed=42,
        output_dir=tmp_path / "second",
        cache_dir=None,
        deduplicate=True,
        min_characters=10,
        max_characters=1_000,
        max_mixed_examples=4,
    )
    second = prepare_pretraining_data(
        repeated_config,
        bpe_tokenizer,
        record_loader=loader,
    )

    assert first.metadata.raw_examples == 10
    assert first.metadata.cleaned_examples == 5
    assert first.metadata.selected_examples == 4
    assert first.metadata.train_documents == 3
    assert first.metadata.validation_documents == 1
    assert first.metadata.approximate_tokens > 0
    assert first.metadata.sources[0].observed_schema == {
        "id": "int",
        "language": "str",
        "text": "str",
    }
    assert torch.equal(first.train_sequences, second.train_sequences)
    assert torch.equal(first.validation_sequences, second.validation_sequences)
    assert first.metadata == second.metadata
    assert (config.output_dir / "raw" / "educational_default.jsonl").is_file()
    assert (config.output_dir / "cleaned" / "train.jsonl").is_file()
    assert (config.output_dir / "metadata.json").is_file()
    loaded = load_token_sequences(
        config.output_dir / "tokenized" / "train.pt",
        context_length=config.context_length,
    )
    assert torch.equal(loaded, first.train_sequences)
    assert first.train_sequences.shape[1] == config.context_length + 1
    assert first.validation_sequences.shape[1] == config.context_length + 1
    assert first.train_sequences.min() >= 0
    assert first.train_sequences.max() < bpe_tokenizer.vocab_size


def test_fixed_sequences_shift_once_and_flow_through_minigpt(
    tmp_path: Path,
    sources: tuple[DatasetSourceConfig, DatasetSourceConfig],
    records: dict[str, list[dict[str, object]]],
    bpe_tokenizer: BPETokenizer,
) -> None:
    config = PretrainingDataConfig(
        sources=sources,
        context_length=8,
        validation_fraction=0.25,
        output_dir=tmp_path,
        cache_dir=None,
        min_characters=10,
        max_characters=1_000,
        max_mixed_examples=4,
    )
    prepared = prepare_pretraining_data(
        config,
        bpe_tokenizer,
        record_loader=lambda source: records[source.key],
        save=False,
    )
    dataset = FixedTokenSequenceDataset(
        prepared.train_sequences,
        context_length=config.context_length,
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    inputs, targets = next(iter(loader))
    model_config = ModelConfig(
        vocab_size=bpe_tokenizer.vocab_size,
        context_length=config.context_length,
        embedding_dim=16,
        num_layers=1,
        num_heads=4,
        feed_forward_dim=32,
    )

    logits = JeePeeTee(model_config)(inputs)
    loss = causal_language_model_loss(logits, targets)

    assert inputs.shape == targets.shape == (2, config.context_length)
    assert torch.equal(inputs[:, 1:], targets[:, :-1])
    assert logits.shape == (2, config.context_length, bpe_tokenizer.vocab_size)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)


def test_pretraining_configuration_rejects_unsafe_sizes(
    sources: tuple[DatasetSourceConfig, DatasetSourceConfig],
) -> None:
    with pytest.raises(ValueError, match="validation_fraction"):
        PretrainingDataConfig(sources=sources, context_length=8, validation_fraction=1.0)
    with pytest.raises(ValueError, match="max_characters"):
        PretrainingDataConfig(
            sources=sources,
            context_length=8,
            min_characters=100,
            max_characters=10,
        )


def test_development_catalog_covers_requested_domains_and_documents_licenses() -> None:
    config = development_pretraining_config(
        context_length=64,
        max_examples_per_source=3,
        max_mixed_examples=12,
    )

    assert {source.language for source in config.sources if source.content_kind == "code"} == {
        "python",
        "javascript",
        "typescript",
        "html",
        "css",
        "sql",
    }
    assert {source.content_kind for source in config.sources} == {
        "general",
        "educational",
        "code",
    }
    assert sum(source.mixture_weight for source in config.sources) == pytest.approx(1.0)
    assert all(source.license and source.license_notes for source in config.sources)
    assert all(
        source.license_status == "review_required"
        for source in config.sources
        if source.content_kind == "code"
    )


def test_local_jsonl_loader_streams_records_and_rejects_malformed_data(
    tmp_path: Path,
    sources: tuple[DatasetSourceConfig, DatasetSourceConfig],
) -> None:
    source = sources[0]
    path = tmp_path / "source.jsonl"
    path.write_text(
        "\n".join(json.dumps({"text": text}) for text in GENERAL_TEXTS) + "\n",
        encoding="utf-8",
    )
    loader = build_jsonl_record_loader({source.key: path})

    assert list(loader(source)) == [{"text": text} for text in GENERAL_TEXTS]

    malformed_path = tmp_path / "malformed.jsonl"
    malformed_path.write_text('{"text": "valid"}\nnot-json\n', encoding="utf-8")
    malformed_loader = build_jsonl_record_loader({source.key: malformed_path})
    with pytest.raises(ValueError, match="line 2"):
        list(malformed_loader(source))
