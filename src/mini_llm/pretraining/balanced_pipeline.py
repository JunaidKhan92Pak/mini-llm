"""Token-balanced, cacheable corpus preparation for the Phase 14 model."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from mini_llm.bpe_tokenizer import BPETokenizer
from mini_llm.pretraining.pipeline import (
    CleanDocument,
    FixedTokenSequenceDataset,
    inspect_record_schema,
    normalize_text,
    pack_documents,
    record_matches_filters,
    split_documents,
)

MixtureCategory = Literal[
    "general_educational",
    "programming_code",
    "general_qa",
    "technical_qa",
    "instruction_writing",
]
RecordStyle = Literal["plain", "instruction_response", "messages"]
BALANCED_PIPELINE_VERSION = 1


@dataclass(frozen=True, slots=True)
class BalancedSourceConfig:
    """One licensed source with an explicit share of the final token mixture."""

    key: str
    dataset_name: str
    subset: str | None
    split: str
    text_fields: tuple[str, ...]
    category: MixtureCategory
    language: str
    mixture_weight: float
    max_examples: int
    source_url: str
    license: str
    license_status: Literal["documented", "review_required"]
    license_notes: str
    record_style: RecordStyle = "plain"
    required_values: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def __post_init__(self) -> None:
        if not self.key or not self.dataset_name or not self.text_fields:
            raise ValueError("source key, dataset name, and text fields are required")
        if self.mixture_weight <= 0 or self.max_examples <= 0:
            raise ValueError("source weight and max_examples must be positive")
        if not self.source_url or not self.license or not self.license_notes:
            raise ValueError("source licensing and provenance must be documented")


@dataclass(frozen=True, slots=True)
class BalancedCorpusConfig:
    """Quality, scale, split, and cache choices for a larger local corpus."""

    sources: tuple[BalancedSourceConfig, ...]
    output_dir: Path
    target_tokens: int = 20_000_000
    context_length: int = 128
    validation_fraction: float = 0.02
    seed: int = 42
    min_characters: int = 80
    max_characters: int = 20_000
    min_text_alphanumeric_fraction: float = 0.35
    min_code_alphanumeric_fraction: float = 0.15
    max_repeated_line_fraction: float = 0.5

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError("at least one source is required")
        if len({source.key for source in self.sources}) != len(self.sources):
            raise ValueError("source keys must be unique")
        if abs(sum(source.mixture_weight for source in self.sources) - 1.0) > 1e-9:
            raise ValueError("source mixture weights must sum to 1.0")
        if self.target_tokens <= 0 or self.context_length <= 0:
            raise ValueError("target_tokens and context_length must be positive")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between zero and one")
        if self.min_characters <= 0 or self.max_characters < self.min_characters:
            raise ValueError("character limits are invalid")
        for value in (
            self.min_text_alphanumeric_fraction,
            self.min_code_alphanumeric_fraction,
            self.max_repeated_line_fraction,
        ):
            if not 0 <= value <= 1:
                raise ValueError("quality fractions must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class BalancedSourceMetadata:
    key: str
    dataset_name: str
    subset: str | None
    split: str
    category: str
    source_url: str
    license: str
    license_status: str
    license_notes: str
    raw_examples: int
    retained_examples: int
    rejected_examples: int
    selected_examples: int
    token_count: int
    final_percentage: float
    average_sample_characters: float
    observed_schema: dict[str, str]


@dataclass(frozen=True, slots=True)
class BalancedCorpusMetadata:
    pipeline_version: int
    cache_fingerprint: str
    requested_tokens: int
    effective_token_budget: int
    approximate_tokens: int
    raw_examples: int
    retained_examples: int
    rejected_examples: int
    duplicate_rejections: int
    selected_examples: int
    train_documents: int
    validation_documents: int
    train_sequences: int
    validation_sequences: int
    train_tokens: int
    validation_tokens: int
    context_length: int
    validation_fraction: float
    seed: int
    train_validation_hash_overlap: int
    category_percentages: dict[str, float]
    sources: tuple[BalancedSourceMetadata, ...]


@dataclass(frozen=True, slots=True)
class BalancedPreparedData:
    train_sequences: torch.Tensor
    validation_sequences: torch.Tensor
    metadata: BalancedCorpusMetadata


def phase15_sources(max_examples_per_source: int = 100_000) -> tuple[BalancedSourceConfig, ...]:
    """Return the documented 40/30/15/10/5 Phase 15 source catalog."""

    common = {"max_examples": max_examples_per_source}
    sources = [
        BalancedSourceConfig(
            key="fineweb_edu", dataset_name="HuggingFaceFW/fineweb-edu",
            subset="sample-10BT", split="train", text_fields=("text",),
            category="general_educational", language="en", mixture_weight=0.25,
            source_url="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
            license="ODC-By-1.0", license_status="documented",
            license_notes="Subject to ODC attribution and Common Crawl source terms.",
            required_values=(("language", ("en",)),), **common,
        ),
        BalancedSourceConfig(
            key="cosmopedia_v2", dataset_name="HuggingFaceTB/smollm-corpus",
            subset="cosmopedia-v2", split="train", text_fields=("text",),
            category="general_educational", language="en", mixture_weight=0.15,
            source_url="https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus",
            license="ODC-By-1.0", license_status="documented",
            license_notes="Synthetic educational text; retain attribution and provenance.",
            **common,
        ),
    ]
    code_sources = {
        "python_codesearchnet": (
            "Nan-Do/code-search-net-python",
            "python",
            0.20,
        ),
        "javascript_codesearchnet": (
            "Nan-Do/code-search-net-javascript",
            "javascript",
            0.10,
        ),
    }
    for key, (dataset_name, language, weight) in code_sources.items():
        sources.append(BalancedSourceConfig(
            key=key, dataset_name=dataset_name, subset=language,
            split="train", text_fields=("content",),
            category="programming_code", language=language,
            mixture_weight=weight,
            source_url=f"https://huggingface.co/datasets/{dataset_name}",
            license="Apache-2.0 mirror; original repositories vary",
            license_status="review_required",
            license_notes=(
                "The mirror declares Apache-2.0, but exported rows omit CodeSearchNet's "
                "per-repository license mapping; review before redistribution or release."
            ),
            **common,
        ))
    sources.extend([
        BalancedSourceConfig(
            key="dolly_writing", dataset_name="databricks/databricks-dolly-15k", subset=None,
            split="train", text_fields=("instruction", "context", "response"),
            category="instruction_writing",
            language="en", mixture_weight=0.05,
            source_url="https://huggingface.co/datasets/databricks/databricks-dolly-15k",
            license="CC-BY-SA-3.0", license_status="documented",
            license_notes=(
                "Requires attribution and ShareAlike compliance; includes Wikipedia material."
            ),
            record_style="instruction_response",
            required_values=(
                ("category", ("brainstorming", "creative_writing", "summarization")),
            ),
            **common,
        ),
        BalancedSourceConfig(
            key="code_alpaca", dataset_name="sahil2801/CodeAlpaca-20k", subset=None,
            split="train", text_fields=("instruction", "input", "output"), category="technical_qa",
            language="en/code", mixture_weight=0.10,
            source_url="https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k",
            license="CC-BY-4.0", license_status="documented",
            license_notes="Synthetic coding instructions; retain attribution and review quality.",
            record_style="instruction_response", **common,
        ),
        BalancedSourceConfig(
            key="ultrachat_qa", dataset_name="HuggingFaceH4/ultrachat_200k",
            subset=None, split="train_sft", text_fields=("messages",),
            category="general_qa",
            language="en", mixture_weight=0.15,
            source_url="https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k",
            license="MIT", license_status="documented",
            license_notes=(
                "Model-generated conversations; review provenance, bias, and factual quality."
            ),
            record_style="messages", **common,
        ),
    ])
    return tuple(sources)


def format_record(record: dict[str, Any], source: BalancedSourceConfig) -> str | None:
    """Format plain, QA, or chat records into explicit pretraining text."""

    if source.record_style == "messages":
        messages = record.get(source.text_fields[0])
        if not isinstance(messages, list):
            return None
        lines = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role, content = message.get("role"), message.get("content")
            if isinstance(role, str) and isinstance(content, str) and content.strip():
                lines.append(f"{role.title()}: {content.strip()}")
        return "\n".join(lines) or None

    values = [record.get(field) for field in source.text_fields]
    strings = [value.strip() if isinstance(value, str) else "" for value in values]
    if source.record_style == "instruction_response":
        if not strings or not strings[0] or not strings[-1]:
            return None
        context = f"\nContext: {strings[1]}" if len(strings) > 2 and strings[1] else ""
        return f"Question: {strings[0]}{context}\nAnswer: {strings[-1]}"
    non_empty = [value for value in strings if value]
    return "\n\n".join(non_empty) if non_empty else None


def passes_quality_filters(
    text: str,
    source: BalancedSourceConfig,
    config: BalancedCorpusConfig,
) -> bool:
    """Reject short, oversized, corrupt, low-signal, or line-repeated samples."""

    if not config.min_characters <= len(text) <= config.max_characters:
        return False
    if text.count("\ufffd") / len(text) > 0.001:
        return False
    alphanumeric_fraction = sum(character.isalnum() for character in text) / len(text)
    threshold = (
        config.min_code_alphanumeric_fraction
        if source.category == "programming_code"
        else config.min_text_alphanumeric_fraction
    )
    if alphanumeric_fraction < threshold:
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 4 and 1 - len(set(lines)) / len(lines) > config.max_repeated_line_fraction:
        return False
    return True


def _file_fingerprint(
    config: BalancedCorpusConfig,
    tokenizer: BPETokenizer,
    files: dict[str, Path],
) -> str:
    digest = hashlib.sha256(str(BALANCED_PIPELINE_VERSION).encode())
    digest.update(json.dumps(asdict(config), default=str, sort_keys=True).encode())
    for token in tokenizer.vocabulary:
        digest.update(token.encode("utf-8"))
        digest.update(b"\0")
    for key, path in sorted(files.items()):
        digest.update(key.encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _load_cached(output_dir: Path, fingerprint: str) -> BalancedPreparedData | None:
    metadata_path = output_dir / "metadata.json"
    train_path = output_dir / "tokenized" / "train.pt"
    validation_path = output_dir / "tokenized" / "validation.pt"
    if not all(path.is_file() for path in (metadata_path, train_path, validation_path)):
        return None
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if payload.get("cache_fingerprint") != fingerprint:
        return None
    payload["sources"] = tuple(BalancedSourceMetadata(**item) for item in payload["sources"])
    metadata = BalancedCorpusMetadata(**payload)
    train = torch.load(train_path, map_location="cpu", weights_only=True)
    validation = torch.load(validation_path, map_location="cpu", weights_only=True)
    FixedTokenSequenceDataset(train, context_length=metadata.context_length)
    FixedTokenSequenceDataset(validation, context_length=metadata.context_length)
    return BalancedPreparedData(train, validation, metadata)


def _write_jsonl(path: Path, documents: list[CleanDocument]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for document in documents:
            handle.write(json.dumps(asdict(document), ensure_ascii=False) + "\n")
    temporary.replace(path)


def prepare_balanced_corpus(
    config: BalancedCorpusConfig,
    tokenizer: BPETokenizer,
    source_files: dict[str, Path],
    *,
    reuse_cache: bool = True,
) -> BalancedPreparedData:
    """Clean local JSONL, mix by token budget, split, pack, and cache artifacts."""

    expected = {source.key for source in config.sources}
    if set(source_files) != expected:
        raise ValueError("source_files keys must exactly match configured sources")
    missing = [str(path) for path in source_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing local source files: {missing}")
    fingerprint = _file_fingerprint(config, tokenizer, source_files)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(config.output_dir / "tokenizer.json")
    if reuse_cache and (cached := _load_cached(config.output_dir, fingerprint)) is not None:
        return cached

    documents: dict[str, list[CleanDocument]] = defaultdict(list)
    raw_counts: dict[str, int] = defaultdict(int)
    rejected_counts: dict[str, int] = defaultdict(int)
    schemas: dict[str, dict[str, str]] = {}
    duplicate_rejections = 0
    seen: set[str] = set()
    for source in config.sources:
        with source_files[source.key].open("r", encoding="utf-8") as handle:
            for line in handle:
                if raw_counts[source.key] >= source.max_examples:
                    break
                raw_counts[source.key] += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    rejected_counts[source.key] += 1
                    continue
                if not isinstance(record, dict):
                    rejected_counts[source.key] += 1
                    continue
                schemas.setdefault(source.key, inspect_record_schema(record))
                recorded_dataset = record.get("_dataset")
                recorded_subset = record.get("_config")
                if (
                    isinstance(recorded_dataset, str)
                    and recorded_dataset != source.dataset_name
                ) or (
                    source.subset is not None
                    and isinstance(recorded_subset, str)
                    and recorded_subset != source.subset
                ):
                    rejected_counts[source.key] += 1
                    continue
                if not record_matches_filters(record, source.required_values):
                    rejected_counts[source.key] += 1
                    continue
                formatted = format_record(record, source)
                cleaned = normalize_text(formatted) if formatted is not None else ""
                if not cleaned or not passes_quality_filters(cleaned, source, config):
                    rejected_counts[source.key] += 1
                    continue
                digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
                if digest in seen:
                    duplicate_rejections += 1
                    rejected_counts[source.key] += 1
                    continue
                seen.add(digest)
                documents[source.key].append(
                    CleanDocument(cleaned, source.key, source.category, source.language)
                )

    token_lengths = {
        source.key: [
            len(tokenizer.encode(document.text, add_eos=True))
            for document in documents[source.key]
        ]
        for source in config.sources
    }
    unavailable = [source.key for source in config.sources if not token_lengths[source.key]]
    if unavailable:
        raise ValueError(f"sources produced no usable documents: {unavailable}")
    feasible_tokens = min(
        sum(token_lengths[source.key]) / source.mixture_weight for source in config.sources
    )
    effective_budget = min(config.target_tokens, int(feasible_tokens))

    randomizer = random.Random(config.seed)
    selected: list[CleanDocument] = []
    selected_tokens: dict[str, int] = defaultdict(int)
    selected_counts: dict[str, int] = defaultdict(int)
    for source in config.sources:
        candidates = list(zip(documents[source.key], token_lengths[source.key], strict=True))
        randomizer.shuffle(candidates)
        budget = round(effective_budget * source.mixture_weight)
        for document, token_count in candidates:
            if selected_tokens[source.key] + token_count > budget:
                continue
            selected.append(document)
            selected_tokens[source.key] += token_count
            selected_counts[source.key] += 1
    if len(selected) < 2:
        raise ValueError("token budgets selected fewer than two documents")
    randomizer.shuffle(selected)
    train_documents, validation_documents = split_documents(
        selected, validation_fraction=config.validation_fraction, seed=config.seed
    )
    train_hashes = {hashlib.sha256(item.text.encode()).hexdigest() for item in train_documents}
    validation_hashes = {
        hashlib.sha256(item.text.encode()).hexdigest() for item in validation_documents
    }
    overlap = len(train_hashes & validation_hashes)
    if overlap:
        raise RuntimeError("train/validation document leakage detected")
    train_sequences, train_tokens = pack_documents(
        train_documents, tokenizer, context_length=config.context_length
    )
    validation_sequences, validation_tokens = pack_documents(
        validation_documents, tokenizer, context_length=config.context_length
    )
    total_selected_tokens = sum(selected_tokens.values())
    source_metadata = tuple(BalancedSourceMetadata(
        key=source.key,
        dataset_name=source.dataset_name,
        subset=source.subset,
        split=source.split,
        category=source.category,
        source_url=source.source_url,
        license=source.license,
        license_status=source.license_status,
        license_notes=source.license_notes,
        raw_examples=raw_counts[source.key],
        retained_examples=len(documents[source.key]),
        rejected_examples=rejected_counts[source.key],
        selected_examples=selected_counts[source.key],
        token_count=selected_tokens[source.key],
        final_percentage=(100 * selected_tokens[source.key] / total_selected_tokens),
        average_sample_characters=(
            sum(len(item.text) for item in documents[source.key]) / len(documents[source.key])
        ),
        observed_schema=schemas.get(source.key, {}),
    ) for source in config.sources)
    category_tokens: dict[str, int] = defaultdict(int)
    for source in config.sources:
        category_tokens[source.category] += selected_tokens[source.key]
    metadata = BalancedCorpusMetadata(
        pipeline_version=BALANCED_PIPELINE_VERSION,
        cache_fingerprint=fingerprint,
        requested_tokens=config.target_tokens,
        effective_token_budget=effective_budget,
        approximate_tokens=train_tokens + validation_tokens,
        raw_examples=sum(raw_counts.values()),
        retained_examples=sum(len(items) for items in documents.values()),
        rejected_examples=sum(rejected_counts.values()),
        duplicate_rejections=duplicate_rejections,
        selected_examples=len(selected),
        train_documents=len(train_documents),
        validation_documents=len(validation_documents),
        train_sequences=len(train_sequences),
        validation_sequences=len(validation_sequences),
        train_tokens=train_tokens,
        validation_tokens=validation_tokens,
        context_length=config.context_length,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
        train_validation_hash_overlap=overlap,
        category_percentages={
            category: 100 * count / total_selected_tokens
            for category, count in sorted(category_tokens.items())
        },
        sources=source_metadata,
    )
    prepared = BalancedPreparedData(train_sequences, validation_sequences, metadata)
    _write_jsonl(config.output_dir / "cleaned" / "train.jsonl", train_documents)
    _write_jsonl(config.output_dir / "cleaned" / "validation.jsonl", validation_documents)
    tokenized = config.output_dir / "tokenized"
    tokenized.mkdir(parents=True, exist_ok=True)
    torch.save(train_sequences, tokenized / "train.pt")
    torch.save(validation_sequences, tokenized / "validation.pt")
    (config.output_dir / "metadata.json").write_text(
        json.dumps(asdict(metadata), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return prepared
