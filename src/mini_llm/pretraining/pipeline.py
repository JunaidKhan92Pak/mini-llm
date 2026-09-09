"""Streaming-friendly cleaning and token packing for Mini LLM pretraining."""

import hashlib
import json
import random
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from mini_llm.bpe_tokenizer import BPETokenizer
from mini_llm.pretraining.config import DatasetSourceConfig, PretrainingDataConfig

Record = Mapping[str, Any]
RecordLoader = Callable[[DatasetSourceConfig], Iterable[Record]]


@dataclass(frozen=True, slots=True)
class CleanDocument:
    """Normalized text plus the source metadata needed for auditing."""

    text: str
    source_key: str
    content_kind: str
    language: str


@dataclass(frozen=True, slots=True)
class SourceMetadata:
    """Observed counts, schema, and licensing for one configured source."""

    key: str
    name: str
    subset: str | None
    split: str
    source_url: str
    license: str
    license_status: str
    license_notes: str
    raw_examples: int
    cleaned_examples: int
    selected_examples: int
    approximate_tokens: int
    observed_schema: dict[str, str]


@dataclass(frozen=True, slots=True)
class PretrainingMetadata:
    """Manifest for one processed development corpus."""

    raw_examples: int
    cleaned_examples: int
    selected_examples: int
    approximate_tokens: int
    train_documents: int
    validation_documents: int
    train_sequences: int
    validation_sequences: int
    context_length: int
    validation_fraction: float
    seed: int
    deduplicate: bool
    min_characters: int
    max_characters: int
    sources: tuple[SourceMetadata, ...]


@dataclass(frozen=True, slots=True)
class PreparedPretrainingData:
    """Fixed-length train/validation token sequences and their manifest."""

    train_sequences: torch.Tensor
    validation_sequences: torch.Tensor
    metadata: PretrainingMetadata


class FixedTokenSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Expose pre-packed `[context + 1]` rows as exactly-once-shifted pairs."""

    def __init__(self, sequences: torch.Tensor, *, context_length: int) -> None:
        expected_width = context_length + 1
        if sequences.ndim != 2 or sequences.shape[1] != expected_width:
            raise ValueError(
                f"sequences must have shape (N, {expected_width}), got {tuple(sequences.shape)}"
            )
        if sequences.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"sequences must use an integer dtype, got {sequences.dtype}")
        if sequences.shape[0] == 0:
            raise ValueError("sequences must contain at least one row")
        self.sequences = sequences.detach().clone()

    def __len__(self) -> int:
        return self.sequences.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = self.sequences[index]
        return sequence[:-1], sequence[1:]


def inspect_record_schema(record: Record) -> dict[str, str]:
    """Return a compact field/type view without retaining sample contents."""

    return {str(field): type(value).__name__ for field, value in sorted(record.items())}


def extract_text(record: Record, text_fields: Sequence[str]) -> str | None:
    """Join configured non-empty string fields from a dataset record."""

    values = [
        value.strip()
        for field in text_fields
        if isinstance((value := record.get(field)), str) and value.strip()
    ]
    return "\n\n".join(values) if values else None


def normalize_text(text: str) -> str:
    """Apply conservative normalization that preserves code indentation."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    normalized = re.sub(r"\n{4,}", "\n\n\n", normalized)
    return normalized.strip()


def record_matches_filters(
    record: Record,
    required_values: Sequence[tuple[str, tuple[str, ...]]],
) -> bool:
    """Apply simple allow-list filters for language or record type fields."""

    return all(str(record.get(field, "")).lower() in allowed for field, allowed in required_values)


def _default_record_loader(
    source: DatasetSourceConfig,
    *,
    cache_dir: Path | None,
) -> Iterable[Record]:
    dataset = load_dataset(
        source.name,
        source.subset,
        split=source.split,
        streaming=True,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    return dataset


def build_jsonl_record_loader(
    source_files: Mapping[str, str | Path],
) -> RecordLoader:
    """Build a strict streaming loader for manually downloaded JSONL sources."""

    resolved_files = {key: Path(path) for key, path in source_files.items()}

    def load_records(source: DatasetSourceConfig) -> Iterable[Record]:
        try:
            path = resolved_files[source.key]
        except KeyError as error:
            raise ValueError(f"no local JSONL file configured for {source.key}") from error

        def iterate() -> Iterable[Record]:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"invalid JSON in {path} at line {line_number}"
                        ) from error
                    if not isinstance(record, dict):
                        raise ValueError(
                            f"JSONL record in {path} at line {line_number} must be an object"
                        )
                    yield record

        return iterate()

    return load_records


def _allocate_mixture_counts(
    available: Mapping[str, int],
    weights: Mapping[str, float],
    total: int,
) -> dict[str, int]:
    keys = sorted(key for key, count in available.items() if count > 0)
    if not keys:
        return {}
    total = min(total, sum(available[key] for key in keys))
    weight_total = sum(weights[key] for key in keys)
    ideal = {key: total * weights[key] / weight_total for key in keys}
    counts = {key: min(available[key], int(ideal[key])) for key in keys}
    remaining = total - sum(counts.values())
    while remaining:
        candidates = [key for key in keys if counts[key] < available[key]]
        if not candidates:
            break
        selected = max(candidates, key=lambda key: (ideal[key] - counts[key], weights[key], key))
        counts[selected] += 1
        remaining -= 1
    return counts


def mix_documents(
    documents_by_source: Mapping[str, Sequence[CleanDocument]],
    weights: Mapping[str, float],
    *,
    max_examples: int | None,
    seed: int,
) -> list[CleanDocument]:
    """Select a deterministic bounded mixture according to configured weights."""

    available = {key: len(documents) for key, documents in documents_by_source.items()}
    total = sum(available.values()) if max_examples is None else max_examples
    counts = _allocate_mixture_counts(available, weights, total)
    randomizer = random.Random(seed)
    mixed: list[CleanDocument] = []
    for key in sorted(counts):
        candidates = list(documents_by_source[key])
        randomizer.shuffle(candidates)
        mixed.extend(candidates[: counts[key]])
    randomizer.shuffle(mixed)
    return mixed


def split_documents(
    documents: Sequence[CleanDocument],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[CleanDocument], list[CleanDocument]]:
    """Split documents deterministically before token packing to avoid leakage."""

    if len(documents) < 2:
        raise ValueError("at least two cleaned documents are required for train/validation")
    shuffled = list(documents)
    random.Random(seed).shuffle(shuffled)
    validation_size = max(1, round(len(shuffled) * validation_fraction))
    validation_size = min(validation_size, len(shuffled) - 1)
    return shuffled[validation_size:], shuffled[:validation_size]


def pack_documents(
    documents: Sequence[CleanDocument],
    tokenizer: BPETokenizer,
    *,
    context_length: int,
) -> tuple[torch.Tensor, int]:
    """Tokenize documents with EOS separators and pack fixed `[T + 1]` rows."""

    token_ids: list[int] = []
    for document in documents:
        token_ids.extend(tokenizer.encode(document.text, add_eos=True))
    width = context_length + 1
    usable_tokens = len(token_ids) - (len(token_ids) % width)
    if usable_tokens == 0:
        raise ValueError(f"documents produced fewer than {width} tokens")
    sequences = torch.tensor(token_ids[:usable_tokens], dtype=torch.long).view(-1, width)
    if sequences.min() < 0 or sequences.max() >= tokenizer.vocab_size:
        raise RuntimeError("tokenizer produced IDs outside its reported vocabulary")
    return sequences, len(token_ids)


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)


def _save_artifacts(
    prepared: PreparedPretrainingData,
    raw_records: Mapping[str, Sequence[Record]],
    train_documents: Sequence[CleanDocument],
    validation_documents: Sequence[CleanDocument],
    output_dir: Path,
) -> None:
    for key, records in raw_records.items():
        safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
        _write_jsonl(output_dir / "raw" / f"{safe_key}.jsonl", records)
    _write_jsonl(
        output_dir / "cleaned" / "train.jsonl",
        (asdict(document) for document in train_documents),
    )
    _write_jsonl(
        output_dir / "cleaned" / "validation.jsonl",
        (asdict(document) for document in validation_documents),
    )
    tokenized_dir = output_dir / "tokenized"
    tokenized_dir.mkdir(parents=True, exist_ok=True)
    torch.save(prepared.train_sequences, tokenized_dir / "train.pt")
    torch.save(prepared.validation_sequences, tokenized_dir / "validation.pt")
    metadata_payload = asdict(prepared.metadata)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def prepare_pretraining_data(
    config: PretrainingDataConfig,
    tokenizer: BPETokenizer,
    *,
    record_loader: RecordLoader | None = None,
    save: bool = True,
) -> PreparedPretrainingData:
    """Load bounded sources, clean, mix, split, tokenize, pack, and optionally save."""

    loader = record_loader or (
        lambda source: _default_record_loader(source, cache_dir=config.cache_dir)
    )
    raw_records: dict[str, list[Record]] = {}
    documents_by_source: dict[str, list[CleanDocument]] = defaultdict(list)
    schemas: dict[str, dict[str, str]] = {}
    seen_hashes: set[str] = set()

    for source in config.sources:
        raw_records[source.key] = []
        for record in loader(source):
            if len(raw_records[source.key]) >= source.max_examples:
                break
            materialized_record = dict(record)
            raw_records[source.key].append(materialized_record)
            if source.key not in schemas:
                schemas[source.key] = inspect_record_schema(materialized_record)
            if not record_matches_filters(materialized_record, source.required_values):
                continue
            extracted = extract_text(materialized_record, source.text_fields)
            if extracted is None:
                continue
            cleaned = normalize_text(extracted)
            if not config.min_characters <= len(cleaned) <= config.max_characters:
                continue
            digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
            if config.deduplicate and digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            documents_by_source[source.key].append(
                CleanDocument(
                    text=cleaned,
                    source_key=source.key,
                    content_kind=source.content_kind,
                    language=source.language,
                )
            )

    mixed = mix_documents(
        documents_by_source,
        {source.key: source.mixture_weight for source in config.sources},
        max_examples=config.max_mixed_examples,
        seed=config.seed,
    )
    train_documents, validation_documents = split_documents(
        mixed,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )
    train_sequences, train_token_count = pack_documents(
        train_documents,
        tokenizer,
        context_length=config.context_length,
    )
    validation_sequences, validation_token_count = pack_documents(
        validation_documents,
        tokenizer,
        context_length=config.context_length,
    )

    selected_counts: dict[str, int] = defaultdict(int)
    selected_tokens: dict[str, int] = defaultdict(int)
    for document in mixed:
        selected_counts[document.source_key] += 1
        selected_tokens[document.source_key] += len(tokenizer.encode(document.text, add_eos=True))
    source_metadata = tuple(
        SourceMetadata(
            key=source.key,
            name=source.name,
            subset=source.subset,
            split=source.split,
            source_url=source.source_url,
            license=source.license,
            license_status=source.license_status,
            license_notes=source.license_notes,
            raw_examples=len(raw_records[source.key]),
            cleaned_examples=len(documents_by_source[source.key]),
            selected_examples=selected_counts[source.key],
            approximate_tokens=selected_tokens[source.key],
            observed_schema=schemas.get(source.key, {}),
        )
        for source in config.sources
    )
    metadata = PretrainingMetadata(
        raw_examples=sum(len(records) for records in raw_records.values()),
        cleaned_examples=sum(len(documents) for documents in documents_by_source.values()),
        selected_examples=len(mixed),
        approximate_tokens=train_token_count + validation_token_count,
        train_documents=len(train_documents),
        validation_documents=len(validation_documents),
        train_sequences=train_sequences.shape[0],
        validation_sequences=validation_sequences.shape[0],
        context_length=config.context_length,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
        deduplicate=config.deduplicate,
        min_characters=config.min_characters,
        max_characters=config.max_characters,
        sources=source_metadata,
    )
    prepared = PreparedPretrainingData(train_sequences, validation_sequences, metadata)
    if save:
        _save_artifacts(
            prepared,
            raw_records,
            train_documents,
            validation_documents,
            config.output_dir,
        )
    return prepared


def load_token_sequences(path: str | Path, *, context_length: int) -> torch.Tensor:
    """Load and validate a tensor produced by the Phase 11 artifact writer."""

    sequences = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(sequences, torch.Tensor):
        raise ValueError("token sequence file must contain a tensor")
    FixedTokenSequenceDataset(sequences, context_length=context_length)
    return sequences
