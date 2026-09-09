"""Configuration and vetted source catalog for pretraining data preparation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

LicenseStatus = Literal["documented", "review_required"]
ContentKind = Literal["general", "educational", "code"]


@dataclass(frozen=True, slots=True)
class DatasetSourceConfig:
    """One bounded Hugging Face dataset source in a mixture."""

    name: str
    subset: str | None
    split: str
    text_fields: tuple[str, ...]
    max_examples: int
    mixture_weight: float
    source_url: str
    license: str
    license_status: LicenseStatus
    license_notes: str
    content_kind: ContentKind
    language: str
    required_values: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("dataset name must not be empty")
        if not self.split.strip():
            raise ValueError("dataset split must not be empty")
        if not self.text_fields or any(not field.strip() for field in self.text_fields):
            raise ValueError("text_fields must contain at least one non-empty field")
        if self.max_examples <= 0:
            raise ValueError(f"max_examples must be positive, got {self.max_examples}")
        if self.mixture_weight <= 0.0:
            raise ValueError(
                f"mixture_weight must be positive, got {self.mixture_weight}"
            )
        if not self.source_url.strip() or not self.license.strip():
            raise ValueError("source_url and license must be documented")

    @property
    def key(self) -> str:
        return f"{self.name}:{self.subset or 'default'}"


@dataclass(frozen=True, slots=True)
class PretrainingDataConfig:
    """Reproducible cleaning, mixing, splitting, and packing settings."""

    sources: tuple[DatasetSourceConfig, ...]
    context_length: int
    validation_fraction: float = 0.1
    seed: int = 42
    output_dir: Path = Path("data/processed/dev")
    cache_dir: Path | None = Path("data/cache/huggingface")
    deduplicate: bool = True
    min_characters: int = 40
    max_characters: int = 100_000
    max_mixed_examples: int | None = 1_000

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError("at least one dataset source is required")
        keys = [source.key for source in self.sources]
        if len(keys) != len(set(keys)):
            raise ValueError("dataset source name/subset pairs must be unique")
        if self.context_length <= 0:
            raise ValueError(f"context_length must be positive, got {self.context_length}")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0.0 and 1.0")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")
        if self.min_characters <= 0:
            raise ValueError("min_characters must be positive")
        if self.max_characters < self.min_characters:
            raise ValueError("max_characters must be at least min_characters")
        if self.max_mixed_examples is not None and self.max_mixed_examples <= 1:
            raise ValueError("max_mixed_examples must be greater than one or None")


def development_dataset_sources(
    *,
    max_examples_per_source: int = 20,
) -> tuple[DatasetSourceConfig, ...]:
    """Return the bounded Phase 11 educational and multi-language code mixture."""

    fineweb = DatasetSourceConfig(
        name="HuggingFaceFW/fineweb-edu",
        subset="sample-10BT",
        split="train",
        text_fields=("text",),
        max_examples=max_examples_per_source,
        mixture_weight=0.30,
        source_url="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
        license="ODC-By-1.0",
        license_status="documented",
        license_notes="Also subject to Common Crawl terms of use.",
        content_kind="general",
        language="en",
        required_values=(("language", ("en",)),),
    )
    cosmopedia = DatasetSourceConfig(
        name="HuggingFaceTB/smollm-corpus",
        subset="cosmopedia-v2",
        split="train",
        text_fields=("text",),
        max_examples=max_examples_per_source,
        mixture_weight=0.20,
        source_url="https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus",
        license="ODC-By-1.0",
        license_status="documented",
        license_notes="Synthetic educational text; retain dataset attribution and provenance.",
        content_kind="educational",
        language="en",
    )
    code_languages = ("python", "javascript", "typescript", "html", "css", "sql")
    code_weight = 0.50 / len(code_languages)
    code_sources = tuple(
        DatasetSourceConfig(
            name="bigcode/the-stack-smol-xs",
            subset=language,
            split="train",
            text_fields=("content",),
            max_examples=max_examples_per_source,
            mixture_weight=code_weight,
            source_url="https://huggingface.co/datasets/bigcode/the-stack-smol-xs",
            license="various original repository licenses",
            license_status="review_required",
            license_notes=(
                "The smol-xs rows omit per-file license metadata; review provenance and "
                "license obligations before production training or redistribution."
            ),
            content_kind="code",
            language=language,
        )
        for language in code_languages
    )
    return (fineweb, cosmopedia, *code_sources)


def development_pretraining_config(
    *,
    context_length: int = 128,
    max_examples_per_source: int = 20,
    max_mixed_examples: int = 100,
    validation_fraction: float = 0.1,
    seed: int = 42,
    output_dir: Path = Path("data/processed/dev"),
    cache_dir: Path | None = Path("data/cache/huggingface"),
) -> PretrainingDataConfig:
    """Build the bounded, CPU-friendly public-data configuration for development."""

    return PretrainingDataConfig(
        sources=development_dataset_sources(
            max_examples_per_source=max_examples_per_source,
        ),
        context_length=context_length,
        validation_fraction=validation_fraction,
        seed=seed,
        output_dir=output_dir,
        cache_dir=cache_dir,
        deduplicate=True,
        max_mixed_examples=max_mixed_examples,
    )
