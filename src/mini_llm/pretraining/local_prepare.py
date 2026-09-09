"""CLI for preparing manually downloaded Phase 11 development data."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from mini_llm.bpe_tokenizer import BPETokenizer
from mini_llm.pretraining.config import (
    PretrainingDataConfig,
    development_dataset_sources,
)
from mini_llm.pretraining.pipeline import (
    PreparedPretrainingData,
    build_jsonl_record_loader,
    extract_text,
    normalize_text,
    prepare_pretraining_data,
    record_matches_filters,
)

SOURCE_FILENAMES = {
    "HuggingFaceFW/fineweb-edu:sample-10BT": "fineweb_edu.jsonl",
    "HuggingFaceTB/smollm-corpus:cosmopedia-v2": "cosmopedia_v2.jsonl",
    "bigcode/the-stack-smol-xs:python": "python.jsonl",
    "bigcode/the-stack-smol-xs:javascript": "javascript.jsonl",
    "bigcode/the-stack-smol-xs:typescript": "typescript.jsonl",
    "bigcode/the-stack-smol-xs:html": "html.jsonl",
    "bigcode/the-stack-smol-xs:css": "css.jsonl",
    "bigcode/the-stack-smol-xs:sql": "sql.jsonl",
}


def prepare_local_development_corpus(
    raw_dir: str | Path,
    output_dir: str | Path,
    *,
    vocab_size: int = 2_048,
    context_length: int = 64,
    max_general_examples: int = 100,
    max_code_examples: int = 25,
    max_mixed_examples: int = 300,
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> PreparedPretrainingData:
    """Train BPE locally, process bounded JSONL files, and persist reusable artifacts."""

    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    base_sources = development_dataset_sources(
        max_examples_per_source=max_general_examples,
    )
    sources = tuple(
        replace(
            source,
            max_examples=(
                max_code_examples if source.content_kind == "code" else max_general_examples
            ),
        )
        for source in base_sources
    )
    source_files = {
        source.key: raw_dir / SOURCE_FILENAMES[source.key] for source in sources
    }
    missing = [str(path) for path in source_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing local dataset files: {missing}")
    loader = build_jsonl_record_loader(source_files)

    tokenizer_corpus: list[str] = []
    for source in sources:
        for index, record in enumerate(loader(source)):
            if index >= source.max_examples:
                break
            if not record_matches_filters(record, source.required_values):
                continue
            text = extract_text(record, source.text_fields)
            if text is not None and (cleaned := normalize_text(text)):
                tokenizer_corpus.append(cleaned)
    tokenizer = BPETokenizer.train(
        tokenizer_corpus,
        vocab_size=vocab_size,
        min_frequency=2,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output_dir / "tokenizer.json")

    config = PretrainingDataConfig(
        sources=sources,
        context_length=context_length,
        validation_fraction=validation_fraction,
        seed=seed,
        output_dir=output_dir,
        cache_dir=None,
        deduplicate=True,
        min_characters=40,
        max_characters=100_000,
        max_mixed_examples=max_mixed_examples,
    )
    return prepare_pretraining_data(
        config,
        tokenizer,
        record_loader=loader,
        save=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--vocab-size", type=int, default=2_048)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--max-general-examples", type=int, default=100)
    parser.add_argument("--max-code-examples", type=int, default=25)
    parser.add_argument("--max-mixed-examples", type=int, default=300)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    prepared = prepare_local_development_corpus(
        arguments.raw_dir,
        arguments.output_dir,
        vocab_size=arguments.vocab_size,
        context_length=arguments.context_length,
        max_general_examples=arguments.max_general_examples,
        max_code_examples=arguments.max_code_examples,
        max_mixed_examples=arguments.max_mixed_examples,
        validation_fraction=arguments.validation_fraction,
        seed=arguments.seed,
    )
    print(json.dumps(asdict(prepared.metadata), indent=2))


if __name__ == "__main__":
    main()
