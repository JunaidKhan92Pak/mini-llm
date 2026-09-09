"""Prepare a local Phase 15 corpus after the documented sources are downloaded."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from mini_llm.bpe_tokenizer import BPETokenizer
from mini_llm.pretraining.balanced_pipeline import (
    BalancedCorpusConfig,
    phase15_sources,
    prepare_balanced_corpus,
)

PHASE15_FILENAMES = {
    "fineweb_edu": "fineweb_edu.jsonl",
    "cosmopedia_v2": "cosmopedia_v2.jsonl",
    "python_codesearchnet": "python_codesearchnet.jsonl",
    "javascript_codesearchnet": "javascript_codesearchnet.jsonl",
    "dolly_writing": "dolly_15k.jsonl",
    "code_alpaca": "code_alpaca_20k.jsonl",
    "ultrachat_qa": "ultrachat_200k.jsonl",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--target-tokens", type=int, default=20_000_000)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--max-examples-per-source", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cache", action="store_true")
    arguments = parser.parse_args()

    sources = phase15_sources(arguments.max_examples_per_source)
    source_files = {
        source.key: arguments.raw_dir / PHASE15_FILENAMES[source.key]
        for source in sources
    }
    config = BalancedCorpusConfig(
        sources=sources,
        output_dir=arguments.output_dir,
        target_tokens=arguments.target_tokens,
        context_length=arguments.context_length,
        validation_fraction=arguments.validation_fraction,
        seed=arguments.seed,
    )
    prepared = prepare_balanced_corpus(
        config,
        BPETokenizer.load(arguments.tokenizer),
        source_files,
        reuse_cache=not arguments.no_cache,
    )
    print(json.dumps(asdict(prepared.metadata), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
