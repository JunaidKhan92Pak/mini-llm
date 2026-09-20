"""Bounded, version-pinned TinyStories preparation; not instruction Q&A training."""

import argparse
import hashlib
import json
import random
import shutil
from collections.abc import Iterable, Mapping
from itertools import islice
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from huggingface_hub import HfApi

from mini_llm.bpe_tokenizer import BPETokenizer
from mini_llm.pretraining.pipeline import (
    CleanDocument,
    _write_jsonl,
    normalize_text,
    pack_documents,
)

DATASET = "roneneldan/TinyStories"
STORY_PROMPTS = (
    "Once upon a time, two friends",
    "One day, Lily lost her toy.",
    "Tom and his friend went to the park.",
    "A little dog wanted to help",
)


def expand_stories(
    records: Iterable[Mapping[str, Any]],
    *,
    base_dir: Path,
    output_dir: Path,
    total_train_examples: int = 25000,
    max_new_records: int = 50000,
) -> dict[str, Any]:
    """Extend a pinned corpus without replacing its tokenizer or validation set."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing corpus: {output_dir}")
    parent = json.loads((base_dir / "metadata.json").read_text(encoding="utf-8"))

    def read_rows(path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    train = read_rows(base_dir / "cleaned/train.jsonl")
    validation = read_rows(base_dir / "cleaned/validation.jsonl")
    raw = read_rows(base_dir / "raw/train.jsonl")
    if total_train_examples <= len(train) or max_new_records < total_train_examples - len(train):
        raise ValueError("expansion requires a larger total and sufficient inspection limit")
    tokenizer_path = base_dir / "tokenizer.json"
    digest = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    if digest != parent["tokenizer_sha256"]:
        raise ValueError("parent tokenizer fingerprint changed")
    tokenizer = BPETokenizer.load(tokenizer_path)

    def canonical(text):
        return hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()

    seen = {canonical(row["text"]) for row in train + validation}
    if len(seen) != len(train) + len(validation):
        raise ValueError("parent corpus contains duplicates or split leakage")
    counts = {
        "raw": len(raw),
        "malformed_or_length": parent["train"]["malformed_or_length"],
        "duplicates": parent["train"]["duplicates"],
        "retained": len(train),
    }
    original = len(train)
    # The same pinned source ordering supplied the original bounded raw prefix.
    additional = islice(records, parent["train"]["raw"], parent["train"]["raw"] + max_new_records)
    for record in additional:
        counts["raw"] += 1
        raw.append(record)
        text = record.get("text")
        if not isinstance(text, str):
            counts["malformed_or_length"] += 1
            continue
        text = normalize_text(text.replace("<|endoftext|>", ""))
        if not 80 <= len(text) <= 8000:
            counts["malformed_or_length"] += 1
            continue
        key = canonical(text)
        if key in seen:
            counts["duplicates"] += 1
            continue
        seen.add(key)
        train.append({"text": text})
        if len(train) == total_train_examples:
            break
    if len(train) != total_train_examples:
        raise ValueError(f"bounded source yielded {len(train)} stories, not {total_train_examples}")
    random.Random(parent["seed"]).shuffle(train)
    sequences, tokens = pack_documents(
        [CleanDocument(r["text"], "tinystories", "story", "en") for r in train],
        tokenizer,
        context_length=parent["context_length"],
    )
    output_dir.mkdir(parents=True)
    for folder in ("raw", "cleaned", "tokenized"):
        (output_dir / folder).mkdir()
    _write_jsonl(output_dir / "raw/train.jsonl", raw)
    _write_jsonl(output_dir / "cleaned/train.jsonl", train)
    torch.save(sequences, output_dir / "tokenized/train.pt")
    # Copy these unchanged for exact checkpoint lineage and comparable evaluation.
    for name in (
        "tokenizer.json",
        "raw/validation.jsonl",
        "cleaned/validation.jsonl",
        "tokenized/validation.pt",
    ):
        shutil.copyfile(base_dir / name, output_dir / name)
    counts["retained"] = len(train)
    metadata = {
        **parent,
        "train": {**counts, "tokens": tokens, "sequences": len(sequences)},
        "settings": {
            **parent["settings"],
            "train_examples": total_train_examples,
            "max_new_records": max_new_records,
        },
        "expansion": {
            "parent_corpus": str(base_dir),
            "original_stories_preserved": original,
            "new_stories": len(train) - original,
            "tokenizer_retrained": False,
            "validation_unchanged": True,
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def clean_stories(
    records: Iterable[Mapping[str, Any]],
    *,
    max_examples: int,
    seen: set[str],
    min_characters: int = 80,
    max_characters: int = 8000,
) -> tuple[list[CleanDocument], list[Mapping[str, Any]], dict[str, int]]:
    """Bound inspection and remove canonical duplicates across both source splits."""
    if max_examples <= 0 or not 0 < min_characters <= max_characters:
        raise ValueError("invalid example or character limits")
    documents, raw = [], []
    counts = {"raw": 0, "malformed_or_length": 0, "duplicates": 0, "retained": 0}
    for record in records:
        if counts["raw"] >= max_examples:
            break
        counts["raw"] += 1
        raw.append(record)
        text = record.get("text")
        if not isinstance(text, str):
            counts["malformed_or_length"] += 1
            continue
        text = normalize_text(text.replace("<|endoftext|>", ""))
        if not min_characters <= len(text) <= max_characters:
            counts["malformed_or_length"] += 1
            continue
        digest = hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()
        if digest in seen:
            counts["duplicates"] += 1
            continue
        seen.add(digest)
        documents.append(CleanDocument(text, "tinystories", "story", "en"))
    counts["retained"] = len(documents)
    return documents, raw, counts


def prepare_stories(
    train_records: Iterable[Mapping[str, Any]],
    validation_records: Iterable[Mapping[str, Any]],
    *,
    output_dir: Path,
    revision: str,
    train_examples: int = 5000,
    validation_examples: int = 500,
    vocab_size: int = 4096,
    context_length: int = 256,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit BPE on training only, preserve official validation, save reusable artifacts."""
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing corpus: {output_dir}")
    seen: set[str] = set()
    train, train_raw, train_counts = clean_stories(
        train_records, max_examples=train_examples, seen=seen
    )
    validation, validation_raw, validation_counts = clean_stories(
        validation_records, max_examples=validation_examples, seen=seen
    )
    if not train or not validation:
        raise ValueError("both splits need retained stories")
    random.Random(seed).shuffle(train)
    random.Random(seed + 1).shuffle(validation)
    tokenizer = BPETokenizer.train((d.text for d in train), vocab_size=vocab_size)
    train_sequences, train_tokens = pack_documents(train, tokenizer, context_length=context_length)
    validation_sequences, validation_tokens = pack_documents(
        validation, tokenizer, context_length=context_length
    )
    output_dir.mkdir(parents=True)
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    _write_jsonl(output_dir / "raw" / "train.jsonl", train_raw)
    _write_jsonl(output_dir / "raw" / "validation.jsonl", validation_raw)
    for split, documents, sequences in (
        ("train", train, train_sequences),
        ("validation", validation, validation_sequences),
    ):
        _write_jsonl(
            output_dir / "cleaned" / f"{split}.jsonl", ({"text": d.text} for d in documents)
        )
        (output_dir / "tokenized").mkdir(exist_ok=True)
        torch.save(sequences, output_dir / "tokenized" / f"{split}.pt")
    metadata = {
        "dataset": DATASET,
        "revision": revision,
        "source_url": f"https://huggingface.co/datasets/{DATASET}",
        "license": "CDLA-Sharing-1.0",
        "license_notes": "Synthetic stories; retain source and license when redistributing data.",
        "purpose": "Story completion pretraining; does not establish factual Q&A accuracy.",
        "context_length": context_length,
        "seed": seed,
        "train": {**train_counts, "tokens": train_tokens, "sequences": len(train_sequences)},
        "validation": {
            **validation_counts,
            "tokens": validation_tokens,
            "sequences": len(validation_sequences),
        },
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "tokenizer_training_split": "train",
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "evaluation_prompts": STORY_PROMPTS,
        "settings": {
            "train_examples": train_examples,
            "validation_examples": validation_examples,
            "requested_vocab_size": vocab_size,
            "min_characters": 80,
            "max_characters": 8000,
            "canonical_cross_split_deduplication": True,
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--train-examples", type=int, default=5000)
    parser.add_argument("--validation-examples", type=int, default=500)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--revision", help="Dataset commit; resolved once if omitted")
    parser.add_argument(
        "--expand-from", type=Path, help="Reuse original stories, BPE and validation"
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing corpus: {args.output_dir}")
    if args.expand_from:
        parent = json.loads((args.expand_from / "metadata.json").read_text(encoding="utf-8"))
        if args.revision and args.revision != parent["revision"]:
            raise ValueError("expansion must use the original dataset revision")
        records = load_dataset(DATASET, split="train", revision=parent["revision"], streaming=True)
        print(
            json.dumps(
                expand_stories(
                    records,
                    base_dir=args.expand_from,
                    output_dir=args.output_dir,
                    total_train_examples=args.train_examples,
                ),
                indent=2,
            )
        )
        return
    revision = HfApi().dataset_info(DATASET, revision=args.revision).sha
    records = {
        split: load_dataset(DATASET, split=split, revision=revision, streaming=True)
        for split in ("train", "validation")
    }
    metadata = prepare_stories(
        records["train"],
        records["validation"],
        output_dir=args.output_dir,
        revision=revision,
        train_examples=args.train_examples,
        validation_examples=args.validation_examples,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        seed=args.seed,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
