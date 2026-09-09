# Mini LLM

Mini LLM is an educational project for building a small, decoder-only GPT-style language
model directly with Python and PyTorch. The goal is a technically correct and understandable
implementation that grows from a CPU-friendly debug model toward configurations of roughly
1M, 5M, and 10M parameters.

This repository is currently at **Phase 0: project foundation**. It contains configuration,
runtime reproducibility helpers, and test infrastructure. It does not yet contain a tokenizer,
Transformer, dataset pipeline, training loop, or inference implementation.

## Requirements

- Python 3.11 or newer
- Windows, Linux, or macOS
- CPU-only development is supported; CUDA is optional and is never assumed

[`uv`](https://docs.astral.sh/uv/) is recommended for reproducible local environments, though
the project remains installable with standard Python tooling.

## Setup

```powershell
uv sync --extra dev
```

Activate the environment if desired:

```powershell
.venv\Scripts\Activate.ps1
```

On Linux or macOS, use `source .venv/bin/activate` instead.

## Checks

```powershell
uv run pytest
uv run ruff check .
```

## Current structure

```text
mini-llm/
├── src/mini_llm/       # Importable package, configuration, and runtime helpers
├── tests/              # Automated tests
├── pyproject.toml      # Package metadata, dependencies, and tool configuration
└── README.md
```

Directories for data, checkpoints, training, inference, and evaluation will be introduced only
when their corresponding phases begin.

## Configuration approach

Configuration is represented by immutable, validated dataclasses. Model dimensions are kept in
one place, and invalid combinations such as an embedding dimension that cannot be divided among
the attention heads fail immediately. The development preset intentionally requires an explicit
vocabulary size so that the future tokenizer and model cannot silently disagree.

## Roadmap

1. **Phase 0 — Foundation (current):** packaging, configuration, reproducibility, and tests.
2. **Phase 1 — Tokenizer fundamentals:** deterministic encode/decode behavior and special-token
   handling using a tiny local corpus.
3. **Phase 2 — Transformer components:** causal attention, feed-forward layers, residual paths,
   and shape-focused tests.
4. **Phase 3 — Debug model and objective:** end-to-end forward pass, target shifting, loss, and
   backward-pass verification.
5. **Later phases:** checkpointing, generation, evaluation, curated data, pretraining, scaling,
   and supervised instruction fine-tuning—each gated by tests at the previous scale.

## Reproducibility

Call `seed_everything` before experiments. It seeds Python and PyTorch, including all CUDA devices
when available. Strict deterministic algorithms are opt-in because some operations or platforms
do not support them and because they can reduce performance.

