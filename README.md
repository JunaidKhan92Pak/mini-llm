# JeePeeTee

JeePeeTee is an educational project for building a small, decoder-only GPT-style language
model directly with Python and PyTorch. The goal is a technically correct and understandable
implementation that grows from a CPU-friendly debug model toward configurations of roughly
1M, 5M, and 10M parameters.

This repository is currently at **Phase 4: debug decoder and language objective**. JeePeeTee now
has pre-normalized Transformer blocks, feed-forward networks, residual paths, stacked blocks, final
vocabulary logits, explicit next-token shifting, and a deterministic one-batch overfit gate. It
does not yet contain a production training loop, generation, checkpoints, or real language data.

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
uv run python -m mini_llm.sanity
```

## Current structure

```text
mini-llm/
├── src/mini_llm/       # Package, runtime, tokenizer, and sanity pipeline
│   └── model/          # Embeddings, attention, Transformer blocks, and decoder
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

1. **Phase 0 — Foundation (complete):** packaging, configuration, reproducibility, and tests.
2. **Phase 1 — PyTorch sanity pipeline (complete):** deterministic tensor, gradient, optimizer,
   and loss-decrease verification on the tiny relation `y = 2x + 1`.
3. **Phase 2 — Tokenizer fundamentals (complete):** deterministic encode/decode behavior,
   special-token handling, padding, serialization, and model vocabulary synchronization.
4. **Phase 3 — Embeddings and causal attention (complete):** learned token/position embeddings,
   explicit causal masking, scaled dot-product attention, and multi-head self-attention.
5. **Phase 4 — Debug decoder model and language objective (current):** normalization, feed-forward layers,
   residual paths, stacked blocks, logits, target shifting, and overfit-one-batch verification.
6. **Later phases:** checkpointing, generation, evaluation, curated data, pretraining, scaling,
   and supervised instruction fine-tuning—each gated by tests at the previous scale.

## Phase 1 sanity pipeline

The sanity model is deliberately a single linear layer, not an LLM. It learns the fixed synthetic
relation `y = 2x + 1` using mean-squared error and SGD. If its deterministic loss falls sharply and
its parameters converge toward weight `2` and bias `1`, the essential PyTorch training path is
working. Run it with:

```powershell
uv run python -m mini_llm.sanity
```

## Phase 2 educational tokenizer

`CharacterTokenizer` assigns fixed IDs to `PAD`, `BOS`, `EOS`, and `UNK`, then assigns the
remaining IDs to sorted characters observed in a supplied local corpus. Sorting makes vocabulary
creation independent of corpus order. Known text decodes exactly, while unseen characters map to
`UNK`. Tokenizer JSON files store a type and schema version so incompatible files fail clearly.

This character tokenizer exists to make tokenization mechanics transparent. It is not intended
for meaningful language-model training; a later phase will introduce a practical subword
tokenizer after the surrounding model pipeline is correct.

## Phase 3 embeddings and attention

`TokenPositionEmbedding` converts integer token IDs shaped `(batch, sequence)` into vectors shaped
`(batch, sequence, embedding_dim)`. Learned position vectors are added because self-attention alone
does not know token order.

`scaled_dot_product_attention` computes `softmax(QKᵀ / sqrt(head_dim))V`. Its lower-triangular
causal mask prevents every token from reading future positions. `MultiHeadSelfAttention` projects
Q, K, and V directly with PyTorch linear layers, splits them into configured heads, applies causal
attention, and merges the heads back to the original embedding shape.

## Phase 4 debug decoder

Each `TransformerBlock` uses pre-normalization and two residual paths: one around causal attention
and one around a GELU feed-forward network. `JeePeeTee` stacks the configured number of blocks,
applies a final normalization, and projects every position to raw vocabulary logits shaped
`(batch, sequence, vocab_size)`.

`prepare_next_token_batch` makes the causal objective explicit by shifting each sequence into
inputs `tokens[:, :-1]` and targets `tokens[:, 1:]`. `causal_language_model_loss` applies
cross-entropy to those aligned logits and targets and supports an ignored target index for future
padding-aware batches. The debug trainer repeatedly fits one synthetic token batch; this is a
correctness gate, not a general-purpose training system.

## Reproducibility

Call `seed_everything` before experiments. It seeds Python and PyTorch, including all CUDA devices
when available. Strict deterministic algorithms are opt-in because some operations or platforms
do not support them and because they can reduce performance.
