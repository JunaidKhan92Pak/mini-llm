# JeePeeTee

JeePeeTee is an educational project for building a small, decoder-only GPT-style language
model directly with Python and PyTorch. The goal is a technically correct and understandable
implementation that grows from a CPU-friendly debug model toward configurations of roughly
1M, 5M, and 10M parameters.

This repository has completed **Phase 11: real pretraining dataset pipeline**. `JeePeeTee` combines
token and positional embeddings, configurable stacked causal Transformer blocks, final LayerNorm,
and an untied vocabulary projection that returns raw logits. Unit tests validate each component,
while a CPU integration test covers the complete text-to-loss-and-backward pipeline before real
training begins.

## Requirements

- Python 3.11 or newer
- Windows, Linux, or macOS
- CPU-only development is supported; CUDA is optional and is never assumed
- Hugging Face `tokenizers` for locally trained byte-level BPE

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
│   ├── model/          # Embeddings, attention, Transformer blocks, and decoder
│   ├── pretraining/    # Public-source configs, cleaning, mixing, packing, and manifests
│   └── training/       # Local windows, baseline trainer, metrics, and checkpoints
├── tests/              # Automated tests
├── pyproject.toml      # Package metadata, dependencies, and tool configuration
└── README.md
```

Generated raw, cleaned, and tokenized datasets live under the configured `data/` location and are
ignored by Git. Checkpoints and other generated training artifacts also remain local.

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
5. **Phase 4 — Debug decoder model and language objective (complete):** normalization, feed-forward layers,
   residual paths, stacked blocks, logits, target shifting, and overfit-one-batch verification.
6. **Phase 5 — Baseline local training (complete):** leakage-aware token windows, deterministic
   batches, AdamW training, validation metrics, gradient clipping, and checkpoint resume.
7. **Phase 6 — Complete Transformer block (complete):** isolated pre-norm block verification,
   explicit residual shape guards, causality, dropout behavior, and parameter accounting.
8. **Phase 7 — Complete GPT-style model (complete):** configurable block stacking, final
   normalization, raw vocabulary logits, model-level validation, and parameter accounting.
9. **Phase 8 — Complete model validation and testing (complete):** component-level regression
   tests plus a tokenizer-to-loss-and-backward integration smoke test.
10. **Phase 9 — Tiny overfit / sanity training (complete):** deterministic tiny-text
    memorization, prediction inspection, finite-gradient checks, and checkpoint verification.
11. **Phase 10 — Real BPE tokenizer integration (complete):** locally trained byte-level BPE,
    stable special tokens, deterministic save/load, Unicode coverage, and MiniGPT integration.
12. **Phase 11 — Real pretraining dataset pipeline (complete):** bounded public-data streaming,
    schema inspection, cleaning, deduplication, weighted mixing, deterministic splitting, BPE
    packing, reusable artifacts, and a MiniGPT smoke test.
13. **Later phases:** generation, pretraining, scaling,
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

This character tokenizer remains available for educational tests and debugging. Future corpus
training should use the byte-level BPE tokenizer introduced in Phase 10.

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

## Phase 5 baseline training

The data pipeline first splits one local token stream contiguously into independent training and
validation regions, then creates shifted windows inside each region. No window crosses the split.
The training loader shuffles deterministically with a seeded PyTorch generator; validation order
is fixed.

The baseline trainer uses AdamW on one automatically selected device, tracks global steps and
tokens processed, evaluates periodically, and optionally clips gradients. Versioned checkpoints
contain model configuration, model weights, optimizer state, training arguments, counters, and
PyTorch RNG state. Loading rejects incompatible model configurations rather than silently applying
weights to the wrong architecture.

## Phase 6 complete Transformer block

The reusable block preserves `[B, T, D]` through two pre-normalized residual sublayers:

```text
x = x + CausalMultiHeadAttention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))
```

Residual additions explicitly require identical tensor shapes; they never reshape or rely on
broadcasting. For the development configuration (`D=64`, four heads, feed-forward width `256`),
one block has 49,984 trainable parameters: 16,640 in attention projections, 33,088 in the MLP,
and 256 in the two LayerNorm modules. Dropout contains no parameters.

## Phase 7 complete JeePeeTee model

The model's forward path is intentionally focused:

```text
token IDs [B, T]
→ token + learned position embeddings [B, T, D]
→ pre-norm causal Transformer block × N [B, T, D]
→ final LayerNorm [B, T, D]
→ linear LM head [B, T, V]
```

The forward method returns raw logits—never probabilities—because cross-entropy applies the
required log-softmax. Dataset windows already provide `inputs = sequence[:-1]` and
`targets = sequence[1:]`; the trainer must not shift them a second time.

The embedding table and LM head are deliberately untied for this first baseline. Keeping their
roles independent makes the implementation and parameter accounting clearer. For the development
configuration with vocabulary `128`, context `64`, width `64`, two blocks, four heads, and MLP
width `256`, JeePeeTee has 120,576 trainable parameters: 12,288 embeddings, 99,968 Transformer
blocks, 128 final normalization, and 8,192 LM head. PyTorch's well-tested default initialization is
retained; specialized GPT initialization is deferred until scaling evidence justifies it.

## Phase 8 verification gate

The test suite checks both isolated behavior and cross-component contracts. The end-to-end CPU
gate encodes tiny text, creates exactly-once-shifted input and target windows through a DataLoader,
runs JeePeeTee to obtain raw logits, computes cross-entropy, and performs backward propagation.
It verifies finite outputs and gradients without running a real training job or optimizer loop.

## Phase 9 tiny overfit gate

The Phase 9 runner intentionally memorizes one small, repeated local string with the complete
JeePeeTee architecture. It uses AdamW, gradient clipping, deterministic CPU execution, periodic
loss and teacher-forced next-token accuracy logs, and optional checkpoint round-trip validation.
The inspected prediction is computed on an existing training example; this is not yet
autoregressive text generation.

```powershell
uv run python -m mini_llm.tiny_training
```

The default gate uses a two-layer, 32-dimensional model with context length `16`, batch size `32`,
learning rate `0.01`, and `100` optimizer steps. It is a learning correctness test, not pretraining.

## Phase 10 byte-level BPE tokenizer

`BPETokenizer` trains a new BPE vocabulary from local strings through Hugging Face's lightweight
`tokenizers` library; it never downloads or reuses a pretrained tokenizer. Byte-level coverage
preserves arbitrary Unicode text, while learned merges compress common byte sequences into
subword tokens. A target vocabulary is supplied explicitly when training, with a minimum of `260`
entries: all 256 byte values plus four special tokens.

The special-token order is centralized as `<pad>=0`, `<bos>=1`, `<eos>=2`, and `<unk>=3` and is
validated again when a saved tokenizer is loaded. The model configuration must use the
tokenizer's reported `vocab_size`; `validate_vocab_size` fails clearly on a mismatch. The original
character tokenizer remains available for small educational experiments.

## Phase 11 pretraining data preparation

The development source catalog combines English educational text from FineWeb-Edu and Cosmopedia
v2 with Python, JavaScript, TypeScript, HTML, CSS, and SQL subsets from The Stack Smol XS. Every
source has a configurable cap and mixture weight. Loading uses Hugging Face Datasets streaming so
raising those limits does not require loading a complete upstream dataset first.

FineWeb-Edu and the SmolLM corpus declare ODC-By-1.0; FineWeb-Edu is additionally subject to
Common Crawl terms. The Stack contains files governed by their original repository licenses. The
Smol XS rows do not expose per-file license metadata, so those code sources are explicitly marked
`review_required` in generated manifests and require a provenance/license review before production
training or redistribution.

Preparation writes separate `raw/`, `cleaned/`, and `tokenized/` directories plus `metadata.json`.
The manifest records source URLs, subsets, splits, licensing notes, observed schemas, raw/cleaned/
selected counts, approximate BPE token counts, preprocessing settings, and final sequence counts.
Documents are split before token packing; fixed rows contain `context_length + 1` IDs so the dataset
returns inputs and next-token targets shifted exactly once.

Manually downloaded JSONL files can be prepared reproducibly without contacting the Hub again:

```powershell
uv run python -m mini_llm.pretraining.local_prepare data/raw/manual `
  data/processed/phase11-development --vocab-size 2048 --context-length 64
```

The first local development corpus contained 350 raw documents. Cleaning removed one exact
duplicate, weighted mixing selected 300 documents, and the resulting 2,048-token BPE vocabulary
encoded approximately 829,458 tokens. The deterministic 90/10 document split produced 11,835
training and 925 validation sequences. These generated data artifacts remain git-ignored.

## Reproducibility

Call `seed_everything` before experiments. It seeds Python and PyTorch, including all CUDA devices
when available. Strict deterministic algorithms are opt-in because some operations or platforms
do not support them and because they can reduce performance.

## Phase 12 first real pretraining run

The first bounded real-data experiment uses a 846,912-parameter JeePeeTee model: vocabulary
`2,048`, context `64`, width `96`, four Transformer blocks, four attention heads, MLP width `384`,
and dropout `0.1`. Batches are selected deterministically from the global step, so checkpoint
resume continues with the same next batch rather than silently restarting a shuffled iterator.

The trainer uses causal next-token cross-entropy with dataset rows shifted exactly once, AdamW,
gradient clipping, 20 warmup steps, linear learning-rate decay, fixed-slice train/validation
evaluation, controlled greedy samples, and JSONL metric logs. Phase 12 checkpoints contain model,
optimizer, scheduler and RNG state; counters; model/run/runtime configuration; full tokenizer
configuration and fingerprint; and the Phase 11 dataset manifest.

```powershell
python -m mini_llm.pretraining.run data/processed/phase11-development `
  checkpoints/phase12-final --max-steps 300 --batch-size 8 `
  --learning-rate 0.0003 --warmup-steps 20 --device cpu
```

The recorded CPU run processed 153,600 target tokens in 300 steps. Fixed-slice training loss fell
from `7.7857` to `6.7870`; validation loss fell from `7.7953` to `6.7358` (perplexity `2429.19` to
`842.05`). It was deliberately interrupted at step 100 and successfully resumed from the saved
checkpoint. Greedy prompt completions remained dominated by punctuation, so this run proves stable
real-data learning and integration—not useful text-generation quality or completed pretraining.

## Phase 13 pretrained-model evaluation

Generation now lives outside the model architecture and supports greedy decoding or fixed-seed
sampling with temperature, top-k, top-p, repetition penalty, EOS stopping, maximum-token limits,
control-token suppression, and rolling context cropping. The evaluation suite keeps seven prompts
fixed across runs: English completion, general knowledge, computer science, Python, JavaScript,
code completion, and a short technical explanation.

```powershell
python -m mini_llm.evaluation checkpoints/phase12-final/checkpoint-step-000300.pt `
  data/processed/phase11-development/tokenizer.json `
  checkpoints/phase13-evaluation/report.json --device cpu
```

The step-300 checkpoint had the best logged validation loss, narrowly beating step 200 (`6.7358`
versus `6.7383`). All evaluated logits were finite and modest (`-4.19` to `3.86`), but greedy output
collapsed mainly to periods and no prompt generated EOS. Controlled sampling increased token
diversity but remained incoherent and heavily code/markup-like.

This is underfitting, not memorization or train/validation overfitting: both losses remain high and
their final fixed-slice gap is about `-0.051`. The tokenizer produced zero UNK tokens in the packed
training split. The more important data issue is length imbalance: document weights requested 50%
code, while code contributed about 76% of tokens and HTML/CSS alone contributed about 42%. EOS was
only about 0.035% of training tokens. The 153,600-token run also covered only about 20% of the
769,275 packed training tokens. Before increasing parameter count, rebalance at the token level,
add more useful document boundaries, and train this same model for at least one controlled pass.

## Phase 14 five-million-parameter preset

The existing decoder-only architecture now has three centralized presets at vocabulary size 2,048:

- debug: 366,336 parameters (`D=64`, 2 blocks, 4 heads, FFN 256, context 64);
- mini: 846,912 parameters (`D=96`, 4 blocks, 4 heads, FFN 384, context 64);
- five-million: 5,030,656 parameters (`D=256`, 5 blocks, 8 heads, FFN 1,024,
  context 128, dropout 0.1).

The 5M preset uses a hardware-friendly head dimension of 32 and changes no model behavior. FP32
weights occupy about 19.2 MiB (FP16 about 9.6 MiB); FP32 weights, gradients, and AdamW states have a
lower-bound footprint around 76.8 MiB before activations and temporary tensors. A T4-class 16 GB
GPU should comfortably start at context 128 and batch size 32 in the current FP32 trainer, then tune
upward from measured memory. CPU execution remains suitable for smoke tests but not efficient full
pretraining.

The real Phase 11 data smoke used batch size 2 for three optimizer steps. Train loss moved from
`7.8384` to `7.8290`, validation loss from `7.7851` to `7.7783`, and checkpoint round-trip remained
valid. The trainer now correctly accepts data sequences shorter than a model's maximum context.
This preset is technically ready, but Phase 13's token-mixture imbalance and undertraining findings
still need correction before spending resources on a full 5M run.
