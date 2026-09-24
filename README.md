# JeePeeTee

JeePeeTee is a small decoder-only Transformer built from scratch in PyTorch. The
repository deliberately separates three concerns: preparing text, training a causal
language model, and generating text from a saved checkpoint.

## Run the current model

From PowerShell in the project directory:

```powershell
.\jeeptee
```

The launcher loads `checkpoints/tinystories-25k-v1/best-validation.pt` and its matching
tokenizer. This checkpoint is a **TinyStories completion model**, not a question-answering
assistant. Every prompt is independent and `/quit` exits.

For deterministic output:

```powershell
.\jeeptee --strategy greedy
```

## Run the visual web interface

```powershell
.\jeeptee-web
```

This starts a local page at `http://127.0.0.1:8765` and opens it in the browser. The
page lets you inspect prompt BPE pieces, IDs, positions, and measured token-plus-position
embeddings before generating. During generation it shows context IDs, the last token's
output magnitude after each Transformer layer, five attention links from one head of the
final layer, raw vocabulary scores, and next-token probabilities. The full-width prompt
and prediction section sits above the visualizer. Below it are vertical prediction steps,
selected-step details, and token arrays. The clickable event log spans the full width at
the bottom. Prompt and generated token pieces each have a readable text array and a matching
ordered ID array; individual BPE pieces also appear below them. On mobile, the sections
stack in the same order.
Previous/next controls and slow/normal/fast playback let you narrate each step while the
reply builds one token at a time. The vertical path highlights the active stage, marks
completed stages for the current prediction, and scrolls to keep playback visible.
Use **Inspect prompt** to study tokenization without
running the full model, or **Generate** for the complete inference trace.
The depth effects are an educational representation of the computation order, not a
literal 3D view of every activation inside the network. Keep the terminal open while
using the page; press `Ctrl+C` to stop it.

The browser frontend lives in `web/`, while `mini_llm.web` provides the local HTTP bridge to
PyTorch. A static GitHub Pages deployment can display the interface but cannot run this
PyTorch checkpoint; a public live demo requires a separate Python inference host.

## What currently works

- locally trained byte-level BPE tokenizer;
- validated token and position embeddings;
- pre-normalized causal multi-head attention;
- stacked decoder-only Transformer blocks;
- next-token cross-entropy training;
- finite-gradient checks, gradient clipping, AdamW and learning-rate scheduling;
- deterministic validation, checkpointing and exact resume;
- greedy and controlled sampling generation;
- CPU operation and optional CUDA selection.

The active checkpoint has 15,666,048 parameters, a 4,096-token vocabulary and a
256-token context. It was trained on 25,000 TinyStories training examples, with 500
held-out validation examples.

## Runtime flow

```text
prompt
  -> byte-level BPE token IDs + <bos>
  -> token embeddings + learned position embeddings
  -> 7 pre-normalized causal Transformer blocks
  -> final LayerNorm
  -> vocabulary projection
  -> repetition penalty / temperature / top-k / top-p
  -> next token
  -> repeat until <eos> or the output limit
  -> BPE decode
```

The model predicts the next token. It does not search the training files, access the
internet, retrieve documents, remember previous prompts or verify facts.

## Active source layout

```text
src/mini_llm/
  config.py             validated model/runtime configuration
  bpe_tokenizer.py      byte-level BPE training, save/load, encode/decode
  runtime.py            device and reproducibility helpers
  model/                embeddings, attention, blocks and language model
  generation.py         greedy and sampled autoregressive decoding
  evaluation.py         fixed comparable completion evaluation
  interactive.py        one-prompt command-line interface
  web.py                local HTTP inference and trace API
  pretraining/
    config.py           data-source configuration
    pipeline.py         cleaning, splitting, packing and tensor loading
    tinystories.py      pinned TinyStories preparation
    real_training.py    optimization, metrics and checkpoints
    run.py              training command
```

```text
web/
  index.html             prompt, response and observatory structure
  styles.css             responsive pseudo-3D neural pipeline
  app.js                 API calls and real-token trace playback
```

`data/`, `checkpoints/`, virtual environments and test caches are generated local
artifacts and are excluded from Git.

## Setup and checks

Requirements: Python 3.11+, PyTorch, Hugging Face `tokenizers`, and `datasets`.

```powershell
uv sync --extra dev
uv run ruff check src tests
uv run pytest
```

## Prepare TinyStories

The existing prepared corpus is under `data/processed/tinystories-25k-v1`. To prepare a
new bounded corpus, use a new output directory; the command refuses to overwrite data.

```powershell
uv run python -m mini_llm.pretraining.tinystories data/processed/tinystories-new `
  --train-examples 25000 --validation-examples 500 `
  --context-length 256 --vocab-size 4096
```

## Train

Example CPU-compatible run:

```powershell
uv run python -m mini_llm.pretraining.run `
  data/processed/tinystories-25k-v1 `
  checkpoints/tinystories-new `
  --model-preset fifteen-million `
  --device auto --batch-size 4 --max-steps 6000
```

Training uses rows of `context_length + 1` token IDs. Inputs are `row[:-1]` and targets
are `row[1:]`; targets are shifted exactly once.

## Quality rule

A lower validation loss proves improved next-token prediction, not question-answering.
A future assistant checkpoint must be trained on a documented instruction/Q&A corpus and
must pass separate held-out and paraphrased-answer evaluations before becoming the default.

## Architecture evolution

The current checkpoint remains the reproducible baseline. A modern architecture candidate
(for example pre-RMSNorm, RoPE, grouped-query attention, SwiGLU and tied embeddings) must use
a new architecture version and checkpoint directory. It should be compared with the baseline
using the same tokenizer, corpus, token budget and evaluation suite; it must not silently
replace or load incompatible weights.
