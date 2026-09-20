import hashlib
import json
from dataclasses import asdict

import pytest
import torch

from mini_llm.bpe_tokenizer import BPETokenizer
from mini_llm.config import fifteen_million_model_config
from mini_llm.model import JeePeeTee, causal_language_model_loss
from mini_llm.pretraining.pipeline import FixedTokenSequenceDataset, load_token_sequences
from mini_llm.pretraining.run import initialize_weights
from mini_llm.pretraining.tinystories import clean_stories, expand_stories, prepare_stories


def test_expansion_preserves_original_stories_tokenizer_and_validation(tmp_path):
    train = [{"text": f"Lily found {i} flowers and shared them with Tom. " * 5} for i in range(6)]
    validation = [{"text": "A little dog went home and shared a happy day with its friends. " * 5}]
    base = tmp_path / "base"
    prepare_stories(
        train[:2],
        validation,
        output_dir=base,
        revision="test",
        train_examples=2,
        validation_examples=1,
        context_length=16,
        vocab_size=300,
    )
    output = tmp_path / "expanded"
    metadata = expand_stories(train, base_dir=base, output_dir=output, total_train_examples=6)
    assert metadata["train"]["retained"] == 6
    assert metadata["expansion"]["new_stories"] == 4
    old = {
        json.loads(line)["text"] for line in (base / "cleaned/train.jsonl").read_text().splitlines()
    }
    new = {
        json.loads(line)["text"]
        for line in (output / "cleaned/train.jsonl").read_text().splitlines()
    }
    assert old <= new
    for name in ("tokenizer.json", "tokenized/validation.pt", "cleaned/validation.jsonl"):
        assert (base / name).read_bytes() == (output / name).read_bytes()
    with pytest.raises(ValueError, match="bounded source"):
        expand_stories(
            train[:2], base_dir=base, output_dir=tmp_path / "short", total_train_examples=6
        )
    assert not (tmp_path / "short").exists()
    with pytest.raises(FileExistsError):
        expand_stories(train, base_dir=base, output_dir=output, total_train_examples=6)


def test_warm_start_preserves_weights_and_rejects_wrong_tokenizer(tmp_path):
    from mini_llm.config import debug_model_config

    tokenizer = BPETokenizer.train(["A little dog went home happily."], vocab_size=300)
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    model = JeePeeTee(debug_model_config(tokenizer.vocab_size, context_length=16))
    checkpoint = tmp_path / "parent.pt"
    torch.save(
        {
            "model_config": asdict(model.config),
            "model_state": model.state_dict(),
            "tokenizer_metadata": {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
            "global_step": 4000,
            "tokens_processed": 4096000,
        },
        checkpoint,
    )
    child = JeePeeTee(model.config)
    metadata = initialize_weights(child, checkpoint, path)
    assert metadata["checkpoint_global_step"] == 4000
    assert all(torch.equal(v, child.state_dict()[k]) for k, v in model.state_dict().items())
    different = tmp_path / "different.json"
    different.write_text("{}")
    with pytest.raises(ValueError, match="tokenizer fingerprint"):
        initialize_weights(child, checkpoint, different)


def test_cleaning_is_bounded_and_removes_cross_split_duplicates() -> None:
    text = "Lily and Tom were friends. They helped a small dog find its way home. " * 2
    seen: set[str] = set()
    train, _, counts = clean_stories(
        [{"text": text}, {"text": None}, {"text": "short"}, {"text": text}],
        max_examples=3,
        seen=seen,
    )
    assert len(train) == 1
    assert counts["raw"] == 3
    assert counts["malformed_or_length"] == 2
    validation, _, counts = clean_stories(
        [{"text": text.upper()}],
        max_examples=1,
        seen=seen,
    )
    assert not validation
    assert counts["duplicates"] == 1


def test_preparation_is_reproducible_and_integrates_with_model(tmp_path) -> None:
    train = [
        {"text": f"Lily found {i} flowers and shared them with her friend Tom. " * 5}
        for i in range(20)
    ]
    validation = [
        {"text": f"A little dog saw {i} birds and went home happily. " * 5} for i in range(5)
    ]
    first = prepare_stories(
        train,
        validation,
        output_dir=tmp_path / "first",
        revision="test-revision",
        train_examples=20,
        validation_examples=5,
        vocab_size=300,
        context_length=16,
    )
    second = prepare_stories(
        train,
        validation,
        output_dir=tmp_path / "second",
        revision="test-revision",
        train_examples=20,
        validation_examples=5,
        vocab_size=300,
        context_length=16,
    )
    assert first == second
    assert first["tokenizer_training_split"] == "train"
    tokenizer = BPETokenizer.load(tmp_path / "first" / "tokenizer.json")
    sequences = load_token_sequences(
        tmp_path / "first" / "tokenized" / "train.pt",
        context_length=16,
    )
    torch.testing.assert_close(
        sequences,
        load_token_sequences(
            tmp_path / "second" / "tokenized" / "train.pt",
            context_length=16,
        ),
    )
    inputs, targets = FixedTokenSequenceDataset(sequences, context_length=16)[0]
    assert torch.equal(inputs[1:], targets[:-1])
    assert sequences.min() >= 0
    assert sequences.max() < tokenizer.vocab_size
    from mini_llm.config import debug_model_config

    model = JeePeeTee(debug_model_config(tokenizer.vocab_size, context_length=16))
    loss = causal_language_model_loss(model(inputs.unsqueeze(0)), targets.unsqueeze(0))
    loss.backward()
    assert torch.isfinite(loss)
    manifest = json.loads((tmp_path / "first" / "metadata.json").read_text())
    assert manifest["license"] == "CDLA-Sharing-1.0"
    with pytest.raises(FileExistsError):
        prepare_stories(train, validation, output_dir=tmp_path / "first", revision="test")


def test_fifteen_million_preset_full_context_backward_and_causality() -> None:
    torch.manual_seed(42)
    model = JeePeeTee(fifteen_million_model_config(4096)).eval()
    assert model.parameter_count() == 15_666_048
    inputs = torch.randint(0, 4096, (1, 256))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    before = model.language_model_head.weight.detach().clone()
    logits = model(inputs)
    assert logits.shape == (1, 256, 4096)
    loss = causal_language_model_loss(logits, torch.roll(inputs, -1, dims=1))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert not torch.equal(before, model.language_model_head.weight)
    with torch.no_grad():
        original = model(inputs[:, :8])
        changed = inputs[:, :8].clone()
        changed[:, 4:] = (changed[:, 4:] + 1) % 4096
        altered = model(changed)
    torch.testing.assert_close(original[:, :4], altered[:, :4])
