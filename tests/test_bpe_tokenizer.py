import json
from pathlib import Path

import pytest
import torch

from mini_llm.bpe_tokenizer import MINIMUM_BPE_VOCAB_SIZE, BPETokenizer
from mini_llm.config import ModelConfig, TrainingConfig
from mini_llm.model import JeePeeTee
from mini_llm.tokenizer import SPECIAL_TOKENS
from mini_llm.training import NextTokenDataset

CORPUS = [
    "JeePeeTee learns reusable language patterns.\n" * 4,
    "Byte pair encoding builds practical subword tokens.\n" * 4,
    "Small local corpora keep tokenizer tests deterministic.\n" * 4,
]
VOCAB_SIZE = 300


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    return BPETokenizer.train(CORPUS, vocab_size=VOCAB_SIZE, min_frequency=1)


def test_bpe_encode_decode_and_subword_compression(tokenizer: BPETokenizer) -> None:
    text = "JeePeeTee learns reusable language patterns."
    token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)

    assert token_ids[0] == tokenizer.bos_token_id
    assert token_ids[-1] == tokenizer.eos_token_id
    assert tokenizer.decode(token_ids, skip_special_tokens=True) == text
    assert len(token_ids) - 2 < len(text.encode("utf-8"))


def test_special_tokens_have_stable_centralized_ids(tokenizer: BPETokenizer) -> None:
    assert tokenizer.special_token_ids == dict(zip(SPECIAL_TOKENS, range(4), strict=True))
    assert tokenizer.vocabulary[:4] == SPECIAL_TOKENS
    assert tokenizer.pad([10, 11], length=4) == [10, 11, 0, 0]


def test_unseen_unicode_round_trips_with_valid_ids(tokenizer: BPETokenizer) -> None:
    text = "Previously unseen: rocket 🚀 and Urdu اردو"
    token_ids = tokenizer.encode(text)

    assert token_ids
    assert all(0 <= token_id < tokenizer.vocab_size for token_id in token_ids)
    assert tokenizer.unk_token_id not in token_ids
    assert tokenizer.decode(token_ids) == text


def test_bpe_training_is_deterministic_across_corpus_order() -> None:
    first = BPETokenizer.train(CORPUS, vocab_size=VOCAB_SIZE, min_frequency=1)
    second = BPETokenizer.train(reversed(CORPUS), vocab_size=VOCAB_SIZE, min_frequency=1)

    assert first.vocabulary == second.vocabulary
    assert first.encode(CORPUS[0]) == second.encode(CORPUS[0])


def test_bpe_save_load_preserves_complete_behavior(
    tokenizer: BPETokenizer,
    tmp_path: Path,
) -> None:
    path = tmp_path / "jeeptee-bpe.json"
    tokenizer.save(path)
    loaded = BPETokenizer.load(path)
    text = "JeePeeTee handles 🚀 bytes."

    assert loaded.vocabulary == tokenizer.vocabulary
    assert loaded.special_token_ids == tokenizer.special_token_ids
    assert loaded.encode(text, add_bos=True, add_eos=True) == tokenizer.encode(
        text,
        add_bos=True,
        add_eos=True,
    )
    assert loaded.decode(loaded.encode(text)) == text
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["type"] == "byte-level-bpe"
    assert payload["version"] == 1


def test_configured_vocabulary_and_model_compatibility(tokenizer: BPETokenizer) -> None:
    assert tokenizer.vocab_size == VOCAB_SIZE
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=8,
        embedding_dim=16,
        num_layers=1,
        num_heads=4,
        feed_forward_dim=32,
    )

    tokenizer.validate_vocab_size(config.vocab_size)
    with pytest.raises(ValueError, match="vocabulary mismatch"):
        tokenizer.validate_vocab_size(config.vocab_size + 1)


def test_bpe_ids_flow_through_dataset_and_minigpt(tokenizer: BPETokenizer) -> None:
    token_ids = torch.tensor(tokenizer.encode("".join(CORPUS)), dtype=torch.long)
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=8,
        embedding_dim=16,
        num_layers=2,
        num_heads=4,
        feed_forward_dim=32,
    )
    dataset = NextTokenDataset(token_ids, context_length=config.context_length, stride=2)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=TrainingConfig().batch_size,
        shuffle=False,
    )
    inputs, targets = next(iter(loader))

    logits = JeePeeTee(config)(inputs)

    assert inputs.shape == targets.shape == (8, config.context_length)
    assert torch.equal(inputs[:, 1:], targets[:, :-1])
    assert logits.shape == (8, config.context_length, tokenizer.vocab_size)
    assert torch.isfinite(logits).all()


def test_bpe_rejects_invalid_configuration_and_ids(tokenizer: BPETokenizer) -> None:
    with pytest.raises(ValueError, match="at least"):
        BPETokenizer.train(CORPUS, vocab_size=MINIMUM_BPE_VOCAB_SIZE - 1)
    with pytest.raises(ValueError, match="empty corpus"):
        BPETokenizer.train([""], vocab_size=MINIMUM_BPE_VOCAB_SIZE)
    with pytest.raises(ValueError, match="outside vocabulary"):
        tokenizer.decode([tokenizer.vocab_size])
    with pytest.raises(TypeError, match="must be integers"):
        tokenizer.decode([True])
