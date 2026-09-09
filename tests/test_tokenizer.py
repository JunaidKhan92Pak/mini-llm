import json
from pathlib import Path

import pytest

from mini_llm.config import development_model_config
from mini_llm.tokenizer import SPECIAL_TOKENS, CharacterTokenizer


def test_training_is_deterministic_regardless_of_corpus_order() -> None:
    first = CharacterTokenizer.train(["Jee", "Pee", "Tee!"])
    second = CharacterTokenizer.train(["Tee!", "Pee", "Jee"])

    assert first.vocabulary == second.vocabulary
    assert first.vocabulary[:4] == SPECIAL_TOKENS


def test_encode_decode_round_trip_preserves_known_text_exactly() -> None:
    text = "JeePeeTee says:\nHello, world!"
    tokenizer = CharacterTokenizer.train([text])

    token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)

    assert token_ids[0] == tokenizer.bos_token_id
    assert token_ids[-1] == tokenizer.eos_token_id
    assert tokenizer.decode(token_ids, skip_special_tokens=True) == text


def test_special_token_ids_are_stable_and_unknown_characters_use_unk() -> None:
    tokenizer = CharacterTokenizer.train(["abc"])

    assert tokenizer.pad_token_id == 0
    assert tokenizer.bos_token_id == 1
    assert tokenizer.eos_token_id == 2
    assert tokenizer.unk_token_id == 3
    assert tokenizer.encode("z") == [tokenizer.unk_token_id]
    assert tokenizer.decode([tokenizer.unk_token_id]) == "<unk>"


def test_padding_reaches_exact_length_and_never_truncates() -> None:
    tokenizer = CharacterTokenizer.train(["abc"])
    token_ids = tokenizer.encode("ab")

    padded = tokenizer.pad(token_ids, length=4)

    assert padded == [*token_ids, tokenizer.pad_token_id, tokenizer.pad_token_id]
    with pytest.raises(ValueError, match="shorter"):
        tokenizer.pad(token_ids, length=1)


def test_save_load_round_trip_preserves_behavior(tmp_path: Path) -> None:
    tokenizer = CharacterTokenizer.train(["JeePeeTee!\n"])
    path = tmp_path / "tokenizer.json"

    tokenizer.save(path)
    loaded = CharacterTokenizer.load(path)

    assert loaded.vocabulary == tokenizer.vocabulary
    assert loaded.encode("Jee!", add_eos=True) == tokenizer.encode("Jee!", add_eos=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["type"] == "character"
    assert payload["version"] == 1


def test_load_rejects_incompatible_or_malformed_files(tmp_path: Path) -> None:
    path = tmp_path / "tokenizer.json"
    path.write_text('{"type": "bpe", "version": 1, "vocabulary": []}', encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported tokenizer type"):
        CharacterTokenizer.load(path)

    path.write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        CharacterTokenizer.load(path)


def test_model_vocabulary_must_match_tokenizer() -> None:
    tokenizer = CharacterTokenizer.train(["abc"])
    model_config = development_model_config(vocab_size=tokenizer.vocab_size)

    tokenizer.validate_vocab_size(model_config.vocab_size)
    with pytest.raises(ValueError, match="vocabulary mismatch"):
        tokenizer.validate_vocab_size(model_config.vocab_size + 1)


def test_invalid_token_ids_and_empty_corpus_fail_clearly() -> None:
    tokenizer = CharacterTokenizer.train(["abc"])

    with pytest.raises(ValueError, match="outside vocabulary"):
        tokenizer.decode([tokenizer.vocab_size])
    with pytest.raises(TypeError, match="must be integers"):
        tokenizer.decode([True])
    with pytest.raises(ValueError, match="empty corpus"):
        CharacterTokenizer.train([""])
