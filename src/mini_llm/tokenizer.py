"""A deterministic character tokenizer for learning tokenizer fundamentals."""

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")
TOKENIZER_TYPE = "character"
TOKENIZER_VERSION = 1


class CharacterTokenizer:
    """Map individual characters to token IDs using a fixed vocabulary."""

    def __init__(self, vocabulary: Sequence[str]) -> None:
        tokens = tuple(vocabulary)
        if tokens[: len(SPECIAL_TOKENS)] != SPECIAL_TOKENS:
            raise ValueError(
                "vocabulary must begin with the special tokens "
                f"{SPECIAL_TOKENS}, got {tokens[:len(SPECIAL_TOKENS)]}"
            )
        if len(tokens) != len(set(tokens)):
            raise ValueError("vocabulary tokens must be unique")
        if any(not isinstance(token, str) or not token for token in tokens):
            raise ValueError("vocabulary tokens must be non-empty strings")
        if any(len(token) != 1 for token in tokens[len(SPECIAL_TOKENS) :]):
            raise ValueError("non-special vocabulary tokens must be single characters")

        self._id_to_token = tokens
        self._token_to_id = {token: token_id for token_id, token in enumerate(tokens)}

    @classmethod
    def train(cls, texts: Iterable[str]) -> "CharacterTokenizer":
        """Build a deterministic vocabulary from characters in a small local corpus."""

        characters: set[str] = set()
        for text in texts:
            if not isinstance(text, str):
                raise TypeError(f"training texts must be strings, got {type(text).__name__}")
            characters.update(text)

        if not characters:
            raise ValueError("cannot train a tokenizer from an empty corpus")
        return cls((*SPECIAL_TOKENS, *sorted(characters)))

    @property
    def vocabulary(self) -> tuple[str, ...]:
        return self._id_to_token

    @property
    def vocab_size(self) -> int:
        return len(self._id_to_token)

    @property
    def pad_token_id(self) -> int:
        return self._token_to_id["<pad>"]

    @property
    def bos_token_id(self) -> int:
        return self._token_to_id["<bos>"]

    @property
    def eos_token_id(self) -> int:
        return self._token_to_id["<eos>"]

    @property
    def unk_token_id(self) -> int:
        return self._token_to_id["<unk>"]

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Convert text to IDs, mapping unseen characters to ``UNK``."""

        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")

        token_ids = [self._token_to_id.get(character, self.unk_token_id) for character in text]
        if add_bos:
            token_ids.insert(0, self.bos_token_id)
        if add_eos:
            token_ids.append(self.eos_token_id)
        return token_ids

    def decode(self, token_ids: Iterable[int], *, skip_special_tokens: bool = False) -> str:
        """Convert token IDs to text, optionally omitting all special tokens."""

        decoded: list[str] = []
        special_token_ids = set(range(len(SPECIAL_TOKENS)))
        for token_id in token_ids:
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise TypeError(f"token IDs must be integers, got {type(token_id).__name__}")
            if not 0 <= token_id < self.vocab_size:
                raise ValueError(
                    f"token ID {token_id} is outside vocabulary size {self.vocab_size}"
                )
            if skip_special_tokens and token_id in special_token_ids:
                continue
            decoded.append(self._id_to_token[token_id])
        return "".join(decoded)

    def pad(self, token_ids: Sequence[int], length: int) -> list[int]:
        """Right-pad a sequence to an exact length without silently truncating it."""

        if length < len(token_ids):
            raise ValueError(
                f"cannot pad sequence of length {len(token_ids)} to shorter length {length}"
            )
        return [*token_ids, *([self.pad_token_id] * (length - len(token_ids)))]

    def validate_vocab_size(self, model_vocab_size: int) -> None:
        """Fail clearly if a model and tokenizer do not share a vocabulary size."""

        if model_vocab_size != self.vocab_size:
            raise ValueError(
                "tokenizer/model vocabulary mismatch: "
                f"tokenizer={self.vocab_size}, model={model_vocab_size}"
            )

    def save(self, path: str | Path) -> None:
        """Serialize the tokenizer as deterministic, human-readable JSON."""

        payload = {
            "type": TOKENIZER_TYPE,
            "version": TOKENIZER_VERSION,
            "vocabulary": list(self.vocabulary),
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharacterTokenizer":
        """Load and validate a tokenizer written by :meth:`save`."""

        try:
            payload: Any = json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("tokenizer file is not valid JSON") from error

        if not isinstance(payload, dict):
            raise ValueError("tokenizer file must contain a JSON object")
        if payload.get("type") != TOKENIZER_TYPE:
            raise ValueError(f"unsupported tokenizer type: {payload.get('type')!r}")
        if payload.get("version") != TOKENIZER_VERSION:
            raise ValueError(f"unsupported tokenizer version: {payload.get('version')!r}")
        vocabulary = payload.get("vocabulary")
        if not isinstance(vocabulary, list):
            raise ValueError("tokenizer vocabulary must be a JSON list")
        return cls(vocabulary)
