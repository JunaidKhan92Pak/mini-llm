"""Locally trained byte-level BPE tokenizer for JeePeeTee pretraining."""

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from mini_llm.tokenizer import SPECIAL_TOKENS

BPE_TOKENIZER_TYPE = "byte-level-bpe"
BPE_TOKENIZER_VERSION = 1
MINIMUM_BPE_VOCAB_SIZE = 256 + len(SPECIAL_TOKENS)


class BPETokenizer:
    """A deterministic byte-level BPE tokenizer trained only on local text."""

    def __init__(self, backend: Tokenizer) -> None:
        self._backend = backend
        expected_ids = tuple(range(len(SPECIAL_TOKENS)))
        actual_ids = tuple(backend.token_to_id(token) for token in SPECIAL_TOKENS)
        if actual_ids != expected_ids:
            raise ValueError(
                "BPE special-token IDs must follow their canonical order: "
                f"expected {expected_ids}, got {actual_ids}"
            )

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        *,
        vocab_size: int,
        min_frequency: int = 2,
    ) -> "BPETokenizer":
        """Learn byte-pair merges from local strings without pretrained assets."""

        if not isinstance(vocab_size, int) or isinstance(vocab_size, bool):
            raise TypeError(f"vocab_size must be an integer, got {type(vocab_size).__name__}")
        if vocab_size < MINIMUM_BPE_VOCAB_SIZE:
            raise ValueError(
                f"vocab_size must be at least {MINIMUM_BPE_VOCAB_SIZE} for byte-level BPE, "
                f"got {vocab_size}"
            )
        if not isinstance(min_frequency, int) or isinstance(min_frequency, bool):
            raise TypeError(
                f"min_frequency must be an integer, got {type(min_frequency).__name__}"
            )
        if min_frequency <= 0:
            raise ValueError(f"min_frequency must be positive, got {min_frequency}")

        corpus: list[str] = []
        for text in texts:
            if not isinstance(text, str):
                raise TypeError(f"training texts must be strings, got {type(text).__name__}")
            if text:
                corpus.append(text)
        if not corpus:
            raise ValueError("cannot train a BPE tokenizer from an empty corpus")

        backend = Tokenizer(BPE(unk_token="<unk>"))
        backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
        backend.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=False,
            special_tokens=list(SPECIAL_TOKENS),
            initial_alphabet=ByteLevel.alphabet(),
        )
        backend.train_from_iterator(sorted(corpus), trainer=trainer, length=len(corpus))
        return cls(backend)

    @property
    def vocabulary(self) -> tuple[str, ...]:
        """Return vocabulary tokens ordered by their integer IDs."""

        tokens = tuple(self._backend.id_to_token(index) for index in range(self.vocab_size))
        if any(token is None for token in tokens):
            raise RuntimeError("BPE vocabulary contains a missing token ID")
        return tokens  # type: ignore[return-value]

    @property
    def vocab_size(self) -> int:
        return self._backend.get_vocab_size(with_added_tokens=True)

    @property
    def pad_token_id(self) -> int:
        return self._special_token_id("<pad>")

    @property
    def bos_token_id(self) -> int:
        return self._special_token_id("<bos>")

    @property
    def eos_token_id(self) -> int:
        return self._special_token_id("<eos>")

    @property
    def unk_token_id(self) -> int:
        return self._special_token_id("<unk>")

    @property
    def special_token_ids(self) -> dict[str, int]:
        """Expose centralized special-token IDs without external hard-coding."""

        return {token: self._special_token_id(token) for token in SPECIAL_TOKENS}

    def _special_token_id(self, token: str) -> int:
        token_id = self._backend.token_to_id(token)
        if token_id is None:
            raise RuntimeError(f"BPE tokenizer is missing required special token {token!r}")
        return token_id

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Encode text as BPE IDs, optionally adding sequence boundary tokens."""

        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")
        token_ids = self._backend.encode(text, add_special_tokens=False).ids
        if add_bos:
            token_ids.insert(0, self.bos_token_id)
        if add_eos:
            token_ids.append(self.eos_token_id)
        if any(not 0 <= token_id < self.vocab_size for token_id in token_ids):
            raise RuntimeError("BPE encoder produced an ID outside its vocabulary")
        return token_ids

    def decode(self, token_ids: Iterable[int], *, skip_special_tokens: bool = False) -> str:
        """Decode valid BPE IDs back into text."""

        validated: list[int] = []
        for token_id in token_ids:
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise TypeError(f"token IDs must be integers, got {type(token_id).__name__}")
            if not 0 <= token_id < self.vocab_size:
                raise ValueError(
                    f"token ID {token_id} is outside vocabulary size {self.vocab_size}"
                )
            validated.append(token_id)
        return self._backend.decode(
            validated,
            skip_special_tokens=skip_special_tokens,
        )

    def pad(self, token_ids: Sequence[int], length: int) -> list[int]:
        """Right-pad a sequence without silently truncating tokens."""

        if length < len(token_ids):
            raise ValueError(
                f"cannot pad sequence of length {len(token_ids)} to shorter length {length}"
            )
        return [*token_ids, *([self.pad_token_id] * (length - len(token_ids)))]

    def validate_vocab_size(self, model_vocab_size: int) -> None:
        """Fail clearly when tokenizer and model vocabulary dimensions differ."""

        if model_vocab_size != self.vocab_size:
            raise ValueError(
                "tokenizer/model vocabulary mismatch: "
                f"tokenizer={self.vocab_size}, model={model_vocab_size}"
            )

    def save(self, path: str | Path) -> None:
        """Save the complete tokenizer model and wrapper schema as one JSON file."""

        payload = {
            "type": BPE_TOKENIZER_TYPE,
            "version": BPE_TOKENIZER_VERSION,
            "tokenizer": json.loads(self._backend.to_str()),
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """Load a tokenizer saved by :meth:`save` and validate its schema."""

        try:
            payload: Any = json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("BPE tokenizer file is not valid JSON") from error
        if not isinstance(payload, dict):
            raise ValueError("BPE tokenizer file must contain a JSON object")
        if payload.get("type") != BPE_TOKENIZER_TYPE:
            raise ValueError(f"unsupported tokenizer type: {payload.get('type')!r}")
        if payload.get("version") != BPE_TOKENIZER_VERSION:
            raise ValueError(f"unsupported tokenizer version: {payload.get('version')!r}")
        serialized_backend = payload.get("tokenizer")
        if not isinstance(serialized_backend, dict):
            raise ValueError("BPE tokenizer payload is missing its backend model")
        backend = Tokenizer.from_str(json.dumps(serialized_backend, ensure_ascii=False))
        return cls(backend)
