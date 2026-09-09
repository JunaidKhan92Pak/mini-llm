from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mini_llm.bpe_tokenizer import BPETokenizer
from mini_llm.generation import GenerationConfig, generate


class ScriptedModel(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, choices: list[int]) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            vocab_size=vocab_size,
            context_length=context_length,
        )
        self.anchor = nn.Parameter(torch.zeros(()))
        self.choices = choices
        self.calls = 0
        self.input_lengths: list[int] = []

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        self.input_lengths.append(token_ids.shape[1])
        logits = torch.linspace(-1, 1, self.config.vocab_size).repeat(
            token_ids.shape[0], token_ids.shape[1], 1
        )
        choice = self.choices[min(self.calls, len(self.choices) - 1)]
        logits[:, -1, choice] = 10
        self.calls += 1
        return logits + self.anchor


@pytest.fixture
def tokenizer() -> BPETokenizer:
    return BPETokenizer.train(
        ["hello world repeated text and useful Python code"], vocab_size=300
    )


def test_generation_stops_immediately_after_eos(tokenizer: BPETokenizer) -> None:
    regular_id = tokenizer.encode("hello")[0]
    model = ScriptedModel(
        tokenizer.vocab_size, 8, [regular_id, tokenizer.eos_token_id, regular_id]
    )
    result = generate(
        model, tokenizer, "hello", GenerationConfig(max_new_tokens=10),
        device=torch.device("cpu"),
    )
    assert result.generated_token_ids == (regular_id, tokenizer.eos_token_id)
    assert result.finish_reason == "eos"
    assert model.calls == 2


def test_generation_obeys_max_tokens_and_restores_training_mode(
    tokenizer: BPETokenizer,
) -> None:
    regular_id = tokenizer.encode("world")[0]
    model = ScriptedModel(tokenizer.vocab_size, 8, [regular_id])
    model.train()
    result = generate(
        model, tokenizer, "hello", GenerationConfig(max_new_tokens=3),
        device=torch.device("cpu"),
    )
    assert len(result.generated_token_ids) == 3
    assert result.finish_reason == "max_new_tokens"
    assert model.training


def test_greedy_is_deterministic_and_generated_ids_are_valid(
    tokenizer: BPETokenizer,
) -> None:
    regular_id = tokenizer.encode("world")[0]
    first = generate(
        ScriptedModel(tokenizer.vocab_size, 8, [regular_id]),
        tokenizer,
        "hello",
        GenerationConfig(max_new_tokens=4, strategy="greedy"),
        device=torch.device("cpu"),
    )
    second = generate(
        ScriptedModel(tokenizer.vocab_size, 8, [regular_id]),
        tokenizer,
        "hello",
        GenerationConfig(max_new_tokens=4, strategy="greedy", seed=999),
        device=torch.device("cpu"),
    )
    assert first == second
    assert all(0 <= token_id < tokenizer.vocab_size for token_id in first.generated_token_ids)


def test_long_prompt_is_cropped_to_context_for_every_forward(
    tokenizer: BPETokenizer,
) -> None:
    regular_id = tokenizer.encode("world")[0]
    model = ScriptedModel(tokenizer.vocab_size, 4, [regular_id])
    prompt = "hello world " * 20
    result = generate(
        model, tokenizer, prompt, GenerationConfig(max_new_tokens=5),
        device=torch.device("cpu"),
    )
    assert result.text.startswith(prompt)
    assert model.input_lengths == [4, 4, 4, 4, 4]


def test_seeded_top_k_top_p_sampling_is_repeatable(tokenizer: BPETokenizer) -> None:
    config = GenerationConfig(
        max_new_tokens=8,
        strategy="sample",
        temperature=0.8,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.1,
        seed=123,
    )
    first = generate(
        ScriptedModel(tokenizer.vocab_size, 8, [tokenizer.vocab_size - 1]),
        tokenizer,
        "hello",
        config,
        device=torch.device("cpu"),
    )
    second = generate(
        ScriptedModel(tokenizer.vocab_size, 8, [tokenizer.vocab_size - 1]),
        tokenizer,
        "hello",
        config,
        device=torch.device("cpu"),
    )
    assert first.generated_token_ids == second.generated_token_ids


@pytest.mark.parametrize(
    ("field", "value"),
    [("temperature", 0.0), ("top_k", 0), ("top_p", 1.1), ("repetition_penalty", 0.9)],
)
def test_invalid_generation_settings_are_rejected(field: str, value: float) -> None:
    values = {field: value}
    with pytest.raises(ValueError):
        GenerationConfig(**values)
