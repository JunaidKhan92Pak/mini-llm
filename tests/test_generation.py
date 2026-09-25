import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mini_llm.bpe_tokenizer import BPETokenizer
from mini_llm.config import debug_model_config
from mini_llm.generation import GenerationConfig, generate
from mini_llm.model import JeePeeTee


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
    return BPETokenizer.train(["hello world repeated text and useful Python code"], vocab_size=300)


def test_generation_stops_immediately_after_eos(tokenizer: BPETokenizer) -> None:
    regular_id = tokenizer.encode("hello")[0]
    model = ScriptedModel(tokenizer.vocab_size, 8, [regular_id, tokenizer.eos_token_id, regular_id])
    result = generate(
        model,
        tokenizer,
        "hello",
        GenerationConfig(max_new_tokens=10),
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
        model,
        tokenizer,
        "hello",
        GenerationConfig(max_new_tokens=3),
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
        model,
        tokenizer,
        prompt,
        GenerationConfig(max_new_tokens=5),
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


def test_generation_trace_reports_real_ranked_probabilities(tokenizer: BPETokenizer) -> None:
    regular_id = tokenizer.encode("world")[0]
    result = generate(
        ScriptedModel(tokenizer.vocab_size, 8, [regular_id]),
        tokenizer,
        "hello",
        GenerationConfig(max_new_tokens=2, strategy="greedy", trace_top_n=3),
        device=torch.device("cpu"),
    )

    assert len(result.steps) == 2
    assert result.steps[0].selected_token_id == regular_id
    assert result.steps[0].candidates[0].token_id == regular_id
    assert all(
        left.probability >= right.probability
        for left, right in zip(
            result.steps[0].candidates,
            result.steps[0].candidates[1:],
            strict=False,
        )
    )


def test_generation_trace_captures_embedding_and_each_transformer_layer(
    tokenizer: BPETokenizer,
) -> None:
    torch.manual_seed(7)
    model = JeePeeTee(debug_model_config(tokenizer.vocab_size, context_length=12))
    model.train()
    result = generate(
        model,
        tokenizer,
        "hello",
        GenerationConfig(max_new_tokens=1, strategy="greedy", trace_top_n=3),
        device=torch.device("cpu"),
    )

    step = result.steps[0]
    assert len(step.embedding_preview) == 8
    assert step.embedding_norm is not None and step.embedding_norm > 0
    assert len(step.layer_norms) == model.config.num_layers
    assert all(value > 0 for value in step.layer_norms)
    assert len(step.attention_focus) == min(5, step.context_token_count)
    assert step.projection_input_norm is not None and step.projection_input_norm > 0
    assert step.context_token_ids == tuple(tokenizer.encode("hello", add_bos=True))
    assert len(step.layer_details) == model.config.num_layers
    assert not model.embeddings._forward_hooks
    assert all(not block._forward_hooks for block in model.blocks)
    assert not model.blocks[-1].attention._forward_pre_hooks
    assert not model.final_norm._forward_hooks
    assert all(not module._forward_hooks for module in model.modules())
    assert model.training

    model.eval()
    context = torch.tensor([step.context_token_ids], dtype=torch.long)
    with torch.no_grad():
        hidden = model.embeddings(context)
        traced_hidden = hidden
        for index, (block, detail) in enumerate(
            zip(model.blocks, step.layer_details, strict=True)
        ):
            normalized_input = block.attention_norm(traced_hidden)
            attention_update = block.attention(normalized_input)
            residual = traced_hidden + attention_update
            normalized_residual = block.feed_forward_norm(residual)
            feed_forward_update = block.feed_forward(normalized_residual)
            output = residual + feed_forward_update
            expected_vectors = {
                "input": traced_hidden,
                "attention_norm": normalized_input,
                "attention": attention_update,
                "attention_residual": residual,
                "feed_forward_norm": normalized_residual,
                "feed_forward": feed_forward_update,
                "output": output,
            }
            assert detail.index == index + 1
            assert len(detail.activations) == len(expected_vectors)
            for activation in detail.activations:
                vector = expected_vectors[activation.name][0, -1]
                assert activation.width == vector.numel()
                assert activation.norm == pytest.approx(float(torch.linalg.vector_norm(vector)))
                torch.testing.assert_close(torch.tensor(activation.preview), vector[:8])
            assert detail.activations[-1].norm == step.layer_norms[index]
            assert len(detail.attention_focus) == min(5, len(step.context_token_ids))
            traced_hidden = output
        for block in model.blocks[:-1]:
            hidden = block(hidden)
        attention = model.blocks[-1].attention
        normalized = model.blocks[-1].attention_norm(hidden)[0]
        projected = attention.qkv_projection(normalized).reshape(
            len(step.context_token_ids), 3, attention.num_heads, attention.head_dim
        )
        scores = projected[:, 1, 0] @ projected[-1, 0, 0] / math.sqrt(attention.head_dim)
        expected = torch.softmax(scores, dim=0)
        logits = model(context)[0, -1]

    for focus in step.attention_focus:
        assert focus.probability == pytest.approx(float(expected[focus.position]))
        assert focus.token_id == step.context_token_ids[focus.position]
    for candidate in step.candidates:
        assert candidate.raw_logit == pytest.approx(float(logits[candidate.token_id]))
    untraced = generate(
        model, tokenizer, "hello", GenerationConfig(max_new_tokens=1, strategy="greedy")
    )
    assert untraced.generated_token_ids == result.generated_token_ids


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", 0.0),
        ("top_k", 0),
        ("top_p", 1.1),
        ("repetition_penalty", 0.9),
        ("trace_top_n", -1),
    ],
)
def test_invalid_generation_settings_are_rejected(field: str, value: float) -> None:
    values = {field: value}
    with pytest.raises(ValueError):
        GenerationConfig(**values)
