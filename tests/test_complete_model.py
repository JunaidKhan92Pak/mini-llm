from pathlib import Path

import pytest
import torch

from mini_llm.config import ModelConfig, development_model_config
from mini_llm.model import JeePeeTee, causal_language_model_loss
from mini_llm.runtime import seed_everything
from mini_llm.tokenizer import CharacterTokenizer
from mini_llm.training import NextTokenDataset


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=16,
        context_length=6,
        embedding_dim=16,
        num_layers=3,
        num_heads=4,
        feed_forward_dim=32,
        dropout=0.0,
    )


@pytest.mark.parametrize(("batch_size", "sequence_length"), [(1, 1), (2, 4), (3, 6)])
def test_complete_model_returns_finite_raw_vocabulary_logits(
    model_config: ModelConfig,
    batch_size: int,
    sequence_length: int,
) -> None:
    seed_everything(42)
    model = JeePeeTee(model_config).eval()
    token_ids = torch.randint(
        model_config.vocab_size,
        (batch_size, sequence_length),
    )

    logits = model(token_ids)

    assert logits.shape == (batch_size, sequence_length, model_config.vocab_size)
    assert torch.isfinite(logits).all()
    assert not torch.allclose(
        logits.sum(dim=-1),
        torch.ones((batch_size, sequence_length)),
    )


def test_configured_blocks_are_distinct_and_all_participate_in_backward(
    model_config: ModelConfig,
) -> None:
    model = JeePeeTee(model_config)
    token_ids = torch.randint(model_config.vocab_size, (2, 5))
    targets = torch.randint(model_config.vocab_size, (2, 5))

    loss = causal_language_model_loss(model(token_ids), targets)
    loss.backward()

    assert len(model.blocks) == model_config.num_layers
    assert len({id(block) for block in model.blocks}) == model_config.num_layers
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def test_complete_multi_block_model_remains_causal(model_config: ModelConfig) -> None:
    seed_everything(42)
    model = JeePeeTee(model_config).eval()
    original = torch.tensor([[1, 4, 5, 6, 7, 2]])
    changed_future = torch.tensor([[1, 4, 5, 9, 10, 11]])

    original_logits = model(original)
    changed_logits = model(changed_future)

    assert torch.equal(original_logits[:, :3], changed_logits[:, :3])
    assert not torch.equal(original_logits[:, 3:], changed_logits[:, 3:])


def test_fixed_seed_and_eval_mode_are_deterministic(model_config: ModelConfig) -> None:
    token_ids = torch.tensor([[1, 4, 5, 2]])
    seed_everything(7)
    first_model = JeePeeTee(model_config).eval()
    first_logits = first_model(token_ids)
    seed_everything(7)
    second_model = JeePeeTee(model_config).eval()
    second_logits = second_model(token_ids)

    assert torch.equal(first_logits, second_logits)
    assert torch.equal(second_model(token_ids), second_logits)


def test_model_rejects_invalid_public_inputs(model_config: ModelConfig) -> None:
    model = JeePeeTee(model_config)

    with pytest.raises(ValueError, match="shape"):
        model(torch.ones(model_config.context_length, dtype=torch.long))
    with pytest.raises(TypeError, match="integer dtype"):
        model(torch.ones((1, 2), dtype=torch.float32))
    with pytest.raises(ValueError, match="batch must not be empty"):
        model(torch.empty((0, 2), dtype=torch.long))
    with pytest.raises(ValueError, match="sequence must not be empty"):
        model(torch.empty((1, 0), dtype=torch.long))
    with pytest.raises(ValueError, match="exceeds context length"):
        model(torch.ones((1, model_config.context_length + 1), dtype=torch.long))
    with pytest.raises(ValueError, match="must be within"):
        model(torch.tensor([[model_config.vocab_size]]))
    with pytest.raises(ValueError, match="must be within"):
        model(torch.tensor([[-1]]))


def test_tokenizer_dataset_and_model_share_one_shift_convention() -> None:
    text = "JeePeeTee"
    tokenizer = CharacterTokenizer.train([text])
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=4,
        embedding_dim=16,
        num_layers=2,
        num_heads=4,
        feed_forward_dim=32,
    )
    tokenizer.validate_vocab_size(config.vocab_size)
    dataset = NextTokenDataset(torch.tensor(tokenizer.encode(text)), context_length=4)
    inputs, targets = dataset[0]

    logits = JeePeeTee(config)(inputs.unsqueeze(0))
    loss = causal_language_model_loss(logits, targets.unsqueeze(0))

    assert logits.shape == (1, 4, tokenizer.vocab_size)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_state_dict_round_trip_reproduces_eval_logits(
    model_config: ModelConfig,
    tmp_path: Path,
) -> None:
    seed_everything(42)
    model = JeePeeTee(model_config).eval()
    token_ids = torch.tensor([[1, 4, 5, 2]])
    expected_logits = model(token_ids)
    state_path = tmp_path / "model-state.pt"
    torch.save(model.state_dict(), state_path)

    restored = JeePeeTee(model_config)
    restored.load_state_dict(
        torch.load(state_path, map_location="cpu", weights_only=True),
        strict=True,
    )
    restored.eval()

    assert torch.equal(restored(token_ids), expected_logits)


def test_debug_parameter_breakdown_and_untied_output_head() -> None:
    config = development_model_config(vocab_size=128)
    model = JeePeeTee(config)

    breakdown = model.parameter_breakdown()
    counts = model.parameter_counts()

    assert breakdown == {
        "embeddings": 12_288,
        "transformer_blocks": 99_968,
        "final_normalization": 128,
        "language_model_head": 8_192,
    }
    assert counts == {"total": 120_576, "trainable": 120_576}
    assert model.parameter_count() == counts["trainable"]
    assert sum(breakdown.values()) == counts["trainable"]
    assert (
        model.embeddings.token_embedding.weight.data_ptr()
        != model.language_model_head.weight.data_ptr()
    )


def test_complete_model_runs_on_cpu(model_config: ModelConfig) -> None:
    device = torch.device("cpu")
    model = JeePeeTee(model_config).to(device)
    token_ids = torch.tensor([[1, 4, 5, 2]], device=device)

    logits = model(token_ids)

    assert logits.device == device
