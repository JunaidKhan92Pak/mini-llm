import torch

from mini_llm.config import (
    debug_model_config,
    five_million_model_config,
    mini_model_config,
)
from mini_llm.model import JeePeeTee, causal_language_model_loss

VOCAB_SIZE = 2048


def test_presets_have_stable_parameter_counts() -> None:
    expected = {
        "debug": (debug_model_config(VOCAB_SIZE), 366_336),
        "mini": (mini_model_config(VOCAB_SIZE), 846_912),
        "five_million": (five_million_model_config(VOCAB_SIZE), 5_030_656),
    }
    for config, parameter_count in expected.values():
        model = JeePeeTee(config)
        assert model.parameter_counts() == {
            "total": parameter_count,
            "trainable": parameter_count,
        }
        assert config.embedding_dim % config.num_heads == 0


def test_five_million_forward_backward_and_optimizer_step_are_finite() -> None:
    torch.manual_seed(14)
    config = five_million_model_config(VOCAB_SIZE)
    model = JeePeeTee(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    inputs = torch.randint(0, VOCAB_SIZE, (2, 16))
    targets = torch.randint(0, VOCAB_SIZE, (2, 16))
    before = model.language_model_head.weight.detach().clone()

    logits = model(inputs)
    loss = causal_language_model_loss(logits, targets)
    loss.backward()
    assert logits.shape == (2, 16, VOCAB_SIZE)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    optimizer.step()
    assert not torch.equal(before, model.language_model_head.weight)


def test_five_million_preset_remains_causal() -> None:
    torch.manual_seed(14)
    model = JeePeeTee(five_million_model_config(VOCAB_SIZE)).eval()
    original = torch.randint(0, VOCAB_SIZE, (1, 8))
    changed_future = original.clone()
    changed_future[:, 5:] = torch.randint(0, VOCAB_SIZE, (1, 3))
    with torch.no_grad():
        original_logits = model(original)
        changed_logits = model(changed_future)
    torch.testing.assert_close(original_logits[:, :5], changed_logits[:, :5])
