"""Phase 8 end-to-end validation for the pre-training model pipeline."""

import torch

from mini_llm.config import DataConfig, ModelConfig, TrainingConfig
from mini_llm.model import JeePeeTee, causal_language_model_loss
from mini_llm.runtime import seed_everything
from mini_llm.tokenizer import CharacterTokenizer
from mini_llm.training import build_dataloaders


def test_tiny_text_to_loss_and_backward_pipeline_on_cpu() -> None:
    text = "JeePeeTee learns from tiny text.\n" * 8
    tokenizer = CharacterTokenizer.train([text])
    token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)

    assert tokenizer.decode(token_ids, skip_special_tokens=True) == text

    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=8,
        embedding_dim=16,
        num_layers=2,
        num_heads=4,
        feed_forward_dim=32,
        dropout=0.1,
    )
    training_config = TrainingConfig(batch_size=4, max_steps=1)
    tokenizer.validate_vocab_size(model_config.vocab_size)
    loaders = build_dataloaders(
        torch.tensor(token_ids, dtype=torch.long),
        model_config,
        DataConfig(validation_fraction=0.2, stride=2),
        training_config,
        seed=42,
    )

    inputs, targets = next(iter(loaders.train))
    assert inputs.shape == targets.shape == (
        training_config.batch_size,
        model_config.context_length,
    )
    assert torch.equal(inputs[:, 1:], targets[:, :-1])

    seed_everything(42, deterministic_algorithms=True)
    device = torch.device("cpu")
    model = JeePeeTee(model_config).to(device)
    inputs = inputs.to(device)
    targets = targets.to(device)

    logits = model(inputs)
    loss = causal_language_model_loss(logits, targets)
    loss.backward()

    assert len(model.blocks) == model_config.num_layers
    assert logits.shape == (
        training_config.batch_size,
        model_config.context_length,
        tokenizer.vocab_size,
    )
    assert logits.device == device
    assert torch.isfinite(logits).all()
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
