import random

import pytest
import torch

from mini_llm.runtime import resolve_device, seed_everything


def test_resolve_cpu_device() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_auto_device_matches_environment() -> None:
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device("auto") == torch.device(expected)


def test_requesting_unavailable_cuda_fails_clearly() -> None:
    if torch.cuda.is_available():
        pytest.skip("This failure mode only applies to CPU-only environments")

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_device("cuda")


def test_seed_everything_repeats_python_and_torch_sequences() -> None:
    seed_everything(7)
    first_python_value = random.random()
    first_torch_values = torch.rand(3)

    seed_everything(7)

    assert random.random() == first_python_value
    assert torch.equal(torch.rand(3), first_torch_values)


def test_seed_everything_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        seed_everything(-1)

