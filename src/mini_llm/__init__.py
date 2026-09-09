"""JeePeeTee educational language model package."""

from mini_llm.config import (
    ModelConfig,
    RuntimeConfig,
    SanityTrainingConfig,
    development_model_config,
)
from mini_llm.tokenizer import CharacterTokenizer

__all__ = [
    "CharacterTokenizer",
    "ModelConfig",
    "RuntimeConfig",
    "SanityTrainingConfig",
    "development_model_config",
]
__project_name__ = "JeePeeTee"
__version__ = "0.1.0"
