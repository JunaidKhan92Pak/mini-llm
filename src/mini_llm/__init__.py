"""JeePeeTee educational language model package."""

from mini_llm.config import (
    DataConfig,
    DebugOverfitConfig,
    ModelConfig,
    RuntimeConfig,
    SanityTrainingConfig,
    TrainingConfig,
    development_model_config,
)
from mini_llm.model import JeePeeTee
from mini_llm.tokenizer import CharacterTokenizer

__all__ = [
    "CharacterTokenizer",
    "DataConfig",
    "DebugOverfitConfig",
    "JeePeeTee",
    "ModelConfig",
    "RuntimeConfig",
    "SanityTrainingConfig",
    "TrainingConfig",
    "development_model_config",
]
__project_name__ = "JeePeeTee"
__version__ = "0.1.0"
