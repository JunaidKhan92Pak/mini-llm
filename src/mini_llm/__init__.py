"""JeePeeTee educational language model package."""

from mini_llm.bpe_tokenizer import BPETokenizer
from mini_llm.config import (
    DataConfig,
    DebugOverfitConfig,
    ModelConfig,
    RuntimeConfig,
    SanityTrainingConfig,
    TrainingConfig,
    debug_model_config,
    development_model_config,
    five_million_model_config,
    mini_model_config,
)
from mini_llm.generation import GenerationConfig, GenerationResult, generate
from mini_llm.model import JeePeeTee
from mini_llm.tokenizer import CharacterTokenizer

__all__ = [
    "BPETokenizer",
    "CharacterTokenizer",
    "DataConfig",
    "DebugOverfitConfig",
    "GenerationConfig",
    "GenerationResult",
    "JeePeeTee",
    "ModelConfig",
    "RuntimeConfig",
    "SanityTrainingConfig",
    "TrainingConfig",
    "debug_model_config",
    "development_model_config",
    "five_million_model_config",
    "generate",
    "mini_model_config",
]
__project_name__ = "JeePeeTee"
__version__ = "0.1.0"
