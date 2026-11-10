"""Pick the adapter from model.config.model_type."""
from .deepseek_v2 import DeepseekV2Adapter
from .gpt_oss import GptOssAdapter
from .olmoe import OlmoeAdapter
from .qwen2_moe import Qwen2MoeAdapter
from .qwen3_moe import Qwen3MoeAdapter

ADAPTERS = {
    "olmoe": OlmoeAdapter,
    "qwen3_moe": Qwen3MoeAdapter,
    "qwen2_moe": Qwen2MoeAdapter,
    "deepseek_v2": DeepseekV2Adapter,
    "gpt_oss": GptOssAdapter,
}


def get_adapter(model_or_type):
    mt = model_or_type if isinstance(model_or_type, str) else model_or_type.config.model_type
    if mt not in ADAPTERS:
        raise KeyError(f"no adapter for model_type {mt!r}; known: {sorted(ADAPTERS)}")
    return ADAPTERS[mt]()
