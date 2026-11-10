"""OLMoE (transformers 5.22.2, models/olmoe/modeling_olmoe.py): softmax over N, top-k, renormalise only if
config.norm_topk_prob (False for OLMoE-1B-7B-0924, verify on the downloaded config), gates cast to model dtype.
No shared experts."""
from .base import FusedExpertsAdapter


class OlmoeAdapter(FusedExpertsAdapter):
    BLOCK_CLASS = "OlmoeSparseMoeBlock"
    FAMILY = "olmoe"
