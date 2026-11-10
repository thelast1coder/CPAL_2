"""Qwen3-MoE (transformers 5.22.2): same router code as OLMoE; norm_topk_prob True for Qwen3-30B-A3B
(verify on the downloaded config). Dense layers (mlp_only_layers, decoder_sparse_step) are skipped by
find_moe_blocks. No shared experts."""
from .base import FusedExpertsAdapter


class Qwen3MoeAdapter(FusedExpertsAdapter):
    BLOCK_CLASS = "Qwen3MoeSparseMoeBlock"
    FAMILY = "qwen3_moe"
