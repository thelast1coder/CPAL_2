"""Qwen2-MoE / Qwen1.5-MoE (transformers 5.22.2): OLMoE-style router; one shared expert with a sigmoid
gate, sigmoid(shared_expert_gate(x)) * shared_expert(x). The shared expert always runs and is untouched."""
import torch.nn.functional as F

from .base import FusedExpertsAdapter


class Qwen2MoeAdapter(FusedExpertsAdapter):
    BLOCK_CLASS = "Qwen2MoeSparseMoeBlock"
    FAMILY = "qwen2_moe"

    def shared_forward(self, block, x):
        return F.sigmoid(block.shared_expert_gate(x)) * block.shared_expert(x)
