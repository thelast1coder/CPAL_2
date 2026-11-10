"""DeepSeek-V2 (transformers 5.22.2, models/deepseek_v2/modeling_deepseek_v2.py).

Router: logits in float32 (F.linear on float32 copies), softmax float32, topk_method "greedy" or
"group_limited_greedy" (n_group, topk_group), unsorted top-k, gates multiplied by routed_scaling_factor and
left in float32. Shared experts: block.shared_experts (always run, untouched). Layers before
first_k_dense_replace are dense and are not MoE blocks. Values for DeepSeek-V2-Lite are read from the
downloaded config on the VM (verify).
"""
import torch
import torch.nn.functional as F

from .base import FusedExpertsAdapter


class DeepseekV2Adapter(FusedExpertsAdapter):
    BLOCK_CLASS = "DeepseekV2Moe"
    FAMILY = "deepseek_v2"

    def router_logits(self, block, x):
        return F.linear(x.type(torch.float32), block.gate.weight.type(torch.float32))

    def native_topk(self, block, logits):
        r = block.gate
        scores = logits.softmax(dim=-1, dtype=torch.float32)
        if r.topk_method == "greedy":
            g, ids = torch.topk(scores, k=r.top_k, dim=-1, sorted=False)
        elif r.topk_method == "group_limited_greedy":
            N = scores.shape[-1]
            group_scores = scores.view(-1, r.num_group, N // r.num_group).max(dim=-1).values
            group_idx = torch.topk(group_scores, k=r.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = group_mask.unsqueeze(-1).expand(-1, r.num_group, N // r.num_group).reshape(-1, N)
            scores = scores.masked_fill(~score_mask.bool(), 0.0)
            g, ids = torch.topk(scores, k=r.top_k, dim=-1, sorted=False)
        else:
            raise NotImplementedError(f"topk_method {r.topk_method}")
        return ids, g * r.routed_scaling_factor

    def shared_forward(self, block, x):
        return block.shared_experts(x)
