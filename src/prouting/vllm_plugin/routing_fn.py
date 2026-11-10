"""Pure-torch custom routing function for vLLM's fused MoE layer (PHASE0_SETUP.md 2.9).

Believed signature (verify against the installed vLLM on the VM):
    custom_routing_function(hidden_states, gating_output, topk, renormalize) -> (topk_weights, topk_ids)
It reproduces the model's native top-k from `gating_output` (softmax, optional renormalisation) and then
calls the SAME selection and weighting code as the HF path (`policies.decide`). vLLM itself is never
imported here, so this file is tested on the laptop (T15).
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ..calibrate.stats import gather_cluster_gram, gather_gram
from ..routing.policies import decide


def native_topk_softmax(gating_output: Tensor, topk: int, renormalize: bool) -> tuple[Tensor, Tensor]:
    """softmax in float32, top-k, optional renormalisation (OLMoE / Qwen3-MoE / Qwen2-MoE convention)."""
    p = torch.softmax(gating_output.float(), dim=-1)
    g, ids = torch.topk(p, topk, dim=-1)
    if renormalize:
        g = g / g.sum(-1, keepdim=True)
    return g, ids


def make_routing_fn(G_avg: Tensor, *, select: str = "ols", weights: str = "ls", m: Optional[int] = None,
                    tau: Optional[float] = None, ridge: float = 1e-3, slots: Optional[int] = None,
                    G_clu: Optional[Tensor] = None, centroids: Optional[Tensor] = None, r_bar: float = 0.0):
    """Close over one layer's calibrated G^ (on the GPU) and return a vLLM-style routing function.

    slots: number of returned slots. None returns k slots with zero weights for dropped experts (correct
    outputs, no speed-up); an integer m returns exactly m slots (fixed budget only; kernel support verify).
    """
    def routing_fn(hidden_states: Tensor, gating_output: Tensor, topk: int, renormalize: bool):
        g, ids = native_topk_softmax(gating_output, topk, renormalize)
        dt = torch.float32 if g.is_cuda else torch.float64
        if G_clu is not None and centroids is not None:
            x = hidden_states.float()
            cl = (x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)) @ centroids.to(x.device).T
            Gs = gather_cluster_gram(G_clu.to(ids.device, dt), ids, cl.argmax(-1))
        else:
            Gs = gather_gram(G_avg.to(ids.device, dt), ids)
        keep, c = decide(g.to(dt), Gs, lambda keep: Gs, select=select, weights=weights, m=m, tau=tau,
                         ridge=ridge, r_bar=r_bar)
        c = c.to(torch.float32)
        if slots is None:
            return c, ids.to(torch.int32)
        # exactly `slots` entries per token: kept experts first (by weight magnitude), zero-weight padding
        order = (keep.to(torch.float32) * 2 + c.abs() / (c.abs().max() + 1)).argsort(-1, descending=True)[:, :slots]
        return c.gather(1, order), ids.gather(1, order).to(torch.int32)
    return routing_fn
