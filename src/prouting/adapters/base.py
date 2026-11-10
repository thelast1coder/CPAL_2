"""MoE adapter interface and the shared implementation for transformers 5.x fused-expert blocks.

Written against transformers==5.22.2 (pinned; docs/decisions.md). In that version the OLMoE, Qwen3-MoE,
Qwen2-MoE and DeepSeek-V2 sparse blocks have:
  block.gate     router module with `weight` [N,d] and `top_k` (plus family-specific routing fields)
  block.experts  fused experts: gate_up_proj [N, 2I, d] (concatenated [gate; up]), down_proj [N, d, I],
                 act_fn, flags is_transposed=False and is_concatenated=True set by @use_experts_implementation
Any other layout raises UnknownLayoutError.
"""
from __future__ import annotations

from typing import Any, Optional, Protocol

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class UnknownLayoutError(RuntimeError):
    pass


class MoEAdapter(Protocol):
    def find_moe_blocks(self, model) -> list[tuple[int, str, nn.Module]]: ...
    def num_experts(self, block) -> int: ...
    def top_k(self, block) -> int: ...
    def intermediate_size(self, block) -> int: ...
    def router_logits(self, block, x: Tensor) -> Tensor: ...
    def native_topk(self, block, logits: Tensor) -> tuple[Tensor, Tensor]: ...
    def expert_forward(self, block, e: int, x: Tensor) -> Tensor: ...
    def expert_hidden(self, block, e: int, x: Tensor) -> Tensor: ...
    def expert_weights(self, block, e: int) -> tuple[Tensor, Tensor, Tensor]: ...
    def shared_forward(self, block, x: Tensor) -> Optional[Tensor]: ...
    def wrap_output(self, block, y: Tensor, logits: Tensor) -> Any: ...


class FusedExpertsAdapter:
    """Shared implementation; subclasses set BLOCK_CLASS and override routing / shared experts."""

    BLOCK_CLASS = ""          # class name of the sparse MoE block
    FAMILY = ""

    # ------------------------------------------------------------------ discovery and layout
    def _layers(self, model) -> nn.ModuleList:
        base = getattr(model, "model", model)
        layers = getattr(base, "layers", None)
        if layers is None:
            raise UnknownLayoutError("model has no .model.layers")
        return layers

    def find_moe_blocks(self, model) -> list[tuple[int, str, nn.Module]]:
        out = []
        prefix = "model.layers" if hasattr(model, "model") else "layers"
        for i, layer in enumerate(self._layers(model)):
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                continue
            inner = getattr(mlp, "orig", mlp)   # already-wrapped blocks are reported too
            if type(inner).__name__ == self.BLOCK_CLASS:
                self.check_layout(inner)
                out.append((i, f"{prefix}.{i}.mlp", mlp))
        return out

    def check_layout(self, block) -> None:
        ex = getattr(block, "experts", None)
        gate = getattr(block, "gate", None)
        if ex is None or gate is None or not hasattr(gate, "weight"):
            raise UnknownLayoutError(f"{type(block).__name__}: expected .gate (router with weight) and .experts")
        if isinstance(ex, nn.ModuleList):
            raise UnknownLayoutError("experts stored as ModuleList: this adapter targets fused 3D weights "
                                     "(transformers 5.x); pin transformers==5.22.2")
        if not (hasattr(ex, "gate_up_proj") and hasattr(ex, "down_proj")):
            raise UnknownLayoutError("experts lack gate_up_proj / down_proj")
        if getattr(ex, "is_transposed", False) or not getattr(ex, "is_concatenated", True):
            raise UnknownLayoutError("unsupported expert layout (transposed or interleaved gate/up)")
        if getattr(ex, "has_bias", False):
            raise UnknownLayoutError("expert biases not supported")
        N, twoI, d = ex.gate_up_proj.shape
        if ex.down_proj.shape != (N, d, twoI // 2):
            raise UnknownLayoutError(f"unexpected down_proj shape {tuple(ex.down_proj.shape)}")

    def pre_moe_norm(self, model, layer_idx: int) -> Optional[nn.Module]:
        """Module whose input is the residual stream h entering the MoE sub-block."""
        return getattr(self._layers(model)[layer_idx], "post_attention_layernorm", None)

    # ------------------------------------------------------------------ sizes
    def num_experts(self, block) -> int:
        return block.experts.gate_up_proj.shape[0]

    def top_k(self, block) -> int:
        return int(block.gate.top_k)

    def intermediate_size(self, block) -> int:
        return block.experts.down_proj.shape[-1]

    # ------------------------------------------------------------------ routing (OLMoE / Qwen convention)
    def router_logits(self, block, x: Tensor) -> Tensor:
        return F.linear(x, block.gate.weight)

    def native_topk(self, block, logits: Tensor) -> tuple[Tensor, Tensor]:
        """softmax(float32) -> topk -> optional renormalisation -> cast to logits dtype (as the router)."""
        probs = torch.softmax(logits, dtype=torch.float, dim=-1)
        g, ids = torch.topk(probs, block.gate.top_k, dim=-1)
        if block.gate.norm_topk_prob:
            g = g / g.sum(dim=-1, keepdim=True)
        return ids, g.to(logits.dtype)

    # ------------------------------------------------------------------ experts
    def act_fn(self, block):
        return block.experts.act_fn

    def expert_weights(self, block, e: int) -> tuple[Tensor, Tensor, Tensor]:
        gu = block.experts.gate_up_proj[e]
        I = gu.shape[0] // 2
        return gu[:I], gu[I:], block.experts.down_proj[e]

    def expert_hidden(self, block, e: int, x: Tensor) -> Tensor:
        gate, up = F.linear(x, block.experts.gate_up_proj[e]).chunk(2, dim=-1)
        return block.experts.act_fn(gate) * up

    def expert_forward(self, block, e: int, x: Tensor) -> Tensor:
        return F.linear(self.expert_hidden(block, e, x), block.experts.down_proj[e])

    def shared_forward(self, block, x: Tensor) -> Optional[Tensor]:
        return None

    def wrap_output(self, block, y: Tensor, logits: Tensor) -> Any:
        # transformers 5.22.2: every supported sparse block returns the hidden states tensor only
        return y
