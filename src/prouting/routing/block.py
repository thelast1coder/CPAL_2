"""ReducedMoEBlock: wraps an original MoE block and routes through a policy (PHASE0_SETUP.md 2.2)."""
from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from torch import Tensor, nn

from .policies import LayerCalib, Native, Policy, RouteCtx

# observer(ctx, slot_out [n,M,d] float32 or None, sel_ids, w) -> None
Observer = Callable[[RouteCtx, Optional[Tensor], Tensor, Tensor], None]


class ReducedMoEBlock(nn.Module):
    def __init__(self, orig: nn.Module, adapter, policy: Policy, layer_idx: int,
                 calib: Optional[LayerCalib] = None, shared_state: Optional[dict] = None):
        super().__init__()
        self.orig = orig
        self.adapter = adapter
        self.policy = policy
        self.layer_idx = layer_idx
        self.calib = calib
        self.shared_state = shared_state if shared_state is not None else {}
        self.observers: list[tuple[Observer, bool]] = []   # (fn, needs_slot_outputs)
        self.reset_counters()

    def reset_counters(self) -> None:
        self.computed_pairs = 0
        self.counted_tokens = 0

    def add_observer(self, fn: Observer, needs_slots: bool = True) -> None:
        self.observers.append((fn, needs_slots))

    def clear_observers(self) -> None:
        self.observers = []

    def _valid(self, n: int, device) -> Optional[Tensor]:
        v = self.shared_state.get("valid")
        if v is None or v.numel() != n:
            return None
        return v.reshape(-1).to(device).bool()

    def forward(self, hidden_states: Tensor, *args, **kwargs) -> Any:
        ad, blk = self.adapter, self.orig
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        n = x.shape[0]
        logits = ad.router_logits(blk, x)
        ids, g = ad.native_topk(blk, logits)
        valid = self._valid(n, x.device)
        ctx = RouteCtx(x=x, logits=logits, ids=ids, g=g, layer=self.layer_idx, calib=self.calib,
                       cache={}, adapter=ad, block=blk, valid=valid)
        sel, w = self.policy(ctx)
        nz = w != 0
        need_slots = self.policy.needs_outputs or any(ns for _, ns in self.observers)
        acc = torch.zeros(n, shape[-1], dtype=torch.float32, device=x.device)
        slot_out = torch.zeros(n, sel.shape[1], shape[-1], dtype=torch.float32, device=x.device) if need_slots else None
        for e in torch.unique(sel[nz]).tolist():
            # (slot, token) order, as the transformers eager experts loop
            slots, rows = ((sel == e) & nz).T.nonzero(as_tuple=True)
            E = ad.expert_forward(blk, e, x[rows]).float()
            if need_slots:
                slot_out[rows, slots] = E
            else:
                acc.index_add_(0, rows, w[rows, slots].float().unsqueeze(-1) * E)
        if self.policy.needs_outputs:
            w = self.policy.reweight(ctx, slot_out)
            nz = w != 0
        if need_slots:
            acc = torch.einsum("nm,nmd->nd", w.float(), slot_out)
        if getattr(self.policy, "adds_sketch", False):
            acc = acc + self.policy.extra_output(ctx)
        for fn, _ in self.observers:
            fn(ctx, slot_out, sel, w)
        # realised budget: computed (token, expert) pairs on valid tokens
        if valid is None:
            self.computed_pairs += int(nz.sum())
            self.counted_tokens += n
        else:
            self.computed_pairs += int((nz & valid.unsqueeze(-1)).sum())
            self.counted_tokens += int(valid.sum())
        shared = ad.shared_forward(blk, x)
        if shared is not None:
            acc = acc + shared.float()
        y = acc.to(hidden_states.dtype).reshape(shape)
        return ad.wrap_output(blk, y, logits)
