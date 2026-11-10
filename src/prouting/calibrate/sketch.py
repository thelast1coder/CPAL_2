"""Expert sketches for the G-sk rescue (PROJECT.md 5.6).

For each expert j: R_j = the ceil(fraction * I) intermediate neurons with the largest calibration mean of
|a_jn(x)| * ||W_down,j[:, n]|| over tokens routed to j. Sketch E^_j(x) = W_down,j[:, R_j] a_j,R_j(x).
A scalar kappa_j is fitted by least squares of E_j on E^_j over routed calibration tokens.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class SketchState:
    fraction: float
    neuron_idx: dict[int, Tensor] = field(default_factory=dict)  # expert -> [r] long
    kappa: dict[int, float] = field(default_factory=dict)

    def state_dict(self) -> dict:
        return dict(fraction=self.fraction, neuron_idx=self.neuron_idx, kappa=self.kappa)

    @classmethod
    def from_state_dict(cls, d: dict) -> "SketchState":
        return cls(d["fraction"], d["neuron_idx"], d["kappa"])


class NeuronImportance:
    """Accumulates sum |a_jn| and counts per expert (pass 1)."""

    def __init__(self, num_experts: int, intermediate_size: int):
        self.sum_abs = torch.zeros(num_experts, intermediate_size, dtype=torch.float64)
        self.cnt = torch.zeros(num_experts, dtype=torch.float64)

    @torch.no_grad()
    def update(self, e: int, hidden: Tensor) -> None:
        """hidden [n_e, I]: the SwiGLU product of expert e on its routed tokens."""
        if hidden.shape[0] == 0:
            return
        self.sum_abs[e] += hidden.abs().double().sum(0).cpu()
        self.cnt[e] += hidden.shape[0]

    def choose(self, down_col_norms: dict[int, Tensor], fraction: float) -> dict[int, Tensor]:
        """down_col_norms: expert -> [I] norms of W_down columns. Returns expert -> top neuron ids."""
        out = {}
        for e, cn in down_col_norms.items():
            I = cn.numel()
            r = min(I, max(1, math.ceil(fraction * I)))
            mean_abs = self.sum_abs[e] / max(float(self.cnt[e]), 1.0)
            score = mean_abs * cn.double().cpu()
            # stable tie-break by index: sort descending with a tiny index-dependent offset not needed;
            # torch.topk on CPU is deterministic for a fixed input.
            out[e] = torch.topk(score, r).indices.sort().values
        return out


@torch.no_grad()
def sketch_forward(adapter, block, e: int, x: Tensor, idx: Tensor) -> Tensor:
    """E^_e(x) using only neurons idx. x [n,d] -> [n,d]."""
    Wg, Wu, Wd = adapter.expert_weights(block, e)      # [I,d], [I,d], [d,I]
    idx = idx.to(x.device)
    act = adapter.act_fn(block)
    a = act(x @ Wg[idx].T.to(x.dtype)) * (x @ Wu[idx].T.to(x.dtype))
    return a @ Wd[:, idx].T.to(x.dtype)


class KappaFit:
    """Least-squares scalar kappa_j = sum <E_j, E^_j> / sum ||E^_j||^2 over routed tokens."""

    def __init__(self, num_experts: int):
        self.num = torch.zeros(num_experts, dtype=torch.float64)
        self.den = torch.zeros(num_experts, dtype=torch.float64)

    @torch.no_grad()
    def update(self, e: int, E: Tensor, E_hat: Tensor) -> None:
        self.num[e] += (E.double() * E_hat.double()).sum().cpu()
        self.den[e] += (E_hat.double() ** 2).sum().cpu()

    def finalise(self) -> dict[int, float]:
        k = torch.where(self.den > 0, self.num / self.den.clamp_min(1e-300), torch.ones_like(self.num))
        return {e: float(k[e]) for e in range(k.numel())}
