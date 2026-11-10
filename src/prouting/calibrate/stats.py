"""Calibration statistics: per-layer Gram accumulation and the finalised GramState (PROJECT.md 5.4).

Modes (can run in one pass):
  * in-set: per-token Gram on T scattered into an N x N sum, plus co-occurrence counts and cosines.
  * cluster-conditioned: the same per cluster id, for each clustering (e.g. C = 8 and 16).
  * full Gram: every expert's output on a token subset (P3 and ablation A4), off by default.
All accumulators are float64.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor


@dataclass
class GramState:
    """Finalised calibration Gram for one MoE layer."""
    G_avg: Tensor                      # [N,N] G-avg with back-off applied
    R: Tensor                          # [N,N] calibrated cosine matrix (back-off applied, diag 1)
    norms: Tensor                      # [N] RMS output norm, sqrt(G_avg_ii)
    r_bar: float                       # count-weighted mean off-diagonal cosine
    counts: Tensor                     # [N,N] co-occurrence counts (diag = times in T)
    n_min: int
    backoff_share: float               # share of off-diagonal pairs below n_min (among all pairs)
    tokens: int
    clusters: dict[str, Tensor] = field(default_factory=dict)        # name -> [C,N,N] G-avg per cluster
    cluster_counts: dict[str, Tensor] = field(default_factory=dict)  # name -> [C,N,N]
    full_G: Optional[Tensor] = None    # [N,N] full Gram (mean over tokens), if computed
    full_tokens: int = 0

    def to(self, device=None, dtype=None) -> "GramState":
        def cv(t):
            if t is None:
                return None
            t = t.to(device) if device is not None else t
            return t.to(dtype) if (dtype is not None and t.is_floating_point()) else t
        return GramState(cv(self.G_avg), cv(self.R), cv(self.norms), self.r_bar, cv(self.counts), self.n_min,
                         self.backoff_share, self.tokens,
                         {k: cv(v) for k, v in self.clusters.items()},
                         {k: cv(v) for k, v in self.cluster_counts.items()},
                         cv(self.full_G), self.full_tokens)

    def state_dict(self) -> dict:
        return dict(G_avg=self.G_avg, R=self.R, norms=self.norms, r_bar=self.r_bar, counts=self.counts,
                    n_min=self.n_min, backoff_share=self.backoff_share, tokens=self.tokens,
                    clusters=self.clusters, cluster_counts=self.cluster_counts, full_G=self.full_G,
                    full_tokens=self.full_tokens)

    @classmethod
    def from_state_dict(cls, d: dict) -> "GramState":
        return cls(**d)

    def save(self, path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path) -> "GramState":
        return cls.from_state_dict(torch.load(path, map_location="cpu", weights_only=False))


class GramAccumulator:
    """Accumulates expert-output Gram statistics for one MoE layer with N routed experts."""

    def __init__(self, num_experts: int, clusters: Optional[dict[str, int]] = None, device="cpu"):
        N = num_experts
        self.N = N
        self.device = torch.device(device)
        f64 = dict(dtype=torch.float64, device=self.device)
        self.sum_G = torch.zeros(N, N, **f64)
        self.sum_cos = torch.zeros(N, N, **f64)
        self.count = torch.zeros(N, N, **f64)
        self.tokens = 0
        self.cluster_sizes = dict(clusters or {})
        self.c_sum = {n: torch.zeros(C, N, N, **f64) for n, C in self.cluster_sizes.items()}
        self.c_count = {n: torch.zeros(C, N, N, **f64) for n, C in self.cluster_sizes.items()}
        self.full_sum: Optional[Tensor] = None
        self.full_tokens = 0

    @torch.no_grad()
    def update(self, ids: Tensor, G: Tensor, cluster_ids: Optional[dict[str, Tensor]] = None) -> None:
        """ids [n,k] expert ids of T; G [n,k,k] true per-token Gram; cluster_ids name -> [n]."""
        n, k = ids.shape
        if n == 0:
            return
        ids = ids.to(self.device)
        G = G.to(self.device, torch.float64)
        rows = ids.unsqueeze(-1).expand(n, k, k).reshape(-1)
        cols = ids.unsqueeze(-2).expand(n, k, k).reshape(-1)
        d = torch.diagonal(G, dim1=-2, dim2=-1).clamp_min(1e-300).sqrt()
        C = (G / (d.unsqueeze(-1) * d.unsqueeze(-2))).reshape(-1)
        ones = torch.ones_like(C)
        Gf = G.reshape(-1)
        self.sum_G.index_put_((rows, cols), Gf, accumulate=True)
        self.sum_cos.index_put_((rows, cols), C, accumulate=True)
        self.count.index_put_((rows, cols), ones, accumulate=True)
        self.tokens += n
        if cluster_ids:
            for name, cid in cluster_ids.items():
                if name not in self.c_sum:
                    continue
                cl = cid.to(self.device).long().unsqueeze(-1).unsqueeze(-1).expand(n, k, k).reshape(-1)
                self.c_sum[name].index_put_((cl, rows, cols), Gf, accumulate=True)
                self.c_count[name].index_put_((cl, rows, cols), ones, accumulate=True)

    @torch.no_grad()
    def update_full(self, E_all: Tensor) -> None:
        """E_all [n,N,d]: every expert's output on n tokens (full-Gram mode)."""
        E = E_all.to(self.device, torch.float64)
        S = torch.einsum("nid,njd->ij", E, E)
        self.full_sum = S if self.full_sum is None else self.full_sum + S
        self.full_tokens += E.shape[0]

    def merge(self, other: "GramAccumulator") -> None:
        self.sum_G += other.sum_G
        self.sum_cos += other.sum_cos
        self.count += other.count
        self.tokens += other.tokens
        for n in self.c_sum:
            self.c_sum[n] += other.c_sum[n]
            self.c_count[n] += other.c_count[n]
        if other.full_sum is not None:
            self.full_sum = other.full_sum.clone() if self.full_sum is None else self.full_sum + other.full_sum
            self.full_tokens += other.full_tokens

    @torch.no_grad()
    def finalise(self, n_min: int = 32) -> GramState:
        N = self.N
        cnt = self.count
        eye = torch.eye(N, dtype=torch.bool, device=self.device)
        diag_cnt = torch.diagonal(cnt)
        diag_mean = torch.where(diag_cnt > 0, torch.diagonal(self.sum_G) / diag_cnt.clamp_min(1), torch.zeros_like(diag_cnt))
        # experts never seen in T: back off to the layer's mean squared norm
        seen = diag_cnt > 0
        fill = diag_mean[seen].mean() if bool(seen.any()) else torch.tensor(1.0, dtype=torch.float64, device=self.device)
        diag_mean = torch.where(seen, diag_mean, fill)
        off = ~eye
        off_cnt = cnt[off]
        r_bar = float(self.sum_cos[off].sum() / off_cnt.sum()) if float(off_cnt.sum()) > 0 else 0.0
        norms = diag_mean.clamp_min(0).sqrt()
        backoff_G = r_bar * norms.unsqueeze(-1) * norms.unsqueeze(-2)
        rare = (cnt < n_min) & off
        G_raw = torch.where(cnt > 0, self.sum_G / cnt.clamp_min(1), torch.zeros_like(cnt))
        R_raw = torch.where(cnt > 0, self.sum_cos / cnt.clamp_min(1), torch.zeros_like(cnt))
        G_avg = torch.where(rare, backoff_G, G_raw)
        G_avg[eye] = diag_mean
        R = torch.where(rare, torch.full_like(R_raw, r_bar), R_raw)
        R[eye] = 1.0
        backoff_share = float(rare.sum()) / float(off.sum()) if N > 1 else 0.0
        clusters, ccounts = {}, {}
        for name, csum in self.c_sum.items():
            cc = self.c_count[name]
            Gc = torch.where(cc > 0, csum / cc.clamp_min(1), torch.zeros_like(cc))
            low = cc < n_min
            Gc = torch.where(low, G_avg.unsqueeze(0).expand_as(Gc), Gc)
            clusters[name] = Gc
            ccounts[name] = cc
        full_G = self.full_sum / self.full_tokens if self.full_sum is not None and self.full_tokens > 0 else None
        return GramState(G_avg=G_avg, R=R, norms=norms, r_bar=r_bar, counts=cnt.clone(), n_min=n_min,
                         backoff_share=backoff_share, tokens=self.tokens, clusters=clusters,
                         cluster_counts=ccounts, full_G=full_G, full_tokens=self.full_tokens)


def gather_gram(G_full: Tensor, ids: Tensor) -> Tensor:
    """G^ on T from an N x N matrix: [N,N], ids [n,k] -> [n,k,k]."""
    return G_full[ids.unsqueeze(-1), ids.unsqueeze(-2)]


def gather_cluster_gram(G_c: Tensor, ids: Tensor, cluster: Tensor) -> Tensor:
    """G^ on T from a per-cluster [C,N,N] matrix: ids [n,k], cluster [n] -> [n,k,k]."""
    c = cluster.long().unsqueeze(-1).unsqueeze(-1)
    return G_c[c, ids.unsqueeze(-1), ids.unsqueeze(-2)]


def cos_gram(R: Tensor, mean_norms: Tensor, ids: Tensor, keep_mask: Tensor, kept_norms: Tensor) -> Tensor:
    """G-cos: G^_ij = n_i n_j R^_ij with actual norms for kept experts, calibrated norms otherwise.

    R [N,N], mean_norms [N], ids [n,k], keep_mask [n,k], kept_norms [n,k] (ignored where not kept).
    """
    nrm = torch.where(keep_mask, kept_norms.to(mean_norms.dtype), mean_norms[ids])
    return gather_gram(R, ids) * nrm.unsqueeze(-1) * nrm.unsqueeze(-2)
