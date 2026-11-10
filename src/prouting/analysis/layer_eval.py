"""Offline evaluation of policies from recorded Grams (PHASE0_SETUP.md 2.6, PHASE1_KILL_TEST.md 7).

Every method selects and weights from its own G^ (or none); the error is always measured with the true
per-token Gram through the quadratic form of PROJECT.md 5.1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor

from ..calibrate.stats import cos_gram, gather_cluster_gram, gather_gram
from ..routing import linalg as L
from ..routing.policies import LayerCalib, decide

# Method table of PHASE1_KILL_TEST.md Section 7 (B3 is added by baselines when implementable).
KILL_METHODS: dict[str, dict] = {
    "B1": dict(select="gate", weights="keep", gram=None),
    "B2": dict(select="gate", weights="renorm", gram=None),
    "W-avg": dict(select="gate", weights="ls", gram="avg"),
    "ours-avg": dict(select="ols", weights="ls", gram="avg"),
    "ours-cos": dict(select="ols", weights="ls", gram="cos"),
    "ours-clu": dict(select="ols", weights="ls", gram="clu"),
    "ours-sk": dict(select="ols", weights="ls", gram="sk"),
    "U2": dict(select="gate", weights="ls", gram="oracle"),
    "U1": dict(select="exhaustive", weights="ls", gram="oracle"),
}


def _grams(rec: dict, calib: Optional[LayerCalib], gram: Optional[str], clusters: str, dt):
    """Returns (Gsel, gw_fn) for the offline estimator `gram`."""
    ids = rec["ids"].long()
    Gt = rec["G"].to(dt)
    if gram is None:
        return None, lambda keep: None
    if gram == "oracle":
        return Gt, lambda keep: Gt
    st = calib.gram
    if gram == "avg":
        Ga = gather_gram(st.G_avg.to(dt), ids)
        return Ga, lambda keep: Ga
    if gram == "clu":
        Gc = gather_cluster_gram(st.clusters[clusters].to(dt), ids, rec[clusters].long())
        return Gc, lambda keep: Gc
    if gram == "cos":
        Ga = gather_gram(st.G_avg.to(dt), ids)
        norms = torch.diagonal(Gt, dim1=-2, dim2=-1).clamp_min(0).sqrt()
        return Ga, lambda keep: cos_gram(st.R.to(dt), st.norms.to(dt), ids, keep, norms)
    if gram == "sk":
        Gsk = rec["G_sk"].to(dt)
        Gx = rec["G_cross"].to(dt)        # Gx[i,j] = <E_i, kappa_j E^_j>

        def gw(keep):
            ki, kj = keep.unsqueeze(-1), keep.unsqueeze(-2)
            return torch.where(ki & kj, Gt, torch.where(ki, Gx, torch.where(kj, Gx.transpose(-1, -2), Gsk)))
        return Gsk, gw
    raise ValueError(gram)


def run_method(rec: dict, calib: Optional[LayerCalib], spec: dict, m: Optional[int] = None,
               tau: Optional[float] = None, ridge: float = 1e-3, clusters: str = "c16",
               dt=torch.float64) -> tuple[Tensor, Tensor]:
    """keep_mask [n,k], c_full [n,k] for one method on a record chunk."""
    if "fn" in spec:  # baseline with its own offline rule: fn(rec, calib, m) -> keep, c
        return spec["fn"](rec, calib, m)
    Gsel, gw = _grams(rec, calib, spec["gram"], clusters, dt)
    if spec["weights"] == "ls" and spec["gram"] == "oracle":
        ridge = 0.0
    return decide(rec["g"].to(dt), Gsel, gw, select=spec["select"], weights=spec["weights"], m=m, tau=tau,
                  ridge=ridge, r_bar=(calib.gram.r_bar if calib is not None and calib.gram is not None else 0.0))


def _chunks(rec: dict, size: int):
    n = rec["ids"].shape[0]
    for s in range(0, n, size):
        yield {k: (v[s:s + size] if isinstance(v, Tensor) and v.shape[:1] == (n,) else v) for k, v in rec.items()}


@torch.no_grad()
def eval_method(rec: dict, calib: Optional[LayerCalib], spec: dict, m: Optional[int] = None,
                tau: Optional[float] = None, ridge: float = 1e-3, clusters: str = "c16",
                chunk: int = 8192, return_token: bool = False) -> dict:
    """Layer-level metrics of one method at one budget."""
    dt = torch.float64
    e_all, y2_all, cnt_all, h_all = [], [], [], []
    for r in _chunks(rec, chunk):
        keep, c = run_method(r, calib, spec, m, tau, ridge, clusters, dt)
        Gt = r["G"].to(dt)
        g = r["g"].to(dt)
        e = L.residual_energy(Gt, g, c.to(dt)).clamp_min(0)
        y2 = L.output_energy(Gt, g)
        e_all.append(e)
        y2_all.append(y2)
        cnt_all.append(keep.sum(-1).to(dt))
        if "h2" in r:
            h_all.append(e.sqrt() / r["h2"].to(dt).clamp_min(1e-30).sqrt())
    e, y2, cnt = torch.cat(e_all), torch.cat(y2_all), torch.cat(cnt_all)
    rel_tok = torch.where(y2 > 0, e / y2.clamp_min(1e-300), torch.zeros_like(e))
    out = dict(rel_error_energy=float(e.sum() / y2.sum()), median_token_rel_error=float(rel_tok.median()),
               mean_active=float(cnt.mean()), n_tokens=int(e.numel()))
    if h_all:
        hr = torch.cat(h_all)
        out["median_err_over_resid"] = float(hr.median())
        out["mean_err_over_resid"] = float(hr.mean())
    if return_token:
        out["_token_err"] = e
        out["_token_y2"] = y2
        out["_token_cnt"] = cnt
    return out


def tune_tau(rec_by_layer: dict[int, dict], calib: dict[int, LayerCalib], spec: dict, target_mean: float,
             lo: float = 0.0, hi: float = 1.0, iters: int = 30, chunk: int = 8192) -> tuple[float, float]:
    """One global tau such that the mean active experts over all layers and tokens hits target_mean.

    Bisection on tau (mean kept count is non-increasing in tau). Returns (tau, realised mean).
    """
    def mean_active(tau):
        tot, n = 0.0, 0
        for li, rec in rec_by_layer.items():
            r = eval_method(rec, calib.get(li), spec, m=None, tau=tau, chunk=chunk)
            tot += r["mean_active"] * r["n_tokens"]
            n += r["n_tokens"]
        return tot / max(n, 1)
    a, b = lo, hi
    ma = mean_active(a)
    if ma <= target_mean:
        return a, ma
    for _ in range(iters):
        mid = 0.5 * (a + b)
        if mean_active(mid) > target_mean:
            a = mid
        else:
            b = mid
    return b, mean_active(b)
