"""Routing policies: which experts to compute and with what weights (PHASE0_SETUP.md 2.5).

Interface: policy(ctx) -> (sel_ids [n,M] long, w [n,M]); a zero weight means "not computed".
Policies that need the kept experts' outputs before fixing weights (G-cos, G-sk) set
`needs_outputs = True` and implement `reweight(ctx, kept_out) -> w`, where kept_out [n,M,d] holds the
outputs of computed slots (zeros elsewhere).

The selection-and-weighting core (`Projection.decide`) is a pure tensor function shared by the HF block,
the offline layer evaluator and the vLLM routing function.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
from torch import Tensor

from . import linalg as L
from ..calibrate.stats import GramState, cos_gram, gather_cluster_gram, gather_gram


@dataclass
class LayerCalib:
    """Everything a policy may need for one MoE layer."""
    gram: Optional[GramState] = None
    kmeans: dict = field(default_factory=dict)       # "c8"/"c16" -> SphericalKMeans
    sketch: Any = None                               # SketchState or None


@dataclass
class RouteCtx:
    x: Tensor                     # [n,d] block input (model dtype)
    logits: Tensor                # [n,N]
    ids: Tensor                   # [n,k] native top-k ids
    g: Tensor                     # [n,k] native gates (model convention)
    layer: int
    calib: Optional[LayerCalib] = None
    cache: dict = field(default_factory=dict)
    adapter: Any = None
    block: Any = None
    valid: Optional[Tensor] = None


class Policy:
    name = "policy"
    needs_outputs = False

    def __call__(self, ctx: RouteCtx) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def reweight(self, ctx: RouteCtx, kept_out: Tensor) -> Tensor:
        raise NotImplementedError


class Native(Policy):
    name = "B0"

    def __call__(self, ctx):
        return ctx.ids, ctx.g


class TopMKeep(Policy):
    """B1: top-m by gate, gates kept."""
    name = "B1"

    def __init__(self, m: int):
        self.m = m

    def __call__(self, ctx):
        mask = L.gate_order_mask(ctx.g, self.m)
        return ctx.ids, L.keep_weights(ctx.g.float(), mask).to(ctx.g.dtype)


class TopMRenorm(Policy):
    """B2: top-m by gate, renormalised to the native total."""
    name = "B2"

    def __init__(self, m: int):
        self.m = m

    def __call__(self, ctx):
        mask = L.gate_order_mask(ctx.g, self.m)
        return ctx.ids, L.renorm_weights(ctx.g.float(), mask).to(ctx.g.dtype)


# --------------------------------------------------------------------------------------- core
def decide(g: Tensor, Gsel: Optional[Tensor], gw_fn: Callable[[Tensor], Optional[Tensor]], *, select: str,
           weights: str, m: Optional[int], tau: Optional[float], ridge: float, m_min: int = 1,
           r_bar: float = 0.0) -> tuple[Tensor, Tensor]:
    """Pure selection and weighting on T. Returns keep_mask [n,k] and c_full [n,k] (dtype of Gsel or g).

    g: gates [n,k]; Gsel: Gram used for selection [n,k,k] (None allowed for select="gate");
    gw_fn(keep_mask) -> Gram used for weights (only read for weights="ls").
    """
    n, k = g.shape
    dt = Gsel.dtype if Gsel is not None else g.dtype
    g = g.to(dt)
    m_max = k if m is None else int(m)
    if select == "gate":
        if tau is None:
            keep = L.gate_order_mask(g, m_max)
        else:
            keep = _gate_prefix_adaptive(g, Gsel, tau, m_min, m_max, ridge)
    elif select in ("ols", "omp"):
        fn = L.greedy_ols if select == "ols" else L.greedy_omp
        keep, _, _ = fn(Gsel, g, m_max, tau=tau, m_min=m_min)
    elif select == "exhaustive":
        if tau is not None:
            raise ValueError("exhaustive selection supports fixed m only")
        keep, _ = L.exhaustive_best(Gsel, g, m_max, ridge)
    else:
        raise ValueError(f"unknown select {select}")
    if weights == "keep":
        c = L.keep_weights(g, keep)
    elif weights == "renorm":
        c = L.renorm_weights(g, keep)
    elif weights == "uniform":
        c = L.uniform_coherence_weights(g, keep, r_bar)
    elif weights == "ls":
        Gw = gw_fn(keep)
        c = L.ls_weights(Gw.to(dt), g, keep, ridge)
    else:
        raise ValueError(f"unknown weights {weights}")
    # nothing dropped: the native output is reproduced exactly by the native gates (ridge-free optimum)
    full = keep.all(-1, keepdim=True)
    c = torch.where(full, g, c)
    return keep, c


def _gate_prefix_adaptive(g, Gsel, tau, m_min, m_max, ridge):
    """Gate-order selection with the adaptive residual stop: smallest gate prefix with rel. residual <= tau."""
    n, k = g.shape
    order = g.argsort(-1, descending=True)
    y2 = L.output_energy(Gsel, g)
    chosen = torch.full((n,), m_max, dtype=torch.long, device=g.device)
    found = torch.zeros(n, dtype=torch.bool, device=g.device)
    for mm in range(max(1, m_min), m_max + 1):
        mask = torch.zeros(n, k, dtype=torch.bool, device=g.device)
        mask.scatter_(1, order[:, :mm], True)
        e = L.residual_energy(Gsel, g, L.ls_weights(Gsel, g, mask, ridge))
        ok = (e <= tau * y2) & ~found
        chosen = torch.where(ok, torch.full_like(chosen, mm), chosen)
        found |= ok
    mask = torch.arange(k, device=g.device).unsqueeze(0) < chosen.unsqueeze(-1)
    keep = torch.zeros(n, k, dtype=torch.bool, device=g.device)
    keep.scatter_(1, order, mask)
    return keep


def gram_for(kind: str, calib: Optional[LayerCalib], ids: Tensor, x: Optional[Tensor] = None,
             cluster_ids: Optional[Tensor] = None, clusters: str = "c16", dtype=torch.float64) -> Tensor:
    """Calibrated G^ on T for estimators that do not need kept outputs ('avg', 'clu')."""
    st = calib.gram
    if kind in ("avg", "cos"):
        return gather_gram(st.G_avg.to(ids.device, dtype), ids)
    if kind == "clu":
        if cluster_ids is None:
            cluster_ids = calib.kmeans[clusters].assign(x)
        return gather_cluster_gram(st.clusters[clusters].to(ids.device, dtype), ids, cluster_ids.to(ids.device))
    raise ValueError(f"gram {kind} needs outputs or is offline-only")


class Projection(Policy):
    """Ours: selection on G^ (gate / OLS / OMP / exhaustive) plus weights (keep / renorm / LS / uniform).

    Ours-fixed: Projection(select="ols", weights="ls", m=m). Ours-adaptive: same with tau.
    gram: "avg" | "cos" | "clu" | "sk" | "oracle" (oracle and pool="2k" are offline-only).
    """
    name = "ours"

    def __init__(self, select: str = "ols", weights: str = "ls", gram: str = "avg", m: Optional[int] = None,
                 tau: Optional[float] = None, ridge: float = 1e-3, pool: str = "k", m_min: int = 1,
                 clusters: str = "c16"):
        if m is None and tau is None:
            raise ValueError("give m (fixed budget) or tau (adaptive)")
        if pool != "k":
            raise NotImplementedError("pool='2k' is evaluated offline only (ablation A4)")
        if gram == "oracle":
            raise ValueError("oracle Gram is offline-only (layer_eval)")
        self.select, self.weights, self.gram = select, weights, gram
        self.m, self.tau, self.ridge, self.m_min, self.clusters = m, tau, ridge, m_min, clusters
        self.needs_outputs = weights == "ls" and gram in ("cos", "sk")
        self.name = f"proj[{select},{weights},{gram},m={m},tau={tau}]"

    # online -----------------------------------------------------------------------------------
    def _sel_gram(self, ctx: RouteCtx, dt) -> Optional[Tensor]:
        if self.select == "gate" and self.tau is None:
            return None
        if self.gram == "sk":
            return _sketch_grams(ctx, dt)[0]
        return gram_for(self.gram, ctx.calib, ctx.ids, ctx.x, clusters=self.clusters, dtype=dt)

    def __call__(self, ctx):
        dt = L.work_dtype(ctx.ids.device)
        Gsel = self._sel_gram(ctx, dt)
        if self.needs_outputs:
            gw = lambda keep: torch.eye(ctx.g.shape[1], dtype=dt, device=ctx.g.device).expand(ctx.g.shape[0], -1, -1)
            keep, _ = decide(ctx.g, Gsel, gw, select=self.select, weights="keep", m=self.m, tau=self.tau,
                             ridge=self.ridge, m_min=self.m_min)
            ctx.cache["keep"] = keep
            ctx.cache["Gsel"] = Gsel
            return ctx.ids, keep.to(ctx.g.dtype)  # placeholder non-zero weights: compute these experts
        r_bar = ctx.calib.gram.r_bar if (ctx.calib is not None and ctx.calib.gram is not None) else 0.0
        keep, c = decide(ctx.g, Gsel, lambda keep: Gsel if Gsel is not None else
                         gram_for(self.gram, ctx.calib, ctx.ids, ctx.x, clusters=self.clusters, dtype=dt),
                         select=self.select, weights=self.weights, m=self.m, tau=self.tau, ridge=self.ridge,
                         m_min=self.m_min, r_bar=r_bar)
        return ctx.ids, c.to(ctx.g.dtype)

    def reweight(self, ctx, kept_out):
        dt = L.work_dtype(ctx.ids.device)
        keep = ctx.cache["keep"]
        g = ctx.g.to(dt)
        ko = kept_out.to(dt)
        if self.gram == "cos":
            norms = ko.norm(dim=-1)
            st = ctx.calib.gram
            Gw = cos_gram(st.R.to(ko.device, dt), st.norms.to(ko.device, dt), ctx.ids, keep, norms)
        else:  # sk: true Gram on S x S, <E_i, kappa_j E^_j> on S x D
            _, Ecomb = _sketch_grams(ctx, dt)                       # kappa-scaled sketches [n,k,d]
            V = torch.where(keep.unsqueeze(-1), ko, Ecomb)          # exact where kept, sketch where dropped
            Gw = torch.einsum("nid,njd->nij", V, V)
        c = L.ls_weights(Gw, g, keep, self.ridge)
        return c.to(ctx.g.dtype)


def _sketch_grams(ctx: RouteCtx, dt) -> tuple[Tensor, Tensor]:
    """Sketch Gram [n,k,k] and kappa-scaled sketches [n,k,d] on T (cached per call)."""
    if "sk" in ctx.cache:
        return ctx.cache["sk"]
    from ..calibrate.sketch import sketch_forward
    sk = ctx.calib.sketch
    n, k = ctx.ids.shape
    d = ctx.x.shape[-1]
    Ehat = torch.zeros(n, k, d, dtype=dt, device=ctx.x.device)
    for e in ctx.ids.unique().tolist():
        rows, slots = (ctx.ids == e).nonzero(as_tuple=True)
        out = sketch_forward(ctx.adapter, ctx.block, e, ctx.x[rows], sk.neuron_idx[e])
        Ehat[rows, slots] = out.to(dt) * sk.kappa[e]
    G = torch.einsum("nid,njd->nij", Ehat, Ehat)
    ctx.cache["sk"] = (G, Ehat)
    return G, Ehat


class AddProjection(Policy):
    """C4 add-on: run any base policy, keep its selection, replace its weights with LS weights.

    The target stays the native output y = sum_T g_j E_j. Selections outside T use the full-Gram
    calibration (GramState.full_G); inside T the in-set G-avg is used unless gram="full".
    """

    def __init__(self, base: Policy, gram: str = "avg", ridge: float = 1e-3):
        self.base, self.gram, self.ridge = base, gram, ridge
        self.name = f"{getattr(base, 'name', 'base')}+P"

    def __call__(self, ctx):
        sel, w = self.base(ctx)
        dt = L.work_dtype(ctx.ids.device)
        n, k = ctx.ids.shape
        pool = torch.cat([ctx.ids, sel], dim=1)                             # [n, k+M]
        kept_ids_mask = w != 0
        # T slots kept if their id is selected with non-zero weight
        inT = (ctx.ids.unsqueeze(-1) == sel.unsqueeze(-2)) & kept_ids_mask.unsqueeze(-2)   # [n,k,M]
        keep_T = inT.any(-1)
        dup = (sel.unsqueeze(-1) == ctx.ids.unsqueeze(-2)).any(-1)                         # sel id already in T
        keep_X = kept_ids_mask & ~dup
        keep = torch.cat([keep_T, keep_X], dim=1)
        gp = torch.cat([ctx.g.to(dt), torch.zeros(sel.shape, dtype=dt, device=sel.device)], dim=1)
        st = ctx.calib.gram
        outside = bool(keep_X.any())
        if outside or self.gram == "full":
            if st.full_G is None:
                raise RuntimeError("selection outside T needs the full-Gram calibration (PHASE0_SETUP.md 2.4)")
            Gp = gather_gram(st.full_G.to(sel.device, dt), pool)
        else:
            Gp = gather_gram(st.G_avg.to(sel.device, dt), pool)
        # duplicate pool entries never kept and carry zero gate: identity rows in ls_weights
        c = L.ls_weights(Gp, gp, keep, self.ridge)
        return pool, c.to(ctx.g.dtype)


class SketchAdd(Policy):
    """Honest sketch baseline: y^ = sum_S g_i E_i + sum_D g_j kappa_j E^_j, S = top-m by gate."""

    def __init__(self, m: int):
        self.m = m
        self.name = f"sketch-add[m={m}]"
        self.adds_sketch = True

    def __call__(self, ctx):
        keep = L.gate_order_mask(ctx.g, self.m)
        ctx.cache["sketch_add_drop"] = ~keep
        return ctx.ids, L.keep_weights(ctx.g.float(), keep).to(ctx.g.dtype)

    def extra_output(self, ctx) -> Tensor:
        """sum over dropped j of g_j kappa_j E^_j, [n,d] float32."""
        _, Ehat = _sketch_grams(ctx, torch.float32 if ctx.x.is_cuda else torch.float64)
        drop = ctx.cache["sketch_add_drop"]
        return (Ehat * (ctx.g.to(Ehat.dtype) * drop.to(Ehat.dtype)).unsqueeze(-1)).sum(1).float()
