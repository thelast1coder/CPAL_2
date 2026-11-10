"""Training-free baselines from papers (PROJECT.md Section 10), implemented from docs/related_work_notes.md.

Every class states its reading of the paper. Where the paper is unclear the class has
FIDELITY_UNCHECKED = True and the open question is in docs/related_work_notes.md. Classes that stay inside
the router's top-k set T also provide `offline(rec, calib, m) -> (keep_mask, c_full)` for layer-level
evaluation from recorded statistics (top-32 router probabilities are recorded).
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import torch
from torch import Tensor

from . import linalg as L
from .policies import Policy, RouteCtx


def _probs_on_T(ids: Tensor, top_ids: Tensor, top_p: Tensor) -> Tensor:
    """Router softmax probability of each expert in T, from recorded top-32 ids and probabilities."""
    match = ids.long().unsqueeze(-1) == top_ids.long().unsqueeze(-2)        # [n,k,32]
    return (match.to(top_p.dtype) * top_p.unsqueeze(-2)).sum(-1)


# ------------------------------------------------------------------------------------------------ B3
class K2Fix(Policy):
    """B3, arXiv 2609.04575 (Chen and Yao), Eq. 2: keep S = top-k1 by router probability and weight
    w_i = p_i / sum_{j in T_k2} p_j, p = softmax over all experts. One integer k2 (paper: scan k2 in
    {k1, k, 2k}; default here k2 = k, the trained value). For routers trained without renormalisation
    (OLMoE) the literal formula is applied as written; whether the authors intend that is open (notes)."""
    FIDELITY_UNCHECKED = True

    def __init__(self, m: int, k2: Optional[int] = None):
        self.m, self.k2 = m, k2
        self.name = f"B3[k1={m},k2={k2 or 'k'}]"

    def _weights(self, p_T: Tensor, mass_k2: Tensor, keep: Tensor) -> Tensor:
        return p_T * keep.to(p_T.dtype) / mass_k2.unsqueeze(-1)

    def __call__(self, ctx: RouteCtx):
        p = torch.softmax(ctx.logits.float(), -1)
        k = ctx.ids.shape[1]
        k2 = self.k2 or k
        mass = p.topk(min(k2, p.shape[-1]), -1).values.sum(-1)
        p_T = p.gather(1, ctx.ids)
        keep = L.gate_order_mask(p_T, self.m)
        return ctx.ids, self._weights(p_T, mass, keep).to(ctx.g.dtype)

    def offline(self, rec: dict, calib, m: Optional[int]):
        k = rec["ids"].shape[1]
        k2 = self.k2 or k
        if k2 > rec["top_p"].shape[1]:
            raise ValueError("k2 larger than the recorded top-p list")
        top_p = rec["top_p"].double()
        p_T = _probs_on_T(rec["ids"], rec["top_ids"], top_p)
        mass = top_p[:, :k2].sum(-1)
        keep = L.gate_order_mask(p_T, m if m is not None else self.m)
        return keep, self._weights(p_T, mass, keep)


# ------------------------------------------------------------------------------------------------ B4
class Elbow(Policy):
    """B4, arXiv 2608.04401: Kneedle elbow on the sorted full softmax. p'_i = (p_i - p_min)/(p_max - p_min + eps),
    x'_i = i/(N-1), e = argmax (p'_i - x'_i), k(x) = min(e + 1, K) (0-based e; the paper writes both
    "k = e + 1" and "min(e(x), K)", open question). Weights: raw gates, renormalised only if the model
    renormalises (released code)."""
    FIDELITY_UNCHECKED = True
    name = "B4"

    def __call__(self, ctx: RouteCtx):
        p = torch.softmax(ctx.logits.float(), -1)
        ps = p.sort(-1, descending=True).values
        N = ps.shape[-1]
        pn = (ps - ps[:, -1:]) / (ps[:, :1] - ps[:, -1:] + 1e-12)
        x = torch.arange(N, device=p.device, dtype=p.dtype) / (N - 1)
        e = (pn - x).argmax(-1)
        k = ctx.ids.shape[1]
        kk = (e + 1).clamp(1, k)
        order = ctx.g.float().argsort(-1, descending=True)
        rank = torch.empty_like(order).scatter_(1, order, torch.arange(k, device=p.device).expand_as(order))
        keep = rank < kk.unsqueeze(-1)
        renorm = bool(getattr(ctx.block.gate, "norm_topk_prob", False)) if ctx.block is not None else False
        c = L.renorm_weights(ctx.g.float(), keep) if renorm else L.keep_weights(ctx.g.float(), keep)
        return ctx.ids, c.to(ctx.g.dtype)


# ------------------------------------------------------------------------------------------------ B5
class EAT(Policy):
    """B5, arXiv 2609.33614 (routing rule only; the OPD router fine-tuning is excluded as not training-free).
    History score S_h,i = w1 N_i/sum N + w2 P_i/N_i + w3 C_i (C = EMA of output L2 norm, alpha = 0.95);
    importance I = (1 - beta) p + beta S_h; threshold tau = max(a1 mu_w, mu_w - a2 sigma_w) over the token's
    raw top-k gates. The perplexity-based threshold adjustment is omitted (needs labels at inference,
    open question). Candidates restricted to T (open question). Weights renormalised (Eq. 7)."""
    FIDELITY_UNCHECKED = True
    name = "B5"

    def __init__(self, w=(0.3, 0.3, 0.4), beta: float = 0.6, alpha: float = 0.95, a1: float = 0.5,
                 a2: float = 1.0, tau_scale: float = 1.0):
        self.w, self.beta, self.alpha, self.a1, self.a2, self.tau_scale = w, beta, alpha, a1, a2, tau_scale
        self.state: dict[int, dict] = {}

    def __call__(self, ctx: RouteCtx):
        N = ctx.logits.shape[-1]
        st = self.state.setdefault(ctx.layer, dict(cnt=torch.zeros(N), psum=torch.zeros(N), ema=torch.zeros(N)))
        p = torch.softmax(ctx.logits.float(), -1)
        cnt, psum, ema = (st[k_].to(p.device) for k_ in ("cnt", "psum", "ema"))
        Sh = self.w[0] * cnt / cnt.sum().clamp_min(1) + self.w[1] * psum / cnt.clamp_min(1) + self.w[2] * ema
        imp = ((1 - self.beta) * p + self.beta * Sh).gather(1, ctx.ids)
        gT = p.gather(1, ctx.ids)
        mu, sd = gT.mean(-1, keepdim=True), gT.std(-1, keepdim=True)
        tau = torch.maximum(self.a1 * mu, mu - self.a2 * sd) * self.tau_scale
        keep = imp > tau
        none = ~keep.any(-1)
        keep[none, imp[none].argmax(-1)] = True
        ctx.cache["eat_keep"] = keep
        return ctx.ids, L.renorm_weights(ctx.g.float(), keep).to(ctx.g.dtype)

    def update_history(self, ctx: RouteCtx, kept_out_norms: Tensor) -> None:
        """Called by the runner after the block with per-slot output norms [n,k]."""
        st = self.state[ctx.layer]
        keep = ctx.cache["eat_keep"]
        ids = ctx.ids[keep].cpu()
        st["cnt"].index_add_(0, ids, torch.ones(ids.numel()))
        st["psum"].index_add_(0, ids, torch.softmax(ctx.logits.float(), -1).gather(1, ctx.ids)[keep].cpu())
        nr = kept_out_norms[keep].float().cpu()
        upd = torch.zeros_like(st["ema"]).index_add_(0, ids, nr) / torch.zeros_like(st["ema"]).index_add_(
            0, ids, torch.ones_like(nr)).clamp_min(1)
        hit = torch.zeros_like(st["ema"], dtype=torch.bool)
        hit[ids] = True
        st["ema"] = torch.where(hit, (1 - self.alpha) * st["ema"] + self.alpha * upd, st["ema"])


# ------------------------------------------------------------------------------------------------ B6
class OMPMoE(Policy):
    """B6, arXiv 2609.31631, inference part OMP-MoE-dagger (Eq. 14, Alg. 3): alpha_e = E||F_e||^2 offline
    (here the calibrated mean squared norm, G-avg diagonal); E^_j = g_j^2 alpha_j in router order;
    m* = min m with 1 - S_m/(S_K + zeta) <= eps; keep the native gates (no renormalisation).
    The static pruning part of the paper (memory side) is not a per-token policy and is not implemented."""
    FIDELITY_UNCHECKED = False

    def __init__(self, eps: float = 0.1, zeta: float = 1e-6):
        self.eps, self.zeta = eps, zeta
        self.name = f"B6[eps={eps}]"

    def _keep(self, g: Tensor, alpha_T: Tensor) -> Tensor:
        n, k = g.shape
        order = g.argsort(-1, descending=True)
        en = (g.gather(1, order) ** 2) * alpha_T.gather(1, order)
        S = en.cumsum(-1)
        ok = 1 - S / (S[:, -1:] + self.zeta) <= self.eps
        ok[:, -1] = True
        mstar = ok.to(torch.long).argmax(-1) + 1
        mask_sorted = torch.arange(k, device=g.device).unsqueeze(0) < mstar.unsqueeze(-1)
        return torch.zeros_like(mask_sorted).scatter(1, order, mask_sorted)

    def __call__(self, ctx: RouteCtx):
        alpha = torch.diagonal(ctx.calib.gram.G_avg).to(ctx.g.device, torch.float32)
        keep = self._keep(ctx.g.float(), alpha[ctx.ids])
        return ctx.ids, L.keep_weights(ctx.g.float(), keep).to(ctx.g.dtype)

    def offline(self, rec, calib, m):
        g = rec["g"].double()
        alpha = torch.diagonal(calib.gram.G_avg).double()
        keep = self._keep(g, alpha[rec["ids"].long()])
        return keep, L.keep_weights(g, keep)


# ------------------------------------------------------------------------------------------------ B7
class SERE(Policy):
    """B7, arXiv 2602.07616: per batch, primary set = union of each token's top-S experts; each secondary
    expert u (in some token's T, not primary) is re-routed to v* = argmax_{v primary} Sim[u, v] if
    Sim[u, v*] >= rho, else kept. Router weights unchanged (a re-routed slot keeps u's gate; if v* is
    already in the token's T the two gates add on v*). Sim: normalised Frobenius similarity
    1 - ||A_u - A_v||_F / max, computed from the full-Gram calibration (all experts on calibration tokens)."""
    FIDELITY_UNCHECKED = True

    def __init__(self, s_primary: int = 2, rho: float = 0.5):
        self.s, self.rho = s_primary, rho
        self.name = f"B7[S={s_primary},rho={rho}]"

    @staticmethod
    def similarity(full_G: Tensor) -> Tensor:
        d = torch.diagonal(full_G)
        dist = (d.unsqueeze(0) + d.unsqueeze(1) - 2 * full_G).clamp_min(0).sqrt()
        return 1 - dist / dist.max().clamp_min(1e-30)

    def __call__(self, ctx: RouteCtx):
        st = ctx.calib.gram
        if st.full_G is None:
            raise RuntimeError("SERE needs the full-Gram calibration")
        Sim = self.similarity(st.full_G.double()).to(ctx.ids.device)
        N = Sim.shape[0]
        order = ctx.g.float().argsort(-1, descending=True)
        prim_ids = ctx.ids.gather(1, order[:, : self.s])
        if ctx.valid is not None:
            prim_ids = prim_ids[ctx.valid]
        primary = torch.zeros(N, dtype=torch.bool, device=ctx.ids.device)
        primary[prim_ids.reshape(-1)] = True
        simP = Sim.masked_fill(~primary.unsqueeze(0), -1e9)
        best_sim, best_v = simP.max(-1)                                    # per expert u
        reroute = (~primary) & (best_sim >= self.rho)
        new_ids = torch.where(reroute[ctx.ids], best_v[ctx.ids], ctx.ids)
        w = ctx.g.clone()
        # merge duplicates within a token: add gates onto the first slot holding that id
        k = new_ids.shape[1]
        for a in range(k):
            for b in range(a + 1, k):
                dup = (new_ids[:, a] == new_ids[:, b]) & (w[:, b] != 0) & (w[:, a] != 0)
                w[:, a] = torch.where(dup, w[:, a] + w[:, b], w[:, a])
                w[:, b] = torch.where(dup, torch.zeros_like(w[:, b]), w[:, b])
        return new_ids, w


# ------------------------------------------------------------------------------------------------ B8
class CacheSub(Policy):
    """B8, offloading setting. Two readings:
    mode="cce": CCE Cache-Prior (arXiv 2412.00099, Eqs. 9 and 10): z' = z + lam * Delta_avg * m, m = per-layer
      cache bitmask (top-J of z always included), select top-k of z'; weights from the unmodified router
      (softmax of z on the new set, renormalised only if the model renormalises: open question).
    mode="smoe": SMoE substitution (arXiv 2508.18983, paper plus released code): with s_ref the (k+1)-th
      probability, top-k experts with p >= (1+a) s_ref are kept; other slots keep their expert if cached,
      else take a cached substitute with (1-a) s_ref < p < s_ref, else load; weights = each used expert's own
      probability, renormalised if the model renormalises.
    The cache is an LRU of `cache_size` experts per layer, updated after each call (token order inside a
    batch is ignored: tokens of one call see the same cache state)."""
    FIDELITY_UNCHECKED = True

    def __init__(self, cache_size: int, mode: str = "cce", lam: float = 0.5, J: int = 2, alpha: float = 0.25):
        self.cache_size, self.mode, self.lam, self.J, self.alpha = cache_size, mode, lam, J, alpha
        self.name = f"B8[{mode},C={cache_size}]"
        self.caches: dict[int, OrderedDict] = {}
        self.delta_avg: dict[int, tuple[float, int]] = {}
        self.loads = 0
        self.lookups = 0

    def _mask(self, layer: int, N: int, device) -> Tensor:
        m = torch.zeros(N, dtype=torch.bool, device=device)
        for e in self.caches.get(layer, {}):
            m[e] = True
        return m

    def _touch(self, layer: int, used: Tensor) -> None:
        c = self.caches.setdefault(layer, OrderedDict())
        for e in used.reshape(-1).tolist():
            self.lookups += 1
            if e in c:
                c.move_to_end(e)
            else:
                self.loads += 1
                c[e] = True
                if len(c) > self.cache_size:
                    c.popitem(last=False)

    def __call__(self, ctx: RouteCtx):
        z = ctx.logits.float()
        n, N = z.shape
        k = ctx.ids.shape[1]
        p = torch.softmax(z, -1)
        cached = self._mask(ctx.layer, N, z.device)
        renorm = bool(getattr(ctx.block.gate, "norm_topk_prob", False)) if ctx.block is not None else False
        if self.mode == "cce":
            s, cnt = self.delta_avg.get(ctx.layer, (0.0, 0))
            spread = (z.max(-1).values - z.min(-1).values)
            s, cnt = s + float(spread.sum()), cnt + n
            self.delta_avg[ctx.layer] = (s, cnt)
            m = cached.unsqueeze(0).expand(n, N).clone()
            m.scatter_(1, z.topk(self.J, -1).indices, True)
            zp = z + self.lam * (s / cnt) * m.float()
            ids = zp.topk(k, -1).indices
        else:
            pk = p.topk(k + 1, -1)
            s_ref = pk.values[:, -1:]
            ids = ctx.ids.clone()
            gT = p.gather(1, ids)
            strong = gT >= (1 + self.alpha) * s_ref
            cand = (p > (1 - self.alpha) * s_ref) & (p < s_ref) & cached.unsqueeze(0)
            for slot in range(k):
                need = ~strong[:, slot] & ~cached[ids[:, slot]] & cand.any(-1)
                if not bool(need.any()):
                    continue
                pick = cand.float().argmax(-1)
                ids[:, slot] = torch.where(need, pick, ids[:, slot])
                cand[need, pick[need]] = False
        w = p.gather(1, ids)
        if renorm:
            w = w / w.sum(-1, keepdim=True)
        self._touch(ctx.layer, ids if ctx.valid is None else ids[ctx.valid])
        return ids, w.to(ctx.g.dtype)


# ------------------------------------------------------------------------------------------------ B9
class GateRatioSkip(Policy):
    """B9, NAEE dynamic skipping (arXiv 2402.14800, Lu et al. 2024): for top-2, skip e1 if w_e1 < beta w_e0,
    beta per layer = median of w_e1/w_e0 on calibration data; released code renormalises. Generalised to
    k > 2 (our reading, open question): keep every expert with g_j >= beta_layer * g_max; beta_layer from
    calibration (median ratio) or tuned to a target mean. Weights renormalised."""
    FIDELITY_UNCHECKED = True

    def __init__(self, beta: dict[int, float] | float = 0.5):
        self.beta = beta
        self.name = "B9"

    def _b(self, layer):
        return self.beta.get(layer, 0.5) if isinstance(self.beta, dict) else float(self.beta)

    def _keep(self, g: Tensor, b: float) -> Tensor:
        return g >= b * g.max(-1, keepdim=True).values

    def __call__(self, ctx: RouteCtx):
        keep = self._keep(ctx.g.float(), self._b(ctx.layer))
        return ctx.ids, L.renorm_weights(ctx.g.float(), keep).to(ctx.g.dtype)

    def offline(self, rec, calib, m, layer: int = -1):
        g = rec["g"].double()
        keep = self._keep(g, self._b(layer))
        return keep, L.renorm_weights(g, keep)

    @staticmethod
    def median_ratio(g: Tensor) -> float:
        s = g.float().sort(-1, descending=True).values
        return float((s[:, 1] / s[:, 0].clamp_min(1e-30)).median())
