"""Small batched linear algebra on per-token Gram matrices (PROJECT.md Section 5).

Conventions (all functions):
    G:    [n, k, k]  Gram matrix of expert outputs on the router's top-k set T, G_ij = <E_i, E_j>.
    g:    [n, k]     native gates on T (model's own convention).
    mask: [n, k]     boolean, True = expert kept (in S).
Work happens in the dtype of G (callers pass float64 on CPU, float32 on GPU; see `work_dtype`).
No Python loops over tokens; loops over at most k greedy steps or over subsets are allowed.
"""
from __future__ import annotations

import itertools
from typing import Iterable, Optional, Sequence

import torch
from torch import Tensor


def work_dtype(device: torch.device | str) -> torch.dtype:
    """float64 on CPU, float32 on GPU (PHASE0_SETUP.md 2.3)."""
    return torch.float64 if torch.device(device).type == "cpu" else torch.float32


def _tol(dtype: torch.dtype) -> float:
    return 100.0 * torch.finfo(dtype).eps


def quad_form(G: Tensor, v: Tensor) -> Tensor:
    """v^T G v per token. G [n,k,k], v [n,k] -> [n]."""
    return torch.einsum("ni,nij,nj->n", v, G, v)


def output_energy(G: Tensor, g: Tensor) -> Tensor:
    """||y||^2 = g^T G g (PROJECT.md 5.1)."""
    return quad_form(G, g)


def residual_energy(G: Tensor, g: Tensor, c_full: Tensor) -> Tensor:
    """Exact error e = (c~ - g)^T G (c~ - g) (PROJECT.md 5.1)."""
    return quad_form(G, c_full - g)


def _masked_system(G: Tensor, keep_mask: Tensor) -> Tensor:
    """G on the kept block, identity on dropped rows and columns."""
    m = keep_mask.to(G.dtype)
    A = G * (m.unsqueeze(-1) * m.unsqueeze(-2))
    return A + torch.diag_embed(1.0 - m)


def _robust_solve(A: Tensor, rhs: Tensor, scale: Tensor) -> Tensor:
    """Solve A x = rhs for SPD A [n,k,k], rhs [n,k]; Cholesky, jitter fallback, then lstsq.

    `scale` [n] sets the jitter size (mean kept diagonal).
    """
    L, info = torch.linalg.cholesky_ex(A)
    x = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
    bad = info != 0
    if bad.any():
        eye = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
        idx = bad.nonzero(as_tuple=True)[0]
        Ab, rb, sb = A[idx], rhs[idx], scale[idx].clamp_min(torch.finfo(A.dtype).tiny)
        xb = None
        for p in (1e-10, 1e-8, 1e-6, 1e-4):
            Aj = Ab + (p * sb)[:, None, None] * eye
            Lj, infoj = torch.linalg.cholesky_ex(Aj)
            if bool((infoj == 0).all()):
                xb = torch.cholesky_solve(rb.unsqueeze(-1), Lj).squeeze(-1)
                break
        if xb is None:  # last resort: minimum-norm least squares
            xb = torch.linalg.lstsq(Ab, rb.unsqueeze(-1), driver="gelsd" if A.device.type == "cpu" else None).solution.squeeze(-1)
        x = x.clone()
        x[idx] = xb
    return x


def ls_weights(G: Tensor, g: Tensor, keep_mask: Tensor, ridge: float = 0.0) -> Tensor:
    """Least-squares weights c* = (G_SS + ridge * mean(diag G_SS) I)^-1 G_ST g, zero on dropped.

    Returns c_full [n,k] (PROJECT.md 5.2). Tokens with an empty kept set get all-zero weights.
    """
    keep_mask = keep_mask.bool()
    m = keep_mask.to(G.dtype)
    rhs = torch.einsum("nij,nj->ni", G, g) * m  # rows of G_ST g on S, zero elsewhere
    A = _masked_system(G, keep_mask)
    diag = torch.diagonal(G, dim1=-2, dim2=-1)
    cnt = m.sum(-1)
    mean_diag = (diag * m).sum(-1) / cnt.clamp_min(1.0)
    if ridge > 0:
        A = A + torch.diag_embed((ridge * mean_diag).unsqueeze(-1) * m)
    c = _robust_solve(A, rhs, mean_diag)
    return c * m


def keep_weights(g: Tensor, keep_mask: Tensor) -> Tensor:
    """B1: keep the native gates on S."""
    return g * keep_mask.to(g.dtype)


def renorm_weights(g: Tensor, keep_mask: Tensor) -> Tensor:
    """B2: renormalise kept gates to the native total, c_i = g_i (a + b) / a."""
    m = keep_mask.to(g.dtype)
    a = (g * m).sum(-1, keepdim=True)
    tot = g.sum(-1, keepdim=True)
    return torch.where(a > 0, g * m * tot / a.clamp_min(torch.finfo(g.dtype).tiny), torch.zeros_like(g))


def uniform_coherence_weights(g: Tensor, keep_mask: Tensor, r: Tensor | float) -> Tensor:
    """Closed form under uniform coherence: c_i = g_i + r b / (1 - r + r m) on S (PROJECT.md 5.2)."""
    m_f = keep_mask.to(g.dtype)
    r = torch.as_tensor(r, dtype=g.dtype, device=g.device)
    if r.dim() == 1:
        r = r.unsqueeze(-1)
    b = (g * (1.0 - m_f)).sum(-1, keepdim=True)
    msz = m_f.sum(-1, keepdim=True)
    add = r * b / (1.0 - r + r * msz)
    return (g + add) * m_f


def uniform_gram(k: int, r: Tensor | float, n: int = 1, dtype=torch.float64, device="cpu") -> Tensor:
    """G = (1 - r) I + r 1 1^T, batched to [n,k,k]. r scalar or [n]."""
    r = torch.as_tensor(r, dtype=dtype, device=device).reshape(-1)
    if r.numel() == 1:
        r = r.expand(n)
    eye = torch.eye(k, dtype=dtype, device=device)
    ones = torch.ones(k, k, dtype=dtype, device=device)
    return (1.0 - r)[:, None, None] * eye + r[:, None, None] * ones


def closed_form_errors(g: Tensor, keep_mask: Tensor, r: Tensor | float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Errors of keep (e1), renormalise (e2), projection (e3), and ||y||^2 under uniform coherence."""
    m_f = keep_mask.to(g.dtype)
    r = torch.as_tensor(r, dtype=g.dtype, device=g.device)
    gs, gd = g * m_f, g * (1.0 - m_f)
    a, b = gs.sum(-1), gd.sum(-1)
    qS, qD = (gs**2).sum(-1), (gd**2).sum(-1)
    msz = m_f.sum(-1)
    e1 = (1 - r) * qD + r * b**2
    e2 = (1 - r) * ((b / a) ** 2 * qS + qD)
    e3 = (1 - r) * qD + (1 - r) * r * b**2 / (1 - r + r * msz)
    y2 = (1 - r) * (qS + qD) + r * (a + b) ** 2
    return e1, e2, e3, y2


# ----------------------------------------------------------------------------- greedy pursuit
def _greedy(G: Tensor, g: Tensor, m_max: int, tau: Optional[float], m_min: int,
            pool_mask: Optional[Tensor], criterion: str) -> tuple[Tensor, Tensor, Tensor]:
    n, k, _ = G.shape
    dtype, dev = G.dtype, G.device
    m_max = min(int(m_max), k)
    m_min = max(0, min(int(m_min), m_max))
    pool = torch.ones(n, k, dtype=torch.bool, device=dev) if pool_mask is None else pool_mask.bool().clone()
    Gp = G.clone()                                   # residual Gram after projecting out S
    bp = torch.einsum("nij,nj->ni", G, g)            # G (g - c~), correlation of each atom with the residual
    y2 = quad_form(G, g)
    y2_safe = torch.where(y2 > 0, y2, torch.ones_like(y2))
    R = y2.clone()
    diag0 = torch.diagonal(G, dim1=-2, dim2=-1).clone()
    tol = _tol(dtype)
    keep = torch.zeros(n, k, dtype=torch.bool, device=dev)
    order = torch.full((n, m_max), -1, dtype=torch.long, device=dev)
    path = torch.zeros(n, m_max + 1, dtype=dtype, device=dev)
    rel = torch.where(y2 > 0, R / y2_safe, torch.zeros_like(R))
    path[:, 0] = rel
    active = torch.ones(n, dtype=torch.bool, device=dev)
    ar = torch.arange(n, device=dev)
    for step in range(m_max):
        if tau is not None:
            done = (keep.sum(-1) >= m_min) & (rel <= max(tau, tol))
            active = active & ~done
        avail = pool & ~keep
        active = active & avail.any(-1)
        if not bool(active.any()):
            path[:, step + 1:] = path[:, step:step + 1]
            break
        gdiag = torch.diagonal(Gp, dim1=-2, dim2=-1)
        ok = gdiag > tol * diag0.clamp_min(torch.finfo(dtype).tiny)
        if criterion == "ols":
            score = torch.where(ok, bp**2 / gdiag.where(ok, torch.ones_like(gdiag)), torch.zeros_like(bp))
        elif criterion == "omp":
            score = torch.where(diag0 > 0, bp**2 / diag0.where(diag0 > 0, torch.ones_like(diag0)), torch.zeros_like(bp))
            score = torch.where(ok, score, torch.zeros_like(score))
        else:
            raise ValueError(criterion)
        score = score.masked_fill(~avail, -1.0)
        j = score.argmax(-1)
        gjj = Gp[ar, j, j]
        okj = ok[ar, j] & active
        gjj_safe = torch.where(okj, gjj, torch.ones_like(gjj))
        col = Gp[ar, :, j]                                    # [n,k]
        bj = bp[ar, j]
        drop = torch.where(okj, bj**2 / gjj_safe, torch.zeros_like(bj))
        # rank-one updates (Gram-Schmidt in coefficient space), only for active tokens with ok pivot
        f = okj.to(dtype)
        Gp = Gp - f[:, None, None] * col.unsqueeze(-1) * col.unsqueeze(-2) / gjj_safe[:, None, None]
        bp = bp - f[:, None] * col * (bj / gjj_safe)[:, None]
        R = torch.where(active, (R - drop).clamp_min(0.0), R)
        keep[ar[active], j[active]] = True
        order[ar[active], step] = j[active]
        rel = torch.where(y2 > 0, R / y2_safe, torch.zeros_like(R))
        path[:, step + 1] = rel
    return keep, order, path


def greedy_ols(G: Tensor, g: Tensor, m_max: int, tau: Optional[float] = None, m_min: int = 1,
               pool_mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor, Tensor]:
    """Forward selection by largest residual drop (OLS form, PROJECT.md 5.3).

    Fixed budget: tau=None, stops at m_max. Adaptive: stops once R(S)/(g^T G g) <= tau and |S| >= m_min.
    Returns keep_mask [n,k], order [n,m_max] (-1 = unused), residual_path [n,m_max+1] (relative).
    O(k^2) per step per token via rank-one updates of the projected Gram.
    """
    return _greedy(G, g, m_max, tau, m_min, pool_mask, "ols")


def greedy_omp(G: Tensor, g: Tensor, m_max: int, tau: Optional[float] = None, m_min: int = 1,
               pool_mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor, Tensor]:
    """OMP criterion: largest |<E_j, residual>| / ||E_j|| (ablation A2). Same interface as greedy_ols."""
    return _greedy(G, g, m_max, tau, m_min, pool_mask, "omp")


def gate_order_mask(g: Tensor, m: int) -> Tensor:
    """Keep the m largest gates (thresholding)."""
    n, k = g.shape
    m = min(int(m), k)
    idx = g.topk(m, dim=-1).indices
    mask = torch.zeros(n, k, dtype=torch.bool, device=g.device)
    mask.scatter_(1, idx, True)
    return mask


def _subset_masks(k: int, m: int, device) -> Tensor:
    combos = list(itertools.combinations(range(k), m))
    masks = torch.zeros(len(combos), k, dtype=torch.bool, device=device)
    for i, c in enumerate(combos):
        masks[i, list(c)] = True
    return masks


def exhaustive_best(G: Tensor, g: Tensor, m: int, ridge: float = 0.0) -> tuple[Tensor, Tensor]:
    """Best m-subset of T with least-squares weights from G (oracle U1 when G is the true Gram).

    Returns keep_mask [n,k] and absolute error [n] (error measured with the same G).
    """
    n, k, _ = G.shape
    masks = _subset_masks(k, m, G.device)
    best_err = torch.full((n,), float("inf"), dtype=G.dtype, device=G.device)
    best_idx = torch.zeros(n, dtype=torch.long, device=G.device)
    for s in range(masks.shape[0]):
        mk = masks[s].expand(n, k)
        c = ls_weights(G, g, mk, ridge)
        e = residual_energy(G, g, c)
        better = e < best_err
        best_err = torch.where(better, e, best_err)
        best_idx = torch.where(better, torch.full_like(best_idx, s), best_idx)
    return masks[best_idx], best_err.clamp_min(0.0)


def best_subset_rel_errors(G: Tensor, g: Tensor) -> Tensor:
    """Relative error of the best m-subset with oracle weights for m = 1..k. Returns [n,k]."""
    n, k, _ = G.shape
    y2 = output_energy(G, g)
    y2_safe = torch.where(y2 > 0, y2, torch.ones_like(y2))
    out = torch.zeros(n, k, dtype=G.dtype, device=G.device)
    for m in range(1, k + 1):
        if m == k:
            out[:, m - 1] = 0.0
            continue
        _, e = exhaustive_best(G, g, m)
        out[:, m - 1] = torch.where(y2 > 0, e / y2_safe, torch.zeros_like(e))
    return out


def effective_experts(G: Tensor, g: Tensor, levels: Sequence[float] = (0.90, 0.95, 0.99),
                      rel_errors: Optional[Tensor] = None) -> tuple[Tensor, ...]:
    """m_level(x): smallest m whose best subset (oracle weights) leaves relative error <= 1 - level.

    Returns one [n] long tensor per level (PROJECT.md 5.5).
    """
    rel = best_subset_rel_errors(G, g) if rel_errors is None else rel_errors
    k = rel.shape[-1]
    outs = []
    for lev in levels:
        ok = rel <= (1.0 - lev) + 1e-12
        ok[:, -1] = True
        outs.append(ok.to(torch.long).argmax(-1) + 1)
    return tuple(outs)


def participation_ratio(G: Tensor, g: Tensor) -> Tensor:
    """PR = (sum lambda)^2 / sum lambda^2 for M = diag(g) G diag(g) (PROJECT.md 5.5)."""
    M = g.unsqueeze(-1) * G * g.unsqueeze(-2)
    lam = torch.linalg.eigvalsh(M).clamp_min(0.0)
    s1 = lam.sum(-1)
    s2 = (lam**2).sum(-1)
    return torch.where(s2 > 0, s1**2 / s2.where(s2 > 0, torch.ones_like(s2)), torch.ones_like(s1))


def cosines(G: Tensor) -> Tensor:
    """Cosine matrix from a Gram, [n,k,k]."""
    d = torch.diagonal(G, dim1=-2, dim2=-1).clamp_min(torch.finfo(G.dtype).tiny).sqrt()
    return G / (d.unsqueeze(-1) * d.unsqueeze(-2))


def coherence(G: Tensor) -> tuple[Tensor, Tensor]:
    """(mu, mean r): largest off-diagonal |cosine| and mean off-diagonal cosine inside T."""
    C = cosines(G)
    k = C.shape[-1]
    off = ~torch.eye(k, dtype=torch.bool, device=G.device)
    vals = C[:, off]
    return vals.abs().max(-1).values, vals.mean(-1)


def top_gate_cosine(G: Tensor, g: Tensor) -> Tensor:
    """Cosine between the top-gate expert's output and y."""
    j = g.argmax(-1)
    n = G.shape[0]
    ar = torch.arange(n, device=G.device)
    Gj = G[ar, j]                              # [n,k]
    inner = (Gj * g).sum(-1)
    y2 = output_energy(G, g).clamp_min(torch.finfo(G.dtype).tiny)
    return inner / (G[ar, j, j].clamp_min(torch.finfo(G.dtype).tiny).sqrt() * y2.sqrt())
