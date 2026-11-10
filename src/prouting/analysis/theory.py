"""Theory checks (claim C2, PROJECT.md 5.7): closed-form predictions, lambda_min, exact recovery,
greedy-to-oracle gap and thresholding failure, per layer.

The precise statements of Tropp (2004) and Das and Kempe (2011) are to be confirmed against the sources
(verify); the quantities below follow PROJECT.md 5.7.
"""
from __future__ import annotations

import torch

from ..routing import linalg as L


def uniform_model_gram(G: torch.Tensor, r) -> torch.Tensor:
    """Per-token uniform-coherence model with measured norms: n n^T * ((1 - r) I + r 1 1^T)."""
    n, k, _ = G.shape
    nrm = torch.diagonal(G, dim1=-2, dim2=-1).clamp_min(0).sqrt()
    U = L.uniform_gram(k, r, n=n, dtype=G.dtype, device=G.device)
    return nrm.unsqueeze(-1) * U * nrm.unsqueeze(-2)


@torch.no_grad()
def closed_form_vs_measured(rec: dict, m: int, r_layer: float) -> dict:
    """Predicted (uniform model, layer r, measured gates and norms) vs measured error energy, gate-order S.

    Rules: keep, renormalise, projection (oracle LS on the true Gram). Returns relative error energies.
    """
    G = rec["G"].double()
    g = rec["g"].double()
    keep = L.gate_order_mask(g, m)
    Gm = uniform_model_gram(G, r_layer)
    out = {}
    for name, Gx in (("pred", Gm), ("meas", G)):
        y2 = L.output_energy(Gx, g).sum()
        c_keep = L.keep_weights(g, keep)
        c_ren = L.renorm_weights(g, keep)
        c_ls = L.ls_weights(Gx, g, keep)
        out[name] = dict(keep=float(L.residual_energy(Gx, g, c_keep).sum() / y2),
                         renorm=float(L.residual_energy(Gx, g, c_ren).sum() / y2),
                         proj=float(L.residual_energy(Gx, g, c_ls).sum() / y2))
    return out


@torch.no_grad()
def lambda_min_normalised(G: torch.Tensor) -> torch.Tensor:
    """Smallest eigenvalue of the cosine (normalised) Gram on T, per token."""
    return torch.linalg.eigvalsh(L.cosines(G)).min(-1).values


@torch.no_grad()
def erc_holds(G: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Exact recovery condition on the normalised Gram for support S* = keep:
    max over j outside S* of ||C_{S*S*}^-1 C_{S* j}||_1 < 1 (Tropp 2004, verify statement)."""
    C = L.cosines(G)
    n, k, _ = C.shape
    A = C * (keep.unsqueeze(-1) & keep.unsqueeze(-2)).to(C.dtype) + torch.diag_embed((~keep).to(C.dtype))
    B = C * keep.unsqueeze(-1).to(C.dtype)                      # rows in S*, all columns
    X = torch.linalg.solve(A, B)                                # [n,k,k]: column j = C_SS^-1 C_Sj (rows in S)
    X = X * keep.unsqueeze(-1).to(C.dtype)
    l1 = X.abs().sum(-2)                                        # [n,k]
    l1 = l1.masked_fill(keep, 0.0)
    return l1.max(-1).values < 1.0


@torch.no_grad()
def layer_theory(rec: dict, m_values=(1, 2, 3), chunk: int = 8192) -> dict:
    out = {}
    n = rec["ids"].shape[0]
    lam, thr_fail = [], []
    per_m = {m: dict(erc=[], gap=[], gain=[]) for m in m_values}
    for s in range(0, n, chunk):
        G = rec["G"][s:s + chunk].double()
        g = rec["g"][s:s + chunk].double()
        y2 = L.output_energy(G, g).clamp_min(1e-300)
        lam.append(lambda_min_normalised(G))
        best1, _ = L.exhaustive_best(G, g, 1)
        thr_fail.append(best1.long().argmax(-1) != g.argmax(-1))
        for m in m_values:
            S_opt, e_opt = L.exhaustive_best(G, g, m)
            per_m[m]["erc"].append(erc_holds(G, S_opt))
            keep_g, _, _ = L.greedy_ols(G, g, m)
            e_g = L.residual_energy(G, g, L.ls_weights(G, g, keep_g))
            gate = L.gate_order_mask(g, m)
            e_gate = L.residual_energy(G, g, L.ls_weights(G, g, gate))
            per_m[m]["gap"].append((e_g - e_opt) / y2)
            per_m[m]["gain"].append((e_gate - e_g) / y2)
    lam_t = torch.cat(lam)
    out["lambda_min_median"] = float(lam_t.median())
    out["threshold_failure_rate"] = float(torch.cat(thr_fail).double().mean())
    for m in m_values:
        out[f"m{m}"] = dict(erc_fraction=float(torch.cat(per_m[m]["erc"]).double().mean()),
                            greedy_oracle_gap_mean=float(torch.cat(per_m[m]["gap"]).mean()),
                            pursuit_gain_over_gate_mean=float(torch.cat(per_m[m]["gain"]).mean()))
    return out
