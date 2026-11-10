"""The law (claim C1): effective number of experts per token, by layer and domain (PROJECT.md 5.5)."""
from __future__ import annotations

from typing import Optional

import torch

from ..routing import linalg as L

DOMAINS = {0: "text", 1: "maths", 2: "code"}


def _summ(x: torch.Tensor) -> dict:
    x = x.double()
    return dict(median=float(x.median()), mean=float(x.mean()), p10=float(torch.quantile(x, 0.1)),
                p90=float(torch.quantile(x, 0.9)))


@torch.no_grad()
def layer_law(rec: dict, chunk: int = 8192) -> dict:
    """m90/m95/m99 distributions, PR, coherence (mu, mean r), top-gate cosine for one layer's records."""
    m90s, m95s, m99s, prs, mus, rs, tgs = [], [], [], [], [], [], []
    n = rec["ids"].shape[0]
    for s in range(0, n, chunk):
        G = rec["G"][s:s + chunk].double()
        g = rec["g"][s:s + chunk].double()
        rel = L.best_subset_rel_errors(G, g)
        m90, m95, m99 = L.effective_experts(G, g, rel_errors=rel)
        m90s.append(m90); m95s.append(m95); m99s.append(m99)
        prs.append(L.participation_ratio(G, g))
        mu, r = L.coherence(G)
        mus.append(mu); rs.append(r)
        tgs.append(L.top_gate_cosine(G, g))
    cat = lambda xs: torch.cat(xs)
    m90, m95, m99 = cat(m90s), cat(m95s), cat(m99s)
    k = rec["ids"].shape[1]
    out = dict(n_tokens=n, k=k,
               m90=_summ(m90), m95=_summ(m95), m99=_summ(m99),
               m95_hist=[int((m95 == j).sum()) for j in range(1, k + 1)],
               pr=_summ(cat(prs)), mu=_summ(cat(mus)), mean_r=_summ(cat(rs)), top_gate_cos=_summ(cat(tgs)))
    if "domain" in rec:
        dom = rec["domain"].long()
        out["by_domain"] = {}
        for d, name in DOMAINS.items():
            sel = dom == d
            if int(sel.sum()) == 0:
                continue
            out["by_domain"][name] = dict(n_tokens=int(sel.sum()), m95=_summ(m95[sel]), pr=_summ(cat(prs)[sel]))
    out["_m95"] = m95
    return out
