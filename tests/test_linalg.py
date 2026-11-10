"""T3 to T7: Gram identities, least squares, closed forms, greedy pursuit, adaptive stop."""
import itertools

import torch

from conftest import random_outputs
from prouting.routing import linalg as L


def _gram(E):
    return torch.einsum("nid,njd->nij", E, E)


def test_T3_gram_identities(gen):
    E, g = random_outputs(200, 8, 16, gen)
    G = _gram(E)
    y = torch.einsum("nk,nkd->nd", g, E)
    assert torch.allclose(L.output_energy(G, g), (y**2).sum(-1), rtol=1e-6)
    S = torch.rand(200, 8, generator=gen) < 0.5
    c = torch.randn(200, 8, generator=gen, dtype=torch.float64) * S
    yh = torch.einsum("nk,nkd->nd", c, E)
    assert torch.allclose(L.residual_energy(G, g, c), ((yh - y) ** 2).sum(-1), rtol=1e-6)


def test_T4_ls_weights(gen):
    E, g = random_outputs(100, 8, 32, gen)
    G = _gram(E)
    y = torch.einsum("nk,nkd->nd", g, E)
    keep = torch.zeros(100, 8, dtype=torch.bool)
    keep[:, [0, 3, 5]] = True
    c = L.ls_weights(G, g, keep, ridge=0.0)
    assert (c[~keep] == 0).all()
    r = y - torch.einsum("nk,nkd->nd", c, E)
    inner = torch.einsum("nd,nkd->nk", r, E)[keep].abs()
    scale = (r.norm(dim=-1, keepdim=True) * E.norm(dim=-1))[keep]
    assert (inner / scale).max() < 1e-6
    A = E[:, [0, 3, 5]].transpose(1, 2)                      # [n,d,3]
    ref = torch.linalg.lstsq(A, y.unsqueeze(-1)).solution.squeeze(-1)
    assert torch.allclose(c[:, [0, 3, 5]], ref, rtol=1e-6, atol=1e-9)


def test_T5_special_cases_and_closed_forms(gen):
    n, k = 1000, 8
    g = torch.rand(n, k, generator=gen, dtype=torch.float64)
    m = torch.randint(1, k, (n,), generator=gen)
    keep = torch.rand(n, k, generator=gen).argsort(-1) < m.unsqueeze(-1)
    # G = I: keep gates is optimal
    I = torch.eye(k, dtype=torch.float64).expand(n, k, k)
    assert torch.allclose(L.ls_weights(I, g, keep), g * keep, atol=1e-12)
    # G = all ones: LS solution exact (zero error), renormalisation also exact
    O = torch.ones(n, k, k, dtype=torch.float64)
    c = L.ls_weights(O, g, keep, ridge=0.0)
    assert L.residual_energy(O, g, c).abs().max() < 1e-8 * L.output_energy(O, g).max()
    assert L.residual_energy(O, g, L.renorm_weights(g, keep)).abs().max() < 1e-10
    # uniform coherence: closed forms equal the numeric solver
    r = torch.rand(n, generator=gen, dtype=torch.float64) * 0.98
    U = L.uniform_gram(k, r, n=n)
    c_num = L.ls_weights(U, g, keep, ridge=0.0)
    c_cf = L.uniform_coherence_weights(g, keep, r)
    assert torch.allclose(c_num, c_cf, atol=1e-8)
    e1, e2, e3, y2 = L.closed_form_errors(g, keep, r)
    assert torch.allclose(e1, L.residual_energy(U, g, L.keep_weights(g, keep)), atol=1e-8)
    assert torch.allclose(e2, L.residual_energy(U, g, L.renorm_weights(g, keep)), atol=1e-8)
    assert torch.allclose(e3, L.residual_energy(U, g, c_num), atol=1e-8)
    assert torch.allclose(y2, L.output_energy(U, g), atol=1e-8)
    assert (e2 - e3 >= -1e-10).all()


def test_T5_quoted_ratios():
    """PROJECT.md 5.2: m = 1, equal gates over k = 8: e2/e3 = 4.7, 2.6, 1.8, 1.1 at r = 0.1, 0.3, 0.5, 0.9."""
    g = torch.full((1, 8), 1 / 8, dtype=torch.float64)
    keep = torch.zeros(1, 8, dtype=torch.bool)
    keep[0, 0] = True
    got = []
    for r in (0.1, 0.3, 0.5, 0.9):
        _, e2, e3, _ = L.closed_form_errors(g, keep, r)
        got.append(round(float(e2 / e3), 1))
    assert got == [4.7, 2.6, 1.8, 1.1]


def _brute_forward(G, g, m):
    n, k, _ = G.shape
    out = torch.zeros(n, k, dtype=torch.bool)
    for t in range(n):
        S = []
        for _ in range(m):
            best, bj = None, None
            for j in range(k):
                if j in S:
                    continue
                mk = torch.zeros(1, k, dtype=torch.bool)
                mk[0, S + [j]] = True
                e = float(L.residual_energy(G[t:t + 1], g[t:t + 1], L.ls_weights(G[t:t + 1], g[t:t + 1], mk)))
                if best is None or e < best - 1e-12:
                    best, bj = e, j
            S.append(bj)
        out[t, S] = True
    return out


def test_T6_greedy_ols(gen):
    E, g = random_outputs(60, 8, 24, gen, coherent=1.0)
    G = _gram(E)
    keep, order, path = L.greedy_ols(G, g, m_max=8)
    assert (path[:, 1:] <= path[:, :-1] + 1e-12).all()
    assert path[:, -1].abs().max() < 1e-9
    for m in (1, 2, 3):
        km, _, _ = L.greedy_ols(G, g, m_max=m)
        assert torch.equal(km, _brute_forward(G, g, m))
        e_g = L.residual_energy(G, g, L.ls_weights(G, g, km))
        _, e_x = L.exhaustive_best(G, g, m)
        assert (e_x <= e_g + 1e-9 * L.output_energy(G, g)).all()
    # path value equals the LS residual of the selected prefix
    k2, _, p2 = L.greedy_ols(G, g, m_max=2)
    rel = L.residual_energy(G, g, L.ls_weights(G, g, k2)) / L.output_energy(G, g)
    assert torch.allclose(p2[:, 2], rel, atol=1e-9)
    # OMP variant runs and is non-increasing
    _, _, po = L.greedy_omp(G, g, m_max=8)
    assert (po[:, 1:] <= po[:, :-1] + 1e-12).all()


def test_T6_m1_rule(gen):
    """m = 1 picks argmax (G_jT g)^2 / G_jj."""
    E, g = random_outputs(100, 8, 16, gen)
    G = _gram(E)
    keep, _, _ = L.greedy_ols(G, g, 1)
    b = torch.einsum("nij,nj->ni", G, g)
    j = (b**2 / torch.diagonal(G, dim1=-2, dim2=-1)).argmax(-1)
    assert torch.equal(keep.long().argmax(-1), j)


def test_T7_adaptive_stop(gen):
    E, g = random_outputs(200, 8, 32, gen)
    G = _gram(E)
    k0, _, _ = L.greedy_ols(G, g, m_max=8, tau=0.0)
    assert (k0.sum(-1) == 8).all()                       # full-rank: zero residual needs every expert
    k1, _, _ = L.greedy_ols(G, g, m_max=8, tau=1.0, m_min=1)
    assert (k1.sum(-1) == 1).all()
    means = [float(L.greedy_ols(G, g, 8, tau=t)[0].sum(-1).double().mean()) for t in (0.0, 0.01, 0.05, 0.2, 0.5, 1.0)]
    assert all(a >= b for a, b in zip(means, means[1:]))
    # rank-deficient: two identical experts, tau = 0 stops once the residual is zero
    E2 = E.clone()
    E2[:, 1] = E2[:, 0]
    G2 = _gram(E2)
    k2, _, p = L.greedy_ols(G2, g, m_max=8, tau=0.0)
    assert (k2.sum(-1) <= 7).all()


def test_effective_experts_and_pr(gen):
    E, g = random_outputs(50, 8, 16, gen)
    G = _gram(E)
    m90, m95, m99 = L.effective_experts(G, g)
    assert ((m90 <= m95) & (m95 <= m99) & (m99 <= 8) & (m90 >= 1)).all()
    pr = L.participation_ratio(G, g)
    assert ((pr >= 1 - 1e-9) & (pr <= 8 + 1e-9)).all()
    # one direction only -> PR = 1 and m95 = 1
    E1 = E[:, :1].expand(-1, 8, -1) * torch.rand(50, 8, 1, generator=gen, dtype=torch.float64)
    G1 = _gram(E1)
    assert torch.allclose(L.participation_ratio(G1, g), torch.ones(50, dtype=torch.float64), atol=1e-6)
    assert (L.effective_experts(G1, g)[1] == 1).all()
