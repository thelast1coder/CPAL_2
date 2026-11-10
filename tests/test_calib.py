"""T9 GramAccumulator, T11 sketch, T12 clustering."""
import torch

from conftest import random_outputs
from prouting.adapters import get_adapter
from prouting.calibrate.clusters import InputReservoir, SphericalKMeans
from prouting.calibrate.sketch import KappaFit, NeuronImportance, sketch_forward
from prouting.calibrate.stats import GramAccumulator, GramState, gather_gram
from prouting.utils.tiny import tiny_olmoe


def test_T9_gram_accumulator(tmp_path, gen):
    N, k, d, n = 6, 3, 10, 4000
    # fixed expert directions -> a known Gram (outputs independent of the token)
    V = torch.randn(N, d, generator=gen, dtype=torch.float64)
    true_G = V @ V.T
    ids = torch.stack([torch.randperm(N, generator=gen)[:k] for _ in range(n)])
    E = V[ids]
    G = torch.einsum("nid,njd->nij", E, E)
    acc = GramAccumulator(N, clusters={"c2": 2})
    cl = {"c2": torch.randint(0, 2, (n,), generator=gen)}
    acc.update(ids[:2000], G[:2000], {"c2": cl["c2"][:2000]})
    acc.update(ids[2000:], G[2000:], {"c2": cl["c2"][2000:]})
    st = acc.finalise(n_min=1)
    assert torch.allclose(st.G_avg, true_G, atol=1e-10)
    # counts: diagonal = times in T, off-diagonal = co-occurrences
    for a in range(N):
        assert int(st.counts[a, a]) == int((ids == a).any(-1).sum())
        for b in range(N):
            if a != b:
                assert int(st.counts[a, b]) == int(((ids == a).any(-1) & (ids == b).any(-1)).sum())
    assert torch.allclose(st.clusters["c2"], true_G.expand(2, N, N), atol=1e-10)
    # back-off below n_min
    st2 = acc.finalise(n_min=10**9)
    off = ~torch.eye(N, dtype=torch.bool)
    expect = st2.r_bar * st2.norms.unsqueeze(0) * st2.norms.unsqueeze(1)
    assert torch.allclose(st2.G_avg[off], expect[off])
    assert st2.backoff_share == 1.0
    # gather and round trip
    assert torch.allclose(gather_gram(st.G_avg, ids[:5]), G[:5], atol=1e-10)
    st.save(tmp_path / "g.pt")
    st3 = GramState.load(tmp_path / "g.pt")
    assert torch.equal(st3.G_avg, st.G_avg) and st3.r_bar == st.r_bar


def test_T11_sketch_full_fraction():
    model = tiny_olmoe()
    ad = get_adapter(model)
    _, _, blk = ad.find_moe_blocks(model)[0]
    I = ad.intermediate_size(blk)
    x = torch.randn(40, model.config.hidden_size, generator=torch.Generator().manual_seed(0))
    imp = NeuronImportance(ad.num_experts(blk), I)
    for e in range(ad.num_experts(blk)):
        imp.update(e, ad.expert_hidden(blk, e, x))
    cn = {e: ad.expert_weights(blk, e)[2].norm(dim=0) for e in range(ad.num_experts(blk))}
    idx = imp.choose(cn, 1.0)
    kf = KappaFit(ad.num_experts(blk))
    with torch.no_grad():
        for e in range(ad.num_experts(blk)):
            assert idx[e].numel() == I
            Eh = sketch_forward(ad, blk, e, x, idx[e])
            E = ad.expert_forward(blk, e, x)
            assert torch.allclose(Eh, E, atol=1e-5)
            kf.update(e, E, Eh)
    for v in kf.finalise().values():
        assert abs(v - 1.0) < 1e-5
    # fraction 0.05 picks ceil(0.05 * I) neurons
    assert imp.choose(cn, 0.05)[0].numel() == -(-5 * I // 100)


def test_T12_clustering(gen):
    x = torch.randn(3000, 16, generator=gen)
    a = SphericalKMeans(8, seed=3, n_iter=20).fit(x)
    b = SphericalKMeans(8, seed=3, n_iter=20).fit(x)
    assert torch.equal(a.centroids, b.centroids)
    assert torch.equal(a.assign(a.centroids), torch.arange(8))
    r = InputReservoir(cap=100, seed=0)
    for i in range(10):
        r.add(torch.randn(50, 16, generator=gen))
    assert r.data().shape == (100, 16) and r.seen == 500
