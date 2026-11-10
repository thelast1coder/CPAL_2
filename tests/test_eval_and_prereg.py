"""T13 evaluation helpers, T14 prereg consistency, T15 vLLM routing function (pure torch)."""
import re
from pathlib import Path

import torch
import yaml

from conftest import ROOT, random_outputs
from prouting.calibrate.stats import GramAccumulator
from prouting.eval.gsm8k import extract_answer, fixed_subset_indices, gold_answer, is_correct
from prouting.eval.math500 import last_boxed
from prouting.eval.stats import paired_bootstrap, recovery, wilson_interval
from prouting.routing.policies import Projection, RouteCtx, LayerCalib
from prouting.vllm_plugin.routing_fn import make_routing_fn, native_topk_softmax

GSM_CASES = [
    ("The answer is 42.\n#### 42", "42"),
    ("#### 1,234", "1234"),
    ("So she pays $18 in total.\n#### $18", "18"),
    ("#### 7.", "7"),
    ("Step 1: 3+4=7. Step 2: 7*2=14. #### 14 apples", "14"),
    ("no marker, the result is 3 and then 25", "25"),
    ("#### -5", "-5"),
    ("first #### 3 then corrected #### 4", "4"),
    ("#### 0.5", "0.5"),
    ("Total = 1,000,000 dollars.", "1000000"),
    ("#### 12.00", "12"),
    ("The final answer is: 9.", "9"),
    ("####   56  ", "56"),
    ("We get 2/3 of 9 = 6\n#### 6", "6"),
    ("nothing numeric here", None),
]


def test_T13_gsm8k_extraction():
    for text, want in GSM_CASES:
        assert extract_answer(text) == want, text
    assert gold_answer("Natalia sold ... 48+24 = <<48+24=72>>72 clips.\n#### 72") == "72"
    assert is_correct("72", "72") and is_correct("72.0", "72") and not is_correct(None, "72")
    a = fixed_subset_indices(1319, 250, seed=0)
    assert a == fixed_subset_indices(1319, 250, seed=0) and len(set(a)) == 250


def test_T13_bootstrap_and_recovery():
    # identical arms: zero difference, interval at zero
    a = [1, 0, 1, 1, 0] * 40
    r = paired_bootstrap(a, a, resamples=2000)
    assert r["diff"] == 0 and r["lo"] == 0 and r["hi"] == 0
    # method always right where base wrong: diff = share of such items, interval excludes zero
    base = [0] * 50 + [1] * 50
    meth = [1] * 100
    r = paired_bootstrap(meth, base, resamples=2000)
    assert abs(r["diff"] - 0.5) < 1e-12 and r["lo"] > 0.35
    nat = [1] * 100
    rec = recovery(meth, base, nat, resamples=2000)
    assert abs(rec["recovery"] - 1.0) < 1e-12
    half = [1] * 75 + [0] * 25
    base2 = [1] * 50 + [0] * 50
    assert abs(recovery(half, base2, nat, resamples=200)["recovery"] - 0.5) < 1e-12
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi
    assert last_boxed("so \\boxed{\\frac{1}{2}} done") == "\\frac{1}{2}"


def test_T14_prereg_consistency():
    text = (ROOT / "PREREG_KILL_TEST.md").read_text(encoding="utf-8")
    block = re.search(r"```yaml\n(.*?)```", text.replace("\r\n", "\n"), re.S).group(1)
    cfg = yaml.safe_load((ROOT / "configs" / "prereg_kill_test.yaml").read_text(encoding="utf-8"))
    assert yaml.safe_load(block) == cfg
    assert cfg["k2"]["go_ratio_at_least"] == 2.0 and cfg["k1"]["go"]["median_m95_at_most"] == 3


def test_k2fix_placement_in_projection_family(gen):
    """docs/related_work_notes.md: the k2 rule is c = lambda(x) g_S; k2 = k1 is renormalise, k2 = k is keep
    (renormalised router); least squares coincides with keep only when G_SD g_D = 0."""
    from prouting.routing import linalg as L
    from prouting.routing.baselines import K2Fix
    N, k, n = 16, 8, 200
    logits = torch.randn(n, N, generator=gen, dtype=torch.float64)
    p = torch.softmax(logits, -1)
    pT, ids = p.topk(k, -1)
    g = pT / pT.sum(-1, keepdim=True)                       # renormalised native gates (Qwen3 convention)
    ctx = RouteCtx(x=torch.zeros(n, 4), logits=logits, ids=ids, g=g, layer=0)
    for m in (1, 2, 4):
        keep = L.gate_order_mask(g, m)
        _, w1 = K2Fix(m, k2=m)(ctx)
        assert torch.allclose(w1, L.renorm_weights(g, keep), atol=1e-12)
        _, wk = K2Fix(m, k2=k)(ctx)
        assert torch.allclose(wk, L.keep_weights(g, keep), atol=1e-12)
        _, w16 = K2Fix(m, k2=16)(ctx)
        lam = w16[keep].reshape(n, m) / g[keep].reshape(n, m)
        assert torch.allclose(lam, lam[:, :1].expand_as(lam), atol=1e-12) and (lam < 1).all()
    E, gg = random_outputs(n, k, 16, gen)
    G = torch.einsum("nid,njd->nij", E, E)
    keep = L.gate_order_mask(gg, 2)
    c_ls = L.ls_weights(G, gg, keep)
    D = ~keep
    rhs = torch.einsum("nij,nj->ni", G, gg * D)             # G_SD g_D on S rows
    assert (c_ls - L.keep_weights(gg, keep)).abs().max() > 1e-3
    Gs = G.clone()
    Gs[keep.unsqueeze(-1) & D.unsqueeze(-2)] = 0           # make kept and dropped outputs orthogonal
    Gs[D.unsqueeze(-1) & keep.unsqueeze(-2)] = 0
    assert torch.allclose(L.ls_weights(Gs, gg, keep), L.keep_weights(gg, keep), atol=1e-10)


def test_T15_vllm_routing_fn_matches_hf_policy(gen):
    N, k, d, n = 16, 8, 12, 300
    V = torch.randn(N, d, generator=gen, dtype=torch.float64) + 0.5
    acc = GramAccumulator(N)
    ids0 = torch.stack([torch.randperm(N, generator=gen)[:k] for _ in range(2000)])
    E = V[ids0] * (1 + 0.1 * torch.randn(2000, k, 1, generator=gen, dtype=torch.float64))
    acc.update(ids0, torch.einsum("nid,njd->nij", E, E))
    st = acc.finalise(n_min=1)
    logits = torch.randn(n, N, generator=gen)
    for renorm in (False, True):
        for m, tau in ((1, None), (2, None), (None, 0.05)):
            fn = make_routing_fn(st.G_avg, select="ols", weights="ls", m=m, tau=tau)
            w_v, ids_v = fn(torch.zeros(n, d), logits, k, renorm)
            g, ids = native_topk_softmax(logits, k, renorm)
            pol = Projection(select="ols", weights="ls", gram="avg", m=m, tau=tau)
            ctx = RouteCtx(x=torch.zeros(n, d), logits=logits, ids=ids, g=g, layer=0, calib=LayerCalib(gram=st))
            sel, w = pol(ctx)
            assert torch.equal(ids_v.long(), sel)
            assert torch.allclose(w_v, w.float(), atol=1e-6)
    # fixed-m slot mode returns exactly m slots holding the kept experts
    fn = make_routing_fn(st.G_avg, m=2, slots=2)
    w2, i2 = fn(torch.zeros(n, d), logits, k, True)
    assert w2.shape == (n, 2) and (w2 != 0).sum(-1).max() <= 2
