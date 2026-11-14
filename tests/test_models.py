"""T1, T2, T8, T10: adapters, patched blocks and offline-equals-online on tiny random models."""
import pytest
import torch

from prouting.adapters import get_adapter
from prouting.analysis.layer_eval import KILL_METHODS, eval_method
from prouting.analysis.recorder import AnalysisRecorder
from prouting.calibrate.driver import calibrate
from prouting.routing import linalg as L
from prouting.routing.patch import patch_model
from prouting.routing.policies import Native, Projection, RouteCtx, TopMKeep, TopMRenorm
from prouting.utils.tiny import BUILDERS, random_tokens

CASES = [("olmoe", 4), ("olmoe", 2), ("qwen3_moe", 4), ("qwen2_moe", 4), ("deepseek_v2", 4)]


@pytest.mark.parametrize("family,k", CASES)
def test_T1_native_patch_equals_original(family, k):
    model = BUILDERS[family](k=k)
    ad = get_adapter(model)
    x = random_tokens(3, 20)
    with torch.no_grad():
        ref = model(x)
        h = patch_model(model, ad)
        out = model(x)
        h.restore()
        back = model(x)
    assert type(out) is type(ref)
    assert (out.logits - ref.logits).abs().max() <= 1e-6
    assert torch.equal(back.logits, ref.logits)


@pytest.mark.parametrize("family,k", [("olmoe", 4), ("qwen3_moe", 4)])
def test_T1_bf16_native_patch_is_bit_identical(family, k):
    """K0.2 on real models: in bf16 the wrapper must combine experts exactly as the native eager loop."""
    model = BUILDERS[family](k=k).to(torch.bfloat16)
    ad = get_adapter(model)
    x = random_tokens(3, 20)
    with torch.no_grad():
        ref = model(x)
        h = patch_model(model, ad)
        out = model(x)
        h.restore()
    assert torch.equal(out.logits, ref.logits)


def test_T1_deepseek_group_limited():
    model = BUILDERS["deepseek_v2"](topk_method="group_limited_greedy")
    ad = get_adapter(model)
    x = random_tokens(2, 16)
    with torch.no_grad():
        ref = model(x).logits
        h = patch_model(model, ad)
        out = model(x).logits
        h.restore()
    assert (out - ref).abs().max() <= 1e-6


@pytest.mark.parametrize("family,k", CASES)
def test_T2_native_topk_matches_router(family, k):
    model = BUILDERS[family](k=k)
    ad = get_adapter(model)
    gen = torch.Generator().manual_seed(1)
    for li, _, blk in ad.find_moe_blocks(model):
        x = torch.randn(50, model.config.hidden_size, generator=gen)
        _, scores, idx = blk.gate(x)
        ids, g = ad.native_topk(blk, ad.router_logits(blk, x))
        # compare as sets with gates (DeepSeek top-k is unsorted)
        o1, o2 = ids.sort(-1), idx.sort(-1)
        assert torch.equal(o1.values, o2.values)
        assert torch.equal(g.gather(1, o1.indices), scores.gather(1, o2.indices))
        assert g.dtype == scores.dtype


def _block_and_x(family="olmoe", k=4, n=64, seed=0):
    model = BUILDERS[family](k=k)
    ad = get_adapter(model)
    li, _, blk = ad.find_moe_blocks(model)[0]
    x = torch.randn(1, n, model.config.hidden_size, generator=torch.Generator().manual_seed(seed))
    return model, ad, li, blk, x


def test_T8_policies_through_block():
    model, ad, li, blk, x = _block_and_x(k=4)
    toks = random_tokens(4, 32)
    calib = calibrate(model, ad, [(toks, torch.ones_like(toks))], clusters=(2,), pass1_tokens=64,
                      sketch_fraction=None, n_min=1, kmeans_iters=5, log=lambda *a: None)
    with torch.no_grad():
        ref = blk(x)
        h = patch_model(model, ad, calib=calib)
        w = h.wrapper(li)
        w.policy = TopMKeep(4)
        assert torch.allclose(w(x), ref, atol=1e-6)
        w.policy = Projection(select="ols", weights="ls", gram="avg", m=4, ridge=0.0)
        assert torch.allclose(w(x), ref, atol=1e-5)
        # with drops: block output equals the hand-computed sum_S c_i E_i
        for pol in (TopMRenorm(2), Projection(select="ols", weights="ls", gram="avg", m=2),
                    Projection(select="ols", weights="ls", gram="cos", m=1)):
            w.policy = pol
            out = w(x)[0]
            xf = x[0]
            ids, g = ad.native_topk(blk, ad.router_logits(blk, xf))
            ctx = RouteCtx(x=xf, logits=ad.router_logits(blk, xf), ids=ids, g=g, layer=li, calib=calib[li],
                           adapter=ad, block=blk)
            sel, c = pol(ctx)
            Eall = torch.stack([ad.expert_forward(blk, e, xf) for e in range(ad.num_experts(blk))], 1)
            Esel = Eall.gather(1, sel.unsqueeze(-1).expand(-1, -1, Eall.shape[-1]))
            if pol.needs_outputs:
                c = pol.reweight(ctx, Esel * (c != 0).unsqueeze(-1))
            hand = (c.unsqueeze(-1) * Esel).sum(1)
            assert torch.allclose(out, hand, atol=1e-5)
        h.restore()


def test_T10_offline_equals_online(tmp_path):
    model, ad, li, blk, _ = _block_and_x(k=4)
    toks = random_tokens(8, 32, seed=3)
    batches = [(toks[i:i + 4], torch.ones_like(toks[i:i + 4])) for i in (0, 4)]
    calib = calibrate(model, ad, batches, clusters=(2,), pass1_tokens=64, sketch_fraction=None, n_min=1,
                      kmeans_iters=5, log=lambda *a: None)
    ana = random_tokens(4, 32, seed=9)
    with torch.no_grad():
        h = patch_model(model, ad, calib=calib)
        rec = AnalysisRecorder(h, model, record_sketch=False, clusters=("c2",))
        h.set_valid(torch.ones_like(ana, dtype=torch.bool))
        model(ana)
        rec.flush(tmp_path)
        rec.remove()
        # capture the block inputs for the online comparison
        xs = {}
        hook = h.wrapper(li).register_forward_pre_hook(lambda mod, args: xs.__setitem__("x", args[0]))
        model(ana)
        hook.remove()
        x = xs["x"].reshape(-1, xs["x"].shape[-1])
        w = h.wrapper(li)
        w.policy = Native()
        y_nat = w(x.unsqueeze(0))[0].double()
        from prouting.analysis.recorder import load_shard
        shard = load_shard(tmp_path, li)
        for tag, pol, spec in (("B2", TopMRenorm(2), KILL_METHODS["B2"]),
                               ("ours", Projection(select="ols", weights="ls", gram="avg", m=2), KILL_METHODS["ours-avg"])):
            w.policy = pol
            y = w(x.unsqueeze(0))[0].double()
            online = float(((y - y_nat) ** 2).sum() / (y_nat**2).sum())
            offline = eval_method(shard, calib[li], spec, m=2)["rel_error_energy"]
            assert abs(online - offline) <= 1e-5 * max(online, 1e-12) + 1e-9, (tag, online, offline)
        h.restore()


def test_counters_and_realised_budget():
    model = BUILDERS["olmoe"](k=4)
    ad = get_adapter(model)
    h = patch_model(model, ad, lambda li: TopMRenorm(2))
    x = random_tokens(2, 10)
    with torch.no_grad():
        model(x)
    assert abs(h.realised_budget()["mean"] - 2.0) < 1e-9
    h.restore()
