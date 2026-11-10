"""Two-pass calibration driver (PHASE0_SETUP.md 2.4), shared by scripts/calibrate.py and the tests.

Pass 1 (first `pass1_tokens` valid tokens): layer inputs for k-means (reservoir), neuron importance for
sketches. Pass 2 (all tokens): in-set and cluster-conditioned Grams, sketch kappa. Optional full-Gram mode
on the first `full_gram_tokens` tokens (every expert on every token).
"""
from __future__ import annotations

from typing import Iterable, Optional

import torch

from ..routing.patch import patch_model
from ..routing.policies import LayerCalib, Native
from .clusters import InputReservoir, SphericalKMeans
from .sketch import KappaFit, NeuronImportance, SketchState, sketch_forward
from .stats import GramAccumulator, GramState


def _batches_until(batches, limit):
    seen = 0
    for ids, mask in batches:
        if limit is not None and seen >= limit:
            break
        yield ids, mask
        seen += int(mask.sum())


@torch.no_grad()
def calibrate(model, adapter, batches: list[tuple[torch.Tensor, torch.Tensor]], *, clusters=(8, 16),
              pass1_tokens: int = 100_000, sketch_fraction: Optional[float] = 0.05, n_min: int = 32,
              full_gram_tokens: int = 0, reservoir_cap: int = 32768, kmeans_iters: int = 100, seed: int = 0,
              log=print) -> dict[int, LayerCalib]:
    """batches: list of (input_ids [B,T], attention_mask [B,T]) on any device. Returns layer -> LayerCalib."""
    dev = next(model.parameters()).device
    handle = patch_model(model, adapter, lambda li: Native())
    layers = handle.layers
    blocks = {li: handle.wrapper(li).orig for li in layers}
    N = {li: adapter.num_experts(blocks[li]) for li in layers}
    I = {li: adapter.intermediate_size(blocks[li]) for li in layers}
    try:
        # ---------------------------------------------------------------- pass 1
        res = {li: InputReservoir(reservoir_cap, seed + li) for li in layers}
        imp = {li: NeuronImportance(N[li], I[li]) for li in layers} if sketch_fraction else {}

        def obs1(li):
            def f(ctx, slot_out, sel, w):
                v = ctx.valid if ctx.valid is not None else torch.ones(ctx.x.shape[0], dtype=torch.bool, device=ctx.x.device)
                res[li].add(ctx.x[v])
                if sketch_fraction:
                    for e in torch.unique(ctx.ids[v]).tolist():
                        rows = ((ctx.ids == e).any(-1) & v).nonzero(as_tuple=True)[0]
                        imp[li].update(e, adapter.expert_hidden(blocks[li], e, ctx.x[rows]).float())
            return f
        for li in layers:
            handle.wrapper(li).add_observer(obs1(li), needs_slots=False)
        p1 = 0
        for ids, mask in _batches_until(batches, pass1_tokens):
            handle.set_valid(mask.to(dev))
            model(input_ids=ids.to(dev), attention_mask=mask.to(dev))
            p1 += int(mask.sum())
        for li in layers:
            handle.wrapper(li).clear_observers()
        log(f"[calib] pass 1: {p1} tokens")
        kmeans = {}
        for li in layers:
            if res[li].buf is None:
                kmeans[li] = {}
                continue
            x = res[li].data()
            kmeans[li] = {f"c{C}": SphericalKMeans(C, seed=seed, n_iter=kmeans_iters).fit(x) for C in clusters
                          if x.shape[0] >= C}
        sk_state = {}
        if sketch_fraction:
            for li in layers:
                cn = {e: adapter.expert_weights(blocks[li], e)[2].float().norm(dim=0) for e in range(N[li])}
                sk_state[li] = SketchState(sketch_fraction, imp[li].choose(cn, sketch_fraction), {})
        # ---------------------------------------------------------------- pass 2
        acc = {li: GramAccumulator(N[li], {f"c{C}": C for C in clusters}) for li in layers}
        kap = {li: KappaFit(N[li]) for li in layers}
        full_left = {"n": full_gram_tokens}

        def obs2(li):
            def f(ctx, slot_out, sel, w):
                v = ctx.valid if ctx.valid is not None else torch.ones(ctx.x.shape[0], dtype=torch.bool, device=ctx.x.device)
                E = slot_out[v].double() if slot_out.device.type == "cpu" else slot_out[v]
                G = torch.einsum("nid,njd->nij", E, E)
                cl = {name: km.assign(ctx.x[v]) for name, km in kmeans[li].items()}
                acc[li].update(ctx.ids[v], G, cl)
                if sketch_fraction:
                    st = sk_state[li]
                    xv, idv = ctx.x[v], ctx.ids[v]
                    for e in torch.unique(idv).tolist():
                        rows, slots = (idv == e).nonzero(as_tuple=True)
                        Eh = sketch_forward(adapter, blocks[li], e, xv[rows], st.neuron_idx[e]).float()
                        kap[li].update(e, slot_out[v][rows, slots], Eh)
            return f
        for li in layers:
            handle.wrapper(li).add_observer(obs2(li), needs_slots=True)
        p2 = 0
        for ids, mask in batches:
            handle.set_valid(mask.to(dev))
            model(input_ids=ids.to(dev), attention_mask=mask.to(dev))
            p2 += int(mask.sum())
        log(f"[calib] pass 2: {p2} tokens")
        for li in layers:
            handle.wrapper(li).clear_observers()
        # ---------------------------------------------------------------- full Gram (optional)
        if full_gram_tokens > 0:
            def obs3(li):
                def f(ctx, slot_out, sel, w):
                    v = ctx.valid if ctx.valid is not None else torch.ones(ctx.x.shape[0], dtype=torch.bool, device=ctx.x.device)
                    xv = ctx.x[v][: max(0, full_left[li])]
                    if xv.shape[0] == 0:
                        return
                    Eall = torch.stack([adapter.expert_forward(blocks[li], e, xv).float() for e in range(N[li])], 1)
                    acc[li].update_full(Eall)
                    full_left[li] -= xv.shape[0]
                return f
            for li in layers:
                full_left[li] = full_gram_tokens
                handle.wrapper(li).add_observer(obs3(li), needs_slots=False)
            for ids, mask in _batches_until(batches, full_gram_tokens):
                handle.set_valid(mask.to(dev))
                model(input_ids=ids.to(dev), attention_mask=mask.to(dev))
            for li in layers:
                handle.wrapper(li).clear_observers()
        out = {}
        for li in layers:
            st = acc[li].finalise(n_min=n_min)
            if sketch_fraction:
                sk_state[li].kappa = kap[li].finalise()
            out[li] = LayerCalib(gram=st, kmeans=kmeans[li], sketch=sk_state.get(li))
            log(f"[calib] layer {li}: tokens {st.tokens}, pairs below n_min {st.backoff_share:.3f}, r_bar {st.r_bar:.3f}")
        return out
    finally:
        handle.restore()


def calib_to_dict(calib: dict[int, LayerCalib]) -> dict:
    return {li: dict(gram=c.gram.state_dict(), kmeans={k: v.state_dict() for k, v in c.kmeans.items()},
                     sketch=c.sketch.state_dict() if c.sketch is not None else None) for li, c in calib.items()}


def calib_from_dict(d: dict) -> dict[int, LayerCalib]:
    out = {}
    for li, c in d.items():
        sk = SketchState.from_state_dict(c["sketch"]) if c.get("sketch") else None
        out[int(li)] = LayerCalib(gram=GramState.from_state_dict(c["gram"]),
                                  kmeans={k: SphericalKMeans.from_state_dict(v) for k, v in c["kmeans"].items()},
                                  sketch=sk)
    return out
