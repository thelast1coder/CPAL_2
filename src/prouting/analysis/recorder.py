"""Records per-token, per-layer statistics during a native forward (PHASE0_SETUP.md 2.6).

Per valid token and MoE layer: T ids [k], gates g [k], true Gram G(x) [k,k] (float32), top-32 router
probabilities and ids, ||y||^2 (routed output, computed directly), ||h||^2 (residual stream entering the
block), cluster ids for C = 8 and 16, an optional domain id, and (if sketches are calibrated) the cross
Gram <E_i, kappa_j E^_j> and the sketch Gram <kappa_i E^_i, kappa_j E^_j>.
Expert outputs are kept only transiently.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

from ..routing.policies import RouteCtx, _sketch_grams
from ..utils.io import load_torch, save_torch

TOPP = 32


class AnalysisRecorder:
    def __init__(self, handle, model=None, record_sketch: bool = True, clusters=("c8", "c16")):
        self.handle = handle
        self.buf: dict[int, dict[str, list[Tensor]]] = defaultdict(lambda: defaultdict(list))
        self.record_sketch = record_sketch
        self.clusters = clusters
        self._hooks = []
        shared = handle.shared
        ad = handle.adapter
        # ||h||^2: input to the norm that precedes the MoE block in each decoder layer
        if model is not None:
            for li in handle.layers:
                mod = ad.pre_moe_norm(model, li)
                if mod is None:
                    continue

                def pre_hook(module, args, _li=li):
                    h = args[0]
                    shared[f"h2_{_li}"] = h.reshape(-1, h.shape[-1]).float().pow(2).sum(-1)
                self._hooks.append(mod.register_forward_pre_hook(pre_hook))
        for li in handle.layers:
            handle.wrapper(li).add_observer(self._make_obs(li), needs_slots=True)

    def _make_obs(self, li: int):
        def obs(ctx: RouteCtx, slot_out: Optional[Tensor], sel: Tensor, w: Tensor) -> None:
            if slot_out is None:
                return
            n = ctx.ids.shape[0]
            v = ctx.valid if ctx.valid is not None else torch.ones(n, dtype=torch.bool, device=ctx.ids.device)
            E = slot_out  # [n,k,d] float32, native policy: slots == T
            dt = torch.float64 if E.device.type == "cpu" else torch.float32
            Ed = E.to(dt)
            G = torch.einsum("nid,njd->nij", Ed, Ed)
            y = torch.einsum("nk,nkd->nd", ctx.g.to(dt), Ed)
            y2 = y.pow(2).sum(-1)
            probs = torch.softmax(ctx.logits.float(), dim=-1)
            tp = probs.topk(min(TOPP, probs.shape[-1]), dim=-1)
            b = self.buf[li]
            b["ids"].append(ctx.ids[v].to(torch.int32).cpu())
            b["g"].append(ctx.g[v].float().cpu())
            b["G"].append(G[v].float().cpu())
            b["y2"].append(y2[v].float().cpu())
            b["top_p"].append(tp.values[v].cpu())
            b["top_ids"].append(tp.indices[v].to(torch.int32).cpu())
            h2 = self.handle.shared.get(f"h2_{li}")
            if h2 is not None and h2.numel() == n:
                b["h2"].append(h2[v].cpu())
            dom = self.handle.shared.get("domain")
            if dom is not None and dom.numel() == n:
                b["domain"].append(dom.reshape(-1).to(ctx.ids.device)[v].to(torch.int8).cpu())
            calib = ctx.calib
            if calib is not None:
                for name in self.clusters:
                    km = calib.kmeans.get(name)
                    if km is not None:
                        b[name].append(km.assign(ctx.x)[v].to(torch.int16).cpu())
                if self.record_sketch and calib.sketch is not None:
                    Gsk, Ehat = _sketch_grams(ctx, dt)
                    cross = torch.einsum("nid,njd->nij", Ed, Ehat)     # <E_i, kappa_j E^_j>
                    b["G_cross"].append(cross[v].float().cpu())
                    b["G_sk"].append(Gsk[v].float().cpu())
        return obs

    def n_tokens(self, li: int) -> int:
        return sum(t.shape[0] for t in self.buf[li]["ids"]) if li in self.buf else 0

    def flush(self, out_dir: Path, append: bool = True) -> None:
        """Write per-layer shards layer_XX.pt (concatenating with existing ones when append=True)."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for li, b in self.buf.items():
            data = {k: torch.cat(v) for k, v in b.items() if v}
            path = out_dir / f"layer_{li:02d}.pt"
            if append and path.exists():
                old = load_torch(path)
                data = {k: torch.cat([old[k], data[k]]) for k in data if k in old}
            save_torch(path, data)
        self.buf.clear()

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        for li in self.handle.layers:
            self.handle.wrapper(li).clear_observers()


def load_shard(out_dir: Path, li: int) -> dict:
    return load_torch(Path(out_dir) / f"layer_{li:02d}.pt")


def list_layers(out_dir: Path) -> list[int]:
    return sorted(int(p.stem.split("_")[1]) for p in Path(out_dir).glob("layer_*.pt"))
