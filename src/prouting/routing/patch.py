"""Patch MoE blocks of a loaded HF model in place; `handle.restore()` puts the originals back."""
from __future__ import annotations

from typing import Callable, Iterable, Optional, Union

import torch
from torch import Tensor, nn

from .block import ReducedMoEBlock
from .policies import LayerCalib, Native, Policy


def _set_submodule(model: nn.Module, name: str, new: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child, new)


class PatchHandle:
    def __init__(self, model, adapter, blocks: dict[int, tuple[str, nn.Module, ReducedMoEBlock]], shared: dict):
        self.model = model
        self.adapter = adapter
        self.blocks = blocks          # layer idx -> (name, original, wrapper)
        self.shared = shared

    @property
    def layers(self) -> list[int]:
        return sorted(self.blocks)

    def wrapper(self, layer: int) -> ReducedMoEBlock:
        return self.blocks[layer][2]

    def set_policy(self, policy_factory: Callable[[int], Policy]) -> None:
        for li, (_, _, w) in self.blocks.items():
            w.policy = policy_factory(li)

    def set_calib(self, calib: dict[int, LayerCalib]) -> None:
        for li, (_, _, w) in self.blocks.items():
            w.calib = calib.get(li)

    def set_valid(self, attention_mask: Optional[Tensor]) -> None:
        """Mark valid (non-padding) tokens for counters and observers; None to clear."""
        self.shared["valid"] = None if attention_mask is None else attention_mask.bool()

    def reset_counters(self) -> None:
        for _, _, w in self.blocks.values():
            w.reset_counters()

    def realised_budget(self) -> dict:
        """Mean active routed experts per token: per layer and averaged over layers."""
        per = {li: (w.computed_pairs / w.counted_tokens if w.counted_tokens else float("nan"))
               for li, (_, _, w) in self.blocks.items()}
        vals = [v for v in per.values() if v == v]
        return dict(per_layer=per, mean=sum(vals) / len(vals) if vals else float("nan"))

    def restore(self) -> None:
        for li, (name, orig, _) in self.blocks.items():
            _set_submodule(self.model, name, orig)
        self.blocks = {}


def patch_model(model: nn.Module, adapter, policy_factory: Callable[[int], Policy] = lambda li: Native(),
                layers: Union[str, Iterable[int]] = "all", calib: Optional[dict[int, LayerCalib]] = None) -> PatchHandle:
    """Replace MoE blocks with ReducedMoEBlock wrappers. `layers` is "all" or a list of layer indices."""
    found = adapter.find_moe_blocks(model)
    if not found:
        raise RuntimeError("no MoE blocks found; check the adapter against the installed transformers")
    want = None if layers == "all" else set(layers)
    shared: dict = {}
    blocks = {}
    for li, name, blk in found:
        if want is not None and li not in want:
            continue
        if isinstance(blk, ReducedMoEBlock):
            raise RuntimeError(f"layer {li} already patched")
        wrap = ReducedMoEBlock(blk, adapter, policy_factory(li), li, (calib or {}).get(li), shared)
        _set_submodule(model, name, wrap)
        blocks[li] = (name, blk, wrap)
    return PatchHandle(model, adapter, blocks, shared)
