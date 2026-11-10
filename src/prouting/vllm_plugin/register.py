"""Attach routing functions to every fused MoE layer of a loaded vLLM model (VM only).

The attribute names below are what we believe current vLLM uses (verify on the VM before P2):
fused MoE layers are `vllm.model_executor.layers.fused_moe.FusedMoE` modules with a
`custom_routing_function` attribute; the model may live in a separate engine process, so run with
`enforce_eager=True` and in-process execution first (for example by setting VLLM_ENABLE_V1_MULTIPROCESSING=0,
verify). This file is never imported on the laptop.
"""
from __future__ import annotations

from typing import Callable


def attach(llm, routing_fn_for_layer: Callable[[int], Callable]) -> int:
    """Set `custom_routing_function` on every FusedMoE layer in order of appearance. Returns the count."""
    from vllm.model_executor.layers.fused_moe import FusedMoE  # noqa: verify import path
    model = _get_model(llm)
    n = 0
    for _, mod in model.named_modules():
        if isinstance(mod, FusedMoE):
            mod.custom_routing_function = routing_fn_for_layer(n)
            n += 1
    if n == 0:
        raise RuntimeError("no FusedMoE layers found; check the installed vLLM version")
    return n


def _get_model(llm):
    """Reach the torch model inside an in-process LLM engine (path differs across versions, verify)."""
    for path in ("llm_engine.model_executor.driver_worker.model_runner.model",
                 "llm_engine.model_executor.driver_worker.worker.model_runner.model"):
        obj = llm
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("cannot reach the vLLM model object; update register._get_model for this version")
