"""Shared script plumbing: arguments, configs, paths, model loading, skip-if-done."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import torch
import yaml

from .logging import REPO_ROOT, get_logger
from .seed import set_seed

KILL_MODELS = ("olmoe_1b_7b", "qwen3_30b_a3b")
TINY_BUILD = {"olmoe_1b_7b": ("olmoe", 4), "olmoe_1b_7b_instruct": ("olmoe", 4), "qwen3_30b_a3b": ("qwen3_moe", 4)}


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "experiments" / "kill_test.yaml"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tiny", action="store_true", help="tiny random model and synthetic data (no downloads)")
    p.add_argument("--force", action="store_true", help="recompute finished outputs")
    p.add_argument("--model", default="olmoe_1b_7b", help="kill-test model key")
    p.add_argument("--target", default="stats", choices=["stats", "task"],
                   help="stats model (K1, K2) or task model (K3; OLMoE-Instruct for olmoe_1b_7b)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--mem-gb", type=float, default=None, help="GPU memory cap for this process")
    return p


def setup(args, name: str):
    set_seed(args.seed)
    if args.tiny:
        os.environ["HF_HUB_OFFLINE"] = "1"
    if args.device.startswith("cuda") and args.mem_gb:
        from .gpu import cap_memory
        cap_memory(gb=args.mem_gb)
    log = get_logger(name, REPO_ROOT / "logs" / (("tiny_" if args.tiny else "") + f"{name}_{args.model}.log"))
    cfg = load_yaml(args.config)
    if args.tiny:
        cfg = tiny_overrides(cfg)
    return cfg, log


def load_yaml(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def model_cfg(name: str) -> dict:
    return load_yaml(REPO_ROOT / "configs" / "models" / f"{name}.yaml")


def tiny_overrides(cfg: dict) -> dict:
    cfg = dict(cfg)
    cfg["data"] = {**cfg["data"], "calibration_tokens": 6000, "validation_tokens": 1500, "seq_len": 64}
    cfg["models"] = {k: {**v, "analysis_tokens": 2000} for k, v in cfg["models"].items()}
    cfg["calibration"] = {**cfg["calibration"], "pass1_tokens": 2000, "clusters": [8, 16], "n_min": 4,
                          "batch_size": 16}
    cfg["downstream"] = {**cfg["downstream"], "batch_size": 32, "ppl_tokens": 2000}
    return cfg


def concrete_model(args, cfg: dict, target: Optional[str] = None) -> str:
    """Concrete model name for the kill-test key and role (stats or task)."""
    m = cfg["models"][args.model]
    return m["task_model"] if (target or args.target) == "task" else m["stats_model"]


def paths(args, cfg: dict, target: Optional[str] = None) -> dict[str, Path]:
    """Artifacts per concrete model; results per kill-test key; data per tokenizer family (same documents)."""
    mk = args.model
    cm = concrete_model(args, cfg, target)
    if args.tiny:
        art, res = REPO_ROOT / "artifacts" / "tiny" / cm, REPO_ROOT / "results" / "tiny" / mk
        data = REPO_ROOT / "artifacts" / "tiny" / "data" / mk
    else:
        art, res = REPO_ROOT / "artifacts" / cm, REPO_ROOT / "results" / "kill_test" / mk
        data = REPO_ROOT / "artifacts" / "data" / model_cfg(cm)["tokenizer_family"]
    for p in (art, res, data):
        p.mkdir(parents=True, exist_ok=True)
    return dict(art=art, res=res, data=data, calib=art / f"calib_seed{args.seed}.pt",
                analysis=art / "analysis", validation=art / "validation")


def load_model(name: str, args, dtype: Optional[str] = None):
    """Returns (model, tokenizer or None). Tiny: seeded random model from config classes."""
    if args.tiny:
        from .tiny import BUILDERS
        fam, k = TINY_BUILD[name]
        model = BUILDERS[fam](k=k, seed=args.seed) if fam != "olmoe" else BUILDERS[fam](k=k, seed=args.seed, num_layers=4)
        return model.to(args.device), None
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mc = model_cfg(name)
    dt = getattr(torch, dtype or mc["dtype"])
    hf_id = mc["hf_id_fp8"] if (mc.get("use_fp8") and mc.get("hf_id_fp8")) else mc["hf_id"]
    tok = AutoTokenizer.from_pretrained(hf_id, revision=mc.get("revision"))
    model = AutoModelForCausalLM.from_pretrained(hf_id, revision=mc.get("revision"), dtype=dt,
                                                 device_map=args.device)
    # reference experts kernel (docs/decisions.md); attribute name checked against transformers 5.22.2
    if hasattr(model, "set_experts_implementation"):
        model.set_experts_implementation("eager")
    else:
        model.config._experts_implementation = "eager"
    model.eval()
    return model, tok


def done(path: Path, args) -> bool:
    return Path(path).exists() and not args.force
