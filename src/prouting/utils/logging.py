"""Run records stored in every results file: config, seed, git hash, versions, GPU, wall time."""
from __future__ import annotations

import datetime as _dt
import logging
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def git_info() -> dict:
    h = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    return dict(git_hash=h or "unknown", git_dirty=dirty)


def package_versions() -> dict:
    out = dict(python=sys.version.split()[0], torch=torch.__version__, platform=platform.platform())
    for name in ("transformers", "numpy", "accelerate", "datasets", "lm_eval", "vllm", "safetensors",
                 "huggingface_hub"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "?")
        except Exception:
            pass
    return out


def gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


class RunRecord:
    """Use as: rec = RunRecord(config, seed); ...; d = rec.finish() -> dict for the results file."""

    def __init__(self, config: Any, seed: int):
        self.config = config
        self.seed = seed
        self.t0 = time.time()
        self.start = _dt.datetime.now().isoformat(timespec="seconds")

    def finish(self) -> dict:
        return dict(config=self.config, seed=self.seed, **git_info(), versions=package_versions(),
                    gpu=gpu_name(), start=self.start,
                    end=_dt.datetime.now().isoformat(timespec="seconds"),
                    wall_time_s=round(time.time() - self.t0, 3))


def run_record(config: Any, seed: int) -> RunRecord:
    return RunRecord(config, seed)


def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    h = logging.StreamHandler()
    h.setFormatter(fmt)
    log.addHandler(h)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    return log
