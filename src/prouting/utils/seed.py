"""Seeding."""
from __future__ import annotations

import os
import random

import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
