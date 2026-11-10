import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest
import torch


@pytest.fixture
def gen():
    return torch.Generator().manual_seed(0)


def random_outputs(n, k, d, gen, coherent=0.5):
    """Expert outputs with shared components so they are correlated."""
    base = torch.randn(n, 1, d, generator=gen, dtype=torch.float64)
    E = coherent * base + torch.randn(n, k, d, generator=gen, dtype=torch.float64)
    E = E * (0.5 + torch.rand(n, k, 1, generator=gen, dtype=torch.float64))
    g = torch.rand(n, k, generator=gen, dtype=torch.float64)
    g = g / g.sum(-1, keepdim=True)
    return E, g
