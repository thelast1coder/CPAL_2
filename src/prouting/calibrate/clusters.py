"""Seeded spherical (cosine) mini-batch k-means on L2-normalised layer inputs (G-clu, PROJECT.md 5.4)."""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def _normalise(x: Tensor) -> Tensor:
    x = x.float()
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class SphericalKMeans:
    """Mini-batch k-means with cosine distance. Deterministic given `seed` and the data order."""

    def __init__(self, n_clusters: int, seed: int = 0, batch_size: int = 4096, n_iter: int = 100):
        self.C = n_clusters
        self.seed = seed
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.centroids: Optional[Tensor] = None   # [C,d], unit rows
        self._counts: Optional[Tensor] = None

    def _init(self, x: Tensor, gen: torch.Generator) -> None:
        """k-means++ on a sample (cosine distance 1 - <a,b>)."""
        n = x.shape[0]
        if n < self.C:
            raise ValueError(f"need at least {self.C} points to initialise, got {n}")
        first = int(torch.randint(n, (1,), generator=gen))
        cents = [x[first]]
        d = (1.0 - x @ x[first]).clamp_min(0)
        for _ in range(1, self.C):
            p = d / d.sum() if float(d.sum()) > 0 else torch.full_like(d, 1.0 / n)
            j = int(torch.multinomial(p, 1, generator=gen))
            cents.append(x[j])
            d = torch.minimum(d, (1.0 - x @ x[j]).clamp_min(0))
        self.centroids = _normalise(torch.stack(cents))
        self._counts = torch.zeros(self.C)

    def fit(self, x: Tensor) -> "SphericalKMeans":
        x = _normalise(x.cpu())
        gen = torch.Generator().manual_seed(self.seed)
        sample = x[torch.randperm(x.shape[0], generator=gen)[: min(x.shape[0], 20000)]]
        self._init(sample, gen)
        for _ in range(self.n_iter):
            idx = torch.randint(x.shape[0], (min(self.batch_size, x.shape[0]),), generator=gen)
            self._step(x[idx])
        return self

    def _step(self, xb: Tensor) -> None:
        a = (xb @ self.centroids.T).argmax(-1)
        for c in range(self.C):
            sel = a == c
            nc = int(sel.sum())
            if nc == 0:
                continue
            self._counts[c] += nc
            lr = nc / float(self._counts[c])
            self.centroids[c] = (1 - lr) * self.centroids[c] + lr * xb[sel].mean(0)
        self.centroids = _normalise(self.centroids)

    def assign(self, x: Tensor) -> Tensor:
        """Cluster id [n] for inputs x [n,d] (any device)."""
        C = self.centroids.to(x.device)
        return (_normalise(x) @ C.T).argmax(-1)

    def state_dict(self) -> dict:
        return dict(C=self.C, seed=self.seed, centroids=self.centroids)

    @classmethod
    def from_state_dict(cls, d: dict) -> "SphericalKMeans":
        km = cls(d["C"], d["seed"])
        km.centroids = d["centroids"]
        return km


class InputReservoir:
    """Keeps a seeded uniform sample of up to `cap` layer inputs (fp16, CPU) for k-means fitting."""

    def __init__(self, cap: int = 32768, seed: int = 0):
        self.cap = cap
        self.gen = torch.Generator().manual_seed(seed)
        self.buf: Optional[Tensor] = None
        self.seen = 0

    def add(self, x: Tensor) -> None:
        x = x.detach().to("cpu", torch.float16)
        n = x.shape[0]
        if self.buf is None:
            self.buf = torch.empty(0, x.shape[1], dtype=torch.float16)
        room = self.cap - self.buf.shape[0]
        if room > 0:
            take = x[:room]
            self.buf = torch.cat([self.buf, take])
            x = x[room:]
            self.seen += take.shape[0]
        if x.shape[0] == 0:
            return
        # reservoir sampling, vectorised: item i (global index s) replaces slot u < cap with prob cap/(s+1)
        s = torch.arange(self.seen, self.seen + x.shape[0], dtype=torch.float64)
        u = torch.rand(x.shape[0], generator=self.gen, dtype=torch.float64) * (s + 1)
        slot = u.floor().long()
        keep = slot < self.cap
        self.buf[slot[keep]] = x[keep]
        self.seen += x.shape[0]

    def data(self) -> Tensor:
        return self.buf.float()
