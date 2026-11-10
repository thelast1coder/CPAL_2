"""Statistics for accuracy comparisons: paired bootstrap, recovery ratio, plain intervals (PROJECT.md 9)."""
from __future__ import annotations

import math
from typing import Sequence

import torch


def _as_tensor(x) -> torch.Tensor:
    return torch.as_tensor(list(x) if not isinstance(x, torch.Tensor) else x, dtype=torch.float64)


def paired_bootstrap(a: Sequence[float], b: Sequence[float], resamples: int = 10000, level: float = 0.95,
                     seed: int = 0) -> dict:
    """Paired bootstrap for mean(a) - mean(b) on the same items (a, b are per-item scores in [0, 1]).

    Returns dict with diff, lo, hi (percentile interval), n, resamples, seed. Differences in the units of
    the inputs (multiply by 100 for points).
    """
    a, b = _as_tensor(a), _as_tensor(b)
    if a.shape != b.shape:
        raise ValueError("paired bootstrap needs equal-length inputs")
    n = a.numel()
    d = a - b
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(n, (resamples, n), generator=gen)
    boots = d[idx].mean(-1)
    alpha = (1 - level) / 2
    lo, hi = torch.quantile(boots, torch.tensor([alpha, 1 - alpha], dtype=torch.float64)).tolist()
    return dict(diff=float(d.mean()), lo=lo, hi=hi, n=n, resamples=resamples, level=level, seed=seed)


def recovery(acc_method: Sequence[float], acc_base: Sequence[float], acc_native: Sequence[float],
             resamples: int = 10000, level: float = 0.95, seed: int = 0) -> dict:
    """Recovery = (acc(method) - acc(base)) / (acc(native) - acc(base)), with a paired bootstrap interval.

    Resamples rows jointly; resamples with a zero denominator are dropped (count reported).
    """
    m, b, nv = _as_tensor(acc_method), _as_tensor(acc_base), _as_tensor(acc_native)
    n = m.numel()
    num = float(m.mean() - b.mean())
    den = float(nv.mean() - b.mean())
    point = num / den if den != 0 else float("nan")
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(n, (resamples, n), generator=gen)
    nums = m[idx].mean(-1) - b[idx].mean(-1)
    dens = nv[idx].mean(-1) - b[idx].mean(-1)
    ok = dens != 0
    r = nums[ok] / dens[ok]
    alpha = (1 - level) / 2
    if r.numel() > 0:
        lo, hi = torch.quantile(r, torch.tensor([alpha, 1 - alpha], dtype=torch.float64)).tolist()
    else:
        lo = hi = float("nan")
    return dict(recovery=point, lo=lo, hi=hi, n=n, dropped_resamples=int((~ok).sum()), seed=seed)


def wilson_interval(correct: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}.get(level)
    if z is None:
        from statistics import NormalDist
        z = NormalDist().inv_cdf(1 - (1 - level) / 2)
    p = correct / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (centre - half, centre + half)
