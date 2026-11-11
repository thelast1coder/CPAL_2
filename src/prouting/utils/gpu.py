"""GPU sharing helpers. The GPU is shared with other projects: never kill processes we did not start."""
from __future__ import annotations

import subprocess
from typing import Optional

import torch


def nvidia_smi_summary() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:  # no driver on the laptop
        return f"nvidia-smi unavailable ({type(e).__name__})"


def free_memory_gb(device: int = 0) -> Optional[float]:
    """Free GPU memory in GB (torch.cuda.mem_get_info), or None without CUDA."""
    if not torch.cuda.is_available():
        return None
    free, _total = torch.cuda.mem_get_info(device)
    return free / 1024**3


def wait_for_memory(gb: float, device: int = 0, margin_gb: float = 1.0, poll_s: float = 60.0,
                    max_wait_min: Optional[float] = None, verbose: bool = True) -> float:
    """Queue behind the other projects: block until at least gb + margin_gb of GPU memory is free.

    Returns the minutes waited. Raises TimeoutError after max_wait_min (default: env PROUTING_GPU_WAIT_MAX_MIN,
    else 720). Never touches other processes.
    """
    import os
    import time
    if not torch.cuda.is_available():
        return 0.0
    if max_wait_min is None:
        max_wait_min = float(os.environ.get("PROUTING_GPU_WAIT_MAX_MIN", "720"))
    t0 = time.time()
    while True:
        free = free_memory_gb(device)
        if free >= gb + margin_gb:
            waited = (time.time() - t0) / 60
            if verbose and waited > 0.05:
                print(f"[gpu] {free:.1f} GB free after waiting {waited:.1f} min")
            return waited
        waited = (time.time() - t0) / 60
        if waited > max_wait_min:
            raise TimeoutError(f"[gpu] needed {gb + margin_gb:.1f} GB, only {free:.1f} GB free after {waited:.0f} min")
        if verbose:
            print(f"[gpu] waiting: need {gb + margin_gb:.1f} GB, {free:.1f} GB free; others:\n{nvidia_smi_summary()}",
                  flush=True)
        time.sleep(poll_s)


def cap_memory(fraction: Optional[float] = None, gb: Optional[float] = None, device: int = 0,
               verbose: bool = True) -> Optional[float]:
    """Cap this process's GPU memory with torch.cuda.set_per_process_memory_fraction.

    Give either a fraction of total memory or an absolute number of GB. Prints what else is running.
    Returns the fraction applied (None on CPU).
    """
    if not torch.cuda.is_available():
        return None
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    if gb is not None:
        fraction = min(1.0, gb / total)
    if fraction is None:
        raise ValueError("give fraction or gb")
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    if verbose:
        print(f"[gpu] cap {fraction:.3f} of {total:.1f} GB; free now {free_memory_gb(device):.1f} GB")
        print("[gpu] other compute processes:\n" + nvidia_smi_summary())
    return fraction
