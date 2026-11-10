"""GPU sharing helpers. The H100 is shared with two other projects: never kill processes we did not start."""
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
