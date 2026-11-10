"""MATH-500: boxed answer extraction and equivalence (PHASE0_SETUP.md 2.8).

Equivalence uses the `math-verify` package (import name `math_verify`, verify on the VM). Without it,
falls back to normalised string comparison and says so in the result.
"""
from __future__ import annotations

from typing import Optional

MAX_NEW_TOKENS = 2048
MATH_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def last_boxed(text: str) -> Optional[str]:
    """Content of the last \\boxed{...} (balanced braces), or None."""
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for p in range(j, len(text)):
        if text[p] == "{":
            depth += 1
        elif text[p] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:p]
    return None


def _norm(s: str) -> str:
    return s.replace(" ", "").replace("\\left", "").replace("\\right", "").replace("dfrac", "frac").strip("$.")


def equivalent(pred: Optional[str], gold: str) -> tuple[bool, str]:
    """(is_correct, backend)."""
    if pred is None:
        return False, "none"
    try:
        from math_verify import parse, verify  # (verify API on the VM)
        return bool(verify(parse("$" + gold + "$"), parse("$" + pred + "$"))), "math_verify"
    except ImportError:
        return _norm(pred) == _norm(gold), "string"
