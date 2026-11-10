"""Run lm-evaluation-harness tasks on an already-loaded (and patched) HF model.

Believed API (verify against the installed lm-eval on the VM): `lm_eval.models.huggingface.HFLM(pretrained=model,
tokenizer=tokenizer, batch_size=...)` accepts an instantiated model, and `lm_eval.simple_evaluate(model=lm,
tasks=[...], num_fewshot=...)` returns a dict with `results` and per-sample logs when `log_samples=True`.
"""
from __future__ import annotations

from typing import Optional


def run_tasks(model, tokenizer, tasks: list[str], num_fewshot: Optional[int] = None, batch_size: int = 8,
              limit: Optional[int] = None, seed: int = 0) -> dict:
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    res = lm_eval.simple_evaluate(model=lm, tasks=tasks, num_fewshot=num_fewshot, limit=limit,
                                  log_samples=True, random_seed=seed, numpy_random_seed=seed,
                                  torch_random_seed=seed)
    return res
