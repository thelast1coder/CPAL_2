"""GSM8K evaluation: zero-shot chat prompt, greedy, answer after the last '####' (PHASE0_SETUP.md 2.8)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Optional

import torch

from .prompts import chat_prompt, gsm8k_user_message

_NUM = re.compile(r"-?\$?\s?\d[\d,]*(?:\.\d+)?|-?\.\d+")


def _clean_number(s: str) -> Optional[str]:
    s = s.strip().replace(",", "").replace("$", "").replace(" ", "")
    s = s.rstrip(".")
    if s in ("", "-"):
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return repr(v)


def extract_answer(text: str) -> Optional[str]:
    """Number after the last '####'; otherwise the last number in the text.

    Strips commas, currency signs and a trailing full stop. Returns a canonical string or None.
    """
    if "####" in text:
        tail = text.rsplit("####", 1)[1]
        m = _NUM.search(tail)
        if m:
            got = _clean_number(m.group(0))
            if got is not None:
                return got
    nums = _NUM.findall(text)
    for cand in reversed(nums):
        got = _clean_number(cand)
        if got is not None:
            return got
    return None


def gold_answer(answer_field: str) -> str:
    """Gold number from a GSM8K 'answer' field (after '####')."""
    got = extract_answer(answer_field.rsplit("####", 1)[1] if "####" in answer_field else answer_field)
    if got is None:
        raise ValueError(f"cannot parse gold answer: {answer_field!r}")
    return got


def is_correct(pred: Optional[str], gold: str) -> bool:
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return pred == gold


def fixed_subset_indices(n_total: int, size: int, seed: int = 0) -> list[int]:
    """Seeded random subset of test indices (written once to configs/ and committed)."""
    gen = torch.Generator().manual_seed(seed)
    return sorted(torch.randperm(n_total, generator=gen)[:size].tolist())


@torch.no_grad()
def generate_batch(model, tokenizer, prompts: list[str], max_new_tokens: int = 512) -> list[str]:
    """Greedy, left-padded batched generation. Returns decoded continuations."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    out = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                         pad_token_id=tokenizer.pad_token_id)
    gen = out[:, enc["input_ids"].shape[1]:]
    return tokenizer.batch_decode(gen, skip_special_tokens=True)


def run_gsm8k(model, tokenizer, items: Iterable[dict], model_family: str, out_jsonl: Path,
              batch_size: int = 16, max_new_tokens: int = 512, tag: str = "") -> dict:
    """Evaluate on items (dicts with 'idx', 'question', 'answer'); resumable JSONL of every generation."""
    from ..utils.io import read_jsonl, append_jsonl
    items = list(items)
    done = {r["idx"]: r for r in read_jsonl(out_jsonl)} if Path(out_jsonl).exists() else {}
    todo = [it for it in items if it["idx"] not in done]
    for i in range(0, len(todo), batch_size):
        chunk = todo[i:i + batch_size]
        prompts = [chat_prompt(tokenizer, gsm8k_user_message(it["question"]), model_family) for it in chunk]
        outs = generate_batch(model, tokenizer, prompts, max_new_tokens)
        rows = []
        for it, o in zip(chunk, outs):
            gold = gold_answer(it["answer"])
            pred = extract_answer(o)
            rows.append(dict(idx=it["idx"], tag=tag, pred=pred, gold=gold, correct=is_correct(pred, gold),
                             generation=o))
        append_jsonl(out_jsonl, rows)
        for r in rows:
            done[r["idx"]] = r
    per_item = [bool(done[it["idx"]]["correct"]) for it in items]
    return dict(accuracy=sum(per_item) / max(len(per_item), 1), n=len(per_item), per_item=per_item,
                idx=[it["idx"] for it in items])
