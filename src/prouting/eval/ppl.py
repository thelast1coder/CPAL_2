"""WikiText-2 test perplexity: context 2048, stride 2048, padding excluded (PHASE0_SETUP.md 2.8)."""
from __future__ import annotations

import math
from typing import Optional

import torch


@torch.no_grad()
def perplexity(model, token_ids: torch.Tensor, context: int = 2048, stride: int = 2048,
               max_tokens: Optional[int] = None, handle=None) -> dict:
    """Non-overlapping windows (stride == context): every token scored once, first token of each window
    has no context and is excluded. token_ids: 1-D long tensor of the concatenated test text."""
    ids = token_ids.reshape(-1)
    if max_tokens is not None:
        ids = ids[:max_tokens]
    dev = next(model.parameters()).device
    nll, count = 0.0, 0
    for s in range(0, ids.numel() - 1, stride):
        chunk = ids[s:s + context].unsqueeze(0).to(dev)
        if chunk.shape[1] < 2:
            break
        if handle is not None:
            handle.set_valid(torch.ones_like(chunk, dtype=torch.bool))
        logits = model(chunk).logits.float()
        lp = torch.log_softmax(logits[0, :-1], -1)
        tgt = chunk[0, 1:]
        nll -= float(lp.gather(-1, tgt.unsqueeze(-1)).sum())
        count += tgt.numel()
    if handle is not None:
        handle.set_valid(None)
    return dict(ppl=math.exp(nll / max(count, 1)), nll=nll, tokens=count)


@torch.no_grad()
def next_token_accuracy(model, batch: torch.Tensor, valid: Optional[torch.Tensor] = None, handle=None) -> list[bool]:
    """Per-position next-token accuracy (the tiny run's stand-in for GSM8K). batch [B,T]."""
    dev = next(model.parameters()).device
    batch = batch.to(dev)
    if handle is not None:
        handle.set_valid(valid if valid is not None else torch.ones_like(batch, dtype=torch.bool))
    pred = model(batch).logits.argmax(-1)[:, :-1]
    ok = (pred == batch[:, 1:])
    if valid is not None:
        ok = ok[valid[:, 1:].to(dev).bool()]
    if handle is not None:
        handle.set_valid(None)
    return ok.reshape(-1).cpu().tolist()
