"""Calibration, validation and analysis token sets (PHASE0_SETUP.md 2.7, decisions D8 and D9).

Documents: FineWeb-Edu text (domain 0) and MetaMathQA in the model's chat template (domain 1). Documents are
split (seeded) into analysis, validation and calibration by whole documents, so the splits are disjoint by
construction; disjointness is re-checked by document hash. Maths documents whose question matches a GSM8K
or MATH-500 test question (normalised text) are removed, and every text document is checked for test
questions as substrings. Text: 1024-token chunks from single documents. Maths: one example per sequence,
right-padded; padding is excluded through the attention mask.
"""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import torch

TEXT, MATHS, CODE = 0, 1, 2


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def doc_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


@dataclass
class Doc:
    text: str            # raw text (maths: rendered chat example)
    domain: int
    question: str = ""   # maths only, for the leakage check
    hash: str = ""

    def __post_init__(self):
        if not self.hash:
            self.hash = doc_hash(self.text)


@dataclass
class SplitTokens:
    input_ids: torch.Tensor          # [S, L] long
    attention_mask: torch.Tensor     # [S, L] bool
    domain: torch.Tensor             # [S] int8
    doc_hash: list = field(default_factory=list)

    @property
    def n_tokens(self) -> int:
        return int(self.attention_mask.sum())

    def to_dict(self) -> dict:
        return dict(input_ids=self.input_ids, attention_mask=self.attention_mask, domain=self.domain,
                    doc_hash=self.doc_hash)

    @classmethod
    def from_dict(cls, d: dict) -> "SplitTokens":
        return cls(d["input_ids"], d["attention_mask"].bool(), d["domain"], list(d["doc_hash"]))

    def batches(self, batch_size: int):
        for s in range(0, self.input_ids.shape[0], batch_size):
            yield self.input_ids[s:s + batch_size], self.attention_mask[s:s + batch_size]

    def domain_mask(self) -> torch.Tensor:
        """Per-token domain [S, L] (for the recorder)."""
        return self.domain.view(-1, 1).expand_as(self.input_ids)


def leakage_filter(docs: list[Doc], test_questions: Iterable[str], log=print) -> tuple[list[Doc], dict]:
    """Drop documents containing any test question (normalised substring; exact match for maths questions)."""
    tq = [norm_text(q) for q in test_questions if q.strip()]
    tq_set = set(tq)
    kept, dropped = [], 0
    for d in docs:
        if d.domain == MATHS and norm_text(d.question) in tq_set:
            dropped += 1
            continue
        nt = norm_text(d.text)
        if any(q in nt for q in tq if len(q) > 40):
            dropped += 1
            continue
        kept.append(d)
    info = dict(test_questions=len(tq_set), docs_in=len(docs), docs_dropped=dropped)
    log(f"[data] leakage filter: {info}")
    return kept, info


def tokenise_doc(d: Doc, tokenizer, seq_len: int, max_chunks: int = 1) -> list[list[int]]:
    ids = tokenizer(d.text, add_special_tokens=d.domain != MATHS)["input_ids"]
    if d.domain == MATHS:
        return [ids[:seq_len]]
    return [ids[i:i + seq_len] for i in range(0, max(len(ids), 1), seq_len)][:max_chunks]


def split_documents(docs: list[Doc], tokenizer, quotas: dict[str, int], mix: dict[int, float], seq_len: int,
                    seed: int = 0, min_text_tokens: int = 256) -> dict[str, SplitTokens]:
    """Seeded document-level split. quotas: split name -> valid tokens, filled in the given order.

    Within each split the token share per domain follows `mix` (domain -> fraction).
    """
    rng = random.Random(seed)
    by_dom: dict[int, list[Doc]] = {}
    for d in docs:
        by_dom.setdefault(d.domain, []).append(d)
    for v in by_dom.values():
        rng.shuffle(v)
    ptr = {dm: 0 for dm in by_dom}
    out = {}
    for split, quota in quotas.items():
        rows, doms, hashes = [], [], []
        for dm, frac in mix.items():
            target, got = int(round(quota * frac)), 0
            while got < target:
                if ptr[dm] >= len(by_dom[dm]):
                    raise RuntimeError(f"ran out of domain-{dm} documents for split {split}")
                d = by_dom[dm][ptr[dm]]
                ptr[dm] += 1
                for chunk in tokenise_doc(d, tokenizer, seq_len):
                    if dm == TEXT and len(chunk) < min_text_tokens:
                        continue
                    rows.append(chunk)
                    doms.append(dm)
                    hashes.append(d.hash)
                    got += len(chunk)
        S = len(rows)
        ids = torch.full((S, seq_len), int(tokenizer.pad_token_id or 0), dtype=torch.long)
        mask = torch.zeros(S, seq_len, dtype=torch.bool)
        for i, r in enumerate(rows):
            ids[i, :len(r)] = torch.tensor(r)
            mask[i, :len(r)] = True
        out[split] = SplitTokens(ids, mask, torch.tensor(doms, dtype=torch.int8), hashes)
    check_disjoint(out)
    return out


def check_disjoint(splits: dict[str, SplitTokens]) -> None:
    names = list(splits)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = set(splits[names[i]].doc_hash) & set(splits[names[j]].doc_hash)
            if inter:
                raise AssertionError(f"splits {names[i]} and {names[j]} share {len(inter)} documents")


# ------------------------------------------------------------------------------------------- VM sources
def load_documents(sources: dict, n_text: int, n_maths: int, chat_fn: Callable[[str, str], str], log=print) -> list[Doc]:
    """Stream FineWeb-Edu and load MetaMathQA (VM only; dataset IDs (verify))."""
    from datasets import load_dataset
    fw = sources["fineweb_edu"]
    ds = load_dataset(fw["path"], name=fw.get("name"), split=fw["split"], streaming=True)
    docs = []
    for i, ex in enumerate(ds):
        if i >= n_text:
            break
        docs.append(Doc(ex["text"], TEXT))
    mm = sources["metamathqa"]
    dm = load_dataset(mm["path"], split=mm["split"])
    idx = list(range(len(dm)))
    random.Random(0).shuffle(idx)
    for i in idx[:n_maths]:
        ex = dm[i]
        docs.append(Doc(chat_fn(ex["query"], ex["response"]), MATHS, question=ex["query"]))
    log(f"[data] loaded {n_text} text and {n_maths} maths documents")
    return docs


def test_questions(sources: dict) -> list[str]:
    from datasets import load_dataset
    g = sources["gsm8k"]
    qs = [ex["question"] for ex in load_dataset(g["path"], g["name"], split=g["split"])]
    m = sources.get("math500")
    if m:
        try:
            qs += [ex["problem"] for ex in load_dataset(m["path"], split=m["split"])]
        except Exception as e:  # (verify) dataset ID
            print(f"[data] WARNING: MATH-500 not loaded for the leakage check: {e}")
    return qs


# ------------------------------------------------------------------------------------------- tiny
def tiny_splits(quotas: dict[str, int], seq_len: int = 64, vocab: int = 128, seed: int = 0) -> dict[str, SplitTokens]:
    """Synthetic sequences (no downloads); each sequence is its own 'document'."""
    from ..utils.tiny import structured_tokens
    out = {}
    s = seed
    for split, quota in quotas.items():
        n = max(1, -(-quota // seq_len))
        ids = structured_tokens(n, seq_len, vocab, seed=1000 + s)
        s += 1
        gen = torch.Generator().manual_seed(seed + s)
        lens = torch.randint(seq_len // 2, seq_len + 1, (n,), generator=gen)
        mask = torch.arange(seq_len).unsqueeze(0) < lens.unsqueeze(1)
        dom = torch.randint(0, 2, (n,), generator=gen).to(torch.int8)
        hashes = [doc_hash(f"{split}-{i}-{ids[i].tolist()}") for i in range(n)]
        out[split] = SplitTokens(ids, mask, dom, hashes)
    check_disjoint(out)
    return out
