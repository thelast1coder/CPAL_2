"""Atomic writes and resumable JSONL."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import torch


def _atomic(path: Path, write_fn) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    os.close(fd)
    try:
        write_fn(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _default(o):
    if isinstance(o, torch.Tensor):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_json(path, obj: Any) -> None:
    def w(tmp):
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=_default)
    _atomic(Path(path), w)


def read_json(path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_text(path, text: str) -> None:
    def w(tmp):
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    _atomic(Path(path), w)


def save_torch(path, obj: Any) -> None:
    _atomic(Path(path), lambda tmp: torch.save(obj, tmp))


def load_torch(path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def append_jsonl(path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=_default) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break  # truncated last line after an interruption
    return rows
