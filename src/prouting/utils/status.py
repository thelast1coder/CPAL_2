"""Append dated entries to results/STATUS.md."""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

from .logging import REPO_ROOT

STATUS = REPO_ROOT / "results" / "STATUS.md"


def append_status(line: str, path: Path = STATUS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"- {stamp}: {line}\n")
