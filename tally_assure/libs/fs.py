from __future__ import annotations

from pathlib import Path

def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)
