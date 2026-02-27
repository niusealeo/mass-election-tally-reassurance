from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

def closest_common_ancestor(a: Path, b: Path) -> Path:
    a = a.resolve()
    b = b.resolve()
    ap = a.parts
    bp = b.parts
    n = min(len(ap), len(bp))
    i = 0
    while i < n and ap[i] == bp[i]:
        i += 1
    if i == 0:
        return Path(a.anchor)
    return Path(*ap[:i])

def common_parent(paths: List[Optional[Path]]) -> Optional[Path]:
    ps = [p.resolve() for p in paths if p is not None]
    if not ps:
        return None
    try:
        return Path(os.path.commonpath([str(p) for p in ps]))
    except Exception:
        anc = ps[0]
        for p in ps[1:]:
            anc = closest_common_ancestor(anc, p)
        return anc

def rel_to_common(p: Optional[Path], common: Optional[Path]) -> Optional[str]:
    if p is None:
        return None
    if common is None:
        return str(p)
    try:
        return str(p.resolve().relative_to(common))
    except Exception:
        return str(p)

def relativize_detail(detail: Optional[dict], common: Optional[Path]) -> Optional[dict]:
    """Return a copy of detail where any 'file' fields are relative to `common`.

    Walks dict/list recursively so nested check entries also get rewritten.
    """
    if detail is None:
        return None

    def walk(x: Any):
        if isinstance(x, dict):
            out: Dict[str, Any] = {}
            for k, v in x.items():
                if k == "file" and isinstance(v, str):
                    try:
                        out[k] = rel_to_common(Path(v), common)
                    except Exception:
                        out[k] = v
                else:
                    out[k] = walk(v)
            return out
        if isinstance(x, list):
            return [walk(v) for v in x]
        return x

    return walk(detail)
