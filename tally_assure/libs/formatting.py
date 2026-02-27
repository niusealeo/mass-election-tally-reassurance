from __future__ import annotations

def bool_to_str(v) -> str:
    """Convert boolean-like values to canonical string booleans for CSV/JSON outputs."""
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, str):
        s = v.strip().casefold()
        if s in {"true","false","error"}:
            return s
    return "error"
