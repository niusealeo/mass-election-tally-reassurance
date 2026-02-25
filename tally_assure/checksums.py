from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

def now_stamps() -> Tuple[str, str]:
    utc = datetime.now(timezone.utc)
    local = utc.astimezone()
    utc_s = utc.isoformat(timespec="seconds")
    local_s = local.strftime("%a %d %b %Y %H:%M:%S %Z%z")
    return utc_s, local_s


# ---------- helpers ----------


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def find_totals_row(df: pd.DataFrame) -> Optional[int]:
    if df.shape[0] == 0:
        return None
    first_col = df.columns[0]
    for idx in [df.index[-1], df.index[-2] if df.shape[0] >= 2 else df.index[-1]]:
        val = str(df.loc[idx, first_col]).strip().lower()
        if "total" in val:
            return int(idx)
    return None


def numeric_cols(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        if c == df.columns[0]:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def sum_excluding_totals(df: pd.DataFrame, totals_idx: Optional[int]) -> pd.Series:
    cols = numeric_cols(df)
    if totals_idx is None:
        return df[cols].sum(numeric_only=True)
    df2 = df.drop(index=totals_idx)
    return df2[cols].sum(numeric_only=True)


def compare_series(a: pd.Series, b: pd.Series, tol=0.0) -> Tuple[List[str], List[str]]:
    passed, failed = [], []
    shared = [c for c in a.index if c in b.index]
    for c in shared:
        av = float(a[c]) if pd.notna(a[c]) else 0.0
        bv = float(b[c]) if pd.notna(b[c]) else 0.0
        if abs(av - bv) <= tol:
            passed.append(c)
        else:
            failed.append(c)
    return passed, failed


# ---------- Sainte-Laguë allocation ----------

