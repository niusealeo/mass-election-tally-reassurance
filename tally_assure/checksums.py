from __future__ import annotations

import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple


def now_stamps() -> Tuple[str, str]:
    utc = datetime.now(timezone.utc)
    local = utc.astimezone()
    utc_s = utc.isoformat(timespec="seconds")
    local_s = local.strftime("%a %d %b %Y %H:%M:%S %Z%z")
    return utc_s, local_s


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _decode_best_effort(raw: bytes) -> str:
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return raw.decode(enc, errors="strict")
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


def _infer_delim_and_header(lines: List[str]) -> Tuple[str, int]:
    """
    Some 2002-era CSVs begin with a few metadata lines with only ~2 columns,
    then the real table starts with many columns. Pandas errors like:
      Expected 2 fields in line X, saw 13

    We detect the delimiter that yields the largest max field count, then find the
    earliest row achieving that max and treat it as the header row.
    """
    delims = [",", "\t", ";", "|"]
    best_delim = ","
    best_max = 0
    best_header_idx = 0

    for d in delims:
        counts = []
        for ln in lines:
            if not ln.strip():
                continue
            counts.append(len(ln.rstrip("\n").split(d)))
        if not counts:
            continue
        mx = max(counts)
        if mx > best_max:
            best_max = mx
            best_delim = d
            for i, ln in enumerate(lines):
                if not ln.strip():
                    continue
                if len(ln.rstrip("\n").split(d)) == mx:
                    best_header_idx = i
                    break

    return best_delim, best_header_idx


def read_csv_robust(path: Path) -> pd.DataFrame:
    """
    Robust CSV reader for NZ election data across eras:
    - non-UTF8 encodings (cp1252/latin1)
    - leading metadata lines with fewer columns than the real table
    - delimiter variations (comma/tab/semicolon)
    """
    raw = path.read_bytes()
    text = _decode_best_effort(raw)
    lines = text.splitlines(True)
    sample = lines[:200] if len(lines) > 200 else lines
    delim, header_idx = _infer_delim_and_header(sample)

    from io import StringIO
    sio = StringIO(text)
    for _ in range(header_idx):
        sio.readline()

    # python engine handles irregularities better
    return pd.read_csv(sio, sep=delim, engine="python")


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


def compare_series(a: pd.Series, b: pd.Series, tol: float = 0.0) -> Tuple[List[str], List[str]]:
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


def checksum_candidate_atomic(candidate_csv: Path) -> Tuple[pd.Series, List[str], List[str]]:
    df = read_csv_robust(candidate_csv)
    trow = find_totals_row(df)
    computed = sum_excluding_totals(df, trow)
    if trow is None:
        return computed, [], list(computed.index)
    provided = df.loc[trow, computed.index]
    passed, failed = compare_series(computed, provided, tol=0.0)
    return computed, passed, failed


def checksum_party_atomic(party_csv: Path) -> Tuple[pd.Series, List[str], List[str]]:
    df = read_csv_robust(party_csv)
    trow = find_totals_row(df)
    computed = sum_excluding_totals(df, trow)
    if trow is None:
        return computed, [], list(computed.index)
    provided = df.loc[trow, computed.index]
    passed, failed = compare_series(computed, provided, tol=0.0)
    return computed, passed, failed


def checksum_party_vs_split_total(party_totals: pd.Series, split_csv: Path) -> Tuple[List[str], List[str]]:
    s = read_csv_robust(split_csv)
    if "Total Party Votes" not in s.columns:
        return [], ["NO_TOTAL_PARTY_VOTES_COLUMN"]
    party_col = s.columns[0]
    split_totals = s.set_index(party_col)["Total Party Votes"]
    if s.shape[0] >= 2 and "total" in str(s.iloc[-1, 0]).lower():
        split_totals = split_totals.iloc[:-1]
    passed, failed = [], []
    for p in party_totals.index:
        if p not in split_totals.index:
            continue
        if float(party_totals[p]) == float(split_totals[p]):
            passed.append(p)
        else:
            failed.append(p)
    return passed, failed
