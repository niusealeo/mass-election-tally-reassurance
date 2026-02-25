from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ---------------- timestamps / fs ----------------

def now_stamps() -> Tuple[str, str]:
    utc = datetime.now(timezone.utc)
    local = utc.astimezone()
    utc_s = utc.isoformat(timespec="seconds")
    local_s = local.strftime("%a %d %b %Y %H:%M:%S %Z%z")
    return utc_s, local_s


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------- robust parsing ----------------

def _decode_best_effort(raw: bytes) -> str:
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return raw.decode(enc, errors="strict")
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


def read_csv_robust(path: Path) -> pd.DataFrame:
    """
    General robust CSV reader: handles odd encodings and preambles where possible.
    For Part VIII atomic files, prefer read_csv_atomic().
    """
    raw = path.read_bytes()
    text = _decode_best_effort(raw)
    from io import StringIO
    return pd.read_csv(StringIO(text), engine="python")


def read_csv_atomic(path: Path) -> pd.DataFrame:
    """
    Part VIII atomic files often have:
      line0: "Part VIII - ..."
      line1: "<Electorate> <n>","Candidate Vote Details" OR "...","Party Vote Details"
      then the real CSV header begins.
    We skip the first 2 lines and let the CSV parser handle quoted newlines (some headers wrap).
    """
    raw = path.read_bytes()
    text = _decode_best_effort(raw)
    lines = text.splitlines(True)

    # Skip up to 5 initial metadata lines; pick the skip that yields a plausible header.
    best_df: Optional[pd.DataFrame] = None
    best_score = -1

    from io import StringIO
    for skip in range(0, 6):
        sio = StringIO("".join(lines[skip:]))
        try:
            df = pd.read_csv(sio, engine="python")
        except Exception:
            continue
        # score: prefer frames with "Total Valid" in columns and >= 6 columns
        cols = [str(c).lower() for c in df.columns]
        score = 0
        if any("total valid" in c for c in cols):
            score += 10
        if any("informal" in c for c in cols):
            score += 5
        score += min(df.shape[1], 40)  # more cols usually means correct header
        if score > best_score:
            best_df = df
            best_score = score

    if best_df is None:
        # fallback
        return read_csv_robust(path)
    return best_df


# ---------------- helpers ----------------

def _to_num(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        s2 = s.astype(str).str.replace(",", "", regex=False).str.strip()
    else:
        s2 = s
    return pd.to_numeric(s2, errors="coerce")


def _label_cols(df: pd.DataFrame) -> List[str]:
    # first two columns are "location-ish" / labels in these files
    if df.shape[1] >= 2:
        return [df.columns[1], df.columns[0]]
    return [df.columns[0]]


def find_totals_row(df: pd.DataFrame) -> Optional[int]:
    """
    Totals label appears in the *second* column in Part VIII files.
    We scan label columns and return the LAST row containing 'total'.
    """
    if df.shape[0] == 0:
        return None
    hits: List[int] = []
    for idx in df.index:
        for c in _label_cols(df):
            v = str(df.loc[idx, c]).strip().lower()
            if v and v != "nan" and "total" in v:
                hits.append(int(idx))
                break
    return hits[-1] if hits else None


def candidate_numeric_cols(df: pd.DataFrame) -> Tuple[List[str], str, str]:
    """
    Returns (candidate_cols, total_valid_col, informal_col)
    """
    cols = list(df.columns)
    # preferred names
    total_valid = next((c for c in cols if str(c).strip().lower() == "total valid candidate votes"), cols[-2])
    informal = next((c for c in cols if "informal" in str(c).strip().lower()), cols[-1])

    # candidate columns are after 2 label columns and before the totals columns
    # We locate totals columns by name positions.
    idx_total_valid = cols.index(total_valid)
    idx_informal = cols.index(informal)
    end = min(idx_total_valid, idx_informal)
    cand_cols = cols[2:end]
    return cand_cols, total_valid, informal


def party_numeric_cols(df: pd.DataFrame) -> Tuple[List[str], str, str]:
    cols = list(df.columns)
    total_valid = next((c for c in cols if str(c).strip().lower() == "total valid party votes"), cols[-2])
    informal = next((c for c in cols if "informal" in str(c).strip().lower()), cols[-1])
    idx_total_valid = cols.index(total_valid)
    idx_informal = cols.index(informal)
    end = min(idx_total_valid, idx_informal)
    party_cols = cols[2:end]
    return party_cols, total_valid, informal


def _fail_entry(key: str, expected: float, official: float) -> Dict[str, Any]:
    diff = official - expected
    pct = (diff / expected * 100.0) if expected != 0 else (0.0 if official == 0 else float("inf"))
    return {
        "key": key,
        "expected_qa": expected,
        "official": official,
        "diff_official_minus_expected": diff,
        "pct_diff_of_expected": pct,
    }


# ---------------- roster extraction + porting ----------------

def extract_candidate_roster(df: pd.DataFrame, totals_idx: Optional[int]) -> pd.DataFrame:
    """
    Extract roster after totals row:
      col0 = candidate name
      col1 = party
      one of candidate columns contains their total (usually under their own name)
    """
    if totals_idx is None or df.shape[1] < 5:
        return pd.DataFrame(columns=["candidate", "party", "total_candidate_votes"])

    c0, c1 = df.columns[0], df.columns[1]
    cand_cols, _, _ = candidate_numeric_cols(df)

    # find marker row
    start = totals_idx + 1
    marker = None
    for i in range(start, df.shape[0]):
        v0 = str(df.iloc[i][c0]).strip().lower()
        if "electorate candidate valid votes" in v0:
            marker = i
            break
    if marker is not None:
        start = marker + 1

    rows = []
    for i in range(start, df.shape[0]):
        cand = df.iloc[i][c0]
        party = df.iloc[i][c1]
        if pd.isna(cand) or str(cand).strip() in ["", "nan"]:
            continue
        # find first numeric among candidate columns
        totals = _to_num(df.loc[i, cand_cols]) if cand_cols else pd.Series(dtype=float)
        tot_val = float(totals.dropna().iloc[0]) if len(totals.dropna()) else None
        if tot_val is None:
            continue
        rows.append({
            "candidate": str(cand).strip(),
            "party": "" if pd.isna(party) else str(party).strip(),
            "total_candidate_votes": tot_val,
        })
    return pd.DataFrame(rows)


def extract_party_roster(df: pd.DataFrame, totals_idx: Optional[int]) -> pd.DataFrame:
    """
    Extract party roster after totals row:
      party columns are header; roster section usually lists party in col0 and total in col1 or under its own column.
    We'll read as:
      col0 = party
      col1 = total (numeric) if present; else look for first numeric in party columns.
    """
    if totals_idx is None or df.shape[1] < 4:
        return pd.DataFrame(columns=["party", "total_party_votes"])

    c0, c1 = df.columns[0], df.columns[1]
    party_cols, _, _ = party_numeric_cols(df)

    start = totals_idx + 1
    marker = None
    for i in range(start, df.shape[0]):
        v0 = str(df.iloc[i][c0]).strip().lower()
        if "electorate party valid votes" in v0:
            marker = i
            break
    if marker is not None:
        start = marker + 1

    rows = []
    for i in range(start, df.shape[0]):
        party = df.iloc[i][c0]
        if pd.isna(party) or str(party).strip() in ["", "nan"]:
            continue
        tot1 = _to_num(pd.Series([df.iloc[i][c1]])).iloc[0]
        if pd.notna(tot1):
            rows.append({"party": str(party).strip(), "total_party_votes": float(tot1)})
            continue
        totals = _to_num(df.loc[i, party_cols]) if party_cols else pd.Series(dtype=float)
        tot_val = float(totals.dropna().iloc[0]) if len(totals.dropna()) else None
        if tot_val is None:
            continue
        rows.append({"party": str(party).strip(), "total_party_votes": tot_val})

    return pd.DataFrame(rows)


def port_candidate_roster_csv(candidate_csv: Path, out_csv: Path) -> None:
    df = read_csv_atomic(candidate_csv)
    trow = find_totals_row(df)
    roster = extract_candidate_roster(df, trow)
    roster.to_csv(out_csv, index=False, encoding="utf-8")


def port_party_roster_csv(party_csv: Path, out_csv: Path) -> None:
    df = read_csv_atomic(party_csv)
    trow = find_totals_row(df)
    roster = extract_party_roster(df, trow)
    roster.to_csv(out_csv, index=False, encoding="utf-8")


# ---------------- detailed checksum builders (JSON-friendly) ----------------

def checksum_candidate_atomic_detailed(candidate_csv: Path) -> Dict[str, Any]:
    df = read_csv_atomic(candidate_csv)
    trow = find_totals_row(df)

    cand_cols, total_valid_col, informal_col = candidate_numeric_cols(df)

    # coerce numeric on needed columns
    for c in cand_cols + [total_valid_col, informal_col]:
        df[c] = _to_num(df[c])

    result: Dict[str, Any] = {
        "file": str(candidate_csv),
        "totals_row_index": trow,
        "checks": {
            "row_sum_vs_total_valid_column": {"passed": [], "failed": []},
            "informal_column_sum_vs_totals_row": {"passed": [], "failed": []},
            "totals_row_candidate_cols_vs_atomic_sums": {"passed": [], "failed": []},
            "roster_candidate_totals_vs_atomic_sums": {"passed": [], "failed": []},
            "valid_plus_informal_row_check": {"passed": [], "failed": []},
        },
    }

    if trow is None:
        for k in result["checks"].keys():
            result["checks"][k]["failed"].append({"key": "TOTALS_ROW_NOT_FOUND"})
        return result

    # atomic part = rows before totals row
    atomic = df.iloc[:trow].copy()

    # (A) per-row sum check: sum candidate cols == Total Valid Candidate Votes
    for i in atomic.index:
        label = str(atomic.loc[i, _label_cols(df)[0]]).strip()
        row_sum = float(atomic.loc[i, cand_cols].sum(skipna=True))
        official = float(atomic.loc[i, total_valid_col]) if pd.notna(atomic.loc[i, total_valid_col]) else 0.0
        if row_sum == official:
            result["checks"]["row_sum_vs_total_valid_column"]["passed"].append({"row": int(i), "label": label})
        else:
            result["checks"]["row_sum_vs_total_valid_column"]["failed"].append(
                _fail_entry(f"row:{int(i)}:{label}", row_sum, official)
            )

    # (B) totals row candidate cols vs atomic sums (column-wise)
    atomic_col_sums = atomic[cand_cols + [total_valid_col, informal_col]].sum(numeric_only=True)
    totals_row = df.loc[trow, cand_cols + [total_valid_col, informal_col]]
    for c in cand_cols + [total_valid_col]:
        expected = float(atomic_col_sums[c]) if pd.notna(atomic_col_sums[c]) else 0.0
        official = float(totals_row[c]) if pd.notna(totals_row[c]) else 0.0
        if expected == official:
            result["checks"]["totals_row_candidate_cols_vs_atomic_sums"]["passed"].append({"key": str(c)})
        else:
            result["checks"]["totals_row_candidate_cols_vs_atomic_sums"]["failed"].append(_fail_entry(str(c), expected, official))

    # (C) informal column sum vs totals row informal
    expected_inf = float(atomic_col_sums[informal_col]) if pd.notna(atomic_col_sums[informal_col]) else 0.0
    official_inf = float(totals_row[informal_col]) if pd.notna(totals_row[informal_col]) else 0.0
    if expected_inf == official_inf:
        result["checks"]["informal_column_sum_vs_totals_row"]["passed"].append({"key": informal_col})
    else:
        result["checks"]["informal_column_sum_vs_totals_row"]["failed"].append(_fail_entry(informal_col, expected_inf, official_inf))

    # (D) row after totals row: valid + informal check (often totals_idx+1 with informal blank)
    expected_total_ballots = float(totals_row[total_valid_col] + totals_row[informal_col])
    found = False
    for j in range(trow + 1, min(trow + 6, df.shape[0])):
        tv = df.loc[j, total_valid_col]
        inf = df.loc[j, informal_col]
        # row that contains total ballots usually has tv numeric and inf NaN
        if pd.notna(tv) and (pd.isna(inf) or float(inf) == 0.0):
            official = float(tv)
            found = True
            if official == expected_total_ballots:
                result["checks"]["valid_plus_informal_row_check"]["passed"].append({"row": int(j), "key": total_valid_col})
            else:
                result["checks"]["valid_plus_informal_row_check"]["failed"].append(_fail_entry(f"row_after_totals:{int(j)}", expected_total_ballots, official))
            break
    if not found:
        result["checks"]["valid_plus_informal_row_check"]["failed"].append({"key": "ROW_AFTER_TOTALS_NOT_FOUND"})

    # (E) roster candidate totals vs atomic sums
    roster = extract_candidate_roster(df, trow)
    atomic_by_candidate = {str(c).strip(): float(atomic[c].sum(skipna=True)) for c in cand_cols}
    if roster.empty:
        result["checks"]["roster_candidate_totals_vs_atomic_sums"]["failed"].append({"key": "ROSTER_NOT_FOUND"})
    else:
        for _, r in roster.iterrows():
            name = str(r["candidate"]).strip()
            expected = atomic_by_candidate.get(name)
            official = float(r["total_candidate_votes"])
            if expected is None:
                result["checks"]["roster_candidate_totals_vs_atomic_sums"]["failed"].append({"key": f"MISSING_HEADER:{name}"})
                continue
            if expected == official:
                result["checks"]["roster_candidate_totals_vs_atomic_sums"]["passed"].append({"key": name})
            else:
                result["checks"]["roster_candidate_totals_vs_atomic_sums"]["failed"].append(_fail_entry(name, expected, official))

    return result


def checksum_party_atomic_detailed(party_csv: Path) -> Dict[str, Any]:
    df = read_csv_atomic(party_csv)
    trow = find_totals_row(df)

    party_cols, total_valid_col, informal_col = party_numeric_cols(df)
    for c in party_cols + [total_valid_col, informal_col]:
        df[c] = _to_num(df[c])

    result: Dict[str, Any] = {
        "file": str(party_csv),
        "totals_row_index": trow,
        "checks": {
            "row_sum_vs_total_valid_column": {"passed": [], "failed": []},
            "informal_column_sum_vs_totals_row": {"passed": [], "failed": []},
            "totals_row_party_cols_vs_atomic_sums": {"passed": [], "failed": []},
            "roster_party_totals_vs_atomic_sums": {"passed": [], "failed": []},
            "valid_plus_informal_row_check": {"passed": [], "failed": []},
        },
    }

    if trow is None:
        for k in result["checks"].keys():
            result["checks"][k]["failed"].append({"key": "TOTALS_ROW_NOT_FOUND"})
        return result

    atomic = df.iloc[:trow].copy()

    # per-row sum check
    for i in atomic.index:
        label = str(atomic.loc[i, _label_cols(df)[0]]).strip()
        row_sum = float(atomic.loc[i, party_cols].sum(skipna=True))
        official = float(atomic.loc[i, total_valid_col]) if pd.notna(atomic.loc[i, total_valid_col]) else 0.0
        if row_sum == official:
            result["checks"]["row_sum_vs_total_valid_column"]["passed"].append({"row": int(i), "label": label})
        else:
            result["checks"]["row_sum_vs_total_valid_column"]["failed"].append(_fail_entry(f"row:{int(i)}:{label}", row_sum, official))

    atomic_col_sums = atomic[party_cols + [total_valid_col, informal_col]].sum(numeric_only=True)
    totals_row = df.loc[trow, party_cols + [total_valid_col, informal_col]]

    for c in party_cols + [total_valid_col]:
        expected = float(atomic_col_sums[c]) if pd.notna(atomic_col_sums[c]) else 0.0
        official = float(totals_row[c]) if pd.notna(totals_row[c]) else 0.0
        if expected == official:
            result["checks"]["totals_row_party_cols_vs_atomic_sums"]["passed"].append({"key": str(c)})
        else:
            result["checks"]["totals_row_party_cols_vs_atomic_sums"]["failed"].append(_fail_entry(str(c), expected, official))

    expected_inf = float(atomic_col_sums[informal_col]) if pd.notna(atomic_col_sums[informal_col]) else 0.0
    official_inf = float(totals_row[informal_col]) if pd.notna(totals_row[informal_col]) else 0.0
    if expected_inf == official_inf:
        result["checks"]["informal_column_sum_vs_totals_row"]["passed"].append({"key": informal_col})
    else:
        result["checks"]["informal_column_sum_vs_totals_row"]["failed"].append(_fail_entry(informal_col, expected_inf, official_inf))

    expected_total_ballots = float(totals_row[total_valid_col] + totals_row[informal_col])
    found = False
    for j in range(trow + 1, min(trow + 6, df.shape[0])):
        tv = df.loc[j, total_valid_col]
        inf = df.loc[j, informal_col]
        if pd.notna(tv) and (pd.isna(inf) or float(inf) == 0.0):
            official = float(tv)
            found = True
            if official == expected_total_ballots:
                result["checks"]["valid_plus_informal_row_check"]["passed"].append({"row": int(j), "key": total_valid_col})
            else:
                result["checks"]["valid_plus_informal_row_check"]["failed"].append(_fail_entry(f"row_after_totals:{int(j)}", expected_total_ballots, official))
            break
    if not found:
        result["checks"]["valid_plus_informal_row_check"]["failed"].append({"key": "ROW_AFTER_TOTALS_NOT_FOUND"})

    roster = extract_party_roster(df, trow)
    atomic_by_party = {str(c).strip(): float(atomic[c].sum(skipna=True)) for c in party_cols}
    if roster.empty:
        result["checks"]["roster_party_totals_vs_atomic_sums"]["failed"].append({"key": "ROSTER_NOT_FOUND"})
    else:
        for _, r in roster.iterrows():
            name = str(r["party"]).strip()
            expected = atomic_by_party.get(name)
            official = float(r["total_party_votes"])
            if expected is None:
                result["checks"]["roster_party_totals_vs_atomic_sums"]["failed"].append({"key": f"MISSING_HEADER:{name}"})
                continue
            if expected == official:
                result["checks"]["roster_party_totals_vs_atomic_sums"]["passed"].append({"key": name})
            else:
                result["checks"]["roster_party_totals_vs_atomic_sums"]["failed"].append(_fail_entry(name, expected, official))

    return result
