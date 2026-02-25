from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


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
    raw = path.read_bytes()
    text = _decode_best_effort(raw)
    from io import StringIO
    return pd.read_csv(StringIO(text), engine="python")


def read_csv_atomic(path: Path) -> pd.DataFrame:
    '''
    Part VIII atomic files can include a small metadata preamble. We try skipping 0..5 lines
    and keep the best-looking parse (heuristic score).
    '''
    raw = path.read_bytes()
    text = _decode_best_effort(raw)
    lines = text.splitlines(True)

    best_df: Optional[pd.DataFrame] = None
    best_score = -1

    from io import StringIO
    for skip in range(0, 6):
        sio = StringIO("".join(lines[skip:]))
        try:
            df = pd.read_csv(sio, engine="python")
        except Exception:
            continue
        cols = [str(c).lower() for c in df.columns]
        score = 0
        if any("total valid" in c for c in cols):
            score += 10
        if any("informal" in c for c in cols):
            score += 5
        score += min(df.shape[1], 40)
        if score > best_score:
            best_df = df
            best_score = score

    return best_df if best_df is not None else read_csv_robust(path)


# ---------------- helpers ----------------

def _to_num(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        s2 = s.astype(str).str.replace(",", "", regex=False).str.strip()
    else:
        s2 = s
    return pd.to_numeric(s2, errors="coerce")


def location_pair(df: pd.DataFrame, i: int) -> List[str]:
    '''
    A size-2 array from the first two columns. No row index.
    '''
    v0 = df.iloc[i, 0] if df.shape[1] >= 1 else ""
    v1 = df.iloc[i, 1] if df.shape[1] >= 2 else ""
    return [
        "" if pd.isna(v0) else str(v0).strip(),
        "" if pd.isna(v1) else str(v1).strip(),
    ]


def _label_cols(df: pd.DataFrame) -> List[str]:
    if df.shape[1] >= 2:
        return [df.columns[1], df.columns[0]]
    return [df.columns[0]]


def find_totals_row(df: pd.DataFrame) -> Optional[int]:
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
    cols = list(df.columns)
    total_valid = next((c for c in cols if str(c).strip().lower() == "total valid candidate votes"), cols[-2])
    informal = next((c for c in cols if "informal" in str(c).strip().lower()), cols[-1])
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


def _scalar_fail(key: str, qa: float, official: float) -> Dict[str, Any]:
    diff = official - qa
    pct = (diff / qa * 100.0) if qa != 0 else (0.0 if official == 0 else float("inf"))
    return {
        "key": key,
        "qa_value": qa,
        "official_value": official,
        "diff_official_minus_qa": diff,
        "pct_diff_of_qa": pct,
    }


def totals_triplet(valid: float, informal: float) -> Dict[str, float]:
    return {
        "valid": float(valid),
        "informal": float(informal),
        "valid_plus_informal": float(valid + informal),
    }


# ---------------- roster extraction + porting ----------------

def extract_candidate_roster(df: pd.DataFrame, totals_idx: Optional[int]) -> pd.DataFrame:
    if totals_idx is None or df.shape[1] < 5:
        return pd.DataFrame(columns=["candidate", "party", "total_candidate_votes"])

    c0, c1 = df.columns[0], df.columns[1]
    cand_cols, _, _ = candidate_numeric_cols(df)

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
    for c in cand_cols + [total_valid_col, informal_col]:
        df[c] = _to_num(df[c])

    result: Dict[str, Any] = {
        "file": str(candidate_csv),
        "totals": None,
        "checks": {
            "voting_place_row_sum_vs_total_valid_column": {"passed": [], "failed": []},
            "informal_column_sum_vs_totals": {"passed": [], "failed": []},
            "totals_row_candidate_cols_vs_atomic_sums": {"passed": [], "failed": []},
            "roster_candidate_totals_vs_atomic_sums": {"passed": [], "failed": []},
            "valid_plus_informal_row_check": {"passed": [], "failed": []},
        },
    }

    if trow is None:
        for k in result["checks"].keys():
            result["checks"][k]["failed"].append({"key": "TOTALS_ROW_NOT_FOUND"})
        return result

    atomic = df.iloc[:trow].copy()

    # provided totals triplet
    provided_total_valid = float(df.loc[trow, total_valid_col]) if pd.notna(df.loc[trow, total_valid_col]) else 0.0
    provided_total_inf = float(df.loc[trow, informal_col]) if pd.notna(df.loc[trow, informal_col]) else 0.0
    result["totals"] = totals_triplet(provided_total_valid, provided_total_inf)

    # (1) per-row valid sum check + record row totals
    for i in range(atomic.shape[0]):
        loc = location_pair(df, i)
        qa_valid = float(atomic.iloc[i][cand_cols].sum(skipna=True))
        official_valid = float(atomic.iloc[i][total_valid_col]) if pd.notna(atomic.iloc[i][total_valid_col]) else 0.0
        row_inf = float(atomic.iloc[i][informal_col]) if pd.notna(atomic.iloc[i][informal_col]) else 0.0

        row_payload = {"location": loc, "row_totals": totals_triplet(official_valid, row_inf)}
        if qa_valid == official_valid:
            result["checks"]["voting_place_row_sum_vs_total_valid_column"]["passed"].append(row_payload)
        else:
            result["checks"]["voting_place_row_sum_vs_total_valid_column"]["failed"].append({**row_payload, **_scalar_fail("valid_row_sum", qa_valid, official_valid)})

    atomic_sums = atomic[cand_cols + [total_valid_col, informal_col]].sum(numeric_only=True)

    # (2) totals-row candidate cols vs atomic sums
    for c in cand_cols + [total_valid_col]:
        qa = float(atomic_sums[c]) if pd.notna(atomic_sums[c]) else 0.0
        official = float(df.loc[trow, c]) if pd.notna(df.loc[trow, c]) else 0.0
        if qa == official:
            result["checks"]["totals_row_candidate_cols_vs_atomic_sums"]["passed"].append({"key": str(c)})
        else:
            result["checks"]["totals_row_candidate_cols_vs_atomic_sums"]["failed"].append(_scalar_fail(str(c), qa, official))

    # (3) informal column sum vs totals informal
    qa_inf = float(atomic_sums[informal_col]) if pd.notna(atomic_sums[informal_col]) else 0.0
    official_inf = float(df.loc[trow, informal_col]) if pd.notna(df.loc[trow, informal_col]) else 0.0
    if qa_inf == official_inf:
        result["checks"]["informal_column_sum_vs_totals"]["passed"].append({"key": informal_col})
    else:
        result["checks"]["informal_column_sum_vs_totals"]["failed"].append(_scalar_fail(informal_col, qa_inf, official_inf))

    # (4) valid + informal extra row check
    qa_total_ballots = float(provided_total_valid + provided_total_inf)
    found = False
    for j in range(trow + 1, min(trow + 6, df.shape[0])):
        tv = df.loc[j, total_valid_col]
        inf = df.loc[j, informal_col]
        if pd.notna(tv) and (pd.isna(inf) or float(inf) == 0.0):
            found = True
            official = float(tv)
            if official == qa_total_ballots:
                result["checks"]["valid_plus_informal_row_check"]["passed"].append({"key": "valid_plus_informal"})
            else:
                result["checks"]["valid_plus_informal_row_check"]["failed"].append(_scalar_fail("valid_plus_informal", qa_total_ballots, official))
            break
    if not found:
        result["checks"]["valid_plus_informal_row_check"]["failed"].append({"key": "ROW_AFTER_TOTALS_NOT_FOUND"})

    # (5) roster totals vs atomic sums
    roster = extract_candidate_roster(df, trow)
    atomic_by_candidate = {str(c).strip(): float(atomic[c].sum(skipna=True)) for c in cand_cols}
    if roster.empty:
        result["checks"]["roster_candidate_totals_vs_atomic_sums"]["failed"].append({"key": "ROSTER_NOT_FOUND"})
    else:
        for _, r in roster.iterrows():
            name = str(r["candidate"]).strip()
            qa = atomic_by_candidate.get(name)
            official = float(r["total_candidate_votes"])
            if qa is None:
                result["checks"]["roster_candidate_totals_vs_atomic_sums"]["failed"].append({"key": f"MISSING_HEADER:{name}"})
                continue
            if qa == official:
                result["checks"]["roster_candidate_totals_vs_atomic_sums"]["passed"].append({"key": name})
            else:
                result["checks"]["roster_candidate_totals_vs_atomic_sums"]["failed"].append(_scalar_fail(name, float(qa), official))

    return result


def checksum_party_atomic_detailed(party_csv: Path) -> Dict[str, Any]:
    df = read_csv_atomic(party_csv)
    trow = find_totals_row(df)

    party_cols, total_valid_col, informal_col = party_numeric_cols(df)
    for c in party_cols + [total_valid_col, informal_col]:
        df[c] = _to_num(df[c])

    result: Dict[str, Any] = {
        "file": str(party_csv),
        "totals": None,
        "checks": {
            "voting_place_row_sum_vs_total_valid_column": {"passed": [], "failed": []},
            "informal_column_sum_vs_totals": {"passed": [], "failed": []},
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

    provided_total_valid = float(df.loc[trow, total_valid_col]) if pd.notna(df.loc[trow, total_valid_col]) else 0.0
    provided_total_inf = float(df.loc[trow, informal_col]) if pd.notna(df.loc[trow, informal_col]) else 0.0
    result["totals"] = totals_triplet(provided_total_valid, provided_total_inf)

    for i in range(atomic.shape[0]):
        loc = location_pair(df, i)
        qa_valid = float(atomic.iloc[i][party_cols].sum(skipna=True))
        official_valid = float(atomic.iloc[i][total_valid_col]) if pd.notna(atomic.iloc[i][total_valid_col]) else 0.0
        row_inf = float(atomic.iloc[i][informal_col]) if pd.notna(atomic.iloc[i][informal_col]) else 0.0

        row_payload = {"location": loc, "row_totals": totals_triplet(official_valid, row_inf)}
        if qa_valid == official_valid:
            result["checks"]["voting_place_row_sum_vs_total_valid_column"]["passed"].append(row_payload)
        else:
            result["checks"]["voting_place_row_sum_vs_total_valid_column"]["failed"].append({**row_payload, **_scalar_fail("valid_row_sum", qa_valid, official_valid)})

    atomic_sums = atomic[party_cols + [total_valid_col, informal_col]].sum(numeric_only=True)

    for c in party_cols + [total_valid_col]:
        qa = float(atomic_sums[c]) if pd.notna(atomic_sums[c]) else 0.0
        official = float(df.loc[trow, c]) if pd.notna(df.loc[trow, c]) else 0.0
        if qa == official:
            result["checks"]["totals_row_party_cols_vs_atomic_sums"]["passed"].append({"key": str(c)})
        else:
            result["checks"]["totals_row_party_cols_vs_atomic_sums"]["failed"].append(_scalar_fail(str(c), qa, official))

    qa_inf = float(atomic_sums[informal_col]) if pd.notna(atomic_sums[informal_col]) else 0.0
    official_inf = float(df.loc[trow, informal_col]) if pd.notna(df.loc[trow, informal_col]) else 0.0
    if qa_inf == official_inf:
        result["checks"]["informal_column_sum_vs_totals"]["passed"].append({"key": informal_col})
    else:
        result["checks"]["informal_column_sum_vs_totals"]["failed"].append(_scalar_fail(informal_col, qa_inf, official_inf))

    qa_total_ballots = float(provided_total_valid + provided_total_inf)
    found = False
    for j in range(trow + 1, min(trow + 6, df.shape[0])):
        tv = df.loc[j, total_valid_col]
        inf = df.loc[j, informal_col]
        if pd.notna(tv) and (pd.isna(inf) or float(inf) == 0.0):
            found = True
            official = float(tv)
            if official == qa_total_ballots:
                result["checks"]["valid_plus_informal_row_check"]["passed"].append({"key": "valid_plus_informal"})
            else:
                result["checks"]["valid_plus_informal_row_check"]["failed"].append(_scalar_fail("valid_plus_informal", qa_total_ballots, official))
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
            qa = atomic_by_party.get(name)
            official = float(r["total_party_votes"])
            if qa is None:
                result["checks"]["roster_party_totals_vs_atomic_sums"]["failed"].append({"key": f"MISSING_HEADER:{name}"})
                continue
            if qa == official:
                result["checks"]["roster_party_totals_vs_atomic_sums"]["passed"].append({"key": name})
            else:
                result["checks"]["roster_party_totals_vs_atomic_sums"]["failed"].append(_scalar_fail(name, float(qa), official))

    return result
