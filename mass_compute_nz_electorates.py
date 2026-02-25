
#!/usr/bin/env python3
"""
Mass computation for NZ electorate split-vote reconstruction + QA checksums.

Inputs (required):
- downloaded_hash_index.json
- electorates_by_term.json

You choose:
- --input-root  : root directory that contains the downloaded directory structure referenced by downloaded_hash_index.json "saved_to"
- --output-root : root directory where outputs will be written (mirrors relative paths of inputs); directories auto-created

Behavior by year/term:
- 2002 (term_47_(2002)): Port XLS split tables to the modern split-vote CSV layout (requires xlrd for .xls).
  (No Sainte-Laguë + puzzle in 2002 by design here.)
- 2005–2023: Build Sainte-Laguë start game state from split-vote percentages + party totals, then solve sliding puzzle to match atomic totals.

Per electorate, performs 3 checksum validations:
1) Candidate atomic totals: recompute by summing voting-place rows vs provided totals row in the same file
2) Party atomic totals: recompute by summing voting-place rows vs provided totals row in the same file
3) Party totals vs split-vote file total party votes (if the split file contains it)

Outputs (mirrors input structure):
- Electorate-level:
  - electorate{N}_checksum_pass_<timestamp>.csv  (only if any checks pass)
  - electorate{N}_checksum_fail_<timestamp>.csv  (only if any checks fail)
  - Start/finish game files for 2005–2023:
    - electorate{N}_start_game_state_with_actual_and_residual.csv
    - electorate{N}_finish_solution_matrix.csv
    - electorate{N}_finish_moves_steps.csv
    - electorate{N}_finish_moves_aggregated.csv
    - electorate{N}_finish_totals_comparison.csv
    - electorate{N}_finish_deviation.csv
    - electorate{N}_finish_variance.csv
  - 2002 only:
    - electorate{N}_ported_split_votes.csv

- Term-level:
  - term_checksum_pass_<timestamp>.csv (only if any electorates fully consistent)
  - term_checksum_fail_<timestamp>.csv (only if any electorates not fully consistent)

- Top-level (under output-root):
  - elections_checksum_pass_<timestamp>.csv (only if any terms fully consistent)
  - elections_checksum_fail_<timestamp>.csv (only if any terms not fully consistent)

Notes:
- This script is designed to be robust to filename variations by using downloaded_hash_index.json.
- For XLS (.xls) parsing (2002), xlrd is required. If xlrd is not installed, the script will raise a clear error.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------- timestamps ----------

def now_stamps() -> Tuple[str, str]:
    utc = datetime.now(timezone.utc)
    local = utc.astimezone()
    utc_s = utc.isoformat(timespec="seconds")
    local_s = local.strftime("%a %d %b %Y %H:%M:%S %Z%z")
    return utc_s, local_s


# ---------- helpers ----------

def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def rel_from_input(input_root: Path, abs_path: Path) -> Path:
    try:
        return abs_path.relative_to(input_root)
    except ValueError:
        return abs_path


def base_name(col: str) -> str:
    return re.sub(r"\s*\(.*\)\s*$", "", str(col)).strip()


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

def allocate_row(total: int, weights: np.ndarray, tol_q=1e-12, tol_dev=1e-12, tol_imp=1e-12):
    w = np.array(weights, dtype=float)
    w[w < 0] = 0.0
    wsum = float(w.sum())
    b = np.zeros(len(w), dtype=float)
    if total <= 0 or wsum <= 0:
        return b, 0, 0, 0, []

    p = w / wsum

    first_solved = 0
    second_solved = 0
    third_events = 0
    third_sizes: List[int] = []

    x = 0
    while x < total:
        q = w / (2.0 * b + 1.0)
        maxq = float(q.max())
        tied = np.flatnonzero(np.abs(q - maxq) <= tol_q)

        if len(tied) == 1:
            b[tied[0]] += 1.0
            x += 1
            continue

        post_share = (b[tied] + 1.0) / (x + 1.0)
        dev = post_share - p[tied]
        min_dev = float(dev.min())
        tied_dev = tied[np.abs(dev - min_dev) <= tol_dev]

        if len(tied_dev) == 1:
            b[tied_dev[0]] += 1.0
            x += 1
            first_solved += 1
            continue

        impacts = 1.0 / (b[tied_dev] + 1.0)
        if min_dev < -tol_dev:
            best = float(impacts.max())
            tied_imp = tied_dev[np.abs(impacts - best) <= tol_imp]
        elif min_dev > tol_dev:
            best = float(impacts.min())
            tied_imp = tied_dev[np.abs(impacts - best) <= tol_imp]
        else:
            tied_imp = tied_dev

        if len(tied_imp) == 1:
            b[tied_imp[0]] += 1.0
            x += 1
            second_solved += 1
            continue

        m = int(len(tied_imp))
        remaining = total - x
        t = int(min(m, remaining))
        b[tied_imp] += t * (1.0 / m)
        x += t
        third_events += 1
        third_sizes.append(m)

    return b, first_solved, second_solved, third_events, third_sizes


def build_start_game_state(split_path: Path, candidate_totals: pd.Series, party_totals: pd.Series,
                           out_path: Path) -> None:
    split = pd.read_csv(split_path)
    party_col = split.columns[0]
    has_total_party_votes = "Total Party Votes" in split.columns

    if has_total_party_votes:
        cols = list(split.columns[2:-1])
    else:
        cols = list(split.columns[1:-1])

    data_rows = split.iloc[:-1].copy()
    parties = data_rows[party_col].astype(str).tolist()

    totals = []
    for p in parties:
        v = party_totals.get(p, np.nan)
        totals.append(float(v) if pd.notna(v) else np.nan)
    if any(math.isnan(x) for x in totals):
        if has_total_party_votes:
            totals = data_rows["Total Party Votes"].astype(float).tolist()
        else:
            raise ValueError("Party totals missing and split file has no Total Party Votes column.")
    totals = np.array(totals, dtype=float)

    W = data_rows[cols].astype(float).to_numpy()

    alloc = np.zeros_like(W, dtype=float)
    fo=[]; so=[]; to=[]; ts=[]
    for i in range(W.shape[0]):
        b, fcnt, scnt, tcnt, sizes = allocate_row(int(round(totals[i])), W[i])
        alloc[i] = b
        fo.append(fcnt); so.append(scnt); to.append(tcnt); ts.append(",".join(map(str, sizes)))

    out = pd.DataFrame({party_col: parties, "Total Party Votes": totals})
    for j, c in enumerate(cols):
        out[c] = alloc[:, j]

    out["allocated_sum"] = out[cols].sum(axis=1)
    out["sum_matches_total"] = np.isclose(out["allocated_sum"], out["Total Party Votes"], atol=1e-9)
    out["first_order_ties_solved"] = fo
    out["second_order_ties_solved"] = so
    out["third_order_events"] = to
    out["third_order_sizes"] = ts

    sum_row = {party_col: "Sum_from_split_vote_percentages", "Total Party Votes": float(out["Total Party Votes"].sum())}
    for c in cols:
        sum_row[c] = float(out[c].sum())
    sum_row["allocated_sum"] = float(out["allocated_sum"].sum())
    sum_row["sum_matches_total"] = True
    for c in ["first_order_ties_solved","second_order_ties_solved","third_order_events","third_order_sizes"]:
        sum_row[c] = ""

    true_row = {party_col: "TRUE_TOTAL", "Total Party Votes": float(out["Total Party Votes"].sum())}
    tmp_sum = 0.0
    for c in cols:
        if c == "Party Vote Only":
            continue
        v = float(candidate_totals.get(base_name(c), candidate_totals.get(c, 0.0)))
        true_row[c] = v
        tmp_sum += v
    true_row["Party Vote Only"] = float(true_row["Total Party Votes"]) - tmp_sum
    true_row["allocated_sum"] = float(true_row["Total Party Votes"])
    true_row["sum_matches_total"] = True
    for c in ["first_order_ties_solved","second_order_ties_solved","third_order_events","third_order_sizes"]:
        true_row[c] = ""

    res_row = {party_col: "RESIDUAL_DEVIATION", "Total Party Votes": ""}
    for c in cols:
        res_row[c] = float(sum_row.get(c, 0.0)) - float(true_row.get(c, 0.0))
    res_row["allocated_sum"] = float(sum_row["allocated_sum"]) - float(true_row["allocated_sum"])
    res_row["sum_matches_total"] = ""
    for c in ["first_order_ties_solved","second_order_ties_solved","third_order_events","third_order_sizes"]:
        res_row[c] = ""

    out_full = pd.concat([out, pd.DataFrame([sum_row, true_row, res_row])], ignore_index=True)
    safe_mkdir(out_path.parent)
    out_full.to_csv(out_path, index=False, encoding="utf-8")


def solve_sliding_puzzle(start_path: Path, split_path: Path, candidate_totals: pd.Series,
                         out_prefix: Path) -> None:
    start = pd.read_csv(start_path)
    split = pd.read_csv(split_path)

    party_col = split.columns[0]
    votes_col = "Total Party Votes"
    cols = [c for c in start.columns if c not in [party_col, votes_col,
                                                  "allocated_sum","sum_matches_total",
                                                  "first_order_ties_solved","second_order_ties_solved",
                                                  "third_order_events","third_order_sizes"]]

    party_rows = start[~start[party_col].isin(["Sum_from_split_vote_percentages","TRUE_TOTAL","RESIDUAL_DEVIATION"])].copy()
    parties = party_rows[party_col].astype(str).tolist()
    row_totals = party_rows[votes_col].astype(float).to_numpy()
    X = party_rows[cols].astype(float).to_numpy()

    # p from split template
    has_total = "Total Party Votes" in split.columns
    if has_total:
        W = split.iloc[:-1][cols].astype(float).to_numpy()
    else:
        W = split.iloc[:-1][cols].astype(float).to_numpy()
    row_sums_W = W.sum(axis=1)
    row_sums_W[row_sums_W == 0] = 1.0
    p = W / row_sums_W[:, None]

    a = row_totals[:, None] * p
    row_w = 1.0 / np.maximum(row_totals, 1.0) ** 2

    targets = np.zeros(X.shape[1], dtype=float)
    pvo_j = cols.index("Party Vote Only")
    for j, col in enumerate(cols):
        if j == pvo_j:
            continue
        targets[j] = float(candidate_totals.get(base_name(col), candidate_totals.get(col, 0.0)))
    grand_total = float(row_totals.sum())
    targets[pvo_j] = grand_total - float(targets.sum() - targets[pvo_j])

    def col_totals(mat): return mat.sum(axis=0)

    def delta_cost(i, A, B, delta):
        xA = X[i, A]; xB = X[i, B]
        aiA = a[i, A]; aiB = a[i, B]
        w = row_w[i]
        dA0 = xA - aiA
        dB0 = xB - aiB
        dA1 = (xA - delta) - aiA
        dB1 = (xB + delta) - aiB
        return ((dA1*dA1 - dA0*dA0) + (dB1*dB1 - dB0*dB0)) * w

    tol = 1e-12
    steps = []
    iters = 0

    while True:
        residual = targets - col_totals(X)
        if np.max(np.abs(residual)) <= 1e-9:
            break

        deficit = np.where(residual > tol)[0]
        excess = np.where(residual < -tol)[0]
        if deficit.size == 0 or excess.size == 0:
            raise RuntimeError("Residuals exist but deficit/excess sets are empty.")

        absr = np.abs(residual)
        frac = absr - np.floor(absr + 1e-15)
        frac = np.where(frac < 1e-12, 0.0, frac)
        frac = np.where((1.0 - frac) < 1e-12, 0.0, frac)
        nonzero_fracs = frac[frac > 0.0]
        delta = float(nonzero_fracs.min()) if nonzero_fracs.size > 0 else 1.0
        if delta > 1.0:
            delta = 1.0

        best = None
        for A in excess:
            if (-residual[A]) + 1e-12 < delta:
                continue
            rows_can_give = np.where(X[:, A] >= delta - 1e-12)[0]
            if rows_can_give.size == 0:
                continue
            for B in deficit:
                if residual[B] + 1e-12 < delta:
                    continue
                for i in rows_can_give:
                    c = delta_cost(i, A, B, delta)
                    if best is None or c < best[0]:
                        best = (c, i, A, B)

        if best is None:
            raise RuntimeError(f"No feasible move found for delta={delta}.")

        c, i, A, B = best
        X[i, A] -= delta
        X[i, B] += delta
        if X[i, A] < 0 and X[i, A] > -1e-10:
            X[i, A] = 0.0

        iters += 1
        steps.append({
            "step": iters,
            "delta": delta,
            "party": parties[i],
            "from": cols[A],
            "to": cols[B],
            "delta_cost": float(c),
            "max_abs_residual_after": float(np.max(np.abs(targets - col_totals(X)))),
        })

    solved = pd.DataFrame(X, columns=cols)
    solved.insert(0, votes_col, row_totals)
    solved.insert(0, party_col, parties)

    tot = {party_col: "Sum_from_split_vote_percentages", votes_col: float(solved[votes_col].sum())}
    for ccol in cols:
        tot[ccol] = float(solved[ccol].sum())

    solved2 = pd.concat([solved, pd.DataFrame([tot])], ignore_index=True)

    comp = pd.DataFrame({
        "column": cols,
        "target_total": targets,
        "final_total": col_totals(X),
        "difference": col_totals(X) - targets,
    })

    shares_real = X / row_totals[:, None]
    dev = shares_real - p
    var = dev**2
    dev_df = pd.DataFrame(dev, columns=cols); dev_df.insert(0, party_col, parties)
    var_df = pd.DataFrame(var, columns=cols); var_df.insert(0, party_col, parties)

    steps_df = pd.DataFrame(steps)
    agg = steps_df.groupby(["party","from","to"], as_index=False).agg(
        units=("delta","sum"),
        moves=("delta","count"),
        total_cost=("delta_cost","sum"),
    ).sort_values(["units","moves"], ascending=False)

    safe_mkdir(out_prefix.parent)
    solved2.to_csv(str(out_prefix) + "_solution_matrix.csv", index=False, encoding="utf-8")
    steps_df.to_csv(str(out_prefix) + "_moves_steps.csv", index=False, encoding="utf-8")
    agg.to_csv(str(out_prefix) + "_moves_aggregated.csv", index=False, encoding="utf-8")
    comp.to_csv(str(out_prefix) + "_totals_comparison.csv", index=False, encoding="utf-8")
    dev_df.to_csv(str(out_prefix) + "_deviation.csv", index=False, encoding="utf-8")
    var_df.to_csv(str(out_prefix) + "_variance.csv", index=False, encoding="utf-8")


def port_2002_xls_to_split_csv(xls_path: Path, out_csv: Path) -> None:
    safe_mkdir(out_csv.parent)
    suffix = xls_path.suffix.lower()

    if suffix == ".xlsx":
        raw = pd.read_excel(xls_path, header=None)
        header_row = None
        for i in range(min(200, raw.shape[0])):
            row = raw.iloc[i].astype(str).str.lower().tolist()
            if any("party vote totals" in cell for cell in row):
                header_row = i
                break
        if header_row is None:
            raise ValueError(f"Could not find 'Party Vote Totals' header row in {xls_path.name}")
        df = pd.read_excel(xls_path, header=header_row)
    elif suffix == ".xls":
        try:
            import xlrd  # noqa
        except Exception as e:
            raise RuntimeError(
                "xlrd is required to read .xls files. Install it (pip install xlrd==1.2.0) or convert .xls to .xlsx."
            ) from e
        import xlrd
        book = xlrd.open_workbook(str(xls_path))
        sheet = book.sheet_by_index(0)
        header_row = None
        for r in range(min(200, sheet.nrows)):
            vals = [str(sheet.cell_value(r, c)).strip().lower() for c in range(sheet.ncols)]
            if any("party vote totals" in v for v in vals):
                header_row = r
                break
        if header_row is None:
            raise ValueError(f"Could not find 'Party Vote Totals' header row in {xls_path.name}")
        headers = [str(sheet.cell_value(header_row, c)).strip() for c in range(sheet.ncols)]
        rows = []
        for r in range(header_row + 1, sheet.nrows):
            first = str(sheet.cell_value(r, 0)).strip()
            if first == "" or first.lower().startswith("total"):
                break
            rows.append([sheet.cell_value(r, c) for c in range(sheet.ncols)])
        df = pd.DataFrame(rows, columns=headers)
    else:
        raise ValueError("Expected .xls or .xlsx for 2002 split file")

    cols = list(df.columns)
    party_col = cols[0]
    tcol = None
    for c in cols:
        if "party vote totals" in str(c).strip().lower():
            tcol = c
            break
    if tcol is None:
        raise ValueError("Could not locate 'Party Vote Totals' column after parsing")

    pct_cols = []
    for c in cols:
        if c in (party_col, tcol):
            continue
        if "%" in str(c):
            pct_cols.append(c)
    if not pct_cols:
        raise ValueError("No percentage columns found in 2002 split sheet")

    def clean_pct_name(c: str) -> str:
        s = str(c).strip()
        s = re.sub(r"%.*$", "", s).strip()
        return s

    out = pd.DataFrame()
    out[party_col] = df[party_col].astype(str)
    out["Total Party Votes"] = pd.to_numeric(df[tcol], errors="coerce").fillna(0).astype(float)

    for c in pct_cols:
        out[clean_pct_name(c)] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)

    out["Total %"] = out[[clean_pct_name(c) for c in pct_cols]].sum(axis=1)
    out.to_csv(out_csv, index=False, encoding="utf-8")


@dataclass
class ElectorateJob:
    termKey: str
    year: int
    electorateFolder: str
    electorateNumber: Optional[int]
    electorateName: Optional[str]
    split_path: Optional[Path]
    cand_path: Optional[Path]
    party_path: Optional[Path]


def parse_year(termKey: str) -> int:
    m = re.search(r"\((\d{4})\)", termKey)
    if not m:
        raise ValueError(f"Cannot parse year from termKey: {termKey}")
    return int(m.group(1))


def build_jobs(hash_index_path: Path, input_root: Path) -> List[ElectorateJob]:
    idx = json.loads(hash_index_path.read_text(encoding="utf-8"))

    groups: Dict[Tuple[str, str], List[dict]] = {}
    for _, meta in idx.items():
        termKey = meta.get("termKey")
        ef = meta.get("electorateFolder")
        saved = meta.get("saved_to")
        if not termKey or not saved or not ef:
            continue
        groups.setdefault((termKey, ef), []).append(meta)

    jobs: List[ElectorateJob] = []
    for (termKey, ef), metas in groups.items():
        year = parse_year(termKey)
        num = None; name = None
        m = re.match(r"(\d{3})_(.+)$", ef)
        if m:
            num = int(m.group(1))
            name = m.group(2).replace("_", " ")

        split_path = None
        cand_path = None
        party_path = None

        for meta in metas:
            p = Path(meta["saved_to"])
            abs_p = (input_root / p) if not p.is_absolute() else p
            fn = abs_p.name.lower()
            if "split" in fn or fn.endswith(".xls") or fn.endswith(".xlsx"):
                split_path = abs_p
            elif "cand" in fn:
                cand_path = abs_p
            elif "party" in fn:
                party_path = abs_p

        jobs.append(ElectorateJob(termKey, year, ef, num, name, split_path, cand_path, party_path))

    return jobs


def checksum_electorate(job: ElectorateJob, out_dir: Path) -> Tuple[bool, bool]:
    utc_s, local_s = now_stamps()
    elect_num = job.electorateNumber if job.electorateNumber is not None else 0
    elect_tag = f"electorate{elect_num}"

    rows_pass = []
    rows_fail = []

    # candidate atomic
    cand_pass_cols=[]; cand_fail_cols=[]
    cand_totals_computed = pd.Series(dtype=float)

    if job.cand_path and job.cand_path.exists():
        df = pd.read_csv(job.cand_path)
        trow = find_totals_row(df)
        cand_totals_computed = sum_excluding_totals(df, trow)
        if trow is not None:
            cand_totals_provided = df.loc[trow, cand_totals_computed.index]
            cand_pass_cols, cand_fail_cols = compare_series(cand_totals_computed, cand_totals_provided, tol=0.0)
        else:
            cand_fail_cols = list(cand_totals_computed.index)
    else:
        cand_fail_cols = ["FILE_MISSING"]

    # party atomic
    party_pass_cols=[]; party_fail_cols=[]
    party_totals_computed = pd.Series(dtype=float)

    if job.party_path and job.party_path.exists():
        dfp = pd.read_csv(job.party_path)
        trowp = find_totals_row(dfp)
        party_totals_computed = sum_excluding_totals(dfp, trowp)
        if trowp is not None:
            party_totals_provided = dfp.loc[trowp, party_totals_computed.index]
            party_pass_cols, party_fail_cols = compare_series(party_totals_computed, party_totals_provided, tol=0.0)
        else:
            party_fail_cols = list(party_totals_computed.index)
    else:
        party_fail_cols = ["FILE_MISSING"]

    # split totals check
    split_pass_cols=[]; split_fail_cols=[]
    if job.split_path and job.split_path.exists() and len(party_totals_computed.index) > 0:
        if job.split_path.suffix.lower() == ".csv":
            s = pd.read_csv(job.split_path)
            if "Total Party Votes" in s.columns:
                party_col = s.columns[0]
                split_party_totals = s.set_index(party_col)["Total Party Votes"]
                # drop trailing total row if present
                if s.shape[0] >= 2 and "total" in str(s.iloc[-1,0]).lower():
                    split_party_totals = split_party_totals.iloc[:-1]
                shared = [p for p in party_totals_computed.index if p in split_party_totals.index]
                for p in shared:
                    if float(party_totals_computed[p]) == float(split_party_totals[p]):
                        split_pass_cols.append(p)
                    else:
                        split_fail_cols.append(p)
            else:
                split_fail_cols = ["NO_TOTAL_PARTY_VOTES_COLUMN"]
        else:
            split_fail_cols = ["SPLIT_NOT_CSV"]
    else:
        split_fail_cols = ["FILE_MISSING"]

    def add_check(kind: str, passed: List[str], failed: List[str]):
        if passed:
            rows_pass.append({
                "termKey": job.termKey, "year": job.year,
                "electorateFolder": job.electorateFolder,
                "electorateNumber": job.electorateNumber,
                "electorateName": job.electorateName,
                "check": kind,
                "passed_items": ";".join(passed),
                "failed_items": "",
                "written_utc": utc_s,
                "written_local": local_s,
            })
        if failed:
            rows_fail.append({
                "termKey": job.termKey, "year": job.year,
                "electorateFolder": job.electorateFolder,
                "electorateNumber": job.electorateNumber,
                "electorateName": job.electorateName,
                "check": kind,
                "passed_items": "",
                "failed_items": ";".join(failed),
                "written_utc": utc_s,
                "written_local": local_s,
            })

    add_check("candidate_atomic_vs_provided_totals", cand_pass_cols, cand_fail_cols)
    add_check("party_atomic_vs_provided_totals", party_pass_cols, party_fail_cols)
    add_check("party_atomic_vs_split_total_party_votes", split_pass_cols, split_fail_cols)

    any_pass = bool(rows_pass)
    any_fail = bool(rows_fail)
    safe_mkdir(out_dir)
    stamp = utc_s.replace(":","").replace("-","")
    if rows_pass:
        pd.DataFrame(rows_pass).to_csv(out_dir / f"{elect_tag}_checksum_pass_{stamp}.csv", index=False, encoding="utf-8")
    if rows_fail:
        pd.DataFrame(rows_fail).to_csv(out_dir / f"{elect_tag}_checksum_fail_{stamp}.csv", index=False, encoding="utf-8")
    return any_pass, any_fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True, help="Folder that CONTAINS the domain folder (e.g. contains electionresults.govt.nz/)")
    ap.add_argument("--output-root", required=True, help="Where to write outputs; mirrors input structure")
    ap.add_argument("--downloaded-hash-index", required=True, help="Path to downloaded_hash_index.json")
    ap.add_argument("--terms", nargs="*", default=None, help="Optional list of termKey(s) to process")
    ap.add_argument("--min-year", type=int, default=2002, help="Ignore terms before this year (default: 2002)")
    ap.add_argument("--max-year", type=int, default=None, help="Ignore terms after this year")
    args = ap.parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    hash_index_path = Path(args.downloaded_hash_index).resolve()

    jobs = build_jobs(hash_index_path, input_root)
    if args.terms:
        jobs = [j for j in jobs if j.termKey in set(args.terms)]
    jobs = [j for j in jobs if j.year >= args.min_year and (args.max_year is None or j.year <= args.max_year)]

    jobs_by_term: Dict[str, List[ElectorateJob]] = {}
    for j in jobs:
        jobs_by_term.setdefault(j.termKey, []).append(j)

    top_pass_terms=[]; top_fail_terms=[]
    utc_top, local_top = now_stamps()
    processed_electorates_total = 0

    for termKey, term_jobs in sorted(jobs_by_term.items(), key=lambda x: x[0]):
        term_pass=[]; term_fail=[]

        for job in sorted(term_jobs, key=lambda j: (j.electorateNumber or 0, j.electorateFolder)):
            in_dir = None
            for p in [job.split_path, job.cand_path, job.party_path]:
                if p and p.exists():
                    in_dir = p.parent
                    break
            if in_dir is None:
                continue
            rel_dir = rel_from_input(input_root, in_dir)
            out_dir = output_root / rel_dir

            _, any_fail = checksum_electorate(job, out_dir)
            processed_electorates_total += 1

            rec = {
                "termKey": termKey, "year": job.year,
                "electorateFolder": job.electorateFolder,
                "electorateNumber": job.electorateNumber,
                "electorateName": job.electorateName,
                "written_utc": utc_top,
                "written_local": local_top,
            }
            if not any_fail:
                rec["status"]="PASS"
                term_pass.append(rec)
            else:
                rec["status"]="FAIL"
                term_fail.append(rec)

            # game outputs
            if job.year == 2002:
                if job.split_path and job.split_path.exists():
                    out_ported = out_dir / f"electorate{job.electorateNumber}_ported_split_votes.csv"
                    try:
                        port_2002_xls_to_split_csv(job.split_path, out_ported)
                    except Exception as e:
                        (out_dir / f"electorate{job.electorateNumber}_ported_split_votes_ERROR.txt").write_text(str(e), encoding="utf-8")
                continue

            if not (job.split_path and job.cand_path and job.party_path):
                continue
            if not (job.split_path.exists() and job.cand_path.exists() and job.party_path.exists()):
                continue
            if job.split_path.suffix.lower() != ".csv":
                continue

            cand_df = pd.read_csv(job.cand_path)
            party_df = pd.read_csv(job.party_path)
            cand_totals = sum_excluding_totals(cand_df, find_totals_row(cand_df))
            party_totals = sum_excluding_totals(party_df, find_totals_row(party_df))

            start_path = out_dir / f"electorate{job.electorateNumber}_start_game_state_with_actual_and_residual.csv"
            try:
                build_start_game_state(job.split_path, cand_totals, party_totals, start_path)
            except Exception as e:
                (out_dir / f"electorate{job.electorateNumber}_start_game_state_ERROR.txt").write_text(str(e), encoding="utf-8")
                continue

            try:
                solve_sliding_puzzle(start_path, job.split_path, cand_totals, out_dir / f"electorate{job.electorateNumber}_finish")
            except Exception as e:
                (out_dir / f"electorate{job.electorateNumber}_finish_ERROR.txt").write_text(str(e), encoding="utf-8")

# term-level outputs (only if we processed at least one electorate in this term)
if term_pass or term_fail:
    # Mirror the term directory: term folder is the parent of electorate folders.
    term_dir_in = None
    for j in term_jobs:
        for p in [j.split_path, j.cand_path, j.party_path]:
            if p and p.exists():
                term_dir_in = p.parent.parent  # electorate dir parent = term dir
                break
        if term_dir_in:
            break

    out_term_dir = output_root / (rel_from_input(input_root, term_dir_in) if term_dir_in else Path(termKey))
    safe_mkdir(out_term_dir)

    utc_s, local_s = now_stamps()
    stamp = utc_s.replace(":","").replace("-","")
    if term_pass:
        pd.DataFrame(term_pass).to_csv(out_term_dir / f"term_checksum_pass_{stamp}.csv", index=False, encoding="utf-8")
    if term_fail:
        pd.DataFrame(term_fail).to_csv(out_term_dir / f"term_checksum_fail_{stamp}.csv", index=False, encoding="utf-8")

    if not term_fail:
        top_pass_terms.append({"termKey": termKey, "status":"PASS", "written_utc": utc_s, "written_local": local_s})
    else:
        top_fail_terms.append({"termKey": termKey, "status":"FAIL", "written_utc": utc_s, "written_local": local_s})

safe_mkdir(output_root)
utc_s, local_s = now_stamps()
stamp = utc_s.replace(":","").replace("-","")

if processed_electorates_total == 0:
    (output_root / f"NO_ELECTORATES_PROCESSED_{stamp}.txt").write_text(
        "No electorates were processed.\n\n"
        "Most likely causes:\n"
        "1) --input-root does not match the prefix of saved_to paths in downloaded_hash_index.json\n"
        "2) The downloaded_hash_index.json points at a different crawl/layout than your local downloads folder\n"
        "3) Files referenced in saved_to are missing on disk\n\n"
        "Quick check:\n"
        "- Open downloaded_hash_index.json\n"
        "- Copy one 'saved_to' value\n"
        "- Verify this exists: (input_root / saved_to)\n",
        encoding="utf-8"
    )
else:
    if top_pass_terms:
        pd.DataFrame(top_pass_terms).to_csv(output_root / f"elections_checksum_pass_{stamp}.csv", index=False, encoding="utf-8")
    if top_fail_terms:
        pd.DataFrame(top_fail_terms).to_csv(output_root / f"elections_checksum_fail_{stamp}.csv", index=False, encoding="utf-8")

print("Done.")


if __name__ == "__main__":
    main()
