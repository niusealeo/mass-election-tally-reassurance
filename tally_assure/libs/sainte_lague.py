from __future__ import annotations

import math
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from .checksums import safe_mkdir

def base_name(col: str) -> str:
    return re.sub(r"\s*\(.*\)\s*$", "", str(col)).strip()


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

