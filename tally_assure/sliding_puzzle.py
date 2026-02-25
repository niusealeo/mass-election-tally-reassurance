from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from .checksums import safe_mkdir

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

