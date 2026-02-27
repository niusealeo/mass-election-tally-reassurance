from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

# Import shared atomic parsers / helpers from the core checksums module.
# This module is imported lazily from tally_assure.checksums to avoid import cycles.
from ...checksums import _read_atomic_candidate_totals, _read_atomic_party_totals, fail_kv


def checksum_splitvote_endstate_2002(
    split_endstate_csv: Path,
    candidate_csv: Path,
    party_csv: Path,
) -> Dict[str, Any]:
    """Checks the endstate splitvote matrix against row/col totals and atomic totals (2002-era)."""
    mat = pd.read_csv(split_endstate_csv)

    required = {
        "Party",
        "Sum_from_split_vote_counts",
        "Total Party Votes",
        "QA_Total_Party_Votes_from_atomic_party",
        "consistent",
    }
    if not required.issubset(set(mat.columns)):
        return {
            "file": str(split_endstate_csv),
            "error": f"Missing required columns: {sorted(required - set(mat.columns))}",
            "checks": {},
        }

    # Separate main rows from bottom summary rows
    def is_summary_party(p: str) -> bool:
        p = str(p).strip().lower()
        return p in {
            "sum_from_split_vote_counts",
            "party vote totals",
            "qa sums from the candidate csv",
            "consistent",
        }

    main = mat[~mat["Party"].astype(str).apply(is_summary_party)].copy()

    # Candidate/informal/partyonly columns are everything except the final 4 bookkeeping cols.
    bookkeeping = [
        "Sum_from_split_vote_counts",
        "Total Party Votes",
        "QA_Total_Party_Votes_from_atomic_party",
        "consistent",
    ]
    cand_cols = [c for c in mat.columns if c not in ["Party"] + bookkeeping]

    for c in cand_cols + bookkeeping[0:3]:
        main.loc[:, c] = pd.to_numeric(main[c], errors="coerce").fillna(0.0)

    # Row consistency
    row_fail = []
    for _, r in main.iterrows():
        party = str(r["Party"]).strip()
        qa_sum = float(r[cand_cols].sum())
        split_total = float(r["Total Party Votes"])
        atomic_total = float(r["QA_Total_Party_Votes_from_atomic_party"])
        ok = (qa_sum == split_total) and (qa_sum == atomic_total)
        if not ok:
            row_fail.append(
                {
                    "party": party,
                    "qa_sum_of_row": qa_sum,
                    "split_total_party_votes": split_total,
                    "atomic_party_total": atomic_total,
                }
            )

    # Column sums vs atomic candidate totals
    atomic_cands = _read_atomic_candidate_totals(candidate_csv)
    col_fail = []
    col_pass = []
    for c in cand_cols:
        key = str(c).strip()
        if key not in atomic_cands:
            continue
        qa = float(main[c].sum())
        official = float(atomic_cands[key])
        if qa == official:
            col_pass.append({"key": key})
        else:
            col_fail.append(fail_kv(key, qa, official))

    # Party totals vs atomic party totals
    atomic_party = _read_atomic_party_totals(party_csv)
    party_fail = []
    party_pass = []
    for _, r in main.iterrows():
        party = str(r["Party"]).strip()
        party_key = party.split(" (", 1)[0].strip() if " (" in party else party
        if party_key not in atomic_party:
            continue
        qa = float(r["Total Party Votes"])
        official = float(atomic_party[party_key])
        if qa == official:
            party_pass.append({"party": party})
        else:
            party_fail.append(fail_kv(party, qa, official))

    def find_row(label: str) -> Optional[pd.Series]:
        hit = mat[mat["Party"].astype(str).str.strip().str.lower() == label.lower()]
        return hit.iloc[0] if len(hit) else None

    qa_row = find_row("Sum_from_split_vote_counts")
    prov_row = find_row("Party vote totals")
    totals_row_fail = []
    totals_row_pass = []
    if qa_row is not None and prov_row is not None:
        for c in cand_cols + ["Total Party Votes"]:
            qa = float(pd.to_numeric(qa_row[c], errors="coerce") or 0.0)
            official = float(pd.to_numeric(prov_row[c], errors="coerce") or 0.0)
            if qa == official:
                totals_row_pass.append({"key": c})
            else:
                totals_row_fail.append(fail_kv(c, qa, official))

    # Final Consistent row (bool) must be included in pass/fail output.
    consistent_row = find_row("Consistent")
    consistent_checks_pass = []
    consistent_checks_fail = []
    if consistent_row is None:
        consistent_checks_fail.append({"key": "CONSISTENT_ROW_NOT_FOUND"})
    else:
        val = (
            consistent_row.get("consistent")
            if isinstance(consistent_row, pd.Series)
            else None
        )
        is_true = bool(val) if pd.notna(val) else False
        if is_true:
            consistent_checks_pass.append({"key": "Consistent", "value": True})
        else:
            consistent_checks_fail.append({"key": "Consistent", "value": False})

    return {
        "file": str(split_endstate_csv),
        "checks": {
            "splitvote_row_consistency_bool": {"passed": [], "failed": row_fail},
            "splitvote_candidate_column_sums_vs_atomic_candidate_totals": {"passed": col_pass, "failed": col_fail},
            "splitvote_party_totals_vs_atomic_party_totals": {"passed": party_pass, "failed": party_fail},
            "splitvote_provided_totals_row_vs_qa_totals_row": {"passed": totals_row_pass, "failed": totals_row_fail},
            "splitvote_endstate_consistent_row": {"passed": consistent_checks_pass, "failed": consistent_checks_fail},
        },
    }
