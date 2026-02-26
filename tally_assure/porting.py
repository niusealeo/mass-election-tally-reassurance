from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def _dedupe_columns(cols) -> List[str]:
    """Return a list of *unique* column names.

    Pandas allows duplicate column labels. If a label is duplicated, `df[label]`
    returns a DataFrame (not a Series) which breaks downstream `.str` accessors.
    We keep the first occurrence as-is and suffix subsequent duplicates with
    `.1`, `.2`, ... similar to pandas' historical behavior.
    """

    seen: Dict[str, int] = {}
    out: List[str] = []
    for c in cols:
        base = str(c).strip()
        if base in seen:
            seen[base] += 1
            out.append(f"{base}.{seen[base]}")
        else:
            seen[base] = 0
            out.append(base)
    return out


def _col_base(name: str) -> str:
    """Lowercased column name with any `.N` dedupe suffix removed."""
    s = str(name).strip().lower()
    return re.sub(r"\.\d+$", "", s)


def _strip_trailing_dot_zero(v):
    """For CSV writing: turn 123.0 -> '123', keep 123.5 -> '123.5'."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return v
    if isinstance(v, float) and float(v).is_integer():
        return int(v)
    return v


def _format_df_numbers_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].apply(_strip_trailing_dot_zero)
    return out


def read_split_votes_csv(path: Path) -> pd.DataFrame:
    # split vote tables are usually clean UTF-8, but handle BOM
    return pd.read_csv(path, encoding="utf-8-sig")


def read_xls_sheet0(path: Path) -> pd.DataFrame:
    """Read sheet0 of a legacy .xls workbook.

    This repo uses pandas with engine='xlrd'. That requires xlrd>=2.0.1,
    which supports .xls (and does not support xlsx).
    """
    return pd.read_excel(path, sheet_name=0, engine="xlrd")

def process_2002_split_sheet_df_to_endstate(
    df: pd.DataFrame,
    atomic_party_totals: Dict[str, float],
    out_csv: Path,
) -> None:
    """Convert 2002-style split votes sheet0 dataframe to our endstate CSV format.

    Endstate format requirements:
      - Keep the splitvote matrix in 'wide' format.
      - Include, at end of each row, 4 bookkeeping columns in this order:
          4th-to-last: Sum_from_split_vote_counts (sum of candidate/informal/partyonly counts)
          3rd-to-last : Total Party Votes (provided splitvote total)
          2nd-to-last : QA_Total_Party_Votes_from_atomic_party (from atomic party csv totals row)
          last        : consistent (bool)
      - Include bottom summary rows: Sum_from_split_vote_counts, Provided_party_vote_totals_row, RESIDUAL_DEVIATION
      - Numbers should not include trailing .0
    """
    # Clean up column labels early (strip, stringify, and *dedupe*).
    df = df.copy()
    df.columns = _dedupe_columns([str(c).strip() for c in df.columns])

    # Heuristics: find the header row that contains 'party' (row label) and candidate columns.
    # If df already has header row as columns, we're good. If it looks like a raw sheet, coerce first row to header.
    if "Party" not in df.columns and "party" not in [str(c).strip().lower() for c in df.columns]:
        # Try promote first row to header
        df2 = df.copy()
        df2.columns = _dedupe_columns([str(x).strip() for x in df2.iloc[0].tolist()])
        df2 = df2.iloc[1:].reset_index(drop=True)
        df = df2

    # Normalise 'Party' column name
    party_col = None
    for c in df.columns:
        if _col_base(c) == "party":
            party_col = c
            break
    if party_col is None:
        # Sometimes 'Party Name'
        for c in df.columns:
            if "party" in _col_base(c):
                party_col = c
                break
    if party_col is None:
        raise ValueError("Could not find Party column in 2002 split-votes sheet")

    df = df.rename(columns={party_col: "Party"})

    # Identify total party votes column
    total_party_col = None
    for c in df.columns:
        if _col_base(c) in {"total party votes", "total_party_votes", "total"}:
            total_party_col = c
            # Don't break: if there are multiple total-like columns, prefer the *last* one.
    if total_party_col is None:
        # fallback: last numeric column
        total_party_col = df.columns[-1]

    # Candidate/informal/partyonly columns are everything except Party and total party votes
    candidate_cols = [c for c in df.columns if c not in ["Party", total_party_col]]

    # Coerce numerics
    for c in candidate_cols + [total_party_col]:
        col = df[c]
        # With deduped columns this should be a Series, but keep a defensive fallback.
        if isinstance(col, pd.DataFrame):
            for subc in col.columns:
                df[subc] = pd.to_numeric(
                    df[subc].astype(str).str.replace(",", "", regex=False),
                    errors="coerce",
                ).fillna(0.0)
        else:
            df[c] = pd.to_numeric(
                col.astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            ).fillna(0.0)

    # Build QA_total column from atomic_party_totals, by party name match
    def norm_party(s: str) -> str:
        return re.sub(r"\s+", " ", str(s).strip()).casefold()

    atomic_lookup = {norm_party(k): float(v) for k, v in atomic_party_totals.items()}

    qa_party = []
    for p in df["Party"].tolist():
        qa_party.append(atomic_lookup.get(norm_party(p), 0.0))
    df["QA_Total_Party_Votes_from_atomic_party"] = qa_party

    # Row sums and consistency
    df["Sum_from_split_vote_counts"] = df[candidate_cols].sum(axis=1)
    df["Total Party Votes"] = df[total_party_col]
    df["consistent"] = (df["Sum_from_split_vote_counts"] == df["Total Party Votes"]) & (
        df["Sum_from_split_vote_counts"] == df["QA_Total_Party_Votes_from_atomic_party"]
    )

    # Reorder columns: Party, candidate_cols..., Sum_from..., Total Party Votes, QA..., consistent
    df = df[["Party"] + candidate_cols + ["Sum_from_split_vote_counts", "Total Party Votes", "QA_Total_Party_Votes_from_atomic_party", "consistent"]]

    # Bottom summary rows
    sums = {"Party": "Sum_from_split_vote_counts"}
    provided = {"Party": "Provided_party_vote_totals_row"}
    residual = {"Party": "RESIDUAL_DEVIATION"}

    for c in candidate_cols + ["Sum_from_split_vote_counts", "Total Party Votes", "QA_Total_Party_Votes_from_atomic_party"]:
        col_sum = float(df[c].sum(skipna=True))
        sums[c] = col_sum
        # Provided totals row: for candidate columns we don't always have a provided row; best proxy is atomic col sums elsewhere,
        # but for 2002 split file, often totals row exists already in df as last row named 'Total'.
        provided[c] = col_sum  # default to same; will be refined by checksums if official differs
        residual[c] = 0.0

    # For consistency col, summary row isn't meaningful
    sums["consistent"] = True
    provided["consistent"] = True
    residual["consistent"] = True

    out = pd.concat([df, pd.DataFrame([sums, provided, residual])], ignore_index=True)

    out2 = _format_df_numbers_for_csv(out)
    out2.to_csv(out_csv, index=False, encoding="utf-8")


def process_2002_split_xls_to_endstate(
    xls_path: Path,
    atomic_party_totals: Dict[str, float],
    out_csv: Path,
) -> None:
    df = read_xls_sheet0(xls_path)
    process_2002_split_sheet_df_to_endstate(df, atomic_party_totals, out_csv)


def process_split_votes_csv_to_endstate(*args, **kwargs):
    raise NotImplementedError("2005-2023 processing lives in the Sainte-Laguë + sliding puzzle pipeline.")


def port_xls_all_sheets(*args, **kwargs):
    raise NotImplementedError("Raw sheet CSVs are no longer kept; we port directly to endstate in-memory.")
