from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def _strip_trailing_dot_zero(v):
    """For CSV writing: turn 123.0 -> '123', keep 123.5 -> '123.5'."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return v
    if isinstance(v, float) and float(v).is_integer():
        return int(v)
    return v


def _strip_trailing_zeros(v):
    """For CSV writing:

    - 123.0 -> 123
    - 123.5000 -> 123.5
    - keep decimal places only if they are non-zero
    """
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if float(v).is_integer():
            return int(v)
        # Use fixed precision then trim; keep as string to avoid binary float artifacts.
        return f"{v:.12f}".rstrip("0").rstrip(".")
    return v


def _format_df_numbers_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].apply(_strip_trailing_zeros)
    return out


def read_split_votes_csv(path: Path) -> pd.DataFrame:
    # split vote tables are usually clean UTF-8, but handle BOM
    return pd.read_csv(path, encoding="utf-8-sig")


def read_xls_sheet0(path: Path) -> pd.DataFrame:
    """Read sheet0 of a split-vote workbook (2002-era).

    Canonical 2002 split-vote structure (confirmed in project PDF + Aoraki sheet):
      - First 4 rows are preamble; the *header row is the 6th row* (0-based index 5).
      - Column 0 (often blank header) contains party names.
      - Column 1 is "Party Vote Totals" (counts).
      - Then repeating pairs of (count column, "% of party vote" column).
      - Tail pairs: Informals (count, %), Party Only (count, %).

    Note: legacy .xls requires xlrd==1.2.0; .xlsx can be read with openpyxl.
    """
    engine = "openpyxl" if path.suffix.lower() in [".xlsx", ".xlsm"] else "xlrd"
    # header=5 -> use the 6th row as header, implicitly skipping preamble.
    return pd.read_excel(path, sheet_name=0, header=5, engine=engine)

def _coerce_num_series(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        s2 = s.astype(str).str.replace(",", "", regex=False).str.strip()
    else:
        s2 = s
    return pd.to_numeric(s2, errors="coerce")

def process_2002_split_sheet_df_to_endstate(
    df: pd.DataFrame,
    atomic_party_totals: Dict[str, float],
    candidate_order: Optional[List[str]],
    out_csv: Path,
    atomic_candidate_totals: Optional[Dict[str, float]],
    party_to_candidate_names: Optional[Dict[str, str]] = None,
) -> None:
    """Convert 2002-style split votes sheet0 dataframe to our endstate CSV format.

    This is *deterministic* for the known 2002 sheet layout:
      - Preamble rows ignored by read_xls_sheet0(header=5).
      - Column 0: Party name (header often blank)
      - Column 1: Party Vote Totals (counts)
      - Then repeating pairs: (count col, "% of party vote" col)
      - Tail pairs: Informals (count, %), Party Only (count, %)
      - Sometimes a final totals row like "Party vote totals".

    Output requirements (current project spec):
      - Keep original column titles for candidate/Informals/Party Only columns.
      - Row order: party rows with candidates first ("Party (Candidate)"), then remaining rows.
      - Bottom rows:
          ... party rows ...
          Sum_from_split_vote_counts        (sum of all rows above, column-wise)
          Party vote totals                 (provided totals row from sheet, if present; otherwise omitted)
          QA sums from the candidate csv    (candidate totals from atomic candidate CSV)
          Consistent                        (bool iff the previous three rows are equal elementwise)
      - No trailing .0s in numeric outputs.
      - "Consistent" row result must be checkable downstream.
    """

    if df.shape[1] < 3:
        raise ValueError("Split-votes sheet too narrow to parse")

    # Identify / rename party column
    party_col = next((c for c in df.columns if str(c).strip().casefold() == "party"), None)
    if party_col is None:
        party_col = df.columns[0]
    df = df.rename(columns={party_col: "Party"})

    # Find Party Vote Totals column (authoritative per-row total party votes)
    cols = list(df.columns)
    party_total_col = next((c for c in cols[1:] if str(c).strip().casefold() == "party vote totals"), None)
    if party_total_col is None:
        raise ValueError('Expected a "Party Vote Totals" column in 2002 split-votes sheet')

    # Drop any leading non-data rows (sometimes blank lines remain after header)
    def _is_data_party(v: object) -> bool:
        if pd.isna(v):
            return False
        s = str(v).strip()
        if not s or s.casefold() == "nan":
            return False
        # ignore title-ish remnants
        if "part viii" in s.casefold() or "candidate vote" in s.casefold():
            return False
        return True

    first_party_row = None
    for i in range(df.shape[0]):
        if _is_data_party(df.loc[i, "Party"]):
            first_party_row = i
            break
    if first_party_row is None:
        raise ValueError("Could not locate first party row in 2002 split-votes sheet")
    df = df.iloc[first_party_row:].copy().reset_index(drop=True)

    # Coerce non-Party columns to numeric
    for c in df.columns[1:]:
        df[c] = _coerce_num_series(df[c])

    # Drop percent columns deterministically
    def _is_percent_col(col: str) -> bool:
        name = str(col).strip().casefold()
        name = re.sub(r"\.\d+$", "", name)  # pandas may dedupe dup headers
        if name in {"% of party vote", "total %"}:
            return True
        if "%" in name and "party vote" in name:
            return True
        return False

    nonparty_cols = [c for c in df.columns[1:] if not _is_percent_col(c)]
    # Keep original titles for count columns (including Informals / Party Only)
    # Candidate count columns are everything after Party Vote Totals.
    count_cols_after_total = [c for c in nonparty_cols if c != party_total_col]

    keep_cols = ["Party", party_total_col] + count_cols_after_total
    df = df[keep_cols].copy()

    # Standardise bookkeeping columns
    df = df.rename(columns={party_total_col: "Total Party Votes"})
    count_cols = count_cols_after_total[:]  # preserve original titles

    # Optional: decorate party row labels with candidate names from roster
    def norm_party(s: str) -> str:
        s = str(s).strip().casefold()
        s = re.sub(r"[^a-z0-9]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    # Map split party label -> atomic key (best-effort), then -> candidate name(s)
    explicit = {
        "act new zealand": "ACT",
        "new zealand first party": "NZ First",
        "jim anderton s progressive coalition": "Progressive Coalition",
        "aotearoa legalise cannabis party": "Legalise Cannabis",
        "mana maori movement": "Mana Maori",
        "outdoor recreation nz": "Outdoor Rec. NZ",
        "christian heritage party": "Christian Heritage",
    }

    def map_party_label(p: object) -> str:
        p0 = "" if pd.isna(p) else str(p).strip()
        n = norm_party(p0)
        # direct explicit map
        if n in explicit:
            return explicit[n]
        # try exact match against atomic party keys
        for k in atomic_party_totals.keys():
            if norm_party(k) == n:
                return k
        # substring match (conservative)
        for k in atomic_party_totals.keys():
            nk = norm_party(k)
            if nk and (nk in n or n in nk):
                return k
        return p0

    if party_to_candidate_names:
        decorated = []
        for p in df["Party"].tolist():
            p_txt = "" if pd.isna(p) else str(p).strip()
            if p_txt.casefold() in {"party vote totals"}:
                decorated.append(p_txt)
                continue
            atomic_key = map_party_label(p_txt)
            cand_names = party_to_candidate_names.get(atomic_key) or party_to_candidate_names.get(p_txt)
            if cand_names:
                decorated.append(f"{p_txt} ({cand_names})")
            else:
                decorated.append(p_txt)
        df["Party"] = decorated

    # Compute per-row Sum_from_split_vote_counts = sum of count columns (candidates + Informals + Party Only)
    for c in count_cols + ["Total Party Votes"]:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)
    df["Sum_from_split_vote_counts"] = df[count_cols].sum(axis=1)
    # QA party totals per row (from atomic party votes CSV), and per-row consistency across:
    #   Sum_from_split_vote_counts, Total Party Votes, QA_Total_Party_Votes_from_atomic_party
    qa_party_col = "QA_Total_Party_Votes_from_atomic_party"
    qa_vals: List[float] = []
    for p in df["Party"].tolist():
        p_txt = "" if pd.isna(p) else str(p).strip()
        p_base = p_txt.split(" (", 1)[0].strip() if " (" in p_txt else p_txt
        p_norm = p_base.casefold()
        if p_norm in {"informal votes", ""}:
            qa_vals.append(float("nan"))
            continue
        if p_norm == "party vote totals":
            qa_vals.append(float(sum(atomic_party_totals.values())) if atomic_party_totals else float("nan"))
            continue
        atomic_key = map_party_label(p_base)
        v = atomic_party_totals.get(atomic_key)
        qa_vals.append(float(v) if v is not None else float("nan"))
    df[qa_party_col] = qa_vals

    def _eq3_or_error(a: float, b: float, c: float) -> object:
        """Return True/False if comparable, else 'error'."""
        if pd.isna(a) or pd.isna(b) or pd.isna(c):
            return "error"
        try:
            return bool(float(a) == float(b) == float(c))
        except Exception:
            return "error"

    # Per-row consistent flag (string 'error' when not computable):
    #   Sum_from_split_vote_counts == Total Party Votes == QA_Total_Party_Votes_from_atomic_party
    df["consistent"] = [
        _eq3_or_error(a, b, c)
        for a, b, c in zip(df["Sum_from_split_vote_counts"], df["Total Party Votes"], df[qa_party_col])
    ]

    # Split out provided totals row if present (by label)

    party_norm = df["Party"].astype(str).str.strip()
    is_totals_row = party_norm.str.casefold() == "party vote totals"
    df_main = df.loc[~is_totals_row].copy()
    df_provided_totals = df.loc[is_totals_row].copy()

    # Row order: rows that have appended candidate parentheses first
    is_informal_votes = df_main["Party"].astype(str).str.strip().str.casefold() == "informal votes"
    has_candidate = df_main["Party"].astype(str).str.contains(r"\(.+\)") & ~is_informal_votes
    df_main["__has_candidate"] = has_candidate
    df_main["__orig"] = range(len(df_main))
    df_main = df_main.sort_values(by=["__has_candidate", "__orig"], ascending=[False, True]).drop(columns=["__has_candidate", "__orig"])

    # --- Summary rows
    numeric_cols = count_cols + ["Sum_from_split_vote_counts", "Total Party Votes", qa_party_col]

    # Sum_from_split_vote_counts summary row: sum of all rows above (df_main), column-wise.
    sum_row = {"Party": "Sum_from_split_vote_counts"}
    for c in numeric_cols:
        sum_row[c] = float(df_main[c].sum(skipna=True))

    # QA sums from candidate csv: populate candidate columns + Informals from atomic candidate totals.
    # Candidate CSV column headers are usually WITHOUT party in parentheses.
    qa_row = {"Party": "QA sums from the candidate csv"}
    # Build a lookup from candidate-csv header -> total
    cand_tot = atomic_candidate_totals or {}

    def _norm_name(x: str) -> str:
        x = str(x).strip()
        x = re.sub(r"\s+", " ", x)
        return x.casefold()

    # Precompute mapping split col -> candidate csv col (strip parenthetical)
    split_to_cand = {}
    for sc in count_cols:
        base = str(sc)
        base2 = base
        if "(" in base and base.rstrip().endswith(")"):
            base2 = base.rsplit("(", 1)[0].strip()
        split_to_cand[sc] = base2

    # Informals key in candidate totals can vary
    informal_key = None
    for k in list(cand_tot.keys()):
        if _norm_name(k) in {"informal candidate votes", "informals", "informal"}:
            informal_key = k
            break

    # Fill candidate columns
    for sc in count_cols:
        if _norm_name(sc).endswith("informals") or _norm_name(sc) in {"informals", "informal candidate votes"}:
            if informal_key is not None:
                qa_row[sc] = float(cand_tot.get(informal_key, float("nan")))
            else:
                qa_row[sc] = float("nan")
            continue
        if "party only" in _norm_name(sc):
            qa_row[sc] = float("nan")  # derived below
            continue

        cc = split_to_cand.get(sc, sc)
        # try exact
        v = None
        for ck in (cc, str(cc).strip(), " ".join(str(cc).split())):
            if ck in cand_tot:
                v = cand_tot[ck]
                break
        if v is None:
            # try casefold match
            for k, val in cand_tot.items():
                if _norm_name(k) == _norm_name(cc):
                    v = val
                    break
        qa_row[sc] = float(v) if v is not None and pd.notna(v) else float("nan")

    # Compute Party Only for QA row if possible (needs total party votes sum)
    total_party_votes_sum = float(sum(atomic_party_totals.values())) if atomic_party_totals else float("nan")

    # Candidate sum + informal for QA
    qa_candidate_sum = 0.0
    for sc in count_cols:
        nsc = _norm_name(sc)
        if "party only" in nsc:
            continue
        v = qa_row.get(sc, float("nan"))
        if pd.notna(v):
            qa_candidate_sum += float(v)

    if informal_key is not None and pd.notna(cand_tot.get(informal_key, float("nan"))):
        # ensure informal included (already included if split has Informals col)
        pass

    if pd.notna(total_party_votes_sum):
        party_only_val = total_party_votes_sum - qa_candidate_sum
        # assign to any split column containing "party only"
        for sc in count_cols:
            if "party only" in _norm_name(sc):
                qa_row[sc] = float(party_only_val)

    # Fill bookkeeping totals
    qa_row["Total Party Votes"] = total_party_votes_sum
    qa_row[qa_party_col] = total_party_votes_sum
    # Sum_from_split_vote_counts in QA row is sum of count cols (incl derived party only)
    qa_row["Sum_from_split_vote_counts"] = float(sum(float(qa_row.get(sc, 0.0)) for sc in count_cols if pd.notna(qa_row.get(sc, float("nan")))))

    # Consistent row (column-wise IFF checks + final corner check)
    # We compute:
    #   - Row-level 'consistent' values for all rows above (party rows + summary rows)
    #   - Last row 'Consistent': per-column consistency of the three summary rows
    #       (Sum_from_split_vote_counts row, Party vote totals row, QA sums from the candidate csv row)
    #   - Final corner cell (Consistent row, consistent column):
    #       TRUE  iff all row-level consistent values and all column-consistency values are TRUE
    #       FALSE iff any of those values are FALSE
    #       "error" if unable to compute any required value
    compare_cols = count_cols + ["Sum_from_split_vote_counts", "Total Party Votes", qa_party_col]

    def _float_or_nan(x) -> float:
        try:
            if pd.isna(x):
                return float("nan")
            return float(x)
        except Exception:
            return float("nan")

    def _eq_vals(vals: List[float]) -> object:
        if any(pd.isna(v) for v in vals):
            return "error"
        try:
            first = float(vals[0])
            return bool(all(float(v) == first for v in vals[1:]))
        except Exception:
            return "error"

    def _and_truth(values: List[object]) -> object:
        # Three-valued AND over True/False/"error"
        has_error = any((v == "error") or pd.isna(v) for v in values)
        if has_error:
            return "error"
        if any(v is False for v in values):
            return False
        return True

    def _row_consistent_from_dict(d: dict) -> object:
        a = _float_or_nan(d.get("Sum_from_split_vote_counts"))
        b = _float_or_nan(d.get("Total Party Votes"))
        c = _float_or_nan(d.get(qa_party_col))
        return _eq_vals([a, b, c])

    # Build provided totals row dict (if present)
    provided_row_dict: Optional[dict] = None
    if not df_provided_totals.empty:
        provided_row_dict = df_provided_totals.iloc[0].to_dict()

    # Attach row-level consistent values to summary rows
    sum_row["consistent"] = _row_consistent_from_dict(sum_row)
    if provided_row_dict is not None:
        provided_row_dict["consistent"] = _row_consistent_from_dict(provided_row_dict)
    qa_row["consistent"] = _row_consistent_from_dict(qa_row)

    # Column-wise consistency (last row) for each compare column
    consistent_row: Dict[str, object] = {"Party": "Consistent"}
    col_consistency_values: List[object] = []

    for c in compare_cols:
        if provided_row_dict is None:
            v = "error"
        else:
            v = _eq_vals([
                _float_or_nan(sum_row.get(c)),
                _float_or_nan(provided_row_dict.get(c)),
                _float_or_nan(qa_row.get(c)),
            ])
        consistent_row[c] = v
        col_consistency_values.append(v)

    # Final corner cell: AND(all row-level consistent values above, all column-consistency values)
    row_consistency_values: List[object] = []
    if "consistent" in df_main.columns:
        row_consistency_values.extend(df_main["consistent"].tolist())
    row_consistency_values.append(sum_row["consistent"])
    row_consistency_values.append(provided_row_dict.get("consistent") if provided_row_dict is not None else "error")
    row_consistency_values.append(qa_row["consistent"])

    corner = _and_truth(row_consistency_values + col_consistency_values)
    consistent_row["consistent"] = corner

    # Assemble final table
    out_rows: List[pd.DataFrame] = [df_main]

    out_rows.append(pd.DataFrame([sum_row], columns=["Party"] + count_cols + ["Sum_from_split_vote_counts", "Total Party Votes", qa_party_col, "consistent"]))

    if provided_row_dict is not None:
        out_rows.append(pd.DataFrame([provided_row_dict], columns=["Party"] + count_cols + ["Sum_from_split_vote_counts", "Total Party Votes", qa_party_col, "consistent"]))

    out_rows.append(pd.DataFrame([qa_row], columns=["Party"] + count_cols + ["Sum_from_split_vote_counts", "Total Party Votes", qa_party_col, "consistent"]))

    out_rows.append(pd.DataFrame([consistent_row], columns=["Party"] + count_cols + ["Sum_from_split_vote_counts", "Total Party Votes", qa_party_col, "consistent"]))

    out_df = pd.concat(out_rows, ignore_index=True, sort=False)

    # Column order: Party + count cols + bookkeeping + consistent
    out_df = out_df[["Party"] + count_cols + ["Sum_from_split_vote_counts", "Total Party Votes", qa_party_col, "consistent"]]

    # Format numbers but keep 'error' strings in place
    out_df = _format_df_numbers_for_csv(out_df)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False, encoding="utf-8")


def process_2002_split_xls_to_endstate(
    xls_path: Path,
    atomic_party_totals: Dict[str, float],
    candidate_order: Optional[List[str]],
    party_to_candidate_names: Optional[Dict[str, str]],
    out_csv: Path,
    atomic_candidate_totals: Optional[Dict[str, float]] = None,
) -> None:
    df = read_xls_sheet0(xls_path)
    process_2002_split_sheet_df_to_endstate(
        df,
        atomic_party_totals,
        candidate_order,
        party_to_candidate_names=party_to_candidate_names,
        out_csv=out_csv,
        atomic_candidate_totals=atomic_candidate_totals,
    )


