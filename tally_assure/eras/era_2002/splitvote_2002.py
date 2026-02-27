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
    # --- 2002 split XLS structure (per project PDF):
    # Party | Party Vote Totals | (Candidate Count, Candidate % of party vote)* | Informals (count, %) | Party Only (count, %) | (sometimes other % columns)
    # We must:
    #   1) find the real header row (the sheet has a preamble)
    #   2) keep ONLY the *count* columns (drop % columns)
    #   3) label candidate columns deterministically (prefer atomic candidate column order)
    #   4) treat Party Vote Totals (count) as "Total Party Votes" in the endstate.

    # df is expected to already have its real header row applied (header=5 in read_xls_sheet0).
    # The first four rows are preamble and are ignored; the column names live in the 6th row.
    # We do not attempt to rediscover headers heuristically here.
    if df.shape[1] < 3:
        raise ValueError('Split-votes sheet too narrow to parse')
    # Ensure we have the Party Vote Totals column (authoritative per-row total).
    if not any(str(c).strip().casefold() == 'party vote totals' for c in df.columns[1:]):
        raise ValueError('Expected a "Party Vote Totals" column in 2002 split-votes sheet')

    party_col = next((c for c in df.columns if str(c).strip().casefold() == "party"), None)
    if party_col is None:
        party_col = df.columns[0]
    df = df.rename(columns={party_col: "Party"})

    # Note: we decorate the party row labels *after* normalising party labels,
    # so the decoration keys match downstream atomic comparisons.

    # Trim leading preamble rows so the first row is a party label
    def _is_data_party(v: object) -> bool:
        if pd.isna(v):
            return False
        s = str(v).strip()
        if not s or s.casefold() == "nan":
            return False
        # header-ish or title-ish rows
        if "split" in s.casefold() and "party" in s.casefold():
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

    cols = list(df.columns)
    if len(cols) < 3:
        raise ValueError("Split-votes sheet too narrow to parse")

    # In 2002 sheets the Party Vote Totals is the first numeric column after Party.
    party_total_col = next((c for c in cols[1:] if str(c).strip().casefold() == 'party vote totals'), cols[1])

    # Coerce all non-Party columns to numeric best-effort
    for c in cols[1:]:
        df[c] = _coerce_num_series(df[c])

    # Identify percent columns and drop them.
    # Identify percent columns and drop them.
    # 2002 split sheets repeat the literal header "% of party vote" for each candidate and for Informals/Party Only.
    def _is_percent_col(col: str) -> bool:
        name = str(col).strip().casefold()
        # pandas may dedupe duplicate headers as '.1', '.2' depending on version/engine
        name = re.sub(r"\.\d+$", "", name)
        if name in {"% of party vote", "total %"}:
            return True
        if "%" in name and "party vote" in name:
            return True
        return False

    nonparty_cols = cols[1:]
    count_cols = [c for c in nonparty_cols if not _is_percent_col(c)]

    # count_cols includes Party Vote Totals (counts) + candidate/informal/party-only count columns.
    after_total = [c for c in count_cols if c != party_total_col]

    # Keep original column titles in the resulting split vote file.
    # (Do not rename using candidate_order.)
    candidate_cols = list(after_total)

    # Keep only Party + counts
    keep_cols = ["Party", party_total_col] + candidate_cols
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy().fillna(0.0)

    # Build QA_total column from atomic_party_totals.
    # 2002 split sheets often use long party names while atomic totals use abbreviations.
    def norm_party(s: str) -> str:
        s = str(s).strip().casefold()
        s = re.sub(r"[^a-z0-9]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

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
        if n in explicit and explicit[n] in atomic_party_totals:
            return explicit[n]
        for k, target in [
            ("act", "ACT"),
            ("labour", "Labour Party"),
            ("national", "National Party"),
            ("green", "Green Party"),
            ("united future", "United Future"),
            ("alliance", "Alliance"),
            ("legalise cannabis", "Legalise Cannabis"),
            ("progressive", "Progressive Coalition"),
            ("new zealand first", "NZ First"),
            ("one nz", "OneNZ Party"),
            ("outdoor", "Outdoor Rec. NZ"),
            ("christian heritage", "Christian Heritage"),
            ("mana maori", "Mana Maori"),
            ("nmp", "NMP"),
        ]:
            if k in n and target in atomic_party_totals:
                return target
        # If already an atomic label, keep it
        if p0 in atomic_party_totals:
            return p0
        return p0

    # Keep the split sheet party label for display/output, but map to atomic party labels for QA lookups.
    df['_Party_display'] = df['Party'].apply(lambda x: '' if pd.isna(x) else str(x).strip())
    df['_Party_key'] = df['_Party_display'].apply(map_party_label)
    df['QA_Total_Party_Votes_from_atomic_party'] = df['_Party_key'].apply(lambda p: float(atomic_party_totals.get(str(p), 0.0)))

    # In the party rows, append the electorate candidate name(s) from the candidate roster so the row
    # label has the form: "Party Name (Candidate Name)" using the *original* party label from the split sheet.
    if party_to_candidate_names:
        def _decorate_party_row(p_display: str, p_key: str) -> str:
            cn = party_to_candidate_names.get(str(p_key).strip())
            return f"{p_display} ({cn})" if cn else p_display
        df['Party'] = [
            _decorate_party_row(d, k) for d, k in zip(df['_Party_display'].tolist(), df['_Party_key'].tolist())
        ]
    else:
        df['Party'] = df['_Party_display']

    # We no longer need helper columns in output.
    df = df.drop(columns=['_Party_display', '_Party_key'], errors='ignore')

    # Row sums and consistency
    df["Sum_from_split_vote_counts"] = df[candidate_cols].sum(axis=1)
    # Party Vote Totals column is the authoritative split per-row total.
    df["Total Party Votes"] = df[party_total_col]
    df["consistent"] = (df["Sum_from_split_vote_counts"] == df["Total Party Votes"]) & (
        df["Sum_from_split_vote_counts"] == df["QA_Total_Party_Votes_from_atomic_party"]
    )

    # Reorder columns: Party, candidate_cols..., Sum_from..., Total Party Votes, QA..., consistent
    df = df[["Party"] + candidate_cols + ["Sum_from_split_vote_counts", "Total Party Votes", "QA_Total_Party_Votes_from_atomic_party", "consistent"]]

    # Reorder rows:
    #  - party rows that have candidate mappings first (Party Name (Candidate))
    #  - then remaining rows.
    party_norm = df["Party"].astype(str).str.strip()
    is_party_vote_totals = party_norm.str.lower() == "party vote totals"
    is_informal_votes = party_norm.str.lower() == "informal votes"

    has_candidate = party_norm.str.contains(r"\(.+\)") & ~(is_party_vote_totals | is_informal_votes)

    df_no_totals = df.loc[~is_party_vote_totals].copy()
    df_totals_row = df.loc[is_party_vote_totals].copy()

    df_no_totals["__has_candidate"] = has_candidate.loc[~is_party_vote_totals].values
    df_no_totals["__orig"] = range(len(df_no_totals))
    df_no_totals = df_no_totals.sort_values(by=["__has_candidate", "__orig"], ascending=[False, True]).drop(columns=["__has_candidate", "__orig"])

    # Bottom summary rows (we place Sum_from_split_vote_counts ABOVE the provided "Party vote totals" row).
    # Compute sums using PARTY rows only (exclude "Informal Votes" and "Party vote totals").
    # Sum_from_split_vote_counts summary row must sum *all rows above* (all non-totals rows).
    df_for_sums = df_no_totals.copy()

    sums = {"Party": "Sum_from_split_vote_counts"}
    qa = {"Party": "QA sums from the candidate csv"}

    def _atomic_lookup(col: str) -> float:
        if not atomic_candidate_totals:
            return float("nan")

    # Totals derived from atomic files (for QA row):
    atomic_party_votes_sum = float(sum(atomic_party_totals.values())) if atomic_party_totals else float('nan')
    # Informal candidate votes total (best-effort lookup from atomic_candidate_totals, if provided)
    atomic_informal_candidate_total = float('nan')
    if atomic_candidate_totals:
        for k in ['Informals', 'Informal Candidate Votes', 'informals', 'informal candidate votes']:
            if k in atomic_candidate_totals:
                try:
                    atomic_informal_candidate_total = float(atomic_candidate_totals[k])
                except Exception:
                    pass
                break

    def _qa_value_for_col(col: str) -> float:
        """Populate QA row values from the candidate CSV (plus derived Party Only if needed)."""
        # For bookkeeping totals, use atomic party vote total (this is what split totals reconcile to).
        if col in {'Sum_from_split_vote_counts', 'Total Party Votes'}:
            return atomic_party_votes_sum
        # Candidate totals (and Informals) come from candidate CSV totals.
        v = _atomic_lookup(col)
        if pd.notna(v):
            return float(v)
        name = str(col).strip().casefold()
        # Derive Party Only from: total party votes - (sum candidates + informal candidate votes).
        if 'party only' in name and pd.notna(atomic_party_votes_sum):
            cand_sum = 0.0
            if atomic_candidate_totals:
                for ck, cv in atomic_candidate_totals.items():
                    if ck in {'Informals', 'Informal Candidate Votes'}:
                        continue
                    # skip non-candidate meta keys
                    if ck.startswith('__'):
                        continue
                    try:
                        cand_sum += float(cv)
                    except Exception:
                        continue
            inf = 0.0 if pd.isna(atomic_informal_candidate_total) else float(atomic_informal_candidate_total)
            return float(atomic_party_votes_sum - (cand_sum + inf))
        return float('nan')
        if col in atomic_candidate_totals:
            return float(atomic_candidate_totals[col])
        k = str(col).strip()
        if k in atomic_candidate_totals:
            return float(atomic_candidate_totals[k])
        k2 = " ".join(k.split())
        if k2 in atomic_candidate_totals:
            return float(atomic_candidate_totals[k2])
        k3 = k.lstrip()
        if k3 in atomic_candidate_totals:
            return float(atomic_candidate_totals[k3])
        return float("nan")

    summary_cols = candidate_cols + ["Sum_from_split_vote_counts", "Total Party Votes", "QA_Total_Party_Votes_from_atomic_party"]
    for c in summary_cols:
        if c == "QA_Total_Party_Votes_from_atomic_party":
            sums[c] = float(df_for_sums[c].sum(skipna=True))
            qa[c] = float("nan")
            continue
        sums[c] = float(df_for_sums[c].sum(skipna=True))
        qa[c] = _qa_value_for_col(c)

    sums["consistent"] = True
    qa["consistent"] = True

    compare_cols = list(candidate_cols) + ["Sum_from_split_vote_counts", "Total Party Votes"]
    if " Informals" in df.columns and " Informals" not in compare_cols:
        compare_cols.insert(len(candidate_cols), " Informals")
    if " Party Only" in df.columns and " Party Only" not in compare_cols:
        compare_cols.insert(len(candidate_cols) + 1, " Party Only")

    sums_vec = [sums.get(c, float("nan")) for c in compare_cols]
    qa_vec = [qa.get(c, float("nan")) for c in compare_cols]
    if not df_totals_row.empty:
        tot_row = df_totals_row.iloc[0].to_dict()
        tot_vec = [float(tot_row.get(c, float("nan"))) if pd.notna(tot_row.get(c, float("nan"))) else float("nan") for c in compare_cols]
    else:
        tot_vec = [float("nan") for _ in compare_cols]

    def _vec_equal(a, b) -> bool:
        for x, y in zip(a, b):
            if pd.isna(x) and pd.isna(y):
                continue
            if pd.isna(x) != pd.isna(y):
                return False
            if float(x) != float(y):
                return False
        return True

    consistent_bool = _vec_equal(sums_vec, tot_vec) and _vec_equal(sums_vec, qa_vec)

    consistent_row = {"Party": "Consistent"}
    for c in df.columns:
        if c == "Party":
            continue
        if c == "consistent":
            consistent_row[c] = bool(consistent_bool)
        else:
            consistent_row[c] = float("nan")

    out = pd.concat([
        df_no_totals,
        pd.DataFrame([sums]),
        df_totals_row,
        pd.DataFrame([qa, consistent_row]),
    ], ignore_index=True)

    out2 = _format_df_numbers_for_csv(out)
    out2.to_csv(out_csv, index=False, encoding="utf-8")


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


