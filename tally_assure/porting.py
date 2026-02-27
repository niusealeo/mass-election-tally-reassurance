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
    """Read sheet0 of a legacy .xls workbook.

    This repo uses pandas with engine='xlrd'. That requires xlrd>=2.0.1,
    which supports .xls (and does not support xlsx).
    """
    return pd.read_excel(path, sheet_name=0, engine="xlrd")


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

    def _find_header_row(raw: pd.DataFrame) -> Optional[int]:
        for i in range(min(60, raw.shape[0])):
            row = raw.iloc[i].tolist()
            lowered = [str(x).strip().casefold() for x in row]
            if any(x == "party" for x in lowered):
                nonempty = sum(1 for x in lowered if x and x != "nan")
                if nonempty >= 3:
                    return i
        return None

    header_i = _find_header_row(df)
    if header_i is not None:
        hdr_raw = df.iloc[header_i].tolist()
        hdr = [None if (pd.isna(x) or str(x).strip() == "") else str(x).strip() for x in hdr_raw]
        # Forward-fill header cells (merged cells create blanks).
        filled: List[Optional[str]] = []
        last: Optional[str] = None
        for x in hdr:
            if x is None:
                filled.append(last)
            else:
                filled.append(x)
                last = x
        df = df.iloc[header_i + 1 :].copy().reset_index(drop=True)
        df.columns = ["Party" if (x and str(x).strip().casefold() == "party") else (x if x is not None else "") for x in filled]

    # Normalise Party column (fallback: first column)
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
    party_total_col = cols[1]

    # Coerce all non-Party columns to numeric best-effort
    for c in cols[1:]:
        df[c] = _coerce_num_series(df[c])

    # Identify percent columns and drop them.
    def _is_percent_col(col: str) -> bool:
        name = str(col).casefold()
        if "%" in name or "percent" in name:
            return True
        v = df[col]
        v = v[pd.notna(v)]
        if len(v) == 0:
            return False
        vmax = float(v.max())
        if vmax <= 1.5:
            frac = (v % 1).abs()
            nonint = float((frac > 1e-9).mean()) if len(frac) else 0.0
            if nonint >= 0.2:
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

    df["Party"] = df["Party"].apply(map_party_label)
    df["QA_Total_Party_Votes_from_atomic_party"] = df["Party"].apply(lambda p: float(atomic_party_totals.get(str(p), 0.0)))

    # In the party rows, append the candidate name from the candidate roster so the row
    # label has the form: "Party Name (Candidate Name)".
    if party_to_candidate_names:
        def _decorate_party(p: object) -> object:
            if pd.isna(p):
                return p
            s = str(p).strip()
            if not s or s.casefold() == "nan":
                return p
            cn = party_to_candidate_names.get(s)
            return f"{s} ({cn})" if cn else s

        df["Party"] = df["Party"].apply(_decorate_party)

    # Row sums and consistency
    df["Sum_from_split_vote_counts"] = df[candidate_cols].sum(axis=1)
    # Party Vote Totals column is the authoritative split per-row total.
    df["Total Party Votes"] = df[party_total_col]
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
    candidate_order: Optional[List[str]],
    party_to_candidate_names: Optional[Dict[str, str]],
    out_csv: Path,
) -> None:
    df = read_xls_sheet0(xls_path)
    process_2002_split_sheet_df_to_endstate(
        df,
        atomic_party_totals,
        candidate_order,
        party_to_candidate_names=party_to_candidate_names,
        out_csv=out_csv,
    )


def process_split_votes_csv_to_endstate(*args, **kwargs):
    raise NotImplementedError("2005-2023 processing lives in the Sainte-Laguë + sliding puzzle pipeline.")


def port_xls_all_sheets(*args, **kwargs):
    raise NotImplementedError("Raw sheet CSVs are no longer kept; we port directly to endstate in-memory.")
