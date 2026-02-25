from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, List

import pandas as pd

from .checksums import safe_mkdir


def port_xls_all_sheets(xls_path: Path, out_dir: Path, prefix: Optional[str] = None) -> List[Path]:
    '''
    Dump every sheet of an .xls/.xlsx workbook to CSV for inspection/QA.

    - Writes header-less CSV (header=None) so it preserves raw layout.
    - Returns list of written CSV paths.
    '''
    safe_mkdir(out_dir)
    suffix = xls_path.suffix.lower()
    base = prefix or xls_path.stem
    outs: List[Path] = []

    if suffix == ".xlsx":
        # pandas can enumerate sheets for xlsx
        xls = pd.ExcelFile(xls_path)
        for i, sheet in enumerate(xls.sheet_names):
            df = pd.read_excel(xls, sheet_name=sheet, header=None)
            out_csv = out_dir / f"{base}_sheet{i}.csv"
            df.to_csv(out_csv, index=False, header=False, encoding="utf-8")
            outs.append(out_csv)
        return outs

    if suffix != ".xls":
        raise ValueError("Expected .xls or .xlsx")

    try:
        import xlrd  # type: ignore
    except Exception as e:
        raise RuntimeError("xlrd is required to read .xls files. Install xlrd==1.2.0.") from e

    book = xlrd.open_workbook(str(xls_path))
    for i in range(book.nsheets):
        sh = book.sheet_by_index(i)
        rows = [[sh.cell_value(r, c) for c in range(sh.ncols)] for r in range(sh.nrows)]
        df = pd.DataFrame(rows)
        out_csv = out_dir / f"{base}_sheet{i}.csv"
        df.to_csv(out_csv, index=False, header=False, encoding="utf-8")
        outs.append(out_csv)
    return outs


def port_2002_xls_to_split_csv(xls_path: Path, out_csv: Path) -> None:
    '''
    Convert the 2002-era split-vote workbook to the *modern* split-votes CSV layout:
    - first column = party name
    - "Total Party Votes" column
    - candidate percentage columns (as floats)
    - "Total %" checksum column
    '''
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

    cleaned = []
    for c in pct_cols:
        name = clean_pct_name(c)
        cleaned.append(name)
        out[name] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)

    out["Total %"] = out[cleaned].sum(axis=1)
    out.to_csv(out_csv, index=False, encoding="utf-8")
