from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .checksums import (
    now_stamps, safe_mkdir, read_csv_robust,
    find_totals_row, sum_excluding_totals, compare_series
)
from .discovery import ElectorateJob, build_jobs
from .sainte_lague import build_start_game_state
from .sliding_puzzle import solve_sliding_puzzle
from .porting import port_2002_xls_to_split_csv

def rel_from_input(input_root: Path, abs_path: Path) -> Path:
    try:
        return abs_path.relative_to(input_root)
    except ValueError:
        return abs_path


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
        df = read_csv_robust(job.cand_path)
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
        dfp = read_csv_robust(job.party_path)
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
            s = read_csv_robust(job.split_path)
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


def run_all(
    input_root: Path,
    output_root: Path,
    downloaded_hash_index_path: Path,
    terms: Optional[List[str]] = None,
    min_year: int = 2002,
    max_year: Optional[int] = None,
) -> None:
    """Orchestrate discovery -> checksums -> reconstructions -> summaries."""
    safe_mkdir(output_root)

    jobs = build_jobs(downloaded_hash_index_path, input_root)

    if terms:
        jobs = [j for j in jobs if j.termKey in set(terms)]
    jobs = [j for j in jobs if j.year >= min_year and (max_year is None or j.year <= max_year)]

    by_term: Dict[str, List[ElectorateJob]] = {}
    for j in jobs:
        by_term.setdefault(j.termKey, []).append(j)

    top_pass_terms: List[dict] = []
    top_fail_terms: List[dict] = []
    processed_electorates_total = 0

    for termKey, term_jobs in sorted(by_term.items(), key=lambda x: x[0]):
        term_pass: List[dict] = []
        term_fail: List[dict] = []

        for job in sorted(term_jobs, key=lambda j: (j.electorateNumber or 0, j.electorateFolder)):
            in_dir = None
            
            print([job.split_path, job.cand_path, job.party_path])
            
            for p in [job.split_path, job.cand_path, job.party_path]:
                if p and p.exists():
                    in_dir = p.parent
                    break
            if in_dir is None:
                continue

            out_dir = output_root / rel_from_input(input_root, in_dir)
            safe_mkdir(out_dir)

            _, any_fail = checksum_electorate(job, out_dir)
            processed_electorates_total += 1

            rec = {
                "termKey": job.termKey,
                "year": job.year,
                "electorateFolder": job.electorateFolder,
                "electorateNumber": job.electorateNumber,
                "electorateName": job.electorateName,
            }
            if any_fail:
                rec["status"] = "FAIL"
                term_fail.append(rec)
            else:
                rec["status"] = "PASS"
                term_pass.append(rec)

            # 2002: port XLS only
            if job.year == 2002 and job.split_path and job.split_path.exists() and job.split_path.suffix.lower() in [".xls", ".xlsx"]:
                try:
                    port_2002_xls_to_split_csv(job.split_path, out_dir / f"electorate{job.electorateNumber or 0}_ported_split_votes.csv")
                except Exception as e:
                    (out_dir / f"electorate{job.electorateNumber or 0}_ported_split_votes_ERROR.txt").write_text(str(e), encoding="utf-8")
                continue

            # 2005-2023: Sainte-Laguë start + sliding puzzle
            if job.year >= 2005:
                try:
                    cand_totals = pd.Series(dtype=float)
                    party_totals = pd.Series(dtype=float)
                    if job.cand_path and job.cand_path.exists():
                        dfc = read_csv_robust(job.cand_path)
                        trow = find_totals_row(dfc)
                        cand_totals = sum_excluding_totals(dfc, trow)
                    if job.party_path and job.party_path.exists():
                        dfp = read_csv_robust(job.party_path)
                        trowp = find_totals_row(dfp)
                        party_totals = sum_excluding_totals(dfp, trowp)

                    if job.split_path and job.split_path.exists() and job.split_path.suffix.lower() == ".csv":
                        start_path = out_dir / f"electorate{job.electorateNumber or 0}_start_game_state_with_actual_and_residual.csv"
                        build_start_game_state(job.split_path, cand_totals, party_totals, start_path)
                        solve_sliding_puzzle(start_path, job.split_path, cand_totals, out_dir / f"electorate{job.electorateNumber or 0}_finish")
                except Exception as e:
                    (out_dir / f"electorate{job.electorateNumber or 0}_reconstruct_ERROR.txt").write_text(str(e), encoding="utf-8")

        if term_pass or term_fail:
            out_term_dir = None
            for j in term_jobs:
                for p in [j.split_path, j.cand_path, j.party_path]:
                    if p and p.exists():
                        elect_out_dir = output_root / rel_from_input(input_root, p.parent)
                        out_term_dir = elect_out_dir.parent
                        break
                if out_term_dir is not None:
                    break
            if out_term_dir is None:
                out_term_dir = output_root / termKey
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
        return

    if top_pass_terms:
        pd.DataFrame(top_pass_terms).to_csv(output_root / f"elections_checksum_pass_{stamp}.csv", index=False, encoding="utf-8")
    if top_fail_terms:
        pd.DataFrame(top_fail_terms).to_csv(output_root / f"elections_checksum_fail_{stamp}.csv", index=False, encoding="utf-8")
