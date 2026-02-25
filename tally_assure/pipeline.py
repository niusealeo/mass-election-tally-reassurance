from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from collections import defaultdict

from .discovery import ElectorateJob, build_jobs
from .checksums import (
    now_stamps, safe_mkdir, write_json,
    checksum_candidate_atomic_detailed, checksum_party_atomic_detailed,
    port_candidate_roster_csv, port_party_roster_csv,
)


def rel_from_input(input_root: Path, abs_path: Path) -> Path:
    try:
        return abs_path.relative_to(input_root)
    except ValueError:
        return abs_path


def run_term(
    termKey: str,
    term_jobs: List[ElectorateJob],
    input_root: Path,
    output_root: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    term_pass: List[Dict[str, Any]] = []
    term_fail: List[Dict[str, Any]] = []
    processed = 0

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
        safe_mkdir(out_dir)

        utc_s, local_s = now_stamps()
        stamp = utc_s.replace(":","").replace("-","")
        elect_num = job.electorateNumber or 0
        elect_tag = f"electorate{elect_num}"

        processed += 1

        election_meta = {
            "termKey": job.termKey,
            "year": job.year,
            "electorateFolder": job.electorateFolder,
            "electorateNumber": job.electorateNumber,
            "electorateName": job.electorateName,
            "written_utc": utc_s,
            "written_local": local_s,
        }

        pass_obj: Dict[str, Any] = {**election_meta, "status": "PASS", "candidate": {}, "party": {}}
        fail_obj: Dict[str, Any] = {**election_meta, "status": "FAIL", "candidate": {}, "party": {}}

        any_fail = False
        any_pass = False

        # Candidate atomic detailed checks
        if job.cand_path and job.cand_path.exists():
            cand = checksum_candidate_atomic_detailed(job.cand_path)
            cand_pass = {k: v["passed"] for k, v in cand["checks"].items()}
            cand_fail = {k: v["failed"] for k, v in cand["checks"].items()}
            cand_has_fail = any(len(v) > 0 for v in cand_fail.values())
            cand_has_pass = any(len(v) > 0 for v in cand_pass.values())
            if cand_has_pass:
                any_pass = True
                pass_obj["candidate"] = {"file": cand["file"], "totals_row_index": cand["totals_row_index"], "passed": cand_pass}
            if cand_has_fail:
                any_fail = True
                fail_obj["candidate"] = {"file": cand["file"], "totals_row_index": cand["totals_row_index"], "failed": cand_fail}
            try:
                port_candidate_roster_csv(job.cand_path, out_dir / f"{elect_tag}_candidate_roster.csv")
            except Exception as e:
                any_fail = True
                fail_obj.setdefault("errors", []).append({"port_candidate_roster_csv": str(e)})
        else:
            any_fail = True
            fail_obj["candidate"] = {"errors": [{"file_missing": "candidate_csv"}]}

        # Party atomic detailed checks
        if job.party_path and job.party_path.exists():
            party = checksum_party_atomic_detailed(job.party_path)
            party_pass = {k: v["passed"] for k, v in party["checks"].items()}
            party_fail = {k: v["failed"] for k, v in party["checks"].items()}
            party_has_fail = any(len(v) > 0 for v in party_fail.values())
            party_has_pass = any(len(v) > 0 for v in party_pass.values())
            if party_has_pass:
                any_pass = True
                pass_obj["party"] = {"file": party["file"], "totals_row_index": party["totals_row_index"], "passed": party_pass}
            if party_has_fail:
                any_fail = True
                fail_obj["party"] = {"file": party["file"], "totals_row_index": party["totals_row_index"], "failed": party_fail}
            try:
                port_party_roster_csv(job.party_path, out_dir / f"{elect_tag}_party_roster.csv")
            except Exception as e:
                any_fail = True
                fail_obj.setdefault("errors", []).append({"port_party_roster_csv": str(e)})
        else:
            any_fail = True
            fail_obj["party"] = {"errors": [{"file_missing": "party_csv"}]}

        if any_pass:
            write_json(out_dir / f"{elect_tag}_checksum_pass_{stamp}.json", pass_obj)
        if any_fail:
            write_json(out_dir / f"{elect_tag}_checksum_fail_{stamp}.json", fail_obj)

        rec = {**election_meta, "status": "FAIL" if any_fail else "PASS"}
        if any_fail:
            term_fail.append(rec)
        else:
            term_pass.append(rec)

    return term_pass, term_fail, processed


def run_all(
    input_root: Path,
    output_root: Path,
    downloaded_hash_index_path: Path,
    terms: Optional[List[str]] = None,
    min_year: int = 2002,
    max_year: Optional[int] = None,
) -> None:
    jobs = build_jobs(downloaded_hash_index_path, input_root)
    if terms:
        jobs = [j for j in jobs if j.termKey in set(terms)]
    jobs = [j for j in jobs if j.year >= min_year and (max_year is None or j.year <= max_year)]

    jobs_by_term: Dict[str, List[ElectorateJob]] = defaultdict(list)
    for j in jobs:
        jobs_by_term[str(j.termKey)].append(j)

    top_pass_terms: List[Dict[str, Any]] = []
    top_fail_terms: List[Dict[str, Any]] = []
    processed_electorates_total = 0

    for termKey, term_jobs in sorted(jobs_by_term.items(), key=lambda x: x[0]):
        term_pass, term_fail, processed = run_term(termKey, term_jobs, input_root, output_root)
        processed_electorates_total += processed

        term_out_dir = output_root / termKey
        safe_mkdir(term_out_dir)

        utc_s, local_s = now_stamps()
        stamp = utc_s.replace(":","").replace("-","")

        if term_pass:
            write_json(term_out_dir / f"term_checksum_pass_{stamp}.json", {"termKey": termKey, "written_utc": utc_s, "written_local": local_s, "electorates": term_pass})
        if term_fail:
            write_json(term_out_dir / f"term_checksum_fail_{stamp}.json", {"termKey": termKey, "written_utc": utc_s, "written_local": local_s, "electorates": term_fail})

        if not term_fail:
            top_pass_terms.append({"termKey": termKey, "status": "PASS", "written_utc": utc_s, "written_local": local_s})
        else:
            top_fail_terms.append({"termKey": termKey, "status": "FAIL", "written_utc": utc_s, "written_local": local_s})

    safe_mkdir(output_root)
    utc_s, local_s = now_stamps()
    stamp = utc_s.replace(":","").replace("-","")

    if processed_electorates_total == 0:
        write_json(output_root / f"NO_ELECTORATES_PROCESSED_{stamp}.json", {
            "written_utc": utc_s,
            "written_local": local_s,
            "error": "No electorates were processed.",
        })
        return

    if top_pass_terms:
        write_json(output_root / f"elections_checksum_pass_{stamp}.json", {"written_utc": utc_s, "written_local": local_s, "terms": top_pass_terms})
    if top_fail_terms:
        write_json(output_root / f"elections_checksum_fail_{stamp}.json", {"written_utc": utc_s, "written_local": local_s, "terms": top_fail_terms})
