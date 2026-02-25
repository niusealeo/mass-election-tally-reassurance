from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .checksums import (
    now_stamps,
    safe_mkdir,
    write_json,
    checksum_candidate_atomic_detailed,
    checksum_party_atomic_detailed,
    checksum_splitvote_endstate_2002,
    _read_atomic_party_totals,
)
from .discovery import ElectorateJob, build_jobs
from .porting import process_2002_split_xls_to_endstate


def _closest_common_ancestor(a: Path, b: Path) -> Path:
    a = a.resolve()
    b = b.resolve()
    ap = a.parts
    bp = b.parts
    n = min(len(ap), len(bp))
    i = 0
    while i < n and ap[i] == bp[i]:
        i += 1
    if i == 0:
        return Path(a.anchor)
    return Path(*ap[:i])


def _relpath_from_common_ancestor(input_file: Path, output_file: Path) -> str:
    anc = _closest_common_ancestor(input_file, output_file.parent)
    try:
        return str(input_file.resolve().relative_to(anc))
    except Exception:
        return str(input_file)


def _load_alphabetic_numbers(electorates_by_term_path: Path) -> Dict[str, Dict[str, int]]:
    """Return mapping: termKey -> normalised electorateName -> alphabeticNumber."""
    raw = json.loads(electorates_by_term_path.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, int]] = {}

    def norm_name(s: str) -> str:
        return re.sub(r"\s+", " ", str(s).strip()).casefold()

    for termKey, payload in raw.items():
        alpha = payload.get("alphabetical_order") or {}
        out[termKey] = {norm_name(k): int(v) for k, v in alpha.items()}
    return out


def _apply_alphabetic_numbers(jobs: List[ElectorateJob], electorates_by_term_path: Optional[Path]) -> None:
    if not electorates_by_term_path or not electorates_by_term_path.exists():
        return
    mapping = _load_alphabetic_numbers(electorates_by_term_path)

    def norm_name(s: str) -> str:
        return re.sub(r"\s+", " ", str(s).strip()).casefold()

    for j in jobs:
        if j.termKey not in mapping:
            continue
        if not j.electorateName:
            continue
        j.alphabeticNumber = mapping[j.termKey].get(norm_name(j.electorateName))


def _term_folder_from_job(job: ElectorateJob, input_root: Path) -> Path:
    # input_root is expected to include electionresults.govt.nz; term folders are under it.
    return input_root / job.termKey


def _electorate_folder_from_job(job: ElectorateJob, input_root: Path) -> Path:
    return _term_folder_from_job(job, input_root) / job.electorateFolder


def _output_term_folder(output_root: Path, termKey: str) -> Path:
    return output_root / termKey


def _output_electorate_folder(output_root: Path, termKey: str, electorateFolder: str) -> Path:
    return output_root / termKey / electorateFolder


def run_all(
    hash_index_path: Path,
    input_root: Path,
    output_root: Path,
    electorates_by_term_path: Optional[Path] = None,
) -> None:
    safe_mkdir(output_root)

    jobs = build_jobs(hash_index_path, input_root)
    _apply_alphabetic_numbers(jobs, electorates_by_term_path)

    # per-term aggregation
    term_pass: Dict[str, List[dict]] = {}
    term_fail: Dict[str, List[dict]] = {}

    for job in sorted(jobs, key=lambda x: (x.termKey, int(x.alphabeticNumber or 10**9), x.electorateFolder)):
        termKey = job.termKey
        out_term = _output_term_folder(output_root, termKey)
        out_elec = _output_electorate_folder(output_root, termKey, job.electorateFolder)
        safe_mkdir(out_elec)

        utc_s, local_s = now_stamps()

        # Atomic checksums + rosters
        cand_detail = None
        party_detail = None
        if job.cand_path and job.cand_path.exists():
            cand_detail = checksum_candidate_atomic_detailed(job.cand_path)
        if job.party_path and job.party_path.exists():
            party_detail = checksum_party_atomic_detailed(job.party_path)

        # Determine if atomic checks are clean
        def has_fail(detail: Optional[dict]) -> bool:
            if not detail or "checks" not in detail:
                return True
            for group in detail["checks"].values():
                if isinstance(group, dict) and group.get("failed"):
                    if len(group["failed"]) > 0:
                        return True
            return False

        atomic_ok = (not has_fail(cand_detail)) and (not has_fail(party_detail))

        # 2002 endstate processing (xls/xlsx split)
        split_endstate_path = None
        split_detail = None
        if job.year == 2002 and job.split_path and job.split_path.exists() and job.party_path and job.party_path.exists():
            split_endstate_path = out_elec / f"{job.electorateFolder}_split_votes_endstate.csv"
            atomic_party_totals = _read_atomic_party_totals(job.party_path)
            process_2002_split_xls_to_endstate(job.split_path, atomic_party_totals, split_endstate_path)
            split_detail = checksum_splitvote_endstate_2002(split_endstate_path, job.cand_path, job.party_path)

        # Write per-electorate checksum jsons
        elec_meta = {
            "termKey": termKey,
            "year": job.year,
            "electorateFolder": job.electorateFolder,
            "electorateNumber": job.electorateNumber,
            "electorateName": job.electorateName,
            "alphabeticNumber": job.alphabeticNumber,
            "timestamps": {"utc": utc_s, "local": local_s},
            "paths": {
                "split": _relpath_from_common_ancestor(job.split_path, out_elec) if job.split_path else None,
                "candidate": _relpath_from_common_ancestor(job.cand_path, out_elec) if job.cand_path else None,
                "party": _relpath_from_common_ancestor(job.party_path, out_elec) if job.party_path else None,
            },
        }

        pass_payload = {**elec_meta, "candidate": cand_detail, "party": party_detail, "splitvote": split_detail}
        fail_payload = {**elec_meta, "candidate": cand_detail, "party": party_detail, "splitvote": split_detail}

        # splitvote ok?
        split_ok = True
        if split_detail and "checks" in split_detail:
            for group in split_detail["checks"].values():
                if isinstance(group, dict) and group.get("failed"):
                    if len(group["failed"]) > 0:
                        split_ok = False

        all_ok = atomic_ok and split_ok

        if all_ok:
            out_pass = out_elec / f"{job.electorateFolder}_checksum_pass_{utc_s.replace(':','').replace('-','')}.json"
            write_json(out_pass, pass_payload)
            term_pass.setdefault(termKey, []).append({"electorateFolder": job.electorateFolder, "alphabeticNumber": job.alphabeticNumber})
        else:
            out_fail = out_elec / f"{job.electorateFolder}_checksum_fail_{utc_s.replace(':','').replace('-','')}.json"
            write_json(out_fail, fail_payload)
            term_fail.setdefault(termKey, []).append({"electorateFolder": job.electorateFolder, "alphabeticNumber": job.alphabeticNumber})

    # per-term summary
    for termKey in sorted(set(list(term_pass.keys()) + list(term_fail.keys()))):
        utc_s, local_s = now_stamps()
        out_term = _output_term_folder(output_root, termKey)
        safe_mkdir(out_term)

        if term_pass.get(termKey):
            write_json(out_term / f"term_checksum_pass_{utc_s.replace(':','').replace('-','')}.json", {
                "termKey": termKey,
                "timestamps": {"utc": utc_s, "local": local_s},
                "passed_electorates": sorted(term_pass.get(termKey, []), key=lambda x: int(x.get("alphabeticNumber") or 10**9)),
            })
        if term_fail.get(termKey):
            write_json(out_term / f"term_checksum_fail_{utc_s.replace(':','').replace('-','')}.json", {
                "termKey": termKey,
                "timestamps": {"utc": utc_s, "local": local_s},
                "failed_electorates": sorted(term_fail.get(termKey, []), key=lambda x: int(x.get("alphabeticNumber") or 10**9)),
            })

    # elections summary (in electionresults.govt.nz folder = output_root)
    utc_s, local_s = now_stamps()
    if term_fail:
        write_json(output_root / f"elections_checksum_fail_{utc_s.replace(':','').replace('-','')}.json", {
            "timestamps": {"utc": utc_s, "local": local_s},
            "failed_terms": sorted(term_fail.keys()),
        })
    if term_pass:
        # pass means "terms that had at least one pass"; not necessarily 100% term
        write_json(output_root / f"elections_checksum_pass_{utc_s.replace(':','').replace('-','')}.json", {
            "timestamps": {"utc": utc_s, "local": local_s},
            "terms_with_any_pass": sorted(term_pass.keys()),
        })
