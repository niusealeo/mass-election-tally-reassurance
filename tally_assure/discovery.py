from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

@dataclass
class ElectorateJob:
    termKey: str
    year: int
    electorateFolder: str
    electorateNumber: Optional[int]
    electorateName: Optional[str]
    alphabeticNumber: Optional[str]
    split_path: Optional[Path]
    cand_path: Optional[Path]
    party_path: Optional[Path]


def parse_year(termKey: str) -> int:
    m = re.search(r"\((\d{4})\)", termKey)
    if not m:
        raise ValueError(f"Cannot parse year from termKey: {termKey}")
    return int(m.group(1))


def build_jobs(hash_index_path: Path, input_root: Path) -> List[ElectorateJob]:
    idx = json.loads(hash_index_path.read_text(encoding="utf-8"))

    groups: Dict[Tuple[str, str], List[dict]] = {}
    for _, meta in idx.items():
        termKey = meta.get("termKey")
        ef = meta.get("electorateFolder")
        saved = meta.get("saved_to")
        if not termKey or not saved or not ef:
            continue
        groups.setdefault((termKey, ef), []).append(meta)

    jobs: List[ElectorateJob] = []
    for (termKey, ef), metas in groups.items():
        year = parse_year(termKey)
        num = None; name = None
        m = re.match(r"(\d{3})_(.+)$", ef)
        if m:
            num = int(m.group(1))
            name = m.group(2).replace("_", " ")

        split_path = None
        cand_path = None
        party_path = None

        for meta in metas:
            p = Path(meta["saved_to"])
            abs_p = (input_root / p) if not p.is_absolute() else p
            fn = abs_p.name.lower()
            if "split" in fn or fn.endswith(".xls") or fn.endswith(".xlsx"):
                split_path = abs_p
            elif "cand" in fn:
                cand_path = abs_p
            elif "party" in fn:
                party_path = abs_p

        jobs.append(ElectorateJob(termKey, year, ef, num, name, None, split_path, cand_path, party_path))

    return jobs
