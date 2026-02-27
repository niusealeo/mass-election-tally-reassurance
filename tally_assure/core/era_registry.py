"""Era registry and dispatch.

The goal is to prevent cross-era regressions: each file-format era has its own parser/cleaner.
Pipeline should only call into the active era module for a given termKey/year.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class Era:
    key: str
    label: str
    min_year: int
    max_year: int

ERA_1996 = Era("1996", "1996 (PDF-scrape era)", 1996, 1996)
ERA_1999 = Era("1999", "1999 (PDF-scrape era)", 1999, 1999)
ERA_2002 = Era("2002", "2002 (splitvote XLS + e9_part8_*.csv)", 2002, 2002)
ERA_2005 = Era("2005", "2005 (elect-splitvote-*.csv + e9_part8_*.csv)", 2005, 2005)
ERA_2008_2023 = Era("2008-2023", "2008–2023 (splitvote CSV era)", 2008, 2023)

def era_for_term_key(term_key: str) -> Optional[Era]:
    """Map termKey strings to an era.

    This is intentionally conservative: if a termKey doesn't match, callers should fail fast
    rather than silently using the wrong parser.
    """
    if not term_key:
        return None
    # Common termKey format: "term_47_(2002)"
    if "(1996)" in term_key:
        return ERA_1996
    if "(1999)" in term_key:
        return ERA_1999
    if "(2002)" in term_key:
        return ERA_2002
    if "(2005)" in term_key:
        return ERA_2005
    # 2008+ (includes 2011, 2014, 2017, 2020, 2023 etc)
    for y in ("2008","2011","2014","2017","2020","2023"):
        if f"({y})" in term_key:
            return ERA_2008_2023
    return None
