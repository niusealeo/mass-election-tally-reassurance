#!/usr/bin/env python3
"""Mass computation for NZ electorate split-vote reconstruction + QA checksums.

Thin CLI wrapper. Core logic lives under `tally_assure/`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tally_assure.pipeline import run_all


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Mass computation for NZ electorate split-vote reconstruction + QA checksums (modular)."
    )
    ap.add_argument("--input-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--downloaded-hash-index", required=True)
    ap.add_argument("--electorates-by-term", default=None, help="Path to electorates_by_term.json (for alphabeticNumber)")
    ap.add_argument("--terms", nargs="*", default=None)
    ap.add_argument("--min-year", type=int, default=2002)
    ap.add_argument("--max-year", type=int, default=None)
    args = ap.parse_args()

    run_all(
        hash_index_path=Path(args.downloaded_hash_index).resolve(),
        input_root=Path(args.input_root).resolve(),
        output_root=Path(args.output_root).resolve(),
        terms=args.terms,
        min_year=args.min_year,
        max_year=args.max_year,
        electorates_by_term_path=(Path(args.electorates_by_term).resolve() if args.electorates_by_term else None),
    )


if __name__ == "__main__":
    main()
