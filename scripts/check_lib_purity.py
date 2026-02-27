#!/usr/bin/env python3
"""Fail if tally_assure/libs imports anything from this repo (other than tally_assure.libs.*).

Run:
  python scripts/check_lib_purity.py
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
LIBS_DIR = REPO_ROOT / "tally_assure" / "libs"

IMPORT_RE = re.compile(r"^\s*(from\s+([\w\.]+)\s+import|import\s+([\w\.]+))")

def main() -> int:
    if not LIBS_DIR.exists():
        print(f"ERROR: libs dir not found: {LIBS_DIR}")
        return 2

    bad = []
    for py in sorted(LIBS_DIR.glob("*.py")):
        txt = py.read_text(encoding="utf-8")
        for ln, line in enumerate(txt.splitlines(), start=1):
            m = IMPORT_RE.match(line)
            if not m:
                continue
            mod = m.group(2) or m.group(3) or ""
            # allow: stdlib/external, and internal libs
            if mod.startswith("tally_assure.") and not mod.startswith("tally_assure.libs"):
                bad.append((py.name, ln, line.strip()))
            if mod.startswith("..") or mod.startswith("."):
                # relative imports are allowed only if they stay within libs (but we can't easily tell).
                # Prefer absolute `tally_assure.libs.*` within libs. Flag any relative import.
                bad.append((py.name, ln, line.strip()))

    if bad:
        print("Found disallowed imports in tally_assure/libs:\n")
        for name, ln, line in bad:
            print(f"  {name}:{ln}: {line}")
        print("\nFix: libs modules may only import stdlib/external packages or tally_assure.libs.* (absolute).")

        return 1

    print("OK: tally_assure/libs is repo-pure.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
