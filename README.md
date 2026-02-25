# mass-computation-tally-assure

`tally-assure` is a project-local, reproducible “npm-style” runner for mass QA + reconstruction of NZ election split-vote tables by electorate and term (2002–2023).

It is designed so that someone can:
1) clone/download the repo,
2) run a single install command (like `npm i`),
3) run a single command to generate outputs (like `npm start`).

---

## What this project does

For each **electorate** in each **term/election** referenced by your `downloaded_hash_index.json`, the tool can:

### A) Quality assurance (QA) checksums (3 checks per electorate)

Using the **atomic voting-place tallies** (candidate and party), it recomputes totals and validates:

1. **Candidate atomic checksum**  
   Sum all voting-place rows in the *candidate atomic file* and compare to the provided totals row in that same file.

2. **Party atomic checksum**  
   Sum all voting-place rows in the *party atomic file* and compare to the provided totals row in that same file.

3. **Split-vote totals checksum**  
   Compare party totals computed from the *party atomic file* to the `Total Party Votes` column in the *split-votes file* (when that column exists).  
   - For 2005 split-vote files (which lack `Total Party Votes`), this check will record a failure reason like `NO_TOTAL_PARTY_VOTES_COLUMN`.

**Important design choice:**  
Even if checksums fail, the pipeline continues and uses the recomputed atomic sums as the “ground truth” for downstream computations.

### B) 2005–2023: Sainte-Laguë reconstruction + sliding puzzle solve

For each electorate (2005–2023), it:
1. Reads the split-vote table (percentages) and **allocates integer/fractional units** row-by-row using your custom **Sainte-Laguë tie-break rules** (including third-order numeric splitting).
2. Writes a **start game state** CSV, including:
   - the Sainte-Laguë reconstructed matrix,
   - `Sum_from_split_vote_percentages` row,
   - `TRUE_TOTAL` row (atomic candidate totals + derived Party Vote Only slack),
   - `RESIDUAL_DEVIATION` row.
3. Solves the “sliding puzzle” by moving at most **1 unit at a time**, preferring **smallest fractional remainders first**, choosing the move that minimizes a weighted squared-error objective (Gallagher-like intent).
4. Writes finish-state outputs + a full step log.

### C) 2002: Port split-vote XLS/XLSX into modern CSV format

For 2002, split-vote information is stored in Excel (`.xls` and/or `.xlsx`). The tool ports this into a modern split-votes CSV layout.

- `.xls` parsing uses **xlrd** (see dependencies below).
- `.xlsx` parsing uses pandas/openpyxl.

The output is written into the matching electorate directory under the output root, mirroring the input structure.

---

## Input requirements

You must provide:

1) **An input root directory** containing the election files (the downloaded directory structure).  
2) **`downloaded_hash_index.json`** describing where each downloaded file is stored (via its `saved_to` path), and its `termKey` and `electorateFolder` metadata.

The script uses `downloaded_hash_index.json` to discover, for each electorate:
- a candidate atomic file (contains “cand” in filename),
- a party atomic file (contains “party” in filename),
- a split-vote file (contains “split” or is `.xls/.xlsx` for 2002).

> This is intentionally filename-robust: it doesn’t hardcode “2017+” names vs “e9_part8” names; it discovers them from the hash index.

---

## Output behavior and directory mirroring

You choose an **output root directory**. The tool writes all results under that directory and **mirrors the input structure**.

If you set `--output-root` equal to `--input-root`, outputs are written back into the same folders.

Outputs are produced at three levels:

### 1) Electorate-level outputs (inside each electorate folder)

Checksum outputs (written only if relevant):
- `electorate{N}_checksum_pass_<timestamp>.csv`  (written if at least one check has passing items)
- `electorate{N}_checksum_fail_<timestamp>.csv`  (written if at least one check has failing items)

Each checksum file includes dual timestamps:
- `written_utc` ISO timestamp
- `written_local` human-readable timestamp with weekday + month

Reconstruction outputs:
- 2002:
  - `electorate{N}_ported_split_votes.csv`
- 2005–2023:
  - `electorate{N}_start_game_state_with_actual_and_residual.csv`
  - `electorate{N}_finish_solution_matrix.csv`
  - `electorate{N}_finish_moves_steps.csv`
  - `electorate{N}_finish_moves_aggregated.csv`
  - `electorate{N}_finish_totals_comparison.csv`
  - `electorate{N}_finish_deviation.csv`
  - `electorate{N}_finish_variance.csv`

If any step fails for an electorate, an `*_ERROR.txt` file is written explaining why.

### 2) Term-level outputs (inside each term folder)

Written after processing all electorates in the term:
- `term_checksum_pass_<timestamp>.csv` (only if at least one electorate is fully consistent)
- `term_checksum_fail_<timestamp>.csv` (only if at least one electorate has any failure)

### 3) Top-level outputs (inside output root)

Written after all terms:
- `elections_checksum_pass_<timestamp>.csv` (only if at least one term is fully consistent)
- `elections_checksum_fail_<timestamp>.csv` (only if at least one term has any failures)

---

## Supported “eras” (practical differences)

This tool supports the real-world NZEC packaging differences:

- **2002**
  - atomic files: `e9_part8_cand_*.csv`, `e9_part8_party_*.csv`
  - split: Excel `.xls/.xlsx`

- **2005**
  - atomic files: `e9_part8_cand_*.csv`, `e9_part8_party_*.csv`
  - split: `elect-splitvote-*.csv`
  - split may not include `Total Party Votes` (checksum #3 will report that)

- **2008–2014**
  - atomic files: `e9_part8_cand_*.csv`, `e9_part8_party_*.csv`
  - split: `elect-splitvote-*.csv`
  - split typically includes `Total Party Votes`

- **2017+**
  - atomic files: `candidate-votes-by-voting-place-*.csv`, `party-votes-by-voting-place-*.csv`
  - split: `split-votes-electorate-*.csv`

The script does not rely on year-based hardcoding; it primarily follows the file discovery from `downloaded_hash_index.json` and only uses the year for the 2002 Excel-port step.

---

## Installation (project-local, like `npm i`)

This repo installs dependencies into a **local virtual environment** at `./.venv`.

### macOS / Linux
```bash
chmod +x install.sh run.sh
./install.sh
```

### Windows (PowerShell)
```powershell
.\install.ps1
```

---

## Run

### Show help
```bash
./run.sh --help
```

### Typical run
```bash
./run.sh \
  --input-root /path/to/all_elections_root \
  --output-root /path/to/output_root \
  --downloaded-hash-index /path/to/downloaded_hash_index.json
```

### Run only certain terms
```bash
./run.sh \
  --input-root /path/to/all_elections_root \
  --output-root /path/to/output_root \
  --downloaded-hash-index /path/to/downloaded_hash_index.json \
  --terms "term_49_(2008)" "term_54_(2023)"
```

---

## Dependencies

Installed by `install.sh` / `install.ps1`:

- `pandas`
- `numpy`
- `openpyxl` (for `.xlsx`)
- `xlrd==1.2.0` (required for legacy `.xls`)

### Why `xlrd==1.2.0`?
`xlrd` version 2.x dropped `.xls` support. `1.2.0` is the last release that can read `.xls` files.

---

## Troubleshooting

### “xlrd is required to read .xls files …”
Install dependencies via the project installer:
```bash
./install.sh
```
or on Windows:
```powershell
.\install.ps1
```

### Missing files for an electorate
If any of the 3 core files (candidate atomic, party atomic, split) can’t be located via `downloaded_hash_index.json`, the electorate may be skipped for reconstruction. Check the generated checksum fail file and the `*_ERROR.txt` files in that electorate output directory.

### Paths not mirroring as expected
The mirroring logic follows `saved_to` paths in `downloaded_hash_index.json`. Ensure `--input-root` is the directory that makes those `saved_to` paths valid on disk.

---

## License / attribution
Add your license details here.


## Important: term filtering

By default the script ignores terms earlier than 2002.
To focus on specific terms, pass `--terms`:

```bash
./run.sh --input-root ... --output-root ... --downloaded-hash-index ... \
  --terms "term_54_(2023)" "term_53_(2020)"
```

Or use a year window:

```bash
./run.sh --input-root ... --output-root ... --downloaded-hash-index ... --min-year 2017
```


## Modular structure

Core logic is split into modules under `tally_assure/`:

- `discovery.py` — locate inputs and build electorate jobs
- `checksums.py` — checksum functions and timestamps
- `sainte_lague.py` — Sainte-Laguë reconstruction
- `sliding_puzzle.py` — solver
- `porting.py` — 2002 XLS→CSV porting
- `pipeline.py` — orchestration
