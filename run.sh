#!/usr/bin/env bash
set -euo pipefail
if [ ! -d ".venv" ]; then
  echo "Run ./install.sh first"
  exit 1
fi
source .venv/bin/activate

# # edit depending whichversion of python, python3, etc. you have installed
python3 mass_compute_nz_electorates.py "$@"
