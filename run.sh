#!/usr/bin/env bash
set -euo pipefail
if [ ! -d ".venv" ]; then
  echo "Run ./install.sh first"
  exit 1
fi
source .venv/bin/activate
python mass_compute_nz_electorates.py "$@"
