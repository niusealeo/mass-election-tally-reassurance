$ErrorActionPreference = "Stop"
if (!(Test-Path ".\.venv")) {
  Write-Host "Run .\install.ps1 first"
  exit 1
}
. .\.venv\Scripts\Activate.ps1
python .\mass_compute_nz_electorates.py @args
