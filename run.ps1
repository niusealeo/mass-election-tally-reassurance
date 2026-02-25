$ErrorActionPreference = "Stop"
if (!(Test-Path ".\.venv")) {
  Write-Host "Run .\install.ps1 first"
  exit 1
}
. .\.venv\Scripts\Activate.ps1

# edit depending whichversion of python, python3, etc. you have installed
python .\mass_compute_nz_electorates.py @args
