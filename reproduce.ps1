# Full rerun of the analysis from the source files, in dependency order (Windows).
#
#   .\reproduce.ps1             # analysis only (source files already in data\)
#   .\reproduce.ps1 -Download   # download the source files first
#
# Source files are read from $env:CANCELLATIONS_DATA_DIR (default: data\) and every
# output is written to $env:CANCELLATIONS_OUTPUT_DIR (default: output\).
param([switch]$Download)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }

function Run([string[]]$ScriptArgs) {
    Write-Host ""
    Write-Host "=== $($ScriptArgs -join ' ')"
    & $py @ScriptArgs
    if ($LASTEXITCODE -ne 0) { throw "Failed: $($ScriptArgs -join ' ')" }
}

if ($Download) {
    Run @("scripts\download_vra_siros.py")
    Run @("scripts\download_vra_anac.py", "--start-year", "2015", "--end-year", "2024")
    Run @("scripts\download_bts_ontime.py", "2018", "2019", "2020", "2021", "2023", "2024")
    Write-Host "ANAC statistics (data\anac_statistics\) must be downloaded manually; see data\README.md."
}

Run @("scripts\build_brazilian_aerodrome_reference.py")
Run @("scripts\build_vra_siros.py")
Run @("scripts\generate_vra_results.py")
Run @("scripts\generate_policy_analyses.py")
Run @("scripts\generate_watchlist_sensitivity.py")
Run @("scripts\generate_encoding_sensitivity.py")
Run @("scripts\generate_bts_validation.py")
Run @("scripts\generate_robustness_analyses.py")
Run @("scripts\generate_comparability_followup.py")
Run @("scripts\reconcile_vra_archives.py")
Run @("scripts\generate_operational_implications.py")
Run @("scripts\generate_operational_context.py")
Run @("scripts\make_tables.py", "--from-output", "--check")
Run @("scripts\verify_outputs.py")
