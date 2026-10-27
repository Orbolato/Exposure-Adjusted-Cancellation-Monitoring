#!/usr/bin/env bash
# Full rerun of the analysis from the source files, in dependency order.
#
#   bash reproduce.sh            # analysis only (source files already in data/)
#   bash reproduce.sh --download # download the source files first
#
# Source files are read from $CANCELLATIONS_DATA_DIR (default: data/) and every
# output is written to $CANCELLATIONS_OUTPUT_DIR (default: output/).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
run() { echo; echo "=== $*"; "$PY" "$@"; }

if [[ "${1:-}" == "--download" ]]; then
  run scripts/download_vra_siros.py
  run scripts/download_vra_anac.py --start-year 2015 --end-year 2024
  run scripts/download_bts_ontime.py 2018 2019 2020 2021 2023 2024
  echo "ANAC statistics (data/anac_statistics/) must be downloaded manually; see data/README.md."
fi

# Reference list of Brazilian aerodromes (from the ANAC statistics).
run scripts/build_brazilian_aerodrome_reference.py
# Harmonized flight-level dataset and the inclusion flow (Table B.2).
run scripts/build_vra_siros.py
# Descriptive results, canceled-record benchmark and Figs. 2, A.2 and A.4-A.7.
run scripts/generate_vra_results.py
# Segmentation diagnostics, algorithm comparison and prospective validation (Figs. 3-4).
run scripts/generate_policy_analyses.py
run scripts/generate_watchlist_sensitivity.py
run scripts/generate_encoding_sensitivity.py
# United States replication (Fig. 5).
run scripts/generate_bts_validation.py
# Registry diagnostics, robustness and comparability analyses (Fig. 1); about 15 minutes.
run scripts/generate_robustness_analyses.py
run scripts/generate_comparability_followup.py
# Reconciliation of the SIROS and open-data portal archives (Table B.1).
run scripts/reconcile_vra_archives.py
# Review workload of the screening lists and options for low-frequency services (Table B.8).
run scripts/generate_operational_implications.py
# Market context from the ANAC statistics (Figs. A.1 and A.3).
run scripts/generate_operational_context.py
# Table bodies from the rerun, compared with those printed in the article.
run scripts/make_tables.py --from-output --check
# Every CSV result compared with the archived results/.
run scripts/verify_outputs.py
