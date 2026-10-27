"""Locations of the source data, reference tables and generated outputs.

Every script resolves its inputs and outputs through this module. The defaults keep
everything inside the repository; two environment variables move the large parts
elsewhere without editing code:

``CANCELLATIONS_DATA_DIR``
    Folder holding the downloaded source files (default: ``data/``), organized as
    ``siros/``, ``legacy_portal/``, ``bts/`` and ``anac_statistics/``.
``CANCELLATIONS_OUTPUT_DIR``
    Folder for everything the pipeline writes (default: ``output/``): harmonized
    caches, analysis tables, figures and LaTeX table bodies.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get("CANCELLATIONS_DATA_DIR", ROOT / "data")).resolve()
OUTPUT_DIR = Path(os.environ.get("CANCELLATIONS_OUTPUT_DIR", ROOT / "output")).resolve()

# Source files (downloaded; not versioned).
SIROS_DIR = DATA_DIR / "siros"                  # VRA monthly files from ANAC's SIROS archive
LEGACY_DIR = DATA_DIR / "legacy_portal"         # VRA files from ANAC's open-data portal, by year
BTS_DIR = DATA_DIR / "bts"                      # BTS on-time performance extracts (Parquet)
STATS_DIR = DATA_DIR / "anac_statistics"        # ANAC air transport statistics, one CSV per year

# Reference tables derived from public sources (versioned).
REFERENCE_DIR = ROOT / "reference"

# Generated outputs (not versioned).
INTERIM_DIR = OUTPUT_DIR / "interim"            # harmonized flight-level caches
ANALYSIS_DIR = OUTPUT_DIR / "analysis_outputs"  # CSV results; same layout as results/
FIGURE_DIR = OUTPUT_DIR / "figures"             # figure PDFs
TABLE_DIR = OUTPUT_DIR / "tables"               # LaTeX table bodies

# Archived results of the published analysis (versioned).
RESULTS_DIR = ROOT / "results"
