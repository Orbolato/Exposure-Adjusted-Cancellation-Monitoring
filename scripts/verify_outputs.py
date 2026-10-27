"""Compare the CSV results of a full rerun with the archived results of the article.

Every CSV file under ``results/`` (except ``results/tables/``) is compared with the
file at the same relative path under ``<output>/analysis_outputs/``: same columns,
same row count, identical text values, and numeric values equal within a relative
tolerance of 1e-9 (floating-point summation order can differ across platforms and
library versions). The script also lists the figures written by the rerun.

    python scripts/verify_outputs.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from paths import ANALYSIS_DIR, FIGURE_DIR, RESULTS_DIR

RTOL = 1e-9
ATOL = 1e-12
EXPECTED_FIGURES = [
    "figure02.pdf", "figure03.pdf", "figure07.pdf", "figure08.pdf", "figure09.pdf", "figure22.pdf",
    "figure24.pdf", "figure26.pdf", "figure29a.pdf", "figure29b.pdf", "figure29c.pdf", "figure29d.pdf",
    "figure30a.pdf", "figure30b.pdf", "figure31a.pdf", "figure31b.pdf", "figure32.pdf",
]


def compare(archived: pd.DataFrame, rerun: pd.DataFrame) -> str | None:
    if list(archived.columns) != list(rerun.columns):
        return f"columns differ: {list(archived.columns)} vs {list(rerun.columns)}"
    if len(archived) != len(rerun):
        return f"row count differs: {len(archived)} vs {len(rerun)}"
    for col in archived.columns:
        a, b = archived[col], rerun[col]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            if not np.allclose(a.to_numpy(float), b.to_numpy(float), rtol=RTOL, atol=ATOL, equal_nan=True):
                worst = np.nanmax(np.abs(a.to_numpy(float) - b.to_numpy(float)))
                return f"column {col!r} differs (max absolute difference {worst:.3g})"
        elif not a.astype(str).equals(b.astype(str)):
            return f"column {col!r} differs"
    return None


def main() -> None:
    archived_files = sorted(p for p in RESULTS_DIR.rglob("*.csv") if "tables" not in p.parts)
    problems = []
    for path in archived_files:
        rel = path.relative_to(RESULTS_DIR)
        rerun_path = ANALYSIS_DIR / rel
        if not rerun_path.exists():
            problems.append(f"{rel}: not produced by the rerun")
            continue
        issue = compare(pd.read_csv(path, low_memory=False), pd.read_csv(rerun_path, low_memory=False))
        if issue:
            problems.append(f"{rel}: {issue}")
    missing_figures = [f for f in EXPECTED_FIGURES if not (FIGURE_DIR / f).exists()]

    print(f"compared {len(archived_files)} archived CSV files with {ANALYSIS_DIR}")
    for p in problems:
        print("  DIFFERENT", p)
    print(f"figures written: {len(EXPECTED_FIGURES) - len(missing_figures)} of {len(EXPECTED_FIGURES)} in {FIGURE_DIR}")
    for f in missing_figures:
        print("  MISSING", f)
    if problems or missing_figures:
        sys.exit(1)
    print("every archived result was reproduced")


if __name__ == "__main__":
    main()
