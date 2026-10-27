"""Categorical-encoding robustness check for the phase-level operational-cell segmentation.

Carrier group and origin aerodrome are nominal variables. The main segmentation
encodes them by prevalence, which places categories of similar frequency close to
each other on a single axis even when they are operationally unrelated. This script
repeats the phase-level segmentation with one-hot indicators for carrier and origin:
the numerical block (rate, departure-time coordinates and weekend indicator) is
standardized and the unscaled indicators are appended, so that two cells differing
in carrier or in origin are at squared distance 2 on that block regardless of how
many categories exist. The prospective version of this comparison is part of
``generate_robustness_analyses.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

import cancellation_monitoring as cm
from generate_policy_analyses import to_monitoring_frame
from generate_vra_results import OUT_DIR, PERIOD_ORDER, load_vra

K = 3
ENCODINGS = {"Prevalence encoding": cm.CellSpec("Prevalence"), "One-hot encoding": cm.CellSpec("One-hot", encoding="onehot")}


def segment_summary(assigned: pd.DataFrame) -> dict:
    top = assigned[assigned["rate_cluster"] == assigned["rate_cluster"].max()]
    return {
        "cells": len(top),
        "rate_pct": cm.weighted_rate(top),
        "share_of_scheduled_pct": float(top["scheduled"].sum() / assigned["scheduled"].sum() * 100),
        "share_of_cancellations_pct": float(top["cancelled"].sum() / assigned["cancelled"].sum() * 100),
        "members": set(zip(*(top[c] for c in cm.CELL_KEYS))),
    }


def compare_by_period(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period in PERIOD_ORDER:
        sub = frame if period == "Overall" else frame[frame["period"] == period]
        cells = cm.build_cells(sub)
        if len(cells) < K + 1:
            continue
        res = {name: segment_summary(cm.fit_segments(cells, spec, K)) for name, spec in ENCODINGS.items()}
        a, b = res["Prevalence encoding"]["members"], res["One-hot encoding"]["members"]
        for name, r in res.items():
            rows.append({"period": period, "encoding": name, "cells": r["cells"], "watchlist_rate_pct": r["rate_pct"],
                         "share_of_scheduled_pct": r["share_of_scheduled_pct"],
                         "share_of_cancellations_pct": r["share_of_cancellations_pct"],
                         "jaccard_between_encodings": len(a & b) / len(a | b)})
    return pd.DataFrame(rows)


def main() -> None:
    frame = to_monitoring_frame(load_vra())
    by_period = compare_by_period(frame)
    by_period.to_csv(OUT_DIR / "encoding_sensitivity_segments.csv", index=False)
    print(by_period.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
