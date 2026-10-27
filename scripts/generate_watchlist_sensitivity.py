"""Sensitivity of the prospective validation to the number of segments and the exposure cut.

The frozen year-t evaluation is repeated for alternative numbers of clusters and
for alternative minimum-exposure thresholds, so that the operational conclusion can
be separated from the two discretionary parameters of the workflow. The rate
ranking does not depend on k; its top-decile results change only with the
minimum exposure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

import cancellation_monitoring as cm
from generate_policy_analyses import to_monitoring_frame
from generate_vra_results import OUT_DIR, load_vra

K_VALUES = [2, 3, 4, 5]
MIN_SCHEDULED_VALUES = [200, 500, 1000]


def main() -> None:
    frame = to_monitoring_frame(load_vra())
    by_year = {int(y): g for y, g in frame.groupby("year")}
    pairs = [(y, y + 1) for y in sorted(by_year) if y + 1 in by_year]

    rows = []
    for k in K_VALUES:
        for min_sched in MIN_SCHEDULED_VALUES:
            spec = cm.CellSpec(f"k={k}, min {min_sched}", min_sched=min_sched)
            results = []
            for base_year, next_year in pairs:
                result = cm.prospective_evaluation(by_year[base_year], by_year[next_year], spec, k=k)
                if result is not None:
                    results.append({"next_year": next_year, **result[0]})
            summary = pd.DataFrame(results)
            if summary.empty:
                continue
            recent, early = summary[summary["next_year"] >= 2020], summary[summary["next_year"] < 2020]
            rows.append({
                "k": k, "min_scheduled": min_sched, "year_pairs": len(summary),
                "mean_watch_lift_2016_2019": float(early["watch_lift"].mean()),
                "mean_watch_lift_2020_2024": float(recent["watch_lift"].mean()),
                "mean_watch_coverage_t1_pct": float(summary["watch_coverage_t1_pct"].mean()),
                "mean_watch_capture_t1_pct": float(summary["watch_capture_t1_pct"].mean()),
                "mean_capture_gap_equal_coverage_pp": float(summary["capture_gap_equal_realized_coverage_pp"].mean()),
                "mean_top_decile_lift_2016_2019": float(early["top_decile_lift"].mean()),
                "mean_top_decile_lift_2020_2024": float(recent["top_decile_lift"].mean()),
                "mean_retained_exposure_pct": float(summary["base_retained_exposure_pct"].mean()),
                "mean_spearman": float(summary["spearman"].mean()),
            })
    pd.DataFrame(rows).to_csv(OUT_DIR / "watchlist_k_exposure_sensitivity.csv", index=False)
    print("Wrote", OUT_DIR / "watchlist_k_exposure_sensitivity.csv")


if __name__ == "__main__":
    main()
