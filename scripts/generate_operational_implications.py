"""Operational quantities for the policy discussion (Section 6.6 and Table B.8).

For every Brazilian year pair, with the baseline prospective design (all list elements
fixed in year t), this script describes the top-decile screening list and the watch
list in the units in which a review program is planned: cells, origin airports,
carrier groups, pooled cells and operating carriers to be contacted, the year t+1
flights and cancellations they hold, and the share of flagged cells whose realized
year t+1 rate exceeded the population rate. It also quantifies two complementary
procedures for the low-frequency services excluded by the minimum-exposure rule:
coarser carrier-airport cells and two-year pooling.

Outputs (analysis_outputs/vra/robustness/):
    operational_workload.csv
    low_frequency_options.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import cancellation_monitoring as cm
from generate_vra_results import OUT_DIR
from vra_frames import brazil_frame

OUT = OUT_DIR / "robustness"
BASELINE = cm.CellSpec("Baseline")
KEYS = BASELINE.keys


def frozen_lists(base_df: pd.DataFrame, next_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Complete year-t lists, rebuilt with the same routines as prospective_evaluation."""
    base_g, _, _ = cm.assign_carrier_groups(base_df, next_df, BASELINE)
    cells = cm.build_cells(base_g, BASELINE.min_sched, KEYS)
    assigned = cm.fit_segments(cells, BASELINE, 3)
    assigned["watch"] = assigned["rate_cluster"] == int(assigned["rate_cluster"].max())
    ordered = cm.ranked_order(assigned, KEYS)
    assigned["top_decile"] = cm.budget_flags(ordered, cm.TOP_SHARE).reindex(assigned.index)
    return assigned, base_g


def workload(br: pd.DataFrame) -> pd.DataFrame:
    by_year = {int(y): g for y, g in br.groupby("year")}
    years = sorted(by_year)
    rows = []
    for t, t1 in zip(years[:-1], years[1:]):
        row, pop = cm.prospective_evaluation(by_year[t], by_year[t1], BASELINE)
        assigned, base_g = frozen_lists(by_year[t], by_year[t1])
        system = row["system_rate_t1_pct"] / 100
        for flag in ("top_decile", "watch"):
            lst = assigned[assigned[flag]]
            assert len(lst) == row[f"{flag}_cells"], "list not reproduced"
            flagged_keys = lst[KEYS].assign(_f=True)
            fl = base_g[KEYS + ["airline"]].merge(flagged_keys, on=KEYS, how="inner")
            pooled = lst["carrier_group"].eq("Others")
            pooled_fl = fl[fl["carrier_group"].eq("Others")]
            # Operating carriers holding at least 5% of the year-t flights of some flagged cell.
            share_in_cell = fl.groupby(KEYS + ["airline"], observed=True).size() / fl.groupby(KEYS, observed=True).size()
            material = share_in_cell[share_in_cell >= 0.05].reset_index()["airline"].nunique()
            ev = pop[pop[flag]]
            s1, c1 = ev["scheduled_t1"].sum(), ev["cancelled_t1"].sum()
            above = ev["rate_pct_t1"] / 100 > system
            rows.append({
                "base_year": t, "next_year": t1, "rule": flag,
                "cells": len(lst), "origin_airports": lst["origin"].nunique(),
                "carrier_groups": lst["carrier_group"].nunique(),
                "pooled_cells": int(pooled.sum()), "pooled_cell_share_pct": float(pooled.mean() * 100),
                "operating_carriers_t": fl["airline"].nunique(),
                "operating_carriers_in_pooled_cells_t": pooled_fl["airline"].nunique(),
                "operating_carriers_ge5pct_of_a_cell_t": int(material),
                "scheduled_t": int(lst["scheduled"].sum()),
                "evaluated_cells": len(ev),
                "scheduled_t1": int(s1), "cancelled_t1": int(c1),
                "population_rate_t1_pct": float(system * 100),
                "expected_t1_at_population_rate": float(s1 * system),
                "above_benchmark_t1": float(c1 - s1 * system),
                "share_cells_above_population_rate_pct": float(above.mean() * 100),
                "share_flights_in_cells_above_population_rate_pct": float(ev.loc[above, "scheduled_t1"].sum() / s1 * 100),
                "share_cells_above_twice_population_rate_pct": float((ev["rate_pct_t1"] / 100 > 2 * system).mean() * 100),
                "lift": row[f"{flag}_lift"], "national_capture_pct": row[f"{flag}_national_capture_pct"],
            })
    return pd.DataFrame(rows)


def low_frequency(br: pd.DataFrame) -> pd.DataFrame:
    """Excluded single-year services (fixed grouping, as in the exclusions table) and two complementary procedures."""
    rows = []
    years = sorted(br["year"].unique())
    for y in years:
        cur = br[br["year"] == y]
        n, c = len(cur), cur["is_cancelled"].sum()
        cell_size = cur.groupby(KEYS, observed=True)["is_cancelled"].transform("size")
        excl = cell_size < cm.MIN_SCHEDULED
        ca_size = cur.groupby(["carrier_group", "origin"], observed=True)["is_cancelled"].transform("size")
        ap_size = cur.groupby("origin", observed=True)["is_cancelled"].transform("size")
        rec = {
            "year": int(y),
            "excluded_flights_pct": float(excl.mean() * 100),
            "excluded_cancellations_pct": float(cur.loc[excl, "is_cancelled"].sum() / c * 100),
            "excluded_rate_pct": float(cur.loc[excl, "is_cancelled"].mean() * 100),
            "excluded_in_carrier_airport_ge500_pct": float((excl & (ca_size >= cm.MIN_SCHEDULED)).sum() / excl.sum() * 100),
            "excluded_canc_in_carrier_airport_ge500_pct": float(cur.loc[excl & (ca_size >= cm.MIN_SCHEDULED), "is_cancelled"].sum()
                                                                / cur.loc[excl, "is_cancelled"].sum() * 100),
            "excluded_in_airport_ge500_pct": float((excl & (ap_size >= cm.MIN_SCHEDULED)).sum() / excl.sum() * 100),
        }
        if y - 1 in years:
            two = br[br["year"].isin([y - 1, y])]
            two_size = two.groupby(KEYS, observed=True)["is_cancelled"].size().rename("two_year")
            joined = cur[KEYS].merge(two_size.reset_index(), on=KEYS, how="left")["two_year"].to_numpy()
            pooled_ok = excl.to_numpy() & (joined >= cm.MIN_SCHEDULED)
            rec["excluded_in_two_year_cells_ge500_pct"] = float(pooled_ok.sum() / excl.sum() * 100)
            rec["excluded_canc_in_two_year_cells_ge500_pct"] = float(cur.loc[pooled_ok, "is_cancelled"].sum()
                                                                     / cur.loc[excl, "is_cancelled"].sum() * 100)
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    br, _ = brazil_frame(save_reference=False)
    cols = ["airline", "carrier_group_fixed", "carrier_group", "operator_category", "origin", "dest", "day_type",
            "departure_period", "quarter", "month", "is_cancelled", "year"]
    lean = br[cols]
    work = workload(lean.drop(columns="carrier_group"))
    work.to_csv(OUT / "operational_workload.csv", index=False)
    low = low_frequency(lean)
    low.to_csv(OUT / "low_frequency_options.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(work.round(2).to_string(index=False))
    print(low.round(2).to_string(index=False))
    print("means:\n", low.mean(numeric_only=True).round(2))


if __name__ == "__main__":
    main()
