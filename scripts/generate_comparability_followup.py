"""Follow-up tables for the comparability analysis.

* carrier-group rates and phase segments for Brazilian domestic passenger flights;
* the operator composition of the top exposure decile in each Brazilian pair;
* the make-up of each year's high-rate segment.

Run after ``generate_robustness_analyses.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

import cancellation_monitoring as cm
from generate_robustness_analyses import BASELINE, CARRIER_NAMES, OUT, year_cells, year_pairs
from vra_frames import brazil_frame


def domestic_passenger(br: pd.DataFrame) -> None:
    """Descriptive rates and phase segments with the full-period grouping."""
    dp = br[br["is_domestic"] & ~br["is_cargo_line"]].assign(carrier_group=lambda d: d["carrier_group_fixed"])
    rates = dp.groupby(["period", "carrier_group"]).agg(scheduled=("is_cancelled", "size"), cancelled=("is_cancelled", "sum"))
    rates["rate_pct"] = rates["cancelled"] / rates["scheduled"] * 100
    rates.to_csv(OUT / "domestic_passenger_carrier_rates.csv")
    dp.groupby("year")["is_cancelled"].agg(["size", "sum", "mean"]).to_csv(OUT / "domestic_passenger_yearly.csv")

    rows = []
    for phase in ["Pre-pandemic", "Pandemic", "Post-pandemic"]:
        sub = dp[dp["period"] == phase]
        assigned = cm.fit_segments(cm.build_cells(sub), BASELINE)
        system = assigned["cancelled"].sum() / assigned["scheduled"].sum() * 100
        for rank, g in assigned.groupby("rate_cluster"):
            rows.append({
                "phase": phase, "segment": int(rank), "cells": len(g), "rate_pct": cm.weighted_rate(g),
                "system_rate_pct": system, "share_scheduled_pct": g["scheduled"].sum() / assigned["scheduled"].sum() * 100,
                "share_cancelled_pct": g["cancelled"].sum() / assigned["cancelled"].sum() * 100,
                "dominant_group": g.groupby("carrier_group")["scheduled"].sum().idxmax(),
                "others_share_pct": g.loc[g["carrier_group"] == "Others", "scheduled"].sum() / g["scheduled"].sum() * 100,
                "dominant_period": g.groupby("departure_period")["scheduled"].sum().idxmax(),
            })
    pd.DataFrame(rows).to_csv(OUT / "domestic_passenger_phase_segments.csv", index=False)


def top_decile_composition(br: pd.DataFrame) -> None:
    """Operator make-up of the evaluation-year flights in the frozen top-decile cells."""
    by_year = {int(y): g for y, g in br.groupby("year")}
    rows = []
    for base_year, next_year in year_pairs(br):
        _, pop, nxt = cm.prospective_evaluation(by_year[base_year], by_year[next_year], BASELINE, return_flights=True)
        sub = nxt.loc[cm.flagged_flights(nxt, pop, BASELINE.keys, "top_decile")]
        label = np.where(sub["carrier_group"] == "Others", "Others: " + sub["operator_category"].astype(str), "Retained carriers")
        shares = pd.Series(label).value_counts(normalize=True) * 100
        rows.append({"base_year": base_year, "next_year": next_year, "flights": len(sub),
                     "international_share_pct": (1 - sub["is_domestic"].mean()) * 100,
                     "cargo_line_share_pct": sub["is_cargo_line"].mean() * 100, **shares.to_dict()})
    pd.DataFrame(rows).fillna(0).to_csv(OUT / "top_decile_composition.csv", index=False)


def yearly_watchlist_composition(br: pd.DataFrame) -> None:
    """Carrier and departure-period make-up of each year's high-rate segment (year-t grouping)."""
    rows = []
    for year, sub in br.groupby("year"):
        assigned = cm.fit_segments(year_cells(sub, BASELINE), BASELINE)
        top = assigned[assigned["rate_cluster"] == assigned["rate_cluster"].max()]
        rest = assigned[assigned["rate_cluster"] < assigned["rate_cluster"].max()]
        shares = top.groupby("carrier_group")["scheduled"].sum() / top["scheduled"].sum() * 100
        rows.append({
            "year": int(year), "cells": len(top), "rate_pct": cm.weighted_rate(top), "rest_rate_pct": cm.weighted_rate(rest),
            "coverage_pct": top["scheduled"].sum() / assigned["scheduled"].sum() * 100,
            "min_cell_rate_pct": top["rate_pct"].min(), "median_cell_rate_pct": top["rate_pct"].median(),
            "rest_median_cell_rate_pct": rest["rate_pct"].median(),
            "dominant_period": top.groupby("departure_period")["scheduled"].sum().idxmax(),
            "weekend_share_pct": top.loc[top["day_type"] == "Weekend", "scheduled"].sum() / top["scheduled"].sum() * 100,
            **{f"share_{CARRIER_NAMES.get(g, g)}_pct": v for g, v in shares.items()},
        })
    pd.DataFrame(rows).fillna(0).to_csv(OUT / "yearly_watchlist_composition.csv", index=False)


def main_groups_share(br: pd.DataFrame) -> None:
    """Share of flagged evaluation-year flights operated by Azul, Gol and TAM, pooled and individual carriers."""
    by_year = {int(y): g for y, g in br.groupby("year")}
    rows = []
    for base_year, next_year in year_pairs(br):
        rec = {"pair": f"{base_year}-{next_year}"}
        for name, spec in (("base", BASELINE), ("ind", cm.CellSpec("Individual carriers", carrier_rule="individual"))):
            _, pop, nxt = cm.prospective_evaluation(by_year[base_year], by_year[next_year], spec, return_flights=True)
            for flag in ("watch", "top_decile"):
                flagged = nxt.loc[cm.flagged_flights(nxt, pop, spec.keys, flag)]
                rec[f"{name}_{flag}_main3"] = round(flagged["airline"].isin(["AZU", "GLO", "TAM"]).mean() * 100, 1)
        rows.append(rec)
    pd.DataFrame(rows).to_csv(OUT / "main_groups_share_flagged.csv", index=False)


def main() -> None:
    br, _ = brazil_frame(save_reference=False)
    domestic_passenger(br)
    top_decile_composition(br)
    yearly_watchlist_composition(br)
    main_groups_share(br)
    print("Follow-up outputs written to", OUT)


if __name__ == "__main__":
    main()
