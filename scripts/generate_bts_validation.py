"""External validation of the monitoring workflow on the US DOT/BTS registry.

The same operational-cell routines applied to the Brazilian ANAC/VRA records are
applied here to the US Bureau of Transportation Statistics On-Time Reporting
Carrier On-Time Performance files. Three evaluation years represent the
pre-pandemic, pandemic and post-pandemic phases, each paired with the preceding
year so that the prospective validation can be replicated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

import cancellation_monitoring as cm
from generate_vra_results import FIG_DIRS, savefig
from paths import ANALYSIS_DIR, BTS_DIR

RAW_DIR = BTS_DIR
OUT_DIR = ANALYSIS_DIR / "bts"
VRA_OUT = ANALYSIS_DIR / "vra"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = cm.RANDOM_STATE
K_RANGE = range(2, 11)
CARRIER_SHARE_THRESHOLD = 0.05
PERIOD_YEARS = {"Pre-pandemic": 2019, "Pandemic": 2021, "Post-pandemic": 2024}
YEAR_PAIRS = [(2018, 2019), (2020, 2021), (2023, 2024)]


def load_bts() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"No BTS extracts found in {RAW_DIR}")
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)

    hour = (df["CRSDepTime"] // 100).astype("float")
    hour = hour.where(hour < 24, 0.0)
    df["departure_period"] = pd.cut(
        hour,
        bins=[0, 6, 12, 18, 24],
        labels=["Night", "Morning", "Afternoon", "Evening"],
        include_lowest=True,
        right=False,
    ).astype("string")
    df["day_type"] = np.where(df["DayOfWeek"].isin([6, 7]), "Weekend", "Weekday")
    df["is_cancelled"] = df["Cancelled"].astype(float).eq(1.0)
    df["year"] = df["Year"].astype(int)
    df = df.rename(columns={"Origin": "origin"})

    # Full-period grouping, used for the descriptive tables: carriers holding at
    # least 5% of the registry's scheduled flights are kept individually and the
    # remainder is pooled. The prospective analysis instead fixes the groups with
    # the assignment year's shares (cancellation_monitoring.assign_carrier_groups).
    df["airline"] = df["Reporting_Airline"].astype(str)
    share = df["airline"].value_counts(normalize=True)
    major = set(share[share >= CARRIER_SHARE_THRESHOLD].index)
    df["carrier_group_fixed"] = np.where(df["airline"].isin(major), df["airline"], "Others")
    df["carrier_group"] = df["carrier_group_fixed"]

    cols = ["airline", "carrier_group", "carrier_group_fixed", "origin", "day_type", "departure_period", "is_cancelled", "year"]
    return df[cols].dropna().reset_index(drop=True)


def descriptive_summary(frame: pd.DataFrame) -> None:
    rows = []
    for year, group in frame.groupby("year"):
        rows.append(
            {
                "year": int(year),
                "scheduled_flights": len(group),
                "cancellations": int(group["is_cancelled"].sum()),
                "cancellation_rate_pct": float(group["is_cancelled"].mean() * 100),
                "carriers": int(group["carrier_group"].nunique()),
                "origin_airports": int(group["origin"].nunique()),
            }
        )
    pd.DataFrame(rows).to_csv(OUT_DIR / "bts_yearly_summary.csv", index=False)


def k_selection(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, year in PERIOD_YEARS.items():
        cells = cm.build_cells(frame[frame["year"] == year])
        if len(cells) < max(K_RANGE) + 1:
            continue
        feat, _ = cm.cell_features(cells)
        x, _ = cm.scaled_matrix(feat)
        table = cm.validity_indices(x, K_RANGE)
        table.insert(0, "period", period)
        table.insert(1, "year", year)
        table["n_cells"] = len(cells)
        rows.append(table)
    result = pd.concat(rows, ignore_index=True)
    result.to_csv(OUT_DIR / "bts_cell_cluster_validity_indices.csv", index=False)

    picks = []
    for (period, year), group in result.groupby(["period", "year"], sort=False):
        picks.append(
            {
                "period": period,
                "year": int(year),
                "best_silhouette_k": int(group.loc[group["silhouette"].idxmax(), "k"]),
                "best_calinski_k": int(group.loc[group["calinski_harabasz"].idxmax(), "k"]),
                "best_davies_bouldin_k": int(group.loc[group["davies_bouldin"].idxmin(), "k"]),
                "gap_criterion_k": cm.first_gap_k(group[["k", "gap", "gap_sk"]]),
                "best_gmm_bic_k": int(group.loc[group["gmm_bic"].idxmin(), "k"]),
            }
        )
    pd.DataFrame(picks).to_csv(OUT_DIR / "bts_cell_cluster_k_selection.csv", index=False)
    return result


def cell_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    rows, metric_rows, segment_rows = [], [], []
    for period, year in PERIOD_YEARS.items():
        cells = cm.build_cells(frame[frame["year"] == year])
        if len(cells) < 10:
            continue
        feat, _ = cm.cell_features(cells)
        x, _ = cm.scaled_matrix(feat)
        assigned = cm.fit_watchlist(cells)
        silhouette = cm.validity_indices(x, [3]).loc[0, "silhouette"]
        system_rate = assigned["cancelled"].sum() / assigned["scheduled"].sum() * 100
        for rank, group in assigned.groupby("rate_cluster"):
            rows.append(
                {
                    "period": period,
                    "year": year,
                    "rate_cluster": int(rank),
                    "n_cells": len(group),
                    "scheduled": int(group["scheduled"].sum()),
                    "cancelled": int(group["cancelled"].sum()),
                    "weighted_rate_pct": cm.weighted_rate(group),
                    "share_of_scheduled_pct": float(group["scheduled"].sum() / assigned["scheduled"].sum() * 100),
                    "share_of_cancellations_pct": float(group["cancelled"].sum() / assigned["cancelled"].sum() * 100),
                    "dominant_carrier": group.groupby("carrier_group")["scheduled"].sum().idxmax(),
                    "dominant_day_type": group.groupby("day_type")["scheduled"].sum().idxmax(),
                    "dominant_departure_period": group.groupby("departure_period")["scheduled"].sum().idxmax(),
                    "silhouette_k3": float(silhouette),
                    "system_rate_pct": float(system_rate),
                }
            )
        metrics, labels = cm.compare_algorithms(x, k=3)
        metrics.insert(0, "period", period)
        metric_rows.append(metrics)
        segments = cm.high_rate_agreement(cells, labels)
        segments.insert(0, "period", period)
        segment_rows.append(segments)

    profiles = pd.DataFrame(rows)
    profiles.to_csv(OUT_DIR / "bts_rate_cell_cluster_profiles.csv", index=False)
    pd.concat(metric_rows, ignore_index=True).to_csv(OUT_DIR / "bts_cell_algorithm_comparison.csv", index=False)
    pd.concat(segment_rows, ignore_index=True).to_csv(OUT_DIR / "bts_cell_algorithm_high_rate_segment.csv", index=False)
    return profiles


def out_of_sample(frame: pd.DataFrame) -> pd.DataFrame:
    """Frozen year-t lists scored on year t+1, with the same routine used for Brazil."""
    by_year = {int(y): g for y, g in frame.groupby("year")}
    spec = cm.CellSpec("Baseline")
    rows, segments = [], []
    for base_year, next_year in YEAR_PAIRS:
        result = cm.prospective_evaluation(by_year[base_year], by_year[next_year], spec)
        if result is None:
            continue
        row, pop = result
        rows.append({"base_year": base_year, "next_year": next_year, **row})
        for rank, g in pop.groupby("rate_cluster"):
            segments.append({"base_year": base_year, "next_year": next_year, "rate_cluster": int(rank),
                             "rate_t1_pct": g["cancelled_t1"].sum() / g["scheduled_t1"].sum() * 100,
                             "coverage_t1_pct": g["scheduled_t1"].sum() / pop["scheduled_t1"].sum() * 100})
    summary = pd.DataFrame(rows)
    pd.DataFrame(segments).to_csv(OUT_DIR / "bts_watchlist_out_of_sample_clusters.csv", index=False)
    summary.to_csv(OUT_DIR / "bts_watchlist_out_of_sample_summary.csv", index=False)
    return summary


def cross_country_table(profiles: pd.DataFrame, us_summary: pd.DataFrame) -> pd.DataFrame:
    """Align the Brazilian and US watch-list results in a single comparison table."""
    rows = []
    br_acc = pd.read_csv(VRA_OUT / "watchlist_policy_accounting.csv")
    br_sil = pd.read_csv(VRA_OUT / "cell_cluster_validity_indices.csv")
    br_sil3 = br_sil[br_sil["k"] == 3].set_index("period")["silhouette"]
    for _, row in br_acc.iterrows():
        rows.append(
            {
                "country": "Brazil (ANAC/VRA)",
                "period": row["period"],
                "cells": int(row["cells"]),
                "silhouette_k3": float(br_sil3.get(row["period"], float("nan"))),
                "system_rate_pct": row["system_rate_pct"],
                "watchlist_rate_pct": row["watchlist_rate_pct"],
                "lift": row["watchlist_rate_pct"] / row["system_rate_pct"],
                "share_of_scheduled_pct": row["share_of_scheduled_pct"],
                "share_of_cancellations_pct": row["share_of_cancellations_pct"],
            }
        )
    for period in PERIOD_YEARS:
        sub = profiles[profiles["period"] == period]
        if sub.empty:
            continue
        top = sub[sub["rate_cluster"] == sub["rate_cluster"].max()].iloc[0]
        rows.append(
            {
                "country": "United States (DOT/BTS)",
                "period": period,
                "cells": int(sub["n_cells"].sum()),
                "silhouette_k3": float(top["silhouette_k3"]),
                "system_rate_pct": float(top["system_rate_pct"]),
                "watchlist_rate_pct": float(top["weighted_rate_pct"]),
                "lift": float(top["weighted_rate_pct"] / top["system_rate_pct"]),
                "share_of_scheduled_pct": float(top["share_of_scheduled_pct"]),
                "share_of_cancellations_pct": float(top["share_of_cancellations_pct"]),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "cross_country_watchlist_comparison.csv", index=False)

    br_oos = pd.read_csv(VRA_OUT / "watchlist_out_of_sample_summary.csv")
    oos = pd.concat(
        [
            br_oos.assign(country="Brazil (ANAC/VRA)"),
            us_summary.assign(country="United States (DOT/BTS)"),
        ],
        ignore_index=True,
    )
    oos.to_csv(OUT_DIR / "cross_country_out_of_sample.csv", index=False)
    return table


def figure_cross_country(table: pd.DataFrame) -> None:
    periods = ["Pre-pandemic", "Pandemic", "Post-pandemic"]
    countries = ["Brazil (ANAC/VRA)", "United States (DOT/BTS)"]
    colors = {"Brazil (ANAC/VRA)": "#1f5a8a", "United States (DOT/BTS)": "#9d4d3f"}
    positions = np.arange(len(periods))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    for offset, country in zip([-width / 2, width / 2], countries):
        sub = table[table["country"] == country].set_index("period").reindex(periods)
        axes[0].bar(positions + offset, sub["watchlist_rate_pct"], width=width, color=colors[country], label=country)
        axes[0].plot(positions + offset, sub["system_rate_pct"], linestyle="none", marker="_", markersize=18, color="black")
        axes[1].bar(positions + offset, sub["share_of_cancellations_pct"], width=width, color=colors[country], label=country)
        axes[1].plot(positions + offset, sub["share_of_scheduled_pct"], linestyle="none", marker="_", markersize=18, color="black")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(periods)
    axes[0].set_xlabel("Period")
    axes[0].set_ylabel("Cancellation rate (%)")
    axes[0].legend(fontsize=14, frameon=False)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(periods)
    axes[1].set_xlabel("Period")
    axes[1].set_ylabel("Watch-list share (%)")
    axes[1].legend(fontsize=14, frameon=False)
    axes[1].grid(axis="y", alpha=0.25)
    savefig("figure31.pdf")

    short_periods = ["Pre", "Pandemic", "Post"]
    short_country = {"Brazil (ANAC/VRA)": "Brazil", "United States (DOT/BTS)": "United States"}
    plt.figure(figsize=(4.9, 3.7))
    for offset, country in zip([-width / 2, width / 2], countries):
        sub = table[table["country"] == country].set_index("period").reindex(periods)
        plt.bar(positions + offset, sub["watchlist_rate_pct"], width=width, color=colors[country], label=short_country[country])
        plt.plot(positions + offset, sub["system_rate_pct"], linestyle="none", marker="_", markersize=14, color="black")
    plt.xticks(positions, short_periods)
    plt.xlabel("Period")
    plt.ylabel("Cancellation rate (%)")
    plt.legend(fontsize=11, frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.01))
    plt.grid(axis="y", alpha=0.25)
    savefig("figure31a.pdf")

    plt.figure(figsize=(4.9, 3.7))
    for offset, country in zip([-width / 2, width / 2], countries):
        sub = table[table["country"] == country].set_index("period").reindex(periods)
        plt.bar(positions + offset, sub["share_of_cancellations_pct"], width=width, color=colors[country], label=short_country[country])
        plt.plot(positions + offset, sub["share_of_scheduled_pct"], linestyle="none", marker="_", markersize=14, color="black")
    plt.xticks(positions, short_periods)
    plt.xlabel("Period")
    plt.ylabel("Watch-list share (%)")
    plt.legend(fontsize=11, frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.01))
    plt.grid(axis="y", alpha=0.25)
    savefig("figure31b.pdf")


def main() -> None:
    frame = load_bts()
    descriptive_summary(frame)
    print("summary done", flush=True)
    k_selection(frame)
    print("k-selection done", flush=True)
    profiles = cell_profiles(frame)
    print("profiles done", flush=True)
    us_summary = out_of_sample(frame)
    print("out-of-sample done", flush=True)
    table = cross_country_table(profiles, us_summary)
    figure_cross_country(table)
    print("Outputs written to", OUT_DIR)
    print("Figures written to", ", ".join(str(p) for p in FIG_DIRS))


if __name__ == "__main__":
    main()
