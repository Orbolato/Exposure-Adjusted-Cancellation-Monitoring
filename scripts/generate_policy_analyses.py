"""Cluster-validity, algorithm-comparison and out-of-sample analyses (Brazil).

The script extends the descriptive workflow with the diagnostics required to
treat the exposure-based operational-cell clustering as a monitoring instrument:
a data-driven choice of the number of clusters, a comparison against alternative
partitioning algorithms, and an out-of-sample test of the high-rate watch list.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

import cancellation_monitoring as cm
from generate_vra_results import (
    FIG_DIRS,
    FINAL_FEATURES,
    OUT_DIR,
    PERIOD_ORDER,
    encoded_matrix,
    load_vra,
    make_frequency_maps,
    period_frame,
    savefig,
)

RANDOM_STATE = cm.RANDOM_STATE
K_RANGE = range(2, 11)
# Ward linkage and k-medoids build a full pairwise distance matrix, so the
# event-level benchmark comparison runs on a fixed random sample.
EVENT_SAMPLE = 5_000


def to_monitoring_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Map the harmonized VRA table onto the registry-independent column names.

    ``carrier_group`` is the full-period grouping used for descriptive tables;
    ``airline`` allows the prospective analysis to fix groups from period t alone.
    """
    sb = df[df["is_br_origin"]].copy()
    sb["carrier_group"] = sb["airline_grouped"]
    sb["carrier_group_fixed"] = sb["airline_grouped"]
    cols = ["airline", "carrier_group", "carrier_group_fixed", "origin", "day_type", "departure_period", "is_cancelled", "year", "period"]
    return sb[cols].dropna()


# ------------------------------------------------------------------ k selection


def k_selection(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period in PERIOD_ORDER:
        sub = frame if period == "Overall" else frame[frame["period"] == period]
        cells = cm.build_cells(sub)
        if len(cells) < max(K_RANGE) + 1:
            continue
        feat, _ = cm.cell_features(cells)
        x, _ = cm.scaled_matrix(feat)
        table = cm.validity_indices(x, K_RANGE)
        table.insert(0, "period", period)
        table["n_cells"] = len(cells)
        rows.append(table)
    result = pd.concat(rows, ignore_index=True)
    result.to_csv(OUT_DIR / "cell_cluster_validity_indices.csv", index=False)

    picks = []
    for period, group in result.groupby("period", sort=False):
        picks.append(
            {
                "period": period,
                "best_silhouette_k": int(group.loc[group["silhouette"].idxmax(), "k"]),
                "best_calinski_k": int(group.loc[group["calinski_harabasz"].idxmax(), "k"]),
                "best_davies_bouldin_k": int(group.loc[group["davies_bouldin"].idxmin(), "k"]),
                "gap_criterion_k": cm.first_gap_k(group[["k", "gap", "gap_sk"]]),
                "best_gmm_bic_k": int(group.loc[group["gmm_bic"].idxmin(), "k"]),
            }
        )
    pd.DataFrame(picks).to_csv(OUT_DIR / "cell_cluster_k_selection.csv", index=False)
    return result


def figure_k_selection(table: pd.DataFrame) -> None:
    metrics = [
        ("silhouette", "Silhouette coefficient"),
        ("calinski_harabasz", "Calinski-Harabasz index"),
        ("davies_bouldin", "Davies-Bouldin index"),
        ("gap", "Gap statistic"),
    ]
    colors = {"Overall": "#1f5a8a", "Pre-pandemic": "#6f7d3c", "Pandemic": "#9d4d3f", "Post-pandemic": "#5d4b8a"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (metric, label) in zip(axes.ravel(), metrics):
        for period in PERIOD_ORDER:
            sub = table[table["period"] == period]
            if sub.empty:
                continue
            ax.plot(sub["k"], sub[metric], marker="o", color=colors[period], label=period)
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.25)
    axes.ravel()[0].legend(fontsize=14, frameon=False)
    savefig("figure29.pdf")

    for (metric, label), suffix in zip(metrics, ["a", "b", "c", "d"]):
        plt.figure(figsize=(4.8, 3.6))
        for period in PERIOD_ORDER:
            sub = table[table["period"] == period]
            if sub.empty:
                continue
            plt.plot(sub["k"], sub[metric], marker="o", color=colors[period], label=period)
        plt.xlabel("Number of clusters")
        plt.ylabel(label)
        plt.grid(axis="y", alpha=0.25)
        if suffix == "a":
            plt.legend(fontsize=11, frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.01))
        savefig(f"figure29{suffix}.pdf")


# -------------------------------------------------------- algorithm comparison


def algorithm_comparison(frame: pd.DataFrame, canc_main: pd.DataFrame) -> None:
    metric_rows, segment_rows = [], []
    for period in PERIOD_ORDER:
        sub = frame if period == "Overall" else frame[frame["period"] == period]
        cells = cm.build_cells(sub)
        if len(cells) < 10:
            continue
        feat, _ = cm.cell_features(cells)
        x, _ = cm.scaled_matrix(feat)
        metrics, labels = cm.compare_algorithms(x, k=3)
        metrics.insert(0, "period", period)
        metric_rows.append(metrics)
        segments = cm.high_rate_agreement(cells, labels)
        segments.insert(0, "period", period)
        segment_rows.append(segments)
    pd.concat(metric_rows, ignore_index=True).to_csv(OUT_DIR / "cell_algorithm_comparison.csv", index=False)
    pd.concat(segment_rows, ignore_index=True).to_csv(OUT_DIR / "cell_algorithm_high_rate_segment.csv", index=False)

    # Same comparison for the canceled-event benchmark model, on a fixed sample.
    rng = np.random.default_rng(RANDOM_STATE)
    event_rows = []
    maps = make_frequency_maps(canc_main[FINAL_FEATURES].dropna())
    for period in PERIOD_ORDER:
        data = period_frame(canc_main, period)[FINAL_FEATURES].dropna()
        if len(data) > EVENT_SAMPLE:
            data = data.iloc[rng.choice(len(data), size=EVENT_SAMPLE, replace=False)]
        x = encoded_matrix(data, "freq", maps)
        metrics, _ = cm.compare_algorithms(x, k=3)
        metrics.insert(0, "period", period)
        metrics["sample_records"] = len(data)
        event_rows.append(metrics)
    pd.concat(event_rows, ignore_index=True).to_csv(OUT_DIR / "event_algorithm_comparison.csv", index=False)


# ----------------------------------------------------------- out-of-sample test


def out_of_sample(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Frozen year-t lists scored on year t+1 for every consecutive pair of years."""
    by_year = {int(y): g for y, g in frame.groupby("year")}
    pairs = [(y, y + 1) for y in sorted(by_year) if y + 1 in by_year]
    spec = cm.CellSpec("Baseline")
    summary_rows, segment_rows, curves = [], [], []
    for base_year, next_year in pairs:
        result = cm.prospective_evaluation(by_year[base_year], by_year[next_year], spec)
        if result is None:
            continue
        row, pop = result
        summary_rows.append({"base_year": base_year, "next_year": next_year, **row})
        for rank, g in pop.groupby("rate_cluster"):
            segment_rows.append({"base_year": base_year, "next_year": next_year, "rate_cluster": int(rank),
                                 "rate_t1_pct": g["cancelled_t1"].sum() / g["scheduled_t1"].sum() * 100,
                                 "coverage_t1_pct": g["scheduled_t1"].sum() / pop["scheduled_t1"].sum() * 100})
        curve = cm.ranking_curve(pop, max_share=0.5)
        curve["base_year"], curve["next_year"] = base_year, next_year
        curves.append(curve)
    summary, segments, curves = pd.DataFrame(summary_rows), pd.DataFrame(segment_rows), pd.concat(curves, ignore_index=True)
    summary.to_csv(OUT_DIR / "watchlist_out_of_sample_summary.csv", index=False)
    segments.to_csv(OUT_DIR / "watchlist_out_of_sample_clusters.csv", index=False)
    curves.to_csv(OUT_DIR / "ranking_capture_curves.csv", index=False)
    return segments, summary, curves


def figure_out_of_sample(cluster_table: pd.DataFrame, summary: pd.DataFrame, curves: pd.DataFrame) -> None:
    short = {1: "Low-rate", 2: "Intermediate-rate", 3: "High-rate"}
    colors = {1: "#6f7d3c", 2: "#1f5a8a", 3: "#9d4d3f"}
    pivot = cluster_table.pivot(index="next_year", columns="rate_cluster", values="rate_t1_pct")
    positions = np.arange(len(pivot.index))
    width = 0.27

    plt.figure(figsize=(4.9, 3.7))
    for offset, cluster in zip([-width, 0.0, width], sorted(pivot.columns)):
        plt.bar(positions + offset, pivot[cluster], width=width, color=colors[cluster], label=short[cluster])
    plt.xticks(positions, pivot.index.astype(int), rotation=45)
    plt.xlabel("Evaluation year (t+1)")
    plt.ylabel("Realized rate (%)")
    plt.legend(fontsize=10, frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.01), columnspacing=0.8, handlelength=1.2)
    plt.grid(axis="y", alpha=0.25)
    savefig("figure30a.pdf")

    plt.figure(figsize=(4.9, 3.7))
    for (b, n), g in curves.groupby(["base_year", "next_year"]):
        late = n >= 2020
        plt.plot(g["coverage_t1_pct"], g["capture_t1_pct"], color="#9d4d3f" if late else "#1f5a8a", linewidth=1.1, alpha=0.8,
                 label=None)
    plt.plot([], [], color="#1f5a8a", linewidth=1.3, label="Ranking, t+1 in 2016-19")
    plt.plot([], [], color="#9d4d3f", linewidth=1.3, label="Ranking, t+1 in 2020-24")
    for late, marker in ((False, "o"), (True, "s")):
        s = summary[(summary["next_year"] >= 2020) == late]
        plt.scatter(s["watch_coverage_t1_pct"], s["watch_capture_t1_pct"], marker=marker, s=28, zorder=3,
                    facecolor="white", edgecolor="#9d4d3f" if late else "#1f5a8a", linewidth=1.3,
                    label="Watch list, t+1 in 2020-24" if late else "Watch list, t+1 in 2016-19")
    plt.plot([0, 50], [0, 50], color="#888888", linestyle=":", linewidth=1)
    plt.xlim(0, 40)
    plt.ylim(0, 80)
    plt.xlabel("Flights covered (%)")
    plt.ylabel("Cancellations captured (%)")
    plt.legend(fontsize=9, frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(-0.02, 1.0), columnspacing=0.8, handlelength=1.4)
    plt.grid(alpha=0.25)
    savefig("figure30b.pdf")


# ------------------------------------------------------------ policy accounting


def policy_accounting(frame: pd.DataFrame) -> None:
    """Exposure accounting for the high-rate segment, by period."""
    rows = []
    for period in PERIOD_ORDER:
        sub = frame if period == "Overall" else frame[frame["period"] == period]
        cells = cm.build_cells(sub)
        if len(cells) < 10:
            continue
        assigned = cm.fit_watchlist(cells)
        top = assigned[assigned["rate_cluster"] == assigned["rate_cluster"].max()]
        system_rate = assigned["cancelled"].sum() / assigned["scheduled"].sum()
        rows.append(
            {
                "period": period,
                "cells": len(assigned),
                "watchlist_cells": len(top),
                "watchlist_scheduled": int(top["scheduled"].sum()),
                "watchlist_cancelled": int(top["cancelled"].sum()),
                "watchlist_rate_pct": cm.weighted_rate(top),
                "system_rate_pct": float(system_rate * 100),
                "share_of_scheduled_pct": float(top["scheduled"].sum() / assigned["scheduled"].sum() * 100),
                "share_of_cancellations_pct": float(top["cancelled"].sum() / assigned["cancelled"].sum() * 100),
                "excess_cancellations_vs_system": int(round(top["cancelled"].sum() - top["scheduled"].sum() * system_rate)),
            }
        )
    pd.DataFrame(rows).to_csv(OUT_DIR / "watchlist_policy_accounting.csv", index=False)


def main() -> None:
    df = load_vra()
    frame = to_monitoring_frame(df)
    canc = df[df["is_cancelled"] & df["is_br_origin"]].copy()
    counts = canc["origin"].value_counts()
    cumulative = counts.cumsum() / counts.sum()
    selected = counts.loc[cumulative <= 0.80].index.tolist()
    canc_main = canc[canc["origin"].isin(selected)].copy()

    validity = k_selection(frame)
    figure_k_selection(validity)
    print("k-selection done", flush=True)
    algorithm_comparison(frame, canc_main)
    print("algorithm comparison done", flush=True)
    cluster_table, summary, curves = out_of_sample(frame)
    figure_out_of_sample(cluster_table, summary, curves)
    print("out-of-sample done", flush=True)
    policy_accounting(frame)

    print("Outputs written to", OUT_DIR)
    print("Figures written to", ", ".join(str(p) for p in FIG_DIRS))


if __name__ == "__main__":
    main()
