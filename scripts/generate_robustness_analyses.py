"""Registry-comparability diagnostics and robustness analyses of the screening rules.

The script complements ``generate_policy_analyses.py`` with the analyses needed
to judge how far the monitoring results depend on the registry and on the
analyst's choices:

* the scope of the registry by designation (DI) and status, and monthly
  diagnostics around the documented changes in how the VRA is produced (SIROS
  registration applicable from 25 March 2018; operated flights taken from the
  statistical database from April 2020);
* the cancellation justification codes of IAC 1504, used descriptively for the
  months in which they are published (January 2015 to March 2020);
* the composition of the pooled Others carrier group and what the
  minimum-exposure rule removes;
* the prospective validation repeated under alternative cell definitions,
  carrier groupings, departure windows, clocks, populations and segmentation
  choices, with the flight-level overlap of each alternative with the baseline;
* airport-cluster bootstrap intervals for the watch list and the rate ranking;
* complete-list and conditional stability of the flagged sets, with entries and
  exits;
* the prospective validation within and across registry windows.

Every prospective evaluation uses ``cancellation_monitoring.prospective_evaluation``,
which fixes groups, eligibility, labels, rankings and budgets in period t.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(str(Path(__file__).resolve().parent))

import cancellation_monitoring as cm
from generate_vra_results import OUT_DIR as VRA_OUT, load_vra, savefig
from vra_frames import CODE_GROUPS, GROUP_ORDER, brazil_frame

OUT = VRA_OUT / "robustness"
OUT.mkdir(parents=True, exist_ok=True)
SEED = cm.RANDOM_STATE
N_BOOT = 2000
BASELINE = cm.CellSpec("Baseline")

# IAC 1504, Annex 2: justification codes for canceled flights or stops. The labels
# below are written into reason_codes_overall.csv and keep the spelling of the
# archived results.
CODE_LABELS = {
    "XA": "Scheduled - national holiday",
    "XB": "Authorized",
    "XI": "Origin airport closed",
    "XJ": "Destination airport closed",
    "XL": "No booked passengers (regional lines)",
    "XM": "Aircraft connection - outbound cancelled, airport closed",
    "XN": "Technical or operational reasons",
    "XO": "Origin airport below limits",
    "XT": "Destination airport below limits",
    "XR": "Code-share flight cancelled",
    "XS": "Aircraft connection - outbound cancelled, weather",
}
CARRIER_NAMES = {"AZU": "Azul", "GLO": "Gol", "TAM": "TAM", "ONE": "Avianca Brasil"}

SIROS_DATE = pd.Timestamp("2018-03-25")
STATS_SOURCE_DATE = pd.Timestamp("2020-04-01")
LAST_CODED_YM = 202003


# ------------------------------------------------------------------ helpers


def year_pairs(frame: pd.DataFrame) -> list[tuple[int, int]]:
    years = sorted(int(y) for y in frame["year"].unique())
    return [(y, y + 1) for y in years if y + 1 in years]


def flight_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else float("nan")


def run_spec(frame: pd.DataFrame, spec: cm.CellSpec, k: int | dict = 3, baseline_flags: dict | None = None) -> pd.DataFrame:
    """Prospective validation for every consecutive year pair under one specification."""
    rows = []
    by_year = {int(y): g for y, g in frame.groupby("year")}
    for base_year, next_year in year_pairs(frame):
        k_used = k[base_year] if isinstance(k, dict) else k
        result = cm.prospective_evaluation(by_year[base_year], by_year[next_year], spec, k_used, return_flights=True)
        if result is None:
            continue
        row, pop, nxt = result
        row.update({"base_year": base_year, "next_year": next_year, "k": k_used})
        if baseline_flags is not None and (base_year, next_year) in baseline_flags:
            ref = baseline_flags[(base_year, next_year)]
            for flag in ("watch", "top_decile"):
                flags = pd.Series(cm.flagged_flights(nxt, pop, spec.keys, flag), index=nxt.index)
                flags = flags.reindex(ref.index, fill_value=False).to_numpy(bool)
                row[f"{flag}_flight_jaccard_vs_baseline"] = flight_jaccard(flags, ref[flag].to_numpy(bool))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(table: pd.DataFrame, label: str) -> dict:
    early, late = table[table["next_year"] < 2020], table[table["next_year"] >= 2020]
    out = {"specification": label, "pairs": len(table), "early_pairs": len(early), "late_pairs": len(late),
           "mean_base_cells": table["base_cells"].mean(), "mean_retained_exposure_pct": table["base_retained_exposure_pct"].mean(),
           "mean_population_share_cancellations_pct": table["population_share_national_cancellations_pct"].mean()}
    for tag, part in (("2016_2019", early), ("2020_2024", late)):
        for flag in ("watch", "threshold", "top_decile"):
            out[f"mean_{flag}_lift_{tag}"] = part[f"{flag}_lift"].mean()
            out[f"mean_{flag}_coverage_t1_{tag}"] = part[f"{flag}_coverage_t1_pct"].mean()
            out[f"mean_{flag}_capture_t1_{tag}"] = part[f"{flag}_capture_t1_pct"].mean()
        out[f"mean_watch_coverage_t_{tag}"] = part["watch_coverage_t_pct"].mean()
        out[f"mean_capture_gap_equal_coverage_pp_{tag}"] = part["capture_gap_equal_realized_coverage_pp"].mean()
        out[f"pairs_watch_above_ranking_{tag}"] = int((part["capture_gap_equal_realized_coverage_pp"] > 0).sum())
        out[f"mean_spearman_{tag}"] = part["spearman"].mean()
        if "watch_flight_jaccard_vs_baseline" in part:
            out[f"mean_watch_flight_jaccard_{tag}"] = part["watch_flight_jaccard_vs_baseline"].mean()
            out[f"mean_top_decile_flight_jaccard_{tag}"] = part["top_decile_flight_jaccard_vs_baseline"].mean()
    out["min_top_decile_lift"] = table["top_decile_lift"].min()
    out["max_top_decile_lift"] = table["top_decile_lift"].max()
    out["min_spearman"] = table["spearman"].min()
    return out


# -------------------------------------------------------- registry diagnostics


def registry_scope() -> None:
    """Planned Brazilian-origin legs by designation (DI) and status, by year."""
    df = load_vra(scope="all")
    df = df[df["is_br_origin"]]
    di = df.groupby(["year", "di"], dropna=False).size().unstack(fill_value=0)
    di.to_csv(OUT / "registry_scope_di_by_year.csv")
    st = df.groupby(["year", "status"], dropna=False).size().unstack(fill_value=0)
    st.to_csv(OUT / "registry_scope_status_by_year.csv")
    reg = df[df["is_regular"]]
    rows = []
    for year, g in df.groupby("year"):
        r = reg[reg["year"] == year]
        rows.append({"year": int(year), "planned_legs": len(g), "regular_legs": len(r),
                     "regular_share_pct": len(r) / len(g) * 100,
                     "carriers_all_designations": int(g["airline"].nunique()), "carriers_regular": int(r["airline"].nunique()),
                     "non_regular_cancelled": int((g["is_cancelled"] & ~g["is_regular"]).sum()),
                     "regular_unknown_status": int((~r["known_status"]).sum()),
                     "regular_restored_dates": int(r["sched_date_restored"].sum())})
    pd.DataFrame(rows).to_csv(OUT / "registry_scope_summary.csv", index=False)


def registry_monthly(br: pd.DataFrame) -> pd.DataFrame:
    monthly = br.groupby("ym").agg(
        scheduled=("is_cancelled", "size"),
        cancelled=("is_cancelled", "sum"),
        coded=("reason_code", "count"),
        carriers=("airline", "nunique"),
        international_share=("is_domestic", lambda s: 1 - s.mean()),
        cargo_line_share=("is_cargo_line", "mean"),
    )
    groups = br.dropna(subset=["reason_group"]).groupby(["ym", "reason_group"]).size().unstack(fill_value=0)
    monthly = monthly.join(groups, how="left").fillna(0)
    monthly["rate_pct"] = monthly["cancelled"] / monthly["scheduled"] * 100
    monthly["coded_share_pct"] = np.where(monthly["cancelled"] > 0, monthly["coded"] / monthly["cancelled"] * 100, np.nan)
    all_months = pd.Index([y * 100 + m for y in range(2015, 2025) for m in range(1, 13)], name="ym")
    monthly = monthly.reindex(all_months)
    assert monthly["scheduled"].fillna(0).gt(1_000).all(), "a calendar month has no or very few legs"
    monthly["date"] = pd.to_datetime(monthly.index.astype(str) + "01", format="%Y%m%d")
    monthly.to_csv(OUT / "registry_monthly.csv")

    rows = []
    for label, start, end in [
        ("Jan 2017 - Feb 2018 (HOTRAN approval)", 201701, 201802),
        ("Mar 2018 (transition month)", 201803, 201803),
        ("Apr 2018 - Feb 2020 (SIROS registration)", 201804, 202002),
        ("Mar 2020 (pandemic onset)", 202003, 202003),
        ("Apr 2020 - Dec 2024 (operated flights from statistical database)", 202004, 202412),
    ]:
        w = monthly.loc[(monthly.index >= start) & (monthly.index <= end)]
        rows.append({
            "window": label, "months": len(w), "mean_monthly_scheduled": w["scheduled"].mean(),
            "rate_pct": w["cancelled"].sum() / w["scheduled"].sum() * 100,
            "coded_share_pct": w["coded"].sum() / w["cancelled"].sum() * 100,
            **{f"rate_{g.split()[0].lower()}_pct": w[g].sum() / w["scheduled"].sum() * 100 for g in GROUP_ORDER},
            "mean_monthly_carriers": w["carriers"].mean(),
        })
    pd.DataFrame(rows).to_csv(OUT / "registry_regime_summary.csv", index=False)

    yearly = br.groupby("year").agg(carriers=("airline", "nunique"), scheduled=("is_cancelled", "size"),
                                    cancelled=("is_cancelled", "sum"), months_with_data=("month", "nunique"))
    per_carrier = br.groupby(["year", "airline"]).size()
    yearly["carriers_below_500"] = per_carrier[per_carrier < 500].groupby("year").size()
    yearly.to_csv(OUT / "registry_yearly_scope.csv")
    return monthly


def figure_registry(monthly: pd.DataFrame) -> None:
    colors = {"Administrative or commercial": "#9d4d3f", "Technical or operational": "#1f5a8a", "Weather": "#6f7d3c",
              "Airport closure": "#c9a227", "Reason not published": "#9a9a9a"}
    m = monthly.copy()
    numeric = m.columns.drop("date")
    m[numeric] = m[numeric].astype("float64")
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.2), sharex=True)

    ax = axes[0]
    ax.bar(m["date"], m["scheduled"] / 1000, width=25, color="#b9c7d6", label="Scheduled flights (thousands)")
    ax.set_ylabel("Scheduled flights (thousands)")
    ax2 = ax.twinx()
    ax2.plot(m["date"], m["rate_pct"], color="#1f1f1f", linewidth=1.8, label="Cancellation rate (%)")
    ax2.set_ylabel("Cancellation rate (%)")
    ax.set_ylim(0, 125)
    ax2.set_ylim(0, max(27, float(m["rate_pct"].max()) * 1.1))
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False, fontsize=12)

    ax = axes[1]
    contrib = pd.DataFrame(index=m.index)
    for g in GROUP_ORDER:
        contrib[g] = m[g] / m["scheduled"] * 100
    contrib["Reason not published"] = (m["cancelled"] - m["coded"]) / m["scheduled"] * 100
    ax.stackplot(m["date"], *[contrib[c].fillna(0) for c in contrib.columns], labels=list(contrib.columns),
                 colors=[colors[c] for c in contrib.columns], alpha=0.9, step="mid")
    ax.set_ylabel("Cancellation rate by reason (%)")
    ax.legend(loc="upper right", frameon=False, fontsize=11, ncol=2)

    for a in axes:
        a.axvline(SIROS_DATE, color="#333333", linestyle="--", linewidth=1.2)
        a.axvline(STATS_SOURCE_DATE, color="#333333", linestyle=":", linewidth=1.4)
        a.grid(axis="y", alpha=0.25)
    top = axes[0].get_ylim()[1]
    axes[0].text(SIROS_DATE - pd.Timedelta(days=20), top * 0.97, "SIROS registration\n(25 Mar 2018)", fontsize=11, va="top", ha="right")
    axes[0].text(STATS_SOURCE_DATE + pd.Timedelta(days=25), top * 0.97, "Operated flights from the\nstatistical database;\nno reasons (Apr 2020)",
                 fontsize=11, va="top")
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].set_xlabel("Month of scheduled departure")
    savefig("figure32.pdf")


def reason_codes(br: pd.DataFrame) -> None:
    window = br[br["ym"] <= LAST_CODED_YM]
    coded = window.dropna(subset=["reason_code"])
    coded.groupby(["year", "reason_code"]).size().unstack(fill_value=0).to_csv(OUT / "reason_codes_by_year.csv")
    totals = coded["reason_code"].value_counts()
    table = pd.DataFrame({"cancellations": totals, "share_pct": totals / totals.sum() * 100})
    table["label"] = table.index.map(CODE_LABELS)
    table["group"] = table.index.map(CODE_GROUPS)
    table.to_csv(OUT / "reason_codes_overall.csv")
    shares = totals / totals.sum()
    pd.DataFrame([{"coded_cancellations": int(totals.sum()), "cancellations_in_window": int(window["is_cancelled"].sum()),
                   "top2_share_pct": float(shares.iloc[:2].sum() * 100), "hhi": float((shares ** 2).sum() * 10000)}]
                 ).to_csv(OUT / "reason_codes_concentration.csv", index=False)

    sched = window.groupby("year").size()
    cancelled = window.groupby("year")["is_cancelled"].sum()
    grp = coded.groupby(["year", "reason_group"]).size().unstack(fill_value=0).reindex(columns=GROUP_ORDER, fill_value=0)
    out = grp.copy()
    out["months"] = window.groupby("year")["month"].nunique()
    out["coded_share_pct"] = grp.sum(axis=1) / cancelled * 100
    for g in GROUP_ORDER:
        out[f"share_{g}"] = grp[g] / grp.sum(axis=1) * 100
        out[f"rate_{g}"] = grp[g] / sched * 100
    out["rate_excluding_administrative_pct"] = (grp.sum(axis=1) - grp["Administrative or commercial"]) / sched * 100
    out.to_csv(OUT / "reason_groups_by_year.csv")


def reason_codes_watchlist(br: pd.DataFrame) -> None:
    """Reason groups of year t+1 cancellations in flagged and in other evaluated cells."""
    by_year = {int(y): g for y, g in br.groupby("year")}
    rows = []
    for base_year, next_year in [(2015, 2016), (2016, 2017), (2017, 2018), (2018, 2019)]:
        _, pop, nxt = cm.prospective_evaluation(by_year[base_year], by_year[next_year], BASELINE, return_flights=True)
        evaluated = cm.flagged_flights(nxt, pop.assign(_all=True), BASELINE.keys, "_all")
        for flag in ("watch", "top_decile"):
            flags = cm.flagged_flights(nxt, pop, BASELINE.keys, flag)
            counts = {}
            for label, mask in (("Flagged", flags), ("Other evaluated cells", evaluated & ~flags)):
                sub = nxt.loc[mask]
                c = sub["reason_group"].value_counts().reindex(GROUP_ORDER, fill_value=0)
                counts[label] = c
                rec = {"base_year": base_year, "next_year": next_year, "rule": flag, "cells": label, "scheduled": len(sub),
                       "cancelled": int(sub["is_cancelled"].sum()), "coded": int(c.sum())}
                for g in GROUP_ORDER:
                    rec[f"share_{g}"] = c[g] / c.sum() * 100
                    rec[f"rate_{g}"] = c[g] / len(sub) * 100
                rows.append(rec)
            table = np.vstack([counts["Flagged"].to_numpy(), counts["Other evaluated cells"].to_numpy()])
            chi2 = stats.chi2_contingency(table)[0]
            rows[-1]["cramers_v"] = float(np.sqrt(chi2 / (table.sum() * (min(table.shape) - 1))))
    pd.DataFrame(rows).to_csv(OUT / "reason_groups_watchlist.csv", index=False)


# ------------------------------------------------------------ Others and cells


def others_composition(br: pd.DataFrame) -> None:
    """Descriptive composition of the full-period Others group, by phase."""
    others = br[br["carrier_group_fixed"] == "Others"]
    phase_order = ["Pre-pandemic", "Pandemic", "Post-pandemic"]
    rows = []
    for phase in phase_order:
        sub = others[others["period"] == phase]
        per = sub.groupby("airline").agg(n=("is_cancelled", "size"), c=("is_cancelled", "sum"), category=("operator_category", "first"))
        per["rate"] = per["c"] / per["n"] * 100
        big = per[per["n"] >= 500]
        rows.append({"phase": phase, "category": "All Others", "carriers": len(per), "scheduled": int(per["n"].sum()),
                     "cancelled": int(per["c"].sum()), "rate_pct": per["c"].sum() / per["n"].sum() * 100,
                     "carriers_500plus": len(big), "min_rate": big["rate"].min(), "p25_rate": big["rate"].quantile(0.25),
                     "median_rate": big["rate"].median(), "p75_rate": big["rate"].quantile(0.75), "max_rate": big["rate"].max()})
        for cat, g in per.groupby("category"):
            gb = g[g["n"] >= 500]
            rows.append({"phase": phase, "category": cat, "carriers": len(g), "scheduled": int(g["n"].sum()),
                         "cancelled": int(g["c"].sum()), "rate_pct": g["c"].sum() / g["n"].sum() * 100,
                         "carriers_500plus": len(gb), "min_rate": gb["rate"].min() if len(gb) else np.nan,
                         "median_rate": gb["rate"].median() if len(gb) else np.nan, "max_rate": gb["rate"].max() if len(gb) else np.nan})
        per.sort_values("n", ascending=False).to_csv(OUT / f"others_carriers_{phase.lower().replace('-', '_')}.csv")
    pd.DataFrame(rows).to_csv(OUT / "others_composition_by_phase.csv", index=False)
    others.groupby("year").agg(carriers=("airline", "nunique"), scheduled=("is_cancelled", "size"),
                               cancelled=("is_cancelled", "sum")).to_csv(OUT / "others_by_year.csv")

    comp_rows = []
    for phase in phase_order:
        sub = br[br["period"] == phase].assign(carrier_group=lambda d: d["carrier_group_fixed"])
        assigned = cm.fit_segments(cm.build_cells(sub), BASELINE)
        top = assigned[assigned["rate_cluster"] == assigned["rate_cluster"].max()]
        flights = sub.merge(top[cm.CELL_KEYS], on=cm.CELL_KEYS, how="inner")
        total_n, total_c = len(flights), int(flights["is_cancelled"].sum())
        label = np.where(flights["carrier_group"] == "Others", "Others: " + flights["operator_category"].astype(str), flights["carrier_group"])
        for name, g in flights.groupby(label):
            comp_rows.append({"phase": phase, "level": "category", "name": name, "share_scheduled_pct": len(g) / total_n * 100,
                              "share_cancelled_pct": g["is_cancelled"].sum() / total_c * 100, "rate_pct": g["is_cancelled"].mean() * 100})
        for name, g in flights.groupby("airline"):
            comp_rows.append({"phase": phase, "level": "carrier", "name": name, "share_scheduled_pct": len(g) / total_n * 100,
                              "share_cancelled_pct": g["is_cancelled"].sum() / total_c * 100, "rate_pct": g["is_cancelled"].mean() * 100,
                              "category": g["operator_category"].iat[0]})
    pd.DataFrame(comp_rows).sort_values(["phase", "level", "share_scheduled_pct"], ascending=[True, True, False]).to_csv(
        OUT / "watchlist_composition_by_operator.csv", index=False)


def exposure_exclusions(br: pd.DataFrame) -> None:
    """What the minimum-exposure rule removes, by phase and by year (full-period grouping)."""
    keys = cm.CELL_KEYS
    frame = br.assign(carrier_group=br["carrier_group_fixed"])
    rows = []
    units = [("Full period", "2015-2024", frame)] + [("Phase", p, frame[frame["period"] == p]) for p in ["Pre-pandemic", "Pandemic", "Post-pandemic"]]
    units += [("Year", str(y), g) for y, g in frame.groupby("year")]
    for level, label, sub in units:
        all_cells = cm.build_cells(sub, 1, keys)
        for threshold in (200, 500, 1000):
            kept = all_cells[all_cells["scheduled"] >= threshold]
            flights = sub[keys + ["airline", "is_cancelled", "is_domestic"]].merge(kept[keys].assign(_kept=True), on=keys, how="left")
            kept_mask = flights["_kept"].eq(True).to_numpy()
            excl = flights.loc[~kept_mask]
            ap_all, ap_kept = set(sub["origin"].unique()), set(kept["origin"].unique())
            ca_all, ca_kept = set(zip(all_cells["carrier_group"], all_cells["origin"])), set(zip(kept["carrier_group"], kept["origin"]))
            car_all, car_kept = set(sub["airline"].unique()), set(flights.loc[kept_mask, "airline"].unique())
            rows.append({
                "level": level, "unit": label, "min_scheduled": threshold,
                "cells_total": len(all_cells), "cells_excluded": len(all_cells) - len(kept),
                "airports_total": len(ap_all), "airports_excluded": len(ap_all - ap_kept),
                "carrier_airport_total": len(ca_all), "carrier_airport_excluded": len(ca_all - ca_kept),
                "operating_carriers_total": len(car_all), "operating_carriers_excluded": len(car_all - car_kept),
                "scheduled_total": len(flights), "scheduled_excluded": len(excl), "scheduled_excluded_pct": len(excl) / len(flights) * 100,
                "cancelled_total": int(flights["is_cancelled"].sum()), "cancelled_excluded": int(excl["is_cancelled"].sum()),
                "cancelled_excluded_pct": excl["is_cancelled"].sum() / flights["is_cancelled"].sum() * 100,
                "rate_excluded_pct": excl["is_cancelled"].mean() * 100, "rate_retained_pct": flights.loc[kept_mask, "is_cancelled"].mean() * 100,
                "excluded_international_share_pct": (1 - excl["is_domestic"].mean()) * 100 if len(excl) else np.nan,
                "excluded_others_share_pct": (excl["carrier_group"] == "Others").mean() * 100 if len(excl) else np.nan,
            })
    pd.DataFrame(rows).to_csv(OUT / "minimum_exposure_exclusions.csv", index=False)


# ------------------------------------------------------------ inference and k


def bootstrap_table(frame: pd.DataFrame, pairs: list[tuple[int, int]], registry: str) -> pd.DataFrame:
    by_year = {int(y): g for y, g in frame.groupby("year")}
    rows = []
    for base_year, next_year in pairs:
        result = cm.prospective_evaluation(by_year[base_year], by_year[next_year], BASELINE)
        if result is None:
            continue
        row, pop = result
        watch = pop["watch"].to_numpy(bool)
        rest_rate = pop.loc[~watch, "cancelled_t1"].sum() / pop.loc[~watch, "scheduled_t1"].sum() * 100
        boot = cm.airport_bootstrap(pop, n_boot=N_BOOT, seed=SEED)
        keep = ["evaluated_cells", "panel_cells", "system_rate_t1_pct", "population_share_national_cancellations_pct",
                "watch_cells", "watch_coverage_t_pct", "watch_coverage_t1_pct", "watch_capture_t1_pct", "watch_lift",
                "top_decile_cells", "top_decile_coverage_t1_pct", "top_decile_capture_t1_pct", "top_decile_lift",
                "threshold_coverage_t_pct", "threshold_coverage_t1_pct", "threshold_capture_t1_pct", "threshold_lift",
                "ranking_capture_at_watch_coverage_pct", "capture_gap_equal_realized_coverage_pp", "spearman",
                "mannwhitney_p", "mannwhitney_auc"]
        rows.append({"registry": registry, "base_year": base_year, "next_year": next_year, **{k: row[k] for k in keep},
                     "watch_rate_diff_pp": row["watch_rate_t1_pct"] - rest_rate, **boot})
    return pd.DataFrame(rows)


def year_cells(sub: pd.DataFrame, spec: cm.CellSpec) -> pd.DataFrame:
    """Eligible cells of one assignment year, with carrier groups fixed from that year."""
    grouped, _, _ = cm.assign_carrier_groups(sub, sub.iloc[:0], spec)
    return cm.build_cells(grouped, spec.min_sched, spec.keys)


def rule_based_k(frame: pd.DataFrame) -> dict[str, dict[int, int]]:
    """Number of segments chosen from year-t cells alone by three pre-specified criteria (k = 2..10)."""
    choice = {"Gap statistic": {}, "Calinski-Harabasz": {}, "Silhouette": {}}
    rows = []
    for year, sub in frame.groupby("year"):
        cells = year_cells(sub, BASELINE)
        table = cm.validity_indices(cm.feature_matrix(cells, BASELINE), range(2, 11))
        gap_k = cm.first_gap_k(table[["k", "gap", "gap_sk"]])
        choice["Gap statistic"][int(year)] = int(gap_k) if np.isfinite(gap_k) else 10
        choice["Calinski-Harabasz"][int(year)] = int(table.loc[table["calinski_harabasz"].idxmax(), "k"])
        choice["Silhouette"][int(year)] = int(table.loc[table["silhouette"].idxmax(), "k"])
        rows.append({"year": int(year), "cells": len(cells), "gap_rule_met": bool(np.isfinite(gap_k)),
                     **{name: c[int(year)] for name, c in choice.items()}})
    pd.DataFrame(rows).to_csv(OUT / "rule_based_k_by_year.csv", index=False)
    return choice


def year_lists(sub: pd.DataFrame, spec: cm.CellSpec = BASELINE) -> dict[str, set]:
    """Eligible cells and the three flagged sets fixed with one year's information."""
    cells = year_cells(sub, spec)
    assigned = cm.fit_segments(cells, spec)
    assigned["watch"] = assigned["rate_cluster"] == assigned["rate_cluster"].max()
    budget = assigned.loc[assigned["watch"], "scheduled"].sum() / assigned["scheduled"].sum()
    ordered = cm.ranked_order(assigned, spec.keys)
    assigned["threshold"] = cm.budget_flags(ordered, budget).reindex(assigned.index)
    assigned["top_decile"] = cm.budget_flags(ordered, cm.TOP_SHARE).reindex(assigned.index)
    keys = list(map(tuple, assigned[spec.keys].to_numpy()))
    out = {"cells": set(keys)}
    for rule in ("watch", "threshold", "top_decile"):
        out[rule] = {k for k, f in zip(keys, assigned[rule].to_numpy(bool)) if f}
    return out


def membership_stability(frame: pd.DataFrame) -> None:
    """Complete-list and conditional stability of the flagged sets between consecutive years."""
    lists = {int(y): year_lists(sub) for y, sub in frame.groupby("year")}
    rows = []
    years = sorted(lists)
    for a, b in zip(years[:-1], years[1:]):
        la, lb = lists[a], lists[b]
        common = la["cells"] & lb["cells"]
        rec = {"base_year": a, "next_year": b, "eligible_t": len(la["cells"]), "eligible_t1": len(lb["cells"]), "common_cells": len(common)}
        for rule in ("watch", "threshold", "top_decile"):
            fa, fb = la[rule], lb[rule]
            ca, cb = fa & common, fb & common
            exits, entries = fa - fb, fb - fa
            rec.update({
                f"{rule}_size_t": len(fa), f"{rule}_size_t1": len(fb),
                f"{rule}_jaccard_complete": len(fa & fb) / len(fa | fb) if (fa | fb) else np.nan,
                f"{rule}_jaccard_conditional": len(ca & cb) / len(ca | cb) if (ca | cb) else np.nan,
                f"{rule}_retained": len(fa & fb),
                f"{rule}_exits_ineligible": len(exits - lb["cells"]), f"{rule}_exits_reclassified": len(exits & lb["cells"]),
                f"{rule}_entries_new_eligible": len(entries - la["cells"]), f"{rule}_entries_reclassified": len(entries & la["cells"]),
            })
        rows.append(rec)
    pd.DataFrame(rows).to_csv(OUT / "membership_stability.csv", index=False)


def encoding_overlap(frame: pd.DataFrame) -> None:
    """Cell-level overlap of the year-t high-rate segments under the three encodings."""
    rows = []
    for year, sub in frame.groupby("year"):
        cells = year_cells(sub, BASELINE)
        sets = {}
        for enc in ("prevalence", "onehot", "rate_time"):
            assigned = cm.fit_segments(cells, replace(BASELINE, encoding=enc))
            top = assigned[assigned["rate_cluster"] == assigned["rate_cluster"].max()]
            sets[enc] = set(map(tuple, top[BASELINE.keys].to_numpy()))
        j = lambda a, b: len(sets[a] & sets[b]) / len(sets[a] | sets[b])
        rows.append({"year": int(year), "cells": len(cells), "prevalence_cells": len(sets["prevalence"]),
                     "onehot_cells": len(sets["onehot"]), "rate_time_cells": len(sets["rate_time"]),
                     "jaccard_prevalence_onehot": j("prevalence", "onehot"), "jaccard_prevalence_rate_time": j("prevalence", "rate_time")})
    pd.DataFrame(rows).to_csv(OUT / "encoding_overlap_by_year.csv", index=False)


# ------------------------------------------------------------ registry regimes


def window_frame(br: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    return br[(br["ym"] >= start) & (br["ym"] <= end)]


def calendar_months(start: int, end: int) -> int:
    return (end // 100 - start // 100) * 12 + (end % 100 - start % 100) + 1


def regime_analysis(br: pd.DataFrame) -> None:
    """Prospective pairs within and across the documented registry windows.

    The minimum exposure is scaled to the number of calendar months in each window
    (500 flights per twelve months), separately for the assignment window and for the
    evaluation window.
    """
    pairs = [
        ("A", "Within", (201501, 201512), (201601, 201612)),
        ("A", "Within", (201601, 201612), (201701, 201712)),
        ("A to B", "Across", (201701, 201712), (201801, 201812)),
        ("B", "Within", (201804, 201903), (201904, 202003)),
        ("B", "Within", (201804, 201902), (201904, 202002)),
        ("B to C", "Across", (201901, 201912), (202001, 202012)),
        ("C", "Within", (202004, 202103), (202104, 202203)),
        ("C", "Within", (202101, 202112), (202201, 202212)),
        ("C", "Within", (202201, 202212), (202301, 202312)),
        ("C", "Within", (202301, 202312), (202401, 202412)),
    ]
    rows = []
    for regime, kind, (a0, a1), (b0, b1) in pairs:
        base_months, eval_months = calendar_months(a0, a1), calendar_months(b0, b1)
        spec = replace(BASELINE, min_sched=int(round(500 * base_months / 12)), min_sched_eval=int(round(500 * eval_months / 12)))
        result = cm.prospective_evaluation(window_frame(br, a0, a1), window_frame(br, b0, b1), spec)
        if result is None:
            continue
        row, _ = result
        rows.append({"regime": regime, "comparison": kind, "base_window": f"{a0}-{a1}", "evaluation_window": f"{b0}-{b1}",
                     "base_months": base_months, "evaluation_months": eval_months, "min_scheduled_base": spec.min_sched,
                     "min_scheduled_eval": spec.min_sched_eval,
                     **{k: row[k] for k in ["evaluated_cells", "panel_cells", "system_rate_t1_pct", "population_share_national_cancellations_pct",
                                            "watch_cells", "watch_lift", "watch_coverage_t1_pct", "watch_capture_t1_pct",
                                            "threshold_lift", "threshold_capture_t1_pct", "top_decile_lift", "top_decile_coverage_t1_pct",
                                            "top_decile_capture_t1_pct", "ranking_capture_at_watch_coverage_pct",
                                            "capture_gap_equal_realized_coverage_pp", "spearman", "mannwhitney_p"]}})
    pd.DataFrame(rows).to_csv(OUT / "regime_out_of_sample.csv", index=False)

    seg_rows = []
    for label, a0, a1 in [("A: Jan 2015 - Feb 2018", 201501, 201802), ("B: Apr 2018 - Mar 2020", 201804, 202003),
                          ("C (pandemic): Apr 2020 - Apr 2023", 202004, 202304), ("C (post-pandemic): May 2023 - Dec 2024", 202305, 202412)]:
        sub = window_frame(br, a0, a1).assign(carrier_group=lambda d: d["carrier_group_fixed"])
        months = calendar_months(a0, a1)
        cells = cm.build_cells(sub, int(round(500 * months / 12)))
        assigned = cm.fit_segments(cells, BASELINE)
        top = assigned[assigned["rate_cluster"] == assigned["rate_cluster"].max()]
        system = assigned["cancelled"].sum() / assigned["scheduled"].sum() * 100
        rates = sub.groupby("carrier_group")["is_cancelled"].mean() * 100
        dom = sub[sub["is_domestic"] & ~sub["is_cargo_line"]]
        rec = {"window": label, "months": months, "scheduled": len(sub), "cancelled": int(sub["is_cancelled"].sum()),
               "system_rate_all_flights_pct": sub["is_cancelled"].mean() * 100,
               "domestic_passenger_rate_pct": dom["is_cancelled"].mean() * 100, "cells": len(cells),
               "watchlist_cells": len(top), "watchlist_rate_pct": cm.weighted_rate(top), "system_rate_cells_pct": system,
               "lift": cm.weighted_rate(top) / system, "coverage_pct": top["scheduled"].sum() / assigned["scheduled"].sum() * 100,
               "capture_pct": top["cancelled"].sum() / assigned["cancelled"].sum() * 100,
               "others_share_of_watchlist_pct": top.loc[top["carrier_group"] == "Others", "scheduled"].sum() / top["scheduled"].sum() * 100,
               "dominant_period": top.groupby("departure_period")["scheduled"].sum().idxmax()}
        for g in ["Azul", "Gol", "TAM", "Others"]:
            gs = sub[sub["carrier_group"] == g]
            rec[f"scheduled_{g}"] = len(gs)
            rec[f"cancelled_{g}"] = int(gs["is_cancelled"].sum())
            rec[f"rate_{g}_pct"] = rates.get(g, np.nan)
            rec[f"domestic_passenger_rate_{g}_pct"] = dom.loc[dom["carrier_group"] == g, "is_cancelled"].mean() * 100
        seg_rows.append(rec)
    pd.DataFrame(seg_rows).to_csv(OUT / "regime_segments.csv", index=False)


# ----------------------------------------------------------------------- main


def main() -> None:
    br, windows = brazil_frame()
    print(f"Brazilian-origin legs: {len(br):,}; cancelled: {int(br['is_cancelled'].sum()):,}", flush=True)

    registry_scope()
    monthly = registry_monthly(br)
    figure_registry(monthly)
    reason_codes(br)
    reason_codes_watchlist(br)
    print("registry diagnostics done", flush=True)

    others_composition(br)
    exposure_exclusions(br)
    print("composition and exclusions done", flush=True)

    base_cols = ["airline", "carrier_group_fixed", "operator_category", "origin", "dest", "day_type", "departure_period",
                 "quarter", "month", "is_cancelled", "year", "win_shift6", "win_3h", "win_2h", "is_domestic", "is_cargo_line"]
    lean = br[base_cols]

    # Baseline, checked against the stored results of generate_policy_analyses.py.
    baseline = run_spec(lean, BASELINE)
    stored = pd.read_csv(VRA_OUT / "watchlist_out_of_sample_summary.csv")
    assert np.allclose(baseline["top_decile_lift"].to_numpy(), stored["top_decile_lift"].to_numpy()), "baseline not reproduced"
    baseline.to_csv(OUT / "baseline_out_of_sample_extended.csv", index=False)

    by_year = {int(y): g for y, g in lean.groupby("year")}
    baseline_flags = {}
    for base_year, next_year in year_pairs(lean):
        _, pop, nxt = cm.prospective_evaluation(by_year[base_year], by_year[next_year], BASELINE, return_flights=True)
        baseline_flags[(base_year, next_year)] = pd.DataFrame(
            {flag: cm.flagged_flights(nxt, pop, BASELINE.keys, flag) for flag in ("watch", "top_decile")}, index=nxt.index)
    print("baseline reproduced", flush=True)

    def with_cols(**cols):
        return lambda: lean.assign(**{k: br[v] for k, v in cols.items()})

    seasonal = dict(keys=list(cm.CELL_KEYS) + ["quarter"], quarter=True)
    route = dict(keys=["carrier_group", "origin", "dest", "day_type", "departure_period"], prevalence=["carrier_group", "origin", "dest"])
    specs: list[tuple[str, object, cm.CellSpec, int | dict]] = [
        ("Matched panel (min 500 in t+1)", lambda: lean, cm.CellSpec("Matched panel", population="matched"), 3),
        ("Full-period carrier grouping", lambda: lean, cm.CellSpec("Full-period carrier grouping", carrier_rule="fixed"), 3),
        ("Brasilia clock", with_cols(day_type="day_type_brt", departure_period="departure_period_brt"), cm.CellSpec("Brasilia clock"), 3),
        ("Excluding legs with restored dates", lambda: lean[~br["sched_date_restored"].to_numpy()], cm.CellSpec("Excluding restored dates"), 3),
        ("Seasonal: quarter, min 500", lambda: lean, cm.CellSpec("Seasonal: quarter, min 500", **seasonal), 3),
        ("Seasonal: quarter, min 125", lambda: lean, cm.CellSpec("Seasonal: quarter, min 125", min_sched=125, **seasonal), 3),
        ("Destination cells", lambda: lean, cm.CellSpec("Destination cells", keys=["carrier_group", "dest", "day_type", "departure_period"],
                                                        prevalence=["carrier_group", "dest"]), 3),
        ("Route cells, min 500", lambda: lean, cm.CellSpec("Route cells, min 500", **route), 3),
        ("Route cells, min 200", lambda: lean, cm.CellSpec("Route cells, min 200", min_sched=200, **route), 3),
        ("Domestic flights", lambda: lean[lean["is_domestic"].to_numpy()], cm.CellSpec("Domestic flights"), 3),
        ("Domestic passenger flights", lambda: lean[(lean["is_domestic"] & ~lean["is_cargo_line"]).to_numpy()],
         cm.CellSpec("Domestic passenger flights"), 3),
    ]
    for thr in (0.01, 0.025, 0.10):
        specs.append((f"Carrier threshold {thr:.1%}", lambda: lean, cm.CellSpec(f"Carrier threshold {thr:.1%}", group_share=thr), 3))
    specs += [
        ("Others split by operator category", lambda: lean, cm.CellSpec("Others split by operator category", carrier_rule="category"), 3),
        ("Individual carriers", lambda: lean, cm.CellSpec("Individual carriers", carrier_rule="individual"), 3),
    ]
    for name, col, key in [("Departure windows: shifted 6 h", "win_shift6", "shift6"), ("Departure windows: 3 h", "win_3h", "3h"),
                           ("Departure windows: 2 h", "win_2h", "2h")]:
        specs.append((name, lambda: lean, cm.CellSpec(name, keys=["carrier_group", "origin", "day_type", col], time_col=col, time_mid=windows[key]), 3))
    specs += [
        ("Encoding: one-hot", lambda: lean, cm.CellSpec("Encoding: one-hot", encoding="onehot"), 3),
        ("Encoding: rate and time only", lambda: lean, cm.CellSpec("Encoding: rate and time only", encoding="rate_time"), 3),
    ]
    k_choice = rule_based_k(lean)
    for name, choice in k_choice.items():
        specs.append((f"k by {name}", lambda: lean, cm.CellSpec(f"k by {name}"), choice))

    all_rows = [baseline.assign(watch_flight_jaccard_vs_baseline=1.0, top_decile_flight_jaccard_vs_baseline=1.0)]
    summary = [summarize(all_rows[0], "Baseline")]
    for name, build, spec, k in specs:
        table = run_spec(build(), spec, k=k, baseline_flags=baseline_flags)
        table["specification"] = name
        all_rows.append(table)
        summary.append(summarize(table, name))
        print(f"  {name}: done", flush=True)
    pd.concat(all_rows, ignore_index=True).to_csv(OUT / "sensitivity_out_of_sample_pairs.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "sensitivity_out_of_sample_summary.csv", index=False)

    # Cell sparsity for month, quarter and route dimensions (full-period grouping).
    sparsity = []
    fixed = lean.assign(carrier_group=lean["carrier_group_fixed"])
    for name, keys in [("Baseline (annual)", list(cm.CELL_KEYS)), ("Quarter", list(cm.CELL_KEYS) + ["quarter"]),
                       ("Month", list(cm.CELL_KEYS) + ["month"]), ("Route", ["carrier_group", "origin", "dest", "day_type", "departure_period"])]:
        for year, sub in fixed.groupby("year"):
            allc = sub.groupby(keys, observed=True).size()
            for thr in (500, 200, 125, 42):
                kept = allc[allc >= thr]
                sparsity.append({"cells": name, "year": int(year), "min_scheduled": thr, "cells_total": len(allc),
                                 "cells_retained": len(kept), "retained_exposure_pct": kept.sum() / allc.sum() * 100})
    pd.DataFrame(sparsity).to_csv(OUT / "cell_sparsity.csv", index=False)

    # Seasonal check of the annual lists: lift by quarter of the evaluation year.
    q_rows = []
    for base_year, next_year in year_pairs(lean):
        _, pop, nxt = cm.prospective_evaluation(by_year[base_year], by_year[next_year], BASELINE, return_flights=True)
        evaluated = cm.flagged_flights(nxt, pop.assign(_all=True), BASELINE.keys, "_all")
        for flag in ("watch", "top_decile"):
            flags = cm.flagged_flights(nxt, pop, BASELINE.keys, flag)
            for q in range(1, 5):
                inq = (nxt["quarter"] == q).to_numpy()
                s_all, c_all = (evaluated & inq).sum(), nxt.loc[evaluated & inq, "is_cancelled"].sum()
                s_w, c_w = (flags & inq).sum(), nxt.loc[flags & inq, "is_cancelled"].sum()
                if s_all == 0 or s_w == 0 or c_all == 0:
                    continue
                q_rows.append({"base_year": base_year, "next_year": next_year, "rule": flag, "quarter": q,
                               "system_rate_pct": c_all / s_all * 100, "flagged_rate_pct": c_w / s_w * 100,
                               "lift": (c_w / s_w) / (c_all / s_all), "capture_pct": c_w / c_all * 100})
    pd.DataFrame(q_rows).to_csv(OUT / "annual_watchlist_lift_by_quarter.csv", index=False)
    print("sensitivity specifications done", flush=True)

    bootstrap_table(lean, year_pairs(lean), "Brazil (ANAC/VRA)").to_csv(OUT / "bootstrap_brazil.csv", index=False)
    from generate_bts_validation import YEAR_PAIRS, load_bts

    bootstrap_table(load_bts(), YEAR_PAIRS, "United States (DOT/BTS)").to_csv(OUT / "bootstrap_us.csv", index=False)
    print("bootstrap done", flush=True)

    membership_stability(lean)
    encoding_overlap(lean)
    regime_analysis(br[base_cols + ["ym"]])
    print("Outputs written to", OUT)


if __name__ == "__main__":
    main()
