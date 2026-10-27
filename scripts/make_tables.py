"""Build the LaTeX table bodies of the article from the analysis results.

Every number printed in Tables 4-13 and B.1-B.13 is produced here from the CSV files
of the analysis; no table value is typed by hand. Each body (the rows between the
column header and the closing rule) is written to ``<output>/tables/<name>.tex``,
and key numbers used in the text are written to ``<output>/tables/numbers.json``.

    python scripts/make_tables.py                  # from the archived results/
    python scripts/make_tables.py --from-output    # from a full rerun (output/analysis_outputs/)
    python scripts/make_tables.py --check          # also compare with results/tables/

The bodies in ``results/tables/`` are the ones printed in the accepted manuscript.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from paths import ANALYSIS_DIR, RESULTS_DIR, TABLE_DIR

VRA = RESULTS_DIR / "vra"
ROB = VRA / "robustness"
BTS = RESULTS_DIR / "bts"
GROUPS = ["Administrative or commercial", "Technical or operational", "Weather", "Airport closure"]
NUM: dict[str, object] = {}


def f(x, d=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:,.{d}f}"


def n(x):
    return f"{int(round(x)):,}"


def signed(x, d=1):
    if abs(x) < 0.05 and d == 1:
        d = 2
    s = f"{x:.{d}f}"
    return f"${s}$" if x < 0 else s


def interval(lo, hi, d=2):
    return f"{signed(lo, d)} to {signed(hi, d)}"


def pair(r):
    return f"{int(r['base_year'])}--{int(r['next_year'])}"


# ------------------------------------------------------------------ registry


def annual_record() -> str:
    m = pd.read_csv(ROB / "registry_monthly.csv")
    m["year"] = m["ym"] // 100
    y = m.groupby("year")[["scheduled", "cancelled", "coded"] + GROUPS].sum()
    rows = []
    for year, r in y.iterrows():
        rate = r["cancelled"] / r["scheduled"] * 100
        coded = r["coded"] / r["cancelled"] * 100
        label = f"{year}$^{{a}}$" if year == 2020 else str(year)
        if year <= 2019:
            parts = [f(r[g] / r["scheduled"] * 100) for g in GROUPS]
        else:
            parts = ["--"] * 4
        rows.append(f"{label} & {n(r['scheduled'])} & {n(r['cancelled'])} & {f(rate)} & {f(coded, 1)} & " + " & ".join(parts) + r" \\")
        NUM[f"rate_{year}"] = round(rate, 2)
    NUM["total_scheduled"] = int(y["scheduled"].sum())
    NUM["total_cancelled"] = int(y["cancelled"].sum())
    return "\n".join(rows)


def registry_numbers() -> None:
    m = pd.read_csv(ROB / "registry_monthly.csv").set_index("ym")
    for ym in (201802, 201803, 201804, 201604, 201606, 202001, 202002, 202003, 202004):
        NUM[f"monthly_rate_{ym}"] = round(float(m.loc[ym, "rate_pct"]), 2)
        NUM[f"monthly_sched_{ym}"] = int(m.loc[ym, "scheduled"])
    reg = pd.read_csv(ROB / "registry_regime_summary.csv")
    NUM["regime_summary"] = reg.round(3).to_dict(orient="records")
    ys = pd.read_csv(ROB / "registry_yearly_scope.csv")
    NUM["carriers_by_year"] = dict(zip(ys["year"].astype(str), ys["carriers"].astype(int)))
    NUM["carriers_below_500_by_year"] = dict(zip(ys["year"].astype(str), ys["carriers_below_500"].fillna(0).astype(int)))
    rc = pd.read_csv(ROB / "reason_codes_overall.csv", index_col=0)
    NUM["reason_code_shares"] = rc["share_pct"].round(1).to_dict()
    NUM["reason_concentration"] = pd.read_csv(ROB / "reason_codes_concentration.csv").iloc[0].to_dict()
    rg = pd.read_csv(ROB / "reason_groups_by_year.csv", index_col=0)
    NUM["reason_group_shares_by_year"] = rg[[f"share_{g}" for g in GROUPS]].round(1).to_dict(orient="index")
    # Months with weather-coded rates, before and after the registration change.
    w = m.loc[(m.index >= 201701) & (m.index <= 202002) & (m.index != 201803)]
    wr = w["Weather"] / w["scheduled"] * 100
    NUM["weather_monthly_min_max"] = [round(float(wr.min()), 2), round(float(wr.max()), 2)]
    sc = pd.read_csv(ROB / "registry_scope_summary.csv")
    NUM["scope_summary"] = sc.to_dict(orient="records")


def inclusion_table() -> str:
    flow = pd.read_csv(VRA / "inclusion_flow.csv")
    rows = []
    for _, r in flow.iterrows():
        step = r["step"].replace("25-31", "25--31").replace("2015-2024", "2015--2024").replace("Dec 2014 to Jan 2025", "December 2014 to January 2025")
        indent = step.startswith("  ")
        step = step.strip()
        step = step[0].upper() + step[1:]
        step = step.replace("cancelled", "canceled").replace("Cancelled", "Canceled")  # US spelling in the manuscript
        label = (r"\quad " + step) if indent else step
        rows.append(f"{label} & {n(r['records'])} \\\\")
        if r["step"] == "Records without usable scheduled departure":
            missing = pd.read_csv(VRA / "missing_schedule_status.csv")
            for status, label in [("REALIZADO", "marked operated"), ("CANCELADO", "marked canceled"),
                                  ("N\u00c3O INFORMADO", "status not informed")]:
                count = missing.loc[missing["status"].eq(status), "records"].sum()
                if count:
                    rows.append(r"\quad Of these, " + label + " & " + n(count) + r" \\")
            mask = missing["status"].eq("CANCELADO") & missing["di"].astype(str).eq("0") & missing["brazilian_origin"].eq(True)
            count = missing.loc[mask, "records"].sum()
            rows.append(r"\quad Of the canceled records, Brazilian-origin regular & " + n(count) + r" \\")
    NUM["inclusion_flow"] = dict(zip(flow["step"].str.strip(), flow["records"].astype(int)))
    return "\n".join(rows)


def reconciliation_table() -> str:
    y = pd.read_csv(VRA / "archive_reconciliation_yearly.csv")
    rows = []
    for _, r in y.iterrows():
        rows.append(f"{int(r['year'])} & {n(r['siros_legs'])} & {n(r['legacy_legs'])} & {n(r['siros_cancelled'])} & {n(r['legacy_cancelled'])} & "
                    f"{f(r['matched_share_of_siros_pct'], 2)} & {f(r['status_agreement_pct'], 2)} \\\\")
    NUM["reconciliation"] = y.round(3).to_dict(orient="records")
    return "\n".join(rows)


def carrier_rates() -> str:
    s = pd.read_csv(ROB / "regime_segments.csv")
    heads = {
        "A: Jan 2015 - Feb 2018": "Window A: January 2015--February 2018 (HOTRAN approval)",
        "B: Apr 2018 - Mar 2020": "Window B: April 2018--March 2020 (SIROS registration)",
        "C (pandemic): Apr 2020 - Apr 2023": "Window C, pandemic phase: April 2020--April 2023",
        "C (post-pandemic): May 2023 - Dec 2024": "Window C, post-pandemic phase: May 2023--December 2024",
    }
    rows = []
    for _, r in s.iterrows():
        rows.append(r"\multicolumn{5}{l}{\textit{" + heads[r["window"]] + r"}} \\")
        for g in ["Azul", "Gol", "TAM", "Others"]:
            rows.append(f"{g} & {n(r[f'scheduled_{g}'])} & {n(r[f'cancelled_{g}'])} & {f(r[f'rate_{g}_pct'])} & {f(r[f'domestic_passenger_rate_{g}_pct'])} \\\\")
        rows.append(f"All carriers & {n(r['scheduled'])} & {n(r['cancelled'])} & {f(r['system_rate_all_flights_pct'])} & {f(r['domestic_passenger_rate_pct'])} \\\\")
    NUM["regime_segments"] = s.round(3).to_dict(orient="records")
    return "\n".join(rows)


def others_table() -> str:
    o = pd.read_csv(ROB / "others_composition_by_phase.csv")
    comp = pd.read_csv(ROB / "watchlist_composition_by_operator.csv")
    comp = comp[comp["level"] == "category"]
    cats = ["Brazilian passenger", "Brazilian cargo", "Foreign passenger", "Foreign cargo", "All Others"]
    rows = []
    for phase in ["Pre-pandemic", "Pandemic", "Post-pandemic"]:
        rows.append(r"\multicolumn{7}{l}{\textit{" + phase.replace("-p", "-p") + r" phase}} \\")
        sub = o[o["phase"] == phase].set_index("category")
        cs = comp[comp["phase"] == phase].set_index("name")["share_scheduled_pct"]
        for c in cats:
            if c not in sub.index:
                continue
            r = sub.loc[c]
            share = cs[[i for i in cs.index if i.startswith("Others")]].sum() if c == "All Others" else cs.get(f"Others: {c}", 0.0)
            rng = "--" if not np.isfinite(r["median_rate"]) else f"{r['median_rate']:.1f} ({r['min_rate']:.1f}--{r['max_rate']:.1f})"
            rows.append(f"{c} & {int(r['carriers'])} & {n(r['scheduled'])} & {f(r['rate_pct'])} & {int(r['carriers_500plus'])} & {rng} & {f(share, 1)} \\\\")
    NUM["others"] = o.round(2).to_dict(orient="records")
    top = pd.read_csv(ROB / "watchlist_composition_by_operator.csv")
    NUM["post_watch_carriers"] = top[(top["phase"] == "Post-pandemic") & (top["level"] == "carrier")].head(8).round(2).to_dict(orient="records")
    return "\n".join(rows)


# --------------------------------------------------------------- segmentation


def k_selection() -> str:
    k = pd.read_csv(VRA / "cell_cluster_k_selection.csv")
    return "\n".join(f"{r['period']} & {int(r['best_silhouette_k'])} & {int(r['best_calinski_k'])} & {int(r['best_davies_bouldin_k'])} & "
                     f"{int(r['gap_criterion_k'])} & {int(r['best_gmm_bic_k'])} \\\\" for _, r in k.iterrows())


def rate_profiles() -> str:
    p = pd.read_csv(VRA / "rate_cell_cluster_profiles.csv")
    names = {1: "Low-rate", 2: "Intermediate-rate", 3: "High-rate"}
    rows = []
    for period, g in p.groupby("period", sort=False):
        rows.append(r"\multicolumn{8}{l}{\textit{" + (period if period == "Overall" else period + " phase") + r"}} \\")
        s, c = g["scheduled"].sum(), g["cancelled"].sum()
        for _, r in g.iterrows():
            rows.append(f"{names[int(r['rate_cluster'])]} & {int(r['n_cells'])} & {f(r['weighted_rate_pct'])} & {f(r['scheduled'] / s * 100, 1)} & "
                        f"{f(r['cancelled'] / c * 100, 1)} & {r['dominant_airline_group']} & {r['dominant_day_type']} & {r['dominant_departure_period']} \\\\")
    NUM["silhouette_cells"] = pd.read_csv(VRA / "rate_cell_cluster_silhouette.csv").round(3).to_dict(orient="records")
    NUM["policy_accounting"] = pd.read_csv(VRA / "watchlist_policy_accounting.csv").round(3).to_dict(orient="records")
    return "\n".join(rows)


def algorithm_table() -> str:
    seg = pd.read_csv(VRA / "cell_algorithm_high_rate_segment.csv")
    ari = pd.read_csv(VRA / "cell_algorithm_comparison.csv").set_index(["period", "algorithm"])["ari_vs_kmeans"]
    rows = []
    for period, g in seg.groupby("period", sort=False):
        first = True
        for _, r in g.iterrows():
            rows.append(f"{period if first else ''} & {r['algorithm']} & {int(r['high_rate_cells'])} & {f(r['high_rate_weighted_rate_pct'])} & "
                        f"{f(r['share_of_scheduled_pct'], 1)} & {f(r['share_of_cancellations_pct'], 1)} & {f(ari[(period, r['algorithm'])])} & {f(r['jaccard_vs_kmeans'])} \\\\")
            first = False
        rows.append(r"\hline")
    return "\n".join(rows[:-1])


def k_sensitivity() -> str:
    k = pd.read_csv(VRA / "watchlist_k_exposure_sensitivity.csv")
    return "\n".join(f"{int(r['k'])} & {n(r['min_scheduled'])} & {f(r['mean_watch_lift_2016_2019'])} & {f(r['mean_watch_lift_2020_2024'])} & "
                     f"{f(r['mean_watch_coverage_t1_pct'], 1)} & {f(r['mean_watch_capture_t1_pct'], 1)} & {signed(r['mean_capture_gap_equal_coverage_pp'])} & "
                     f"{f(r['mean_top_decile_lift_2016_2019'])} & {f(r['mean_top_decile_lift_2020_2024'])} \\\\" for _, r in k.iterrows())


def encoding_table() -> str:
    e = pd.read_csv(VRA / "encoding_sensitivity_segments.csv")
    rows = []
    for period, g in e.groupby("period", sort=False):
        first = True
        for _, r in g.iterrows():
            enc = "Prevalence" if r["encoding"].startswith("Prevalence") else "One-hot"
            rows.append(f"{period if first else ''} & {enc} & {int(r['cells'])} & {f(r['watchlist_rate_pct'])} & {f(r['share_of_scheduled_pct'], 1)} & "
                        f"{f(r['share_of_cancellations_pct'], 1)} & {f(r['jaccard_between_encodings']) if first else ''} \\\\")
            first = False
        rows.append(r"\hline")
    return "\n".join(rows[:-1])


def profile_table() -> str:
    p = pd.read_csv(VRA / "cluster_dominant_characteristics.csv")
    order = ["Overall", "Pre-pandemic", "Pandemic", "Post-pandemic"]
    rows = []
    for period in order:
        g = p[p["period"] == period].sort_values("cluster")
        first = True
        for _, r in g.iterrows():
            h = r["mean_departure_hour"]
            label = "Weekend" if r["dominant_day_type"] == "Weekend" else ("Weekday evening" if h >= 14 else "Weekday morning")
            rows.append(f"{period if first else ''} & {int(r['cluster'])} & {r['share'] * 100:.1f}\\% & {r['dominant_day_type']}; {r['dominant_day_of_week']}; "
                        f"{r['dominant_airline_group']}; {h:.1f}h & {label} \\\\")
            first = False
        rows.append(r"\hline")
    return "\n".join(rows[:-1])


def diagnostics_table() -> str:
    sil = pd.read_csv(VRA / "rate_cell_cluster_silhouette.csv").set_index("period")["silhouette"]
    ev = pd.read_csv(VRA / "time_representation_sensitivity.csv")
    circ = ev[ev["representation"] == "circular_sine_cosine"].set_index("period")["silhouette_k3"]
    lin = ev[ev["representation"] == "linear_hour"].set_index("period")["silhouette_k3"]
    stab = pd.read_csv(VRA / "kmeans_stability_k3.csv").set_index("period")["mean_pairwise_ari"]
    ap = pd.read_csv(VRA / "airport_selection_sensitivity.csv").set_index("selection")["silhouette_k3"]
    js = pd.read_csv(VRA / "baseline_cluster_assignment_divergence.csv").set_index("period")["js_divergence_from_pre_pandemic"]
    pc = pd.read_csv(VRA / "post_cutoff_sensitivity.csv")
    km = pd.read_csv(VRA / "categorical_kmodes_summary.csv")
    rows = [
        f"Segment separation, cells & Silhouette {sil['Overall']:.3f} overall; {sil['Pre-pandemic']:.3f} pre-pandemic; {sil['Pandemic']:.3f} pandemic; {sil['Post-pandemic']:.3f} post-pandemic \\\\",
        f"Segment separation, events & Silhouette {circ['Overall']:.3f} overall; {circ['Pre-pandemic']:.3f} pre-pandemic; {circ['Pandemic']:.3f} pandemic; {circ['Post-pandemic']:.3f} post-pandemic \\\\",
        f"Random starts, events, $k=3$ & Mean pairwise ARI: {stab['Overall']:.2f} overall; {stab['Pre-pandemic']:.2f} pre-pandemic; {stab['Pandemic']:.2f} pandemic; {stab['Post-pandemic']:.2f} post-pandemic \\\\",
        f"Aerodrome selection, events & Silhouette: {ap['all_airports']:.3f} all aerodromes; {ap['cancellation_share_80']:.3f} main 80\\%; {ap['cancellation_share_90']:.3f} main 90\\%; {ap['scheduled_exposure_share_80']:.3f} exposure 80\\% \\\\",
        f"Time representation, events & Circular-time Silhouette {circ.min():.2f}--{circ.max():.2f} against {lin.min():.2f}--{lin.max():.2f} for a linear hour variable, which treats 23:00 and 00:00 as maximally distant \\\\",
        f"Fixed baseline assignment, events & Jensen--Shannon divergence from the pre-pandemic baseline: {js['Pandemic']:.4f} pandemic; {js['Post-pandemic']:.4f} post-pandemic \\\\",
        f"Alternative phase boundaries, events & Silhouette {pc['silhouette_k3'].min():.3f}--{pc['silhouette_k3'].max():.3f} for post-pandemic boundaries from January to October 2023 \\\\",
        f"Categorical k-modes check, events & Mean mismatches per record: {km['mean_mismatches_per_record'].min():.2f}--{km['mean_mismatches_per_record'].max():.2f} across five categorical fields \\citep{{huang1998}} \\\\",
    ]
    NUM["event_silhouette"] = circ.round(3).to_dict()
    NUM["event_js"] = js.round(4).to_dict()
    return "\n".join(rows)


# -------------------------------------------------------- prospective results


def oos_table() -> str:
    s = pd.read_csv(VRA / "watchlist_out_of_sample_summary.csv")
    b = pd.read_csv(ROB / "bootstrap_brazil.csv").set_index("next_year")
    rows = []
    for _, r in s.iterrows():
        bb = b.loc[r["next_year"]]
        rows.append(
            f"{pair(r)} & {int(r['evaluated_cells'])} & {f(r['population_share_national_cancellations_pct'], 1)} & {f(r['system_rate_t1_pct'])} & "
            f"{f(r['top_decile_coverage_t1_pct'], 1)} & {f(r['top_decile_capture_t1_pct'], 1)} & {f(r['top_decile_lift'])} ({f(bb['top_decile_lift_ci_low'])}--{f(bb['top_decile_lift_ci_high'])}) & "
            f"{int(r['watch_cells'])} & {f(r['watch_coverage_t_pct'], 1)} & {f(r['watch_coverage_t1_pct'], 1)} & {f(r['watch_capture_t1_pct'], 1)} & "
            f"{f(r['watch_lift'])} ({f(bb['watch_lift_ci_low'])}--{f(bb['watch_lift_ci_high'])}) & {f(r['ranking_capture_at_watch_coverage_pct'], 1)} & {f(r['spearman'])} \\\\")
    early, late = s[s["next_year"] < 2020], s[s["next_year"] >= 2020]
    NUM["oos"] = {
        "spearman_min": round(s["spearman"].min(), 2), "spearman_max": round(s["spearman"].max(), 2), "spearman_mean": round(s["spearman"].mean(), 2),
        "td_lift_early": [round(early["top_decile_lift"].min(), 2), round(early["top_decile_lift"].max(), 2), round(early["top_decile_lift"].mean(), 2)],
        "td_lift_late": [round(late["top_decile_lift"].min(), 2), round(late["top_decile_lift"].max(), 2), round(late["top_decile_lift"].mean(), 2)],
        "td_cov_t1": [round(s["top_decile_coverage_t1_pct"].min(), 1), round(s["top_decile_coverage_t1_pct"].max(), 1)],
        "td_capture": [round(s["top_decile_capture_t1_pct"].min(), 1), round(s["top_decile_capture_t1_pct"].max(), 1)],
        "td_national_capture": [round(s["top_decile_national_capture_pct"].min(), 1), round(s["top_decile_national_capture_pct"].max(), 1)],
        "watch_lift_early": [round(early["watch_lift"].min(), 2), round(early["watch_lift"].max(), 2)],
        "watch_lift_late": [round(late["watch_lift"].min(), 2), round(late["watch_lift"].max(), 2), round(late["watch_lift"].mean(), 2)],
        "watch_cov_t1_late": [round(late["watch_coverage_t1_pct"].min(), 1), round(late["watch_coverage_t1_pct"].max(), 1)],
        "watch_cov_t_late": [round(late["watch_coverage_t_pct"].min(), 1), round(late["watch_coverage_t_pct"].max(), 1)],
        "watch_capture_late": [round(late["watch_capture_t1_pct"].min(), 1), round(late["watch_capture_t1_pct"].max(), 1)],
        "watch_national_capture_late": [round(late["watch_national_capture_pct"].min(), 1), round(late["watch_national_capture_pct"].max(), 1)],
        "gap_early": [round(early["capture_gap_equal_realized_coverage_pp"].min(), 1), round(early["capture_gap_equal_realized_coverage_pp"].max(), 1)],
        "gap_late": [round(late["capture_gap_equal_realized_coverage_pp"].min(), 1), round(late["capture_gap_equal_realized_coverage_pp"].max(), 1)],
        "threshold_cov_t1_minus_watch": (s["threshold_coverage_t1_pct"] - s["watch_coverage_t1_pct"]).round(2).tolist(),
        "pop_share_canc": [round(s["population_share_national_cancellations_pct"].min(), 1), round(s["population_share_national_cancellations_pct"].max(), 1)],
        "pop_share_flights": [round(s["population_share_national_flights_pct"].min(), 1), round(s["population_share_national_flights_pct"].max(), 1)],
        "auc_early": [round(early["mannwhitney_auc"].min(), 2), round(early["mannwhitney_auc"].max(), 2)],
        "auc_late": [round(late["mannwhitney_auc"].min(), 2), round(late["mannwhitney_auc"].max(), 2)],
        "mw_p_early": early["mannwhitney_p"].round(4).tolist(),
        "rows": s.round(3).to_dict(orient="records"),
    }
    return "\n".join(rows)


def attrition_table() -> str:
    s = pd.read_csv(VRA / "watchlist_out_of_sample_summary.csv")
    rows = []
    for _, r in s.iterrows():
        rows.append(f"{pair(r)} & {int(r['base_cells'])} & {f(r['base_retained_exposure_pct'], 1)} & {int(r['exit_cells'])} & {int(r['evaluated_cells'])} & "
                    f"{f(r['population_share_national_flights_pct'], 1)} & {f(r['population_share_national_cancellations_pct'], 1)} & {int(r['panel_cells'])} & "
                    f"{f(r['panel_share_national_cancellations_pct'], 1)} & {int(r['entrant_cells'])} & {f(r['entrant_share_national_flights_pct'], 1)} \\\\")
    return "\n".join(rows)


def bootstrap_table() -> str:
    rows = []
    for reg, file in (("Brazil", ROB / "bootstrap_brazil.csv"), ("United States", ROB / "bootstrap_us.csv")):
        b = pd.read_csv(file)
        first = True
        for _, r in b.iterrows():
            p = "$<$0.001" if r["mannwhitney_p"] < 0.001 else f"{r['mannwhitney_p']:.3f}"
            rows.append(f"{reg if first else ''} & {pair(r)} & {int(r['airports'])} & {f(r['top_decile_lift'])} & {interval(r['top_decile_lift_ci_low'], r['top_decile_lift_ci_high'])} & "
                        f"{f(r['watch_lift'])} & {interval(r['watch_lift_ci_low'], r['watch_lift_ci_high'])} & {signed(r['watch_rate_diff_pp'], 2)} & "
                        f"{interval(r['watch_rate_diff_pp_ci_low'], r['watch_rate_diff_pp_ci_high'])} & {signed(r['capture_gap_equal_realized_coverage_pp'])} & "
                        f"{interval(r['capture_gap_pp_ci_low'], r['capture_gap_pp_ci_high'], 1)} & {p} & {f(r['mannwhitney_auc'])} \\\\")
            first = False
        rows.append(r"\hline")
        NUM[f"bootstrap_{reg}"] = b.round(3).to_dict(orient="records")
    return "\n".join(rows[:-1])


def stability_table() -> str:
    s = pd.read_csv(ROB / "membership_stability.csv")
    rows = []
    for _, r in s.iterrows():
        rows.append(f"{pair(r)} & {int(r['eligible_t'])} & {int(r['common_cells'])} & "
                    f"{int(r['watch_size_t'])} & {f(r['watch_jaccard_complete'])} & {f(r['watch_jaccard_conditional'])} & {int(r['watch_exits_ineligible'])}/{int(r['watch_exits_reclassified'])} & "
                    f"{int(r['watch_entries_new_eligible'])}/{int(r['watch_entries_reclassified'])} & "
                    f"{int(r['top_decile_size_t'])} & {f(r['top_decile_jaccard_complete'])} & {f(r['top_decile_jaccard_conditional'])} & "
                    f"{f(r['threshold_jaccard_complete'])} & {f(r['threshold_jaccard_conditional'])} \\\\")
    late, early = s[s["next_year"] >= 2020], s[s["next_year"] < 2020]
    NUM["stability"] = {tag: {c: round(part[c].mean(), 2) for c in s.columns if "jaccard" in c} for tag, part in (("late", late), ("early", early))}
    NUM["stability_rows"] = s.round(3).to_dict(orient="records")
    return "\n".join(rows)


def sensitivity_table() -> str:
    t = pd.read_csv(ROB / "sensitivity_out_of_sample_summary.csv").set_index("specification")
    groups = [
        (None, [("Baseline", "Baseline (year-$t$ carrier groups, origin, day type, six-hour period; min.\\ 500)")]),
        ("Population and data construction", [
            ("Matched panel (min 500 in t+1)", "Matched panel (cells with $\\geq$500 flights in both years)"),
            ("Full-period carrier grouping", "Carrier groups fixed from 2015--2024 shares"),
            ("Brasilia clock", "Brasilia clock instead of origin-local time"),
            ("Excluding legs with restored dates", "Legs with restored scheduled dates excluded")]),
        ("Seasonality", [("Seasonal: quarter, min 500", "Quarter added; min.\\ 500"), ("Seasonal: quarter, min 125", "Quarter added; min.\\ 125")]),
        ("Destination and route", [("Destination cells", "Destination replaces origin"), ("Route cells, min 500", "Origin--destination added; min.\\ 500"),
                                   ("Route cells, min 200", "Origin--destination added; min.\\ 200")]),
        ("Service scope", [("Domestic flights", "Domestic flights only"), ("Domestic passenger flights", "Domestic passenger flights only")]),
        ("Carrier grouping", [("Carrier threshold 1.0%", "Carriers with $\\geq$1\\% of year-$t$ flights retained"),
                              ("Carrier threshold 2.5%", "Carriers with $\\geq$2.5\\% of year-$t$ flights retained"),
                              ("Carrier threshold 10.0%", "Carriers with $\\geq$10\\% of year-$t$ flights retained"),
                              ("Others split by operator category", "Others split into four operator categories"),
                              ("Individual carriers", "Individual operating carriers")]),
        ("Departure windows", [("Departure windows: shifted 6 h", "Six-hour periods shifted by three hours"), ("Departure windows: 3 h", "Three-hour windows"),
                               ("Departure windows: 2 h", "Two-hour windows")]),
        ("Segmentation choices", [("k by Gap statistic", "$k$ chosen yearly by gap statistic"), ("k by Calinski-Harabasz", "$k$ chosen yearly by Calinski--Harabasz"),
                                  ("k by Silhouette", "$k$ chosen yearly by Silhouette"), ("Encoding: one-hot", "One-hot encoding of carrier and airport"),
                                  ("Encoding: rate and time only", "Rate and temporal features only")]),
    ]
    rows = []
    for head, items in groups:
        if head:
            rows.append(r"\multicolumn{11}{l}{\textit{" + head + r"}} \\")
        for key, label in items:
            r = t.loc[key]
            ov = "--" if key == "Baseline" else f(r["mean_top_decile_flight_jaccard_2020_2024"])
            wov = "--" if key == "Baseline" else f(r["mean_watch_flight_jaccard_2020_2024"])
            rows.append(f"{label} & {f(r['mean_retained_exposure_pct'], 1)} & {f(r['mean_population_share_cancellations_pct'], 1)} & "
                        f"{f(r['mean_top_decile_lift_2016_2019'])} & {f(r['mean_top_decile_lift_2020_2024'])} & {ov} & "
                        f"{f(r['mean_watch_lift_2016_2019'])} & {f(r['mean_watch_lift_2020_2024'])} & {f(r['mean_watch_coverage_t1_2020_2024'], 1)} & "
                        f"{signed(r['mean_capture_gap_equal_coverage_pp_2020_2024'])} & {wov} \\\\")
    NUM["sensitivity"] = t.round(3).reset_index().to_dict(orient="records")
    return "\n".join(rows)


def regimes_table() -> str:
    r0 = pd.read_csv(ROB / "regime_out_of_sample.csv")

    def lab(w: str) -> str:
        a, b = w.split("-")
        a, b = int(a), int(b)
        if a % 100 == 1 and b % 100 == 12 and a // 100 == b // 100:
            return str(a // 100)
        mon = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        return f"{mon[a % 100 - 1]} {str(a // 100)[2:]}--{mon[b % 100 - 1]} {str(b // 100)[2:]}"

    rows = []
    for _, r in r0.iterrows():
        rows.append(f"{r['regime']} & {lab(r['base_window'])} & {lab(r['evaluation_window'])} & {int(r['min_scheduled_base'])}/{int(r['min_scheduled_eval'])} & "
                    f"{f(r['system_rate_t1_pct'])} & {f(r['top_decile_lift'])} & {f(r['watch_lift'])} & {f(r['threshold_lift'])} & {f(r['spearman'])} \\\\")
    NUM["regimes"] = r0.round(3).to_dict(orient="records")
    return "\n".join(rows)


def reasons_table() -> str:
    t = pd.read_csv(ROB / "reason_groups_watchlist.csv")
    rows = []
    for (b, nx), g in t.groupby(["base_year", "next_year"], sort=True):
        first = True
        for rule, gg in g.groupby("rule", sort=False):
            rl = "Watch list" if rule == "watch" else "Top decile"
            for _, r in gg.iterrows():
                cells = "Flagged" if r["cells"] == "Flagged" else "Other"
                v = f"{r['cramers_v']:.2f}" if np.isfinite(r.get("cramers_v", np.nan)) else ""
                if cells == "Flagged":
                    v = f"{gg['cramers_v'].dropna().iloc[0]:.2f}"
                rows.append(f"{f'{b}--{nx}' if first else ''} & {rl if cells == 'Flagged' else ''} & {cells} & {n(r['cancelled'])} & "
                            + " & ".join(f(r[f'share_{x}'], 1) for x in GROUPS) + f" & {v if cells == 'Flagged' else ''} \\\\")
                first = False
        rows.append(r"\hline")
    NUM["reasons_cramers_v"] = t.dropna(subset=["cramers_v"])[["next_year", "rule", "cramers_v"]].round(3).to_dict(orient="records")
    return "\n".join(rows[:-1])


def exclusions_table() -> str:
    e = pd.read_csv(ROB / "minimum_exposure_exclusions.csv")
    rows = []
    for thr in (200, 500, 1000):
        sub = e[e["min_scheduled"] == thr]
        for label, unit, level in (("Full period", "2015-2024", "Full period"), ("Pre-pandemic phase", "Pre-pandemic", "Phase"),
                                   ("Pandemic phase", "Pandemic", "Phase"), ("Post-pandemic phase", "Post-pandemic", "Phase")):
            r = sub[(sub["level"] == level) & (sub["unit"] == unit)].iloc[0]
            rows.append(f"{n(thr)} & {label} & {int(r['cells_excluded'])}/{int(r['cells_total'])} & {int(r['airports_excluded'])}/{int(r['airports_total'])} & "
                        f"{int(r['carrier_airport_excluded'])}/{int(r['carrier_airport_total'])} & {int(r['operating_carriers_excluded'])}/{int(r['operating_carriers_total'])} & "
                        f"{f(r['scheduled_excluded_pct'], 1)} & {f(r['cancelled_excluded_pct'], 1)} & {f(r['rate_excluded_pct'])} & {f(r['rate_retained_pct'])} \\\\")
        y = sub[sub["level"] == "Year"].mean(numeric_only=True)
        rows.append(f"{n(thr)} & Single year (mean) & {y['cells_excluded']:.0f}/{y['cells_total']:.0f} & {y['airports_excluded']:.0f}/{y['airports_total']:.0f} & "
                    f"{y['carrier_airport_excluded']:.0f}/{y['carrier_airport_total']:.0f} & {y['operating_carriers_excluded']:.0f}/{y['operating_carriers_total']:.0f} & "
                    f"{f(y['scheduled_excluded_pct'], 1)} & {f(y['cancelled_excluded_pct'], 1)} & {f(y['rate_excluded_pct'])} & {f(y['rate_retained_pct'])} \\\\")
        NUM[f"exclusions_year_mean_{thr}"] = y.round(3).to_dict()
        rows.append(r"\hline")
    return "\n".join(rows[:-1])


def workload_table() -> str:
    """Review workload of the frozen top-decile lists (generate_operational_implications.py)."""
    w = pd.read_csv(ROB / "operational_workload.csv")
    q = pd.read_csv(ROB / "annual_watchlist_lift_by_quarter.csv")
    q1 = q[(q["rule"] == "top_decile") & (q["quarter"] == 1)].set_index("next_year")["lift"]
    watch = w[w["rule"] == "watch"].set_index("next_year")["share_cells_above_population_rate_pct"]
    rows = []
    for _, r in w[w["rule"] == "top_decile"].iterrows():
        ny = int(r["next_year"])
        rows.append(f"{pair(r)} & {int(r['cells'])} & {int(r['origin_airports'])} & {int(r['carrier_groups'])} & {int(r['pooled_cells'])} & "
                    f"{int(r['operating_carriers_ge5pct_of_a_cell_t'])} & {n(r['cancelled_t1'])} & {n(r['above_benchmark_t1'])} & "
                    f"{f(r['share_cells_above_population_rate_pct'], 1)} & {f(r['share_cells_above_twice_population_rate_pct'], 1)} & "
                    f"{f(q1[ny])} & {f(watch[ny], 1)} \\\\")
    NUM["workload"] = w.round(3).to_dict(orient="records")
    NUM["first_quarter_top_decile_lift"] = q1.round(3).to_dict()
    NUM["low_frequency_options"] = pd.read_csv(ROB / "low_frequency_options.csv").round(3).to_dict(orient="records")
    NUM["low_frequency_means"] = pd.read_csv(ROB / "low_frequency_options.csv").mean(numeric_only=True).round(2).to_dict()
    return "\n".join(rows)


# ----------------------------------------------------------------- BTS tables


def us_summary() -> str:
    u = pd.read_csv(BTS / "bts_yearly_summary.csv")
    NUM["us_total"] = int(u["scheduled_flights"].sum())
    return "\n".join(f"{int(r['year'])} & {n(r['scheduled_flights'])} & {n(r['cancellations'])} & {f(r['cancellation_rate_pct'])} & "
                     f"{int(r['carriers'])} & {int(r['origin_airports'])} \\\\" for _, r in u.iterrows())


def cross_country() -> str:
    br = pd.read_csv(VRA / "watchlist_out_of_sample_summary.csv")
    sp = pd.read_csv(ROB / "sensitivity_out_of_sample_pairs.csv")
    dp = sp[sp["specification"] == "Domestic passenger flights"]
    us = pd.read_csv(BTS / "bts_watchlist_out_of_sample_summary.csv")
    rows = []
    for head, t in (("Brazil, all Brazilian-origin flights", br), ("Brazil, domestic passenger flights", dp),
                    ("United States (BTS), domestic flights of reporting carriers", us)):
        rows.append(r"\multicolumn{9}{l}{\textit{" + head + r"}} \\")
        for ny in (2019, 2021, 2024):
            r = t[t["next_year"] == ny].iloc[0]
            rows.append(f"{pair(r)} & {f(r['population_share_national_cancellations_pct'], 1)} & {f(r['top_decile_lift'])} & {f(r['top_decile_capture_t1_pct'], 1)} & "
                        f"{f(r['watch_lift'])} & {f(r['watch_coverage_t1_pct'], 1)} & {f(r['watch_capture_t1_pct'], 1)} & "
                        f"{signed(r['capture_gap_equal_realized_coverage_pp'])} & {f(r['spearman'])} \\\\")
    NUM["us_oos"] = us.round(3).to_dict(orient="records")
    NUM["dp_oos"] = dp[["next_year", "top_decile_lift", "watch_lift", "spearman", "watch_cells", "watch_coverage_t1_pct"]].round(3).to_dict(orient="records")
    cc = pd.read_csv(BTS / "cross_country_watchlist_comparison.csv")
    NUM["cross_country_phase"] = cc.round(3).to_dict(orient="records")
    return "\n".join(rows)


# ------------------------------------------------------------------ injection

BUILDERS = {
    "annual": annual_record, "inclusion": inclusion_table, "reconciliation": reconciliation_table, "carrier-rates": carrier_rates,
    "others": others_table, "k-selection": k_selection, "rate-profiles": rate_profiles, "oos": oos_table, "attrition": attrition_table,
    "sensitivity": sensitivity_table, "regimes": regimes_table, "us-summary": us_summary, "cross-country": cross_country,
    "reasons": reasons_table, "algorithms": algorithm_table, "k-sensitivity": k_sensitivity, "encoding": encoding_table,
    "profiles": profile_table, "diagnostics": diagnostics_table, "exclusions": exclusions_table, "bootstrap": bootstrap_table,
    "stability": stability_table, "workload": workload_table,
}


def main() -> None:
    global VRA, ROB, BTS
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from-output", action="store_true",
                        help="read the CSV files of a full rerun instead of the archived results/")
    parser.add_argument("--check", action="store_true",
                        help="compare every table body with the archived one in results/tables/")
    args = parser.parse_args()

    source = ANALYSIS_DIR if args.from_output else RESULTS_DIR
    VRA, BTS = source / "vra", source / "bts"
    ROB = VRA / "robustness"

    registry_numbers()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    mismatches = []
    for name, build in BUILDERS.items():
        body = build() + "\n"
        (TABLE_DIR / f"{name}.tex").write_text(body, encoding="utf-8")
        if args.check:
            archived = (RESULTS_DIR / "tables" / f"{name}.tex").read_text(encoding="utf-8")
            if body != archived:
                mismatches.append(name)
    (TABLE_DIR / "numbers.json").write_text(json.dumps(NUM, indent=1, default=float), encoding="utf-8")
    print(f"wrote {len(BUILDERS)} table bodies from {source} to {TABLE_DIR}")
    if args.check:
        if mismatches:
            print("differ from the archived tables:", ", ".join(mismatches))
            sys.exit(1)
        print("all table bodies are identical to the archived ones in results/tables/")


if __name__ == "__main__":
    main()
