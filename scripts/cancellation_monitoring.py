"""Reusable exposure-based cancellation-monitoring routines.

The module implements the operational-cell monitoring workflow independently of
any single national registry. It receives a flight-level table with the columns

    ``airline``, ``carrier_group_fixed``, ``origin``, ``day_type``,
    ``departure_period``, ``is_cancelled`` and ``year``

(plus optional columns used by particular cell definitions) and provides
operational-cell construction, internal cluster-validity indices, alternative
clustering algorithms, and the prospective validation of screening rules. The same
functions are applied to the Brazilian ANAC/VRA registry and to the US DOT/BTS
On-Time Performance registry.

Prospective validation. Everything that defines a screening list is fixed with
information from the assignment period t alone: the carrier grouping, the cells
that are eligible, the cell features, the cluster labels (including which cluster
is the high-rate segment), the rate ranking, and the review budgets. The frozen
lists are then scored against the flights of period t+1. The evaluation population
is every eligible period-t cell that still has scheduled flights in period t+1;
cells without flights in t+1 are exits and have no defined future rate. A matched
panel, in which cells must also reach the minimum exposure in t+1, is used for
cell-level rate statistics (Spearman correlation, Mann-Whitney test).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
CELL_KEYS = ["carrier_group", "origin", "day_type", "departure_period"]
PERIOD_HOUR = {"Night": 3, "Morning": 9, "Afternoon": 15, "Evening": 21}
MIN_SCHEDULED = 500
GROUP_SHARE = 0.05
TOP_SHARE = 0.10


# ---------------------------------------------------------------- cell building


def build_cells(df: pd.DataFrame, min_sched: int = MIN_SCHEDULED, keys: list[str] | None = None) -> pd.DataFrame:
    """Aggregate flight records into operational cells with exposure and rate."""
    keys = keys or CELL_KEYS
    cells = (
        df.groupby(keys, observed=True)
        .agg(scheduled=("is_cancelled", "size"), cancelled=("is_cancelled", "sum"))
        .reset_index()
    )
    cells = cells[cells["scheduled"] >= min_sched].copy()
    cells["rate_pct"] = cells["cancelled"] / cells["scheduled"] * 100
    return cells.reset_index(drop=True)


def cell_features(cells: pd.DataFrame, maps: dict[str, pd.Series] | None = None) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Build the baseline cell feature frame and the prevalence maps used to encode it."""
    angle = 2 * np.pi * cells["departure_period"].map(PERIOD_HOUR) / 24
    feat = pd.DataFrame(
        {
            "rate_pct": cells["rate_pct"].to_numpy(),
            "dep_sin": np.sin(angle).to_numpy(),
            "dep_cos": np.cos(angle).to_numpy(),
            "is_weekend": (cells["day_type"] == "Weekend").astype(float).to_numpy(),
        }
    )
    maps = dict(maps) if maps else {}
    for col in ["carrier_group", "origin"]:
        if col not in maps:
            maps[col] = cells[col].value_counts(normalize=True)
        feat[f"{col}_freq"] = cells[col].map(maps[col]).fillna(0.0).to_numpy()
    return feat, maps


def scaled_matrix(feat: pd.DataFrame, scaler: StandardScaler | None = None) -> tuple[np.ndarray, StandardScaler]:
    if scaler is None:
        scaler = StandardScaler().fit(feat)
    return scaler.transform(feat), scaler


def weighted_rate(group: pd.DataFrame) -> float:
    scheduled = group["scheduled"].sum()
    return float(group["cancelled"].sum() / scheduled * 100) if scheduled else float("nan")


def order_clusters_by_rate(cells: pd.DataFrame, label_col: str = "cluster") -> dict[int, int]:
    """Map raw cluster labels to ranks ordered from the lowest to the highest rate."""
    ordered = (
        cells.groupby(label_col)
        .apply(weighted_rate, include_groups=False)
        .sort_values()
        .index.tolist()
    )
    return {label: rank for rank, label in enumerate(ordered, start=1)}


# ------------------------------------------------------- internal validity indices


def gap_statistic(x: np.ndarray, k_values, n_refs: int = 20, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Tibshirani et al. (2001) gap statistic with uniform reference distributions."""
    rng = np.random.default_rng(seed)
    lo, hi = x.min(axis=0), x.max(axis=0)
    rows = []
    for k in k_values:
        dispersion = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x).inertia_
        ref_logs = []
        for _ in range(n_refs):
            ref = rng.uniform(lo, hi, size=x.shape)
            ref_logs.append(np.log(KMeans(n_clusters=k, n_init=5, random_state=seed).fit(ref).inertia_))
        ref_logs = np.asarray(ref_logs)
        rows.append(
            {
                "k": k,
                "gap": float(ref_logs.mean() - np.log(dispersion)),
                "gap_sk": float(ref_logs.std(ddof=1) * np.sqrt(1 + 1 / n_refs)),
            }
        )
    return pd.DataFrame(rows)


def first_gap_k(gap_table: pd.DataFrame) -> int | float:
    """Smallest k satisfying gap(k) >= gap(k+1) - s(k+1)."""
    table = gap_table.sort_values("k").reset_index(drop=True)
    for i in range(len(table) - 1):
        if table.loc[i, "gap"] >= table.loc[i + 1, "gap"] - table.loc[i + 1, "gap_sk"]:
            return int(table.loc[i, "k"])
    return float("nan")


def validity_indices(x: np.ndarray, k_values=range(2, 11), n_refs: int = 20, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Silhouette, Calinski-Harabasz, Davies-Bouldin, gap statistic and GMM BIC."""
    gaps = gap_statistic(x, list(k_values), n_refs=n_refs, seed=seed).set_index("k")
    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(x)
        labels = model.labels_
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=seed, n_init=3).fit(x)
        rows.append(
            {
                "k": k,
                "silhouette": float(silhouette_score(x, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(x, labels)),
                "davies_bouldin": float(davies_bouldin_score(x, labels)),
                "inertia": float(model.inertia_),
                "gap": float(gaps.loc[k, "gap"]),
                "gap_sk": float(gaps.loc[k, "gap_sk"]),
                "gmm_bic": float(gmm.bic(x)),
            }
        )
    return pd.DataFrame(rows)


# -------------------------------------------------------- alternative algorithms


def kmedoids(x: np.ndarray, k: int, max_iter: int = 100, seed: int = RANDOM_STATE) -> np.ndarray:
    """Alternating k-medoids heuristic minimizing summed (unsquared) Euclidean distances."""
    rng = np.random.default_rng(seed)
    distances = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    medoids = rng.choice(len(x), size=k, replace=False)
    labels = np.full(len(x), -1)
    for _ in range(max_iter):
        new_labels = distances[:, medoids].argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for cluster in range(k):
            members = np.where(labels == cluster)[0]
            if len(members) == 0:
                continue
            medoids[cluster] = members[distances[np.ix_(members, members)].sum(axis=1).argmin()]
    return labels


def compare_algorithms(x: np.ndarray, k: int = 3, seed: int = RANDOM_STATE) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Fit K-Means and three alternative partitioning methods on the same features."""
    labels = {
        "K-Means": KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(x),
        "Gaussian mixture": GaussianMixture(n_components=k, covariance_type="full", random_state=seed, n_init=5).fit_predict(x),
        "Ward hierarchical": AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(x),
        "K-Medoids": kmedoids(x, k, seed=seed),
    }
    reference = labels["K-Means"]
    rows = []
    for name, lab in labels.items():
        rows.append(
            {
                "algorithm": name,
                "k": k,
                "silhouette": float(silhouette_score(x, lab)) if len(np.unique(lab)) > 1 else float("nan"),
                "calinski_harabasz": float(calinski_harabasz_score(x, lab)) if len(np.unique(lab)) > 1 else float("nan"),
                "davies_bouldin": float(davies_bouldin_score(x, lab)) if len(np.unique(lab)) > 1 else float("nan"),
                "ari_vs_kmeans": float(adjusted_rand_score(reference, lab)),
            }
        )
    return pd.DataFrame(rows), labels


def high_rate_agreement(cells: pd.DataFrame, labels: dict[str, np.ndarray]) -> pd.DataFrame:
    """Compare the high-rate segment each algorithm isolates."""
    rows = []
    reference_members = None
    for name, lab in labels.items():
        temp = cells.copy()
        temp["cluster"] = lab
        ranks = order_clusters_by_rate(temp)
        temp["rank"] = temp["cluster"].map(ranks)
        top = temp[temp["rank"] == max(ranks.values())]
        members = set(zip(*(top[col] for col in CELL_KEYS)))
        if reference_members is None:
            reference_members = members
        jaccard = len(members & reference_members) / len(members | reference_members)
        rows.append(
            {
                "algorithm": name,
                "high_rate_cells": len(top),
                "high_rate_weighted_rate_pct": weighted_rate(top),
                "share_of_scheduled_pct": float(top["scheduled"].sum() / cells["scheduled"].sum() * 100),
                "share_of_cancellations_pct": float(top["cancelled"].sum() / cells["cancelled"].sum() * 100),
                "jaccard_vs_kmeans": jaccard,
            }
        )
    return pd.DataFrame(rows)


def fit_watchlist(cells: pd.DataFrame, k: int = 3, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Cluster cells with the baseline features and rank clusters from lowest to highest rate."""
    feat, _ = cell_features(cells)
    x, _ = scaled_matrix(feat)
    assigned = cells.copy()
    assigned["cluster"] = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(x)
    ranks = order_clusters_by_rate(assigned)
    assigned["rate_cluster"] = assigned["cluster"].map(ranks)
    return assigned


# ------------------------------------------------- generic cell specifications


@dataclass
class CellSpec:
    """A cell definition and the choices used to segment and screen its cells.

    carrier_rule: "share" retains carriers holding at least ``group_share`` of the
    assignment period's flights and pools the rest; "fixed" uses the full-period
    grouping in ``carrier_group_fixed``; "individual" keeps every operating carrier;
    "category" retains carriers by share and splits the pooled rest by
    ``operator_category`` (a fixed retrospective taxonomy; this sensitivity is
    not a fully prospective classification of operator types).
    """

    name: str
    keys: list[str] = field(default_factory=lambda: list(CELL_KEYS))
    time_col: str = "departure_period"
    time_mid: dict[str, float] = field(default_factory=lambda: dict(PERIOD_HOUR))
    prevalence: list[str] = field(default_factory=lambda: ["carrier_group", "origin"])
    quarter: bool = False
    min_sched: int = MIN_SCHEDULED
    min_sched_eval: int | None = None
    encoding: str = "prevalence"  # "prevalence", "onehot" or "rate_time"
    carrier_rule: str = "share"
    group_share: float = GROUP_SHARE
    population: str = "deployment"  # "deployment" (all surviving cells) or "matched" (panel)


def assign_carrier_groups(base_df: pd.DataFrame, next_df: pd.DataFrame, spec: CellSpec) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Carry grouping into t+1; fixed/category sensitivities condition on retrospective labels."""
    rule = spec.carrier_rule
    if rule == "fixed":
        groups = [base_df["carrier_group_fixed"], next_df["carrier_group_fixed"]]
        retained = sorted(set(base_df["carrier_group_fixed"].unique()) - {"Others"})
    elif rule == "individual":
        groups = [base_df["airline"], next_df["airline"]]
        retained = []
    else:
        shares = base_df["airline"].value_counts(normalize=True)
        retained = sorted(shares[shares >= spec.group_share].index)
        groups = []
        for d in (base_df, next_df):
            g = pd.Series(np.where(d["airline"].isin(retained), d["airline"], "Others"), index=d.index)
            if rule == "category":
                g = g.where(g != "Others", "Others: " + d["operator_category"].astype(str))
            groups.append(g)
    base = base_df.assign(carrier_group=groups[0].to_numpy())
    nxt = next_df.assign(carrier_group=groups[1].to_numpy())
    return base, nxt, retained


def feature_matrix(cells: pd.DataFrame, spec: CellSpec) -> np.ndarray:
    """Standardized features; under one-hot encoding the indicator block is appended unscaled."""
    angle = 2 * np.pi * cells[spec.time_col].map(spec.time_mid).astype(float) / 24
    feat = pd.DataFrame(
        {
            "rate_pct": cells["rate_pct"].to_numpy(),
            "dep_sin": np.sin(angle).to_numpy(),
            "dep_cos": np.cos(angle).to_numpy(),
            "is_weekend": (cells["day_type"] == "Weekend").astype(float).to_numpy(),
        }
    )
    if spec.quarter:
        q_angle = 2 * np.pi * (cells["quarter"].astype(float) - 0.5) / 4
        feat["q_sin"] = np.sin(q_angle).to_numpy()
        feat["q_cos"] = np.cos(q_angle).to_numpy()
    if spec.encoding == "prevalence":
        for col in spec.prevalence:
            feat[f"{col}_freq"] = cells[col].map(cells[col].value_counts(normalize=True)).fillna(0.0).to_numpy()
        return StandardScaler().fit_transform(feat)
    numeric = StandardScaler().fit_transform(feat)
    if spec.encoding == "onehot":
        dummies = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit_transform(cells[spec.prevalence])
        return np.hstack([numeric, dummies])
    return numeric


def fit_segments(cells: pd.DataFrame, spec: CellSpec, k: int = 3, seed: int = RANDOM_STATE) -> pd.DataFrame:
    assigned = cells.copy()
    assigned["cluster"] = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(feature_matrix(cells, spec))
    assigned["rate_cluster"] = assigned["cluster"].map(order_clusters_by_rate(assigned))
    return assigned


def ranked_order(cells: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Cells in descending order of assignment-period rate; ties broken by exposure, then by key."""
    return cells.sort_values(["rate_pct", "scheduled"] + keys, ascending=[False, False] + [True] * len(keys), kind="mergesort")


def budget_flags(ordered: pd.DataFrame, budget: float) -> pd.Series:
    """Flag cells in rank order while cumulative assignment-period exposure stays within the budget.

    The cell whose inclusion would exceed the budget is not flagged, so achieved
    coverage is at or below the budget; if the first cell alone exceeds it, that
    cell is flagged.
    """
    cum = ordered["scheduled"].cumsum() / ordered["scheduled"].sum()
    flags = cum <= budget + 1e-12
    if not flags.any():
        flags.iloc[0] = True
    return flags.reindex(ordered.index)


def prospective_evaluation(base_df: pd.DataFrame, next_df: pd.DataFrame, spec: CellSpec, k: int = 3,
                           return_flights: bool = False) -> tuple | None:
    """Freeze screening lists with period-t information and score them on period t+1.

    Returns (summary row, evaluated cells) and, if ``return_flights``, the grouped
    period-t+1 flights, so that flagged flights can be compared across cell definitions.
    """
    base_df, next_df, retained = assign_carrier_groups(base_df, next_df, spec)
    keys = spec.keys
    base = build_cells(base_df, spec.min_sched, keys)
    if len(base) < k + 1:
        return None
    assigned = fit_segments(base, spec, k)
    top_rank = int(assigned["rate_cluster"].max())
    assigned["watch"] = assigned["rate_cluster"] == top_rank  # fixed at assignment
    budget = assigned.loc[assigned["watch"], "scheduled"].sum() / assigned["scheduled"].sum()
    ordered = ranked_order(assigned, keys)
    assigned["rank_position"] = pd.Series(np.arange(len(ordered)), index=ordered.index).reindex(assigned.index)
    assigned["threshold"] = budget_flags(ordered, budget).reindex(assigned.index)
    assigned["top_decile"] = budget_flags(ordered, TOP_SHARE).reindex(assigned.index)

    nxt = build_cells(next_df, 1, keys).rename(columns={"scheduled": "scheduled_t1", "cancelled": "cancelled_t1", "rate_pct": "rate_pct_t1"})
    frame = assigned.rename(columns={"scheduled": "scheduled_t", "cancelled": "cancelled_t", "rate_pct": "rate_pct_t"}).merge(
        nxt, on=keys, how="left")
    frame[["scheduled_t1", "cancelled_t1"]] = frame[["scheduled_t1", "cancelled_t1"]].fillna(0)
    pop = frame[frame["scheduled_t1"] > 0].copy()
    min_eval = spec.min_sched_eval if spec.min_sched_eval is not None else spec.min_sched
    panel = pop[pop["scheduled_t1"] >= min_eval]
    if spec.population == "matched":
        pop = panel.copy()
    if pop["watch"].sum() == 0 or (~pop["watch"]).sum() == 0 or len(panel) < 5:
        return None

    nat_s, nat_c = float(len(next_df)), float(next_df["is_cancelled"].sum())
    s1, c1 = pop["scheduled_t1"].sum(), pop["cancelled_t1"].sum()
    system = c1 / s1 * 100

    def rule_stats(flag: str) -> dict:
        m = pop[flag]
        rate = pop.loc[m, "cancelled_t1"].sum() / pop.loc[m, "scheduled_t1"].sum() * 100
        return {
            f"{flag}_cells": int(assigned[flag].sum()),
            f"{flag}_coverage_t_pct": float(assigned.loc[assigned[flag], "scheduled"].sum() / assigned["scheduled"].sum() * 100),
            f"{flag}_coverage_t1_pct": float(pop.loc[m, "scheduled_t1"].sum() / s1 * 100),
            f"{flag}_capture_t1_pct": float(pop.loc[m, "cancelled_t1"].sum() / c1 * 100),
            f"{flag}_national_capture_pct": float(pop.loc[m, "cancelled_t1"].sum() / nat_c * 100),
            f"{flag}_rate_t1_pct": float(rate),
            f"{flag}_lift": float(rate / system),
        }

    row = {"specification": spec.name, "retained_carriers": "|".join(retained), "base_cells": len(base),
           "base_retained_exposure_pct": float(base["scheduled"].sum() / len(base_df) * 100),
           "evaluated_cells": len(pop), "exit_cells": int((frame["scheduled_t1"] == 0).sum()),
           "exit_base_exposure_pct": float(frame.loc[frame["scheduled_t1"] == 0, "scheduled_t"].sum() / frame["scheduled_t"].sum() * 100),
           "panel_cells": len(panel), "system_rate_t1_pct": float(system),
           "population_share_national_flights_pct": float(s1 / nat_s * 100),
           "population_share_national_cancellations_pct": float(c1 / nat_c * 100),
           "panel_share_national_flights_pct": float(panel["scheduled_t1"].sum() / nat_s * 100),
           "panel_share_national_cancellations_pct": float(panel["cancelled_t1"].sum() / nat_c * 100)}
    eligible_t1 = build_cells(next_df, min_eval, keys)
    base_keys = set(map(tuple, base[keys].to_numpy()))
    entrants = eligible_t1[[tuple(r) not in base_keys for r in eligible_t1[keys].to_numpy()]]
    row["entrant_cells"] = len(entrants)
    row["entrant_share_national_flights_pct"] = float(entrants["scheduled"].sum() / nat_s * 100)
    for flag in ("watch", "threshold", "top_decile"):
        row.update(rule_stats(flag))

    # Capture of the rate ranking at the watch list's realized t+1 coverage.
    cur = pop.sort_values("rank_position")
    x = np.concatenate([[0.0], cur["scheduled_t1"].cumsum().to_numpy() / s1])
    y = np.concatenate([[0.0], cur["cancelled_t1"].cumsum().to_numpy() / c1])
    ranking_at_watch = float(np.interp(row["watch_coverage_t1_pct"] / 100, x, y) * 100)
    row["ranking_capture_at_watch_coverage_pct"] = ranking_at_watch
    row["capture_gap_equal_realized_coverage_pp"] = row["watch_capture_t1_pct"] - ranking_at_watch

    # Cell-level statistics on the matched panel.
    row["spearman"] = float(stats.spearmanr(panel["rate_pct_t"], panel["rate_pct_t1"]).statistic)
    w, o = panel.loc[panel["watch"], "rate_pct_t1"], panel.loc[~panel["watch"], "rate_pct_t1"]
    if len(w) and len(o):
        u, p = stats.mannwhitneyu(w, o, alternative="greater")
        row["mannwhitney_p"], row["mannwhitney_auc"] = float(p), float(u / (len(w) * len(o)))
    else:
        row["mannwhitney_p"] = row["mannwhitney_auc"] = float("nan")
    if return_flights:
        return row, pop, next_df
    return row, pop


def flagged_flights(next_df: pd.DataFrame, pop: pd.DataFrame, keys: list[str], flag: str = "watch") -> np.ndarray:
    """Boolean flag over period-t+1 flights that belong to evaluated cells flagged by ``flag``."""
    flagged = pop.loc[pop[flag], keys].drop_duplicates().assign(_flag=True)
    joined = next_df[keys].merge(flagged, on=keys, how="left")
    return joined["_flag"].eq(True).to_numpy()


def ranking_curve(pop: pd.DataFrame, max_share: float = 1.0) -> pd.DataFrame:
    """Cumulative t+1 coverage and capture of the frozen rate ranking."""
    cur = pop.sort_values("rank_position")
    s1, c1 = cur["scheduled_t1"].sum(), cur["cancelled_t1"].sum()
    out = pd.DataFrame({"coverage_t1_pct": cur["scheduled_t1"].cumsum().to_numpy() / s1 * 100,
                        "capture_t1_pct": cur["cancelled_t1"].cumsum().to_numpy() / c1 * 100})
    return out[out["coverage_t1_pct"] <= max_share * 100 + 1e-9]


def airport_bootstrap(pop: pd.DataFrame, n_boot: int = 2000, seed: int = RANDOM_STATE) -> dict:
    """Airport-cluster bootstrap of the frozen lists' statistics on the evaluation population.

    Origin airports are resampled with replacement; each drawn airport contributes
    all of its evaluated cells with their period-t assignments, so the statistics are
    conditional on the fitted lists and on the evaluation population.
    """
    cur = pop.sort_values("rank_position").reset_index(drop=True)
    airports, idx = np.unique(cur["origin"].to_numpy(), return_inverse=True)
    rng = np.random.default_rng(seed)
    counts = np.vstack([np.bincount(rng.integers(0, len(airports), len(airports)), minlength=len(airports)) for _ in range(n_boot)])
    w = counts[:, idx].astype(float)
    s1, c1 = cur["scheduled_t1"].to_numpy(float), cur["cancelled_t1"].to_numpy(float)
    ws, wc = w * s1, w * c1
    tot_s, tot_c = ws.sum(axis=1), wc.sum(axis=1)
    out = {"airports": len(airports)}
    with np.errstate(divide="ignore", invalid="ignore"):
        system = tot_c / tot_s
        for flag in ("watch", "top_decile", "threshold"):
            m = cur[flag].to_numpy(bool)
            lift = (wc[:, m].sum(axis=1) / ws[:, m].sum(axis=1)) / system
            lift = lift[np.isfinite(lift)]
            out[f"{flag}_lift_ci_low"], out[f"{flag}_lift_ci_high"] = np.quantile(lift, [0.025, 0.975])
            out[f"{flag}_share_boot_lift_le_1"] = float((lift <= 1).mean())
        m = cur["watch"].to_numpy(bool)
        diff = (wc[:, m].sum(axis=1) / ws[:, m].sum(axis=1) - wc[:, ~m].sum(axis=1) / ws[:, ~m].sum(axis=1)) * 100
        diff = diff[np.isfinite(diff)]
        out["watch_rate_diff_pp_ci_low"], out["watch_rate_diff_pp_ci_high"] = np.quantile(diff, [0.025, 0.975])
        gaps = []
        cs, cc = np.cumsum(ws, axis=1), np.cumsum(wc, axis=1)
        for b in range(n_boot):
            if tot_s[b] == 0 or tot_c[b] == 0:
                continue
            cov = ws[b, m].sum() / tot_s[b]
            x = np.concatenate([[0.0], cs[b] / tot_s[b]])
            y = np.concatenate([[0.0], cc[b] / tot_c[b]])
            gaps.append((wc[b, m].sum() / tot_c[b] - np.interp(cov, x, y)) * 100)
        out["capture_gap_pp_ci_low"], out["capture_gap_pp_ci_high"] = np.quantile(gaps, [0.025, 0.975])
    return out
