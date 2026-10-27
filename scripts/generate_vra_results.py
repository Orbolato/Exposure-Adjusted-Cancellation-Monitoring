"""Descriptive results, canceled-record benchmark and exposure-based segments (Brazil).

Reads the harmonized SIROS-archive VRA (``build_vra_siros.py``) and writes:

* the canceled-record benchmark: K-Means profiles of canceled records at the main
  origin aerodromes, with its encoding and variable-set comparisons and diagnostics
  (Tables B.12 and B.13);
* the exposure-based segmentation of operational cells by phase (Table 8);
* Fig. 2 (highest origin-airport cancellation rates), Fig. A.2 (construction of the
  analysis population) and Figs. A.4-A.7 (aerodrome selection and count-based
  distributions).

It also draws diagnostic figures of the canceled-record benchmark that the article
summarizes in tables instead of showing (figure04 to figure06, figure10 to figure21,
figure25, figure27 and figure28, with their panels). The module provides
``load_vra`` and other helpers used by the later scripts.
"""

from __future__ import annotations

import re
import unicodedata
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


from paths import ANALYSIS_DIR, FIGURE_DIR, INTERIM_DIR, LEGACY_DIR, REFERENCE_DIR, ROOT

VRA_DIR = LEGACY_DIR                                    # open-data portal files (reconciliation only)
CACHE_PATH = INTERIM_DIR / "vra_harmonized.parquet"     # cache of the portal files
FIG_DIRS = [FIGURE_DIR]
OUT_DIR = ANALYSIS_DIR / "vra"
for path in FIG_DIRS + [OUT_DIR, INTERIM_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Figures are published without embedded plot titles (the caption carries the
# description) and with enlarged axis and tick labels for print readability.
plt.rcParams.update(
    {
        "font.size": 15,
        "axes.labelsize": 16,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 14,
        "axes.titlesize": 14,
        "figure.titlesize": 15,
        "axes.labelpad": 6,
        "savefig.bbox": "tight",
    }
)

RANDOM_STATE = 42
MAIN_AIRPORT_SHARE = 0.80
# Brazilian aerodromes span seven ICAO location-indicator blocks, so the study
# population is defined from the ANAC aerodrome registry together with those
# prefixes rather than from the "SB" block alone.
BR_PREFIXES = ("SB", "SD", "SI", "SJ", "SN", "SS", "SW")
AERODROME_REFERENCE = REFERENCE_DIR / "brazilian_aerodromes.csv"
PERIOD_ORDER = ["Overall", "Pre-pandemic", "Pandemic", "Post-pandemic"]
PERIOD_LABELS = {
    "Overall": "Overall phase",
    "Pre-pandemic": "Pre-pandemic",
    "Pandemic": "Pandemic",
    "Post-pandemic": "Post-pandemic",
}
NUMERIC_FEATURES = {"departure_hour", "departure_sin", "departure_cos"}
FINAL_FEATURES = ["day_of_week", "departure_sin", "departure_cos", "airline_grouped"]

def slug(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^A-Za-z0-9]+", "_", text.upper()).strip("_")
    return re.sub(r"_+", "_", text)


COL_MAP = {
    "ICAO_EMPRESA_AEREA": "airline",
    "SIGLA_ICAO_EMPRESA_AEREA": "airline",
    "NUMERO_VOO": "flight_no",
    "CODIGO_AUTORIZACAO_DI": "di",
    "CODIGO_DI": "di",
    "CODIGO_TIPO_LINHA": "line_type",
    "ICAO_AERODROMO_ORIGEM": "origin",
    "SIGLA_ICAO_AEROPORTO_ORIGEM": "origin",
    "ICAO_AERODROMO_DESTINO": "dest",
    "SIGLA_ICAO_AEROPORTO_DESTINO": "dest",
    "PARTIDA_PREVISTA": "sched_dep",
    "PARTIDA_REAL": "real_dep",
    "CHEGADA_PREVISTA": "sched_arr",
    "CHEGADA_REAL": "real_arr",
    "SITUACAO_VOO": "status",
    "CODIGO_JUSTIFICATIVA": "justification",
    "MODELO_EQUIPAMENTO": "aircraft_model",
    "NUMERO_DE_ASSENTOS": "seats",
}


def brazilian_aerodromes() -> set[str]:
    """ICAO codes of Brazilian aerodromes from the ANAC registry."""
    if not AERODROME_REFERENCE.exists():
        raise RuntimeError(
            f"Missing {AERODROME_REFERENCE}. Run scripts/build_brazilian_aerodrome_reference.py first."
        )
    return set(pd.read_csv(AERODROME_REFERENCE)["icao"].astype(str).str.upper())


def mark_brazilian_origin(df: pd.DataFrame) -> pd.DataFrame:
    """Flag records departing from a Brazilian aerodrome."""
    origin = df["origin"].astype("string").fillna("")
    df["is_br_origin"] = origin.isin(brazilian_aerodromes()) | origin.str.startswith(BR_PREFIXES)
    return df


def first_data_row_to_skip(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        first = handle.readline()
    return 1 if first.startswith("Atualizado em") else 0


def standardize_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for col in chunk.columns:
        key = slug(col)
        if key in COL_MAP:
            renamed[col] = COL_MAP[key]
    chunk = chunk.rename(columns=renamed)
    needed = ["airline", "flight_no", "di", "line_type", "origin", "dest", "sched_dep", "real_dep", "sched_arr", "real_arr", "status", "justification", "aircraft_model", "seats"]
    for col in needed:
        if col not in chunk.columns:
            chunk[col] = np.nan
    return chunk[needed].copy()


def read_vra() -> pd.DataFrame:
    frames = []
    for path in sorted(VRA_DIR.rglob("*.csv")):
        skiprows = first_data_row_to_skip(path)
        for chunk in pd.read_csv(
            path,
            sep=";",
            encoding="utf-8-sig",
            skiprows=skiprows,
            chunksize=150_000,
            low_memory=False,
        ):
            chunk = standardize_chunk(chunk)
            chunk["source_file"] = path.name
            chunk["sched_dep"] = pd.to_datetime(chunk["sched_dep"], errors="coerce", format="mixed", dayfirst=True)
            if path.name == "VRA_20260527133655.csv":
                chunk = chunk[chunk["sched_dep"] < pd.Timestamp("2024-04-01")]
            chunk = chunk[(chunk["sched_dep"] >= "2015-01-01") & (chunk["sched_dep"] < "2025-01-01")]
            if not chunk.empty:
                frames.append(chunk)
    if not frames:
        raise RuntimeError(f"No VRA rows found in {VRA_DIR}")

    df = pd.concat(frames, ignore_index=True)
    for col in ["airline", "line_type", "origin", "dest", "status", "justification", "aircraft_model"]:
        df[col] = df[col].astype("string").str.upper().str.strip()
    df["flight_no"] = df["flight_no"].astype("string").str.zfill(4)
    df["year"] = df["sched_dep"].dt.year.astype("int16")
    df["month"] = df["sched_dep"].dt.month.astype("int8")
    df["date"] = df["sched_dep"].dt.date
    df["day_of_week"] = df["sched_dep"].dt.day_name()
    df["departure_hour"] = df["sched_dep"].dt.hour + df["sched_dep"].dt.minute / 60
    angle = 2 * np.pi * df["departure_hour"] / 24
    df["departure_sin"] = np.sin(angle)
    df["departure_cos"] = np.cos(angle)
    df["departure_period"] = pd.cut(
        df["departure_hour"],
        bins=[0, 6, 12, 18, 24],
        labels=["Night", "Morning", "Afternoon", "Evening"],
        include_lowest=True,
        right=False,
    ).astype("string")
    df["is_cancelled"] = df["status"].eq("CANCELADO")
    df = mark_brazilian_origin(df)
    df["airline_grouped"] = df["airline"].map({"AZU": "Azul", "GLO": "Gol", "TAM": "TAM"}).fillna("Others")
    df["day_type"] = np.where(df["day_of_week"].isin(["Saturday", "Sunday"]), "Weekend", "Weekday")
    df["period"] = np.select(
        [
            df["sched_dep"] < pd.Timestamp("2020-03-01"),
            df["sched_dep"] <= pd.Timestamp("2023-04-30 23:59:59"),
        ],
        ["Pre-pandemic", "Pandemic"],
        default="Post-pandemic",
    )
    return df


SIROS_CACHE = INTERIM_DIR / "vra_siros_harmonized.parquet"


def load_vra(scope: str = "analysis") -> pd.DataFrame:
    """Harmonized VRA legs from the SIROS archive (built by ``build_vra_siros.py``).

    ``scope="analysis"`` returns regular legs (DI 0) with a scheduled departure and a
    final status of operated or canceled; ``scope="all"`` returns every planned leg.
    """
    if not SIROS_CACHE.exists():
        raise RuntimeError(f"Missing {SIROS_CACHE}. Run scripts/build_vra_siros.py first.")
    df = pd.read_parquet(SIROS_CACHE)
    if scope == "analysis":
        df = df[df["is_regular"] & df["known_status"]]
    return df.reset_index(drop=True)


def load_legacy_vra(use_cache: bool = True) -> pd.DataFrame:
    """Legacy open-data VRA files, retained only for reconciliation with the SIROS archive."""
    if use_cache and CACHE_PATH.exists():
        return mark_brazilian_origin(pd.read_parquet(CACHE_PATH))
    df = read_vra()
    cached = df.copy()
    cached["date"] = pd.to_datetime(cached["date"])
    for col in cached.columns:
        if cached[col].dtype == object:
            cached[col] = cached[col].astype("string")
    cached.to_parquet(CACHE_PATH, index=False)
    return df


def filtered_cancellations(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["is_cancelled"] & df["is_br_origin"]].copy()


def savefig(name: str) -> None:
    plt.tight_layout()
    for directory in FIG_DIRS:
        plt.savefig(directory / name, bbox_inches="tight")
    plt.close()


def main_airports(canc: pd.DataFrame) -> list[str]:
    counts = canc["origin"].value_counts()
    cum = counts.cumsum() / counts.sum()
    selected = counts.loc[cum <= MAIN_AIRPORT_SHARE].index.tolist()
    if not selected or (cum.loc[selected[-1]] < MAIN_AIRPORT_SHARE and len(selected) < len(counts)):
        selected = counts.index[: len(selected) + 1].tolist()
    return selected


def period_frame(df: pd.DataFrame, period: str) -> pd.DataFrame:
    if period == "Overall":
        return df.copy()
    return df[df["period"] == period].copy()


def sample_for_metric(x: np.ndarray, max_rows: int = 5_000) -> np.ndarray:
    if x.shape[0] <= max_rows:
        return x
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(x.shape[0], size=max_rows, replace=False)
    return x[idx]


def feature_tables(canc_main: pd.DataFrame) -> dict[str, dict[str, pd.DataFrame]]:
    specs = {
        "onehot": ["airline", "line_type", "origin", "day_of_week", "departure_sin", "departure_cos"],
        "freq": ["airline", "line_type", "origin", "day_of_week", "departure_sin", "departure_cos"],
        "manip1": ["airline", "origin", "day_of_week", "departure_sin", "departure_cos"],
        "manip2": ["origin", "day_of_week", "departure_sin", "departure_cos", "airline_grouped"],
        "manip3": FINAL_FEATURES,
    }
    return {name: {p: period_frame(canc_main, p)[cols].dropna() for p in PERIOD_ORDER} for name, cols in specs.items()}


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in NUMERIC_FEATURES]


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col in NUMERIC_FEATURES]


def make_frequency_maps(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {col: df[col].value_counts(normalize=True) for col in categorical_columns(df)}


def frequency_encoded_frame(df: pd.DataFrame, maps: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    encoded = df.copy()
    maps = maps or make_frequency_maps(encoded)
    for col in categorical_columns(encoded):
        encoded[col] = encoded[col].map(maps.get(col, pd.Series(dtype=float))).fillna(0.0)
    return encoded.astype(float)


def encoded_matrix(df: pd.DataFrame, spec: str, maps: dict[str, pd.Series] | None = None) -> np.ndarray:
    if spec == "onehot":
        categorical = categorical_columns(df)
        numeric = numeric_columns(df)
        transformers = []
        if categorical:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical))
        if numeric:
            transformers.append(("num", StandardScaler(), numeric))
        transformer = ColumnTransformer(transformers)
        return transformer.fit_transform(df)

    return StandardScaler().fit_transform(frequency_encoded_frame(df, maps))


def silhouette_table(tables: dict[str, dict[str, pd.DataFrame]], spec: str, k_values=range(2, 9)) -> pd.DataFrame:
    rows = []
    maps = None if spec == "onehot" else make_frequency_maps(tables[spec]["Overall"])
    for period in PERIOD_ORDER:
        df = tables[spec][period]
        if len(df) < 20:
            continue
        x = encoded_matrix(df, spec, maps)
        xs = sample_for_metric(x)
        for k in k_values:
            model = KMeans(n_clusters=k, n_init=5, random_state=RANDOM_STATE)
            labels = model.fit_predict(xs)
            rows.append({"spec": spec, "period": period, "k": k, "silhouette": silhouette_score(xs, labels)})
    return pd.DataFrame(rows)


def plot_silhouettes(table: pd.DataFrame, figure_name: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, period in zip(axes.ravel(), PERIOD_ORDER):
        sub = table[table["period"] == period]
        ax.plot(sub["k"], sub["silhouette"], marker="o", color="#1f5a8a")
        ax.grid(axis="y", alpha=0.25)
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel("Silhouette coefficient")
    savefig(figure_name)


def plot_silhouette_single(table: pd.DataFrame, period: str, figure_name: str) -> None:
    sub = table[table["period"] == period]
    plt.figure(figsize=(4.8, 3.6))
    plt.plot(sub["k"], sub["silhouette"], marker="o", color="#1f5a8a")
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette coefficient")
    plt.grid(axis="y", alpha=0.25)
    savefig(figure_name)


def cluster_profiles(canc_main: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    rows = []
    maps = make_frequency_maps(canc_main[FINAL_FEATURES].dropna())
    for period in PERIOD_ORDER:
        source = period_frame(canc_main, period)
        data = source[FINAL_FEATURES].dropna()
        if len(data) < k:
            continue
        x = encoded_matrix(data, "freq", maps)
        labels = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit_predict(x)
        assigned = source.loc[data.index].copy()
        assigned["cluster"] = labels + 1
        assigned["period"] = period
        rows.append(assigned)
    return pd.concat(rows, ignore_index=True)


def figure02_flow(df: pd.DataFrame, canc: pd.DataFrame, canc_main: pd.DataFrame, selected_airports: list[str], min_sched: int = 500) -> None:
    """Construction of the analysis population and the analysis paths built from it.

    The exposure-based path ranks operational cells by cancellation rate (primary
    screening), with K-Means segmentation as an optional reporting layer, and both are
    validated prospectively. The canceled-record benchmark conditions on cancellation
    and therefore loses the exposure denominator.
    """
    flow = pd.read_csv(OUT_DIR / "inclusion_flow.csv").set_index("step")["records"]
    planned = int(flow.filter(like="Planned legs scheduled").iloc[0])
    df_br = df[df["is_br_origin"]]
    cells = (
        df_br.groupby(["airline_grouped", "origin", "day_type", "departure_period"], observed=True)
        .agg(scheduled=("is_cancelled", "size"))
        .reset_index()
    )
    n_cells = int((cells["scheduled"] >= min_sched).sum())

    fig, ax = plt.subplots(figsize=(11.5, 7.6))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def box(cx, cy, label, count, width=0.30, height=0.13, dashed=False, note=None):
        ax.add_patch(plt.Rectangle((cx - width / 2, cy - height / 2), width, height, facecolor="white",
                                   edgecolor="#333333", linewidth=1.5, linestyle="--" if dashed else "-"))
        ax.text(cx, cy + (0.028 if count else 0.0), label, ha="center", va="center", fontsize=11)
        if count:
            ax.text(cx, cy - 0.036, count, ha="center", va="center", fontsize=11, weight="bold")
        if note:
            ax.text(cx, cy - height / 2 - 0.025, note, ha="center", va="center", fontsize=9.5, style="italic", color="#444444")

    def arrow(x1, y1, x2, y2, dashed=False):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.4, color="#333333", linestyle="--" if dashed else "-"))

    box(0.50, 0.935, "Harmonized planned legs, 2015-2024", f"n = {planned:,}", width=0.46)
    box(0.50, 0.755, "Regular legs with a final status, Brazilian origin", f"n = {len(df_br):,}", width=0.46,
        note="excludes foreign origins, non-regular designations and legs without a final status")
    arrow(0.50, 0.87, 0.50, 0.82)

    # exposure-based path
    arrow(0.42, 0.645, 0.335, 0.62)
    box(0.335, 0.555, f"Operational cells with $\\geq${min_sched} scheduled flights", f"n = {n_cells:,} cells (full period)", width=0.46)
    arrow(0.26, 0.49, 0.17, 0.425)
    arrow(0.41, 0.49, 0.50, 0.425, dashed=True)
    box(0.17, 0.36, "Rate ranking of cells", "primary screening", width=0.29)
    box(0.50, 0.36, "K-Means segmentation", "optional reporting layer", width=0.29, dashed=True)
    arrow(0.17, 0.295, 0.28, 0.215)
    arrow(0.50, 0.295, 0.39, 0.215, dashed=True)
    box(0.335, 0.15, "Prospective validation:\nlists fixed in year t, scored in year t+1", "", width=0.46)

    # canceled-record benchmark
    arrow(0.62, 0.645, 0.83, 0.62)
    box(0.83, 0.555, "Canceled records", f"n = {len(canc):,}", width=0.29)
    box(0.83, 0.36, f"Main origin aerodromes ({len(selected_airports)})", f"n = {len(canc_main):,}", width=0.29)
    box(0.83, 0.15, "Canceled-record\nbenchmark model", "comparison", width=0.29)
    arrow(0.83, 0.49, 0.83, 0.425)
    arrow(0.83, 0.295, 0.83, 0.215)

    savefig("figure02.pdf")


def figure03_airport_elbow(canc: pd.DataFrame) -> None:
    counts = canc["origin"].value_counts()
    cumulative = counts.cumsum() / counts.sum() * 100
    x = np.arange(1, len(cumulative) + 1)
    plt.figure(figsize=(9, 5.4))
    plt.plot(x, cumulative.values, color="#1f5a8a", linewidth=2)
    plt.axhline(80, color="#b4553b", linestyle="--", linewidth=1.5)
    plt.xlabel("Number of origin airports, ordered by cancellations")
    plt.ylabel("Cumulative cancellations (%)")
    plt.grid(axis="y", alpha=0.25)
    savefig("figure03.pdf")


def figure06_year_weekday(canc_main: pd.DataFrame) -> None:
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = canc_main.pivot_table(index="year", columns="day_of_week", values="status", aggfunc="count", fill_value=0)
    pivot = pivot.reindex(columns=order, fill_value=0)
    ax = pivot.plot(kind="bar", stacked=True, figsize=(11, 5.8), colormap="viridis")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of cancellations")
    ax.legend(title="Day of week", ncol=2, fontsize=11, title_fontsize=12)
    savefig("figure06.pdf")


def figure07_airlines(canc_main: pd.DataFrame) -> None:
    counts = canc_main["airline"].value_counts().head(10).sort_values()
    plt.figure(figsize=(9, 5.4))
    plt.barh(counts.index, counts.values, color="#1f5a8a")
    plt.xlabel("Number of cancellations")
    plt.ylabel("Airline")
    plt.grid(axis="x", alpha=0.25)
    savefig("figure07.pdf")


def figure08_airports_2024(canc_main: pd.DataFrame) -> None:
    counts = canc_main[canc_main["year"] == 2024]["origin"].value_counts().head(15).sort_values()
    plt.figure(figsize=(9, 5.8))
    plt.barh(counts.index, counts.values, color="#6f7d3c")
    plt.xlabel("Number of cancellations")
    plt.grid(axis="x", alpha=0.25)
    savefig("figure08.pdf")


def figure09_departure_time(canc_main: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5.4))
    plt.hist(canc_main["departure_hour"].dropna(), bins=np.arange(0, 25, 1), color="#1f5a8a", edgecolor="white")
    plt.xlabel("Scheduled departure hour")
    plt.ylabel("Number of cancellations")
    plt.grid(axis="y", alpha=0.25)
    savefig("figure09.pdf")


def plot_cluster_bars(clustered: pd.DataFrame, column: str, period: str, figure_name: str, title: str) -> None:
    sub = clustered[clustered["period"].eq(period) if period != "Overall" else clustered["period"].eq("Overall")]
    pivot = sub.pivot_table(index="cluster", columns=column, values="status", aggfunc="count", fill_value=0)
    if column == "day_of_week":
        cols = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        pivot = pivot.reindex(columns=[c for c in cols if c in pivot.columns], fill_value=0)
    else:
        pivot = pivot.reindex(columns=[c for c in ["Azul", "Gol", "TAM", "Others"] if c in pivot.columns], fill_value=0)
    ax = pivot.plot(kind="bar", stacked=True, figsize=(9, 5.4), colormap="viridis")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of cancellations")
    ax.legend(title=column.replace("_", " ").title(), fontsize=11, title_fontsize=12)
    savefig(figure_name)


def figure21_departure_box(clustered: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    for ax, period in zip(axes.ravel(), PERIOD_ORDER):
        sub = clustered[clustered["period"] == period]
        data = [sub[sub["cluster"] == c]["departure_hour"].dropna() for c in sorted(sub["cluster"].unique())]
        ax.boxplot(data, tick_labels=sorted(sub["cluster"].unique()), showfliers=False)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Scheduled departure hour")
        ax.grid(axis="y", alpha=0.25)
    savefig("figure21.pdf")
    for period, suffix in zip(PERIOD_ORDER, ["a", "b", "c", "d"]):
        sub = clustered[clustered["period"] == period]
        data = [sub[sub["cluster"] == c]["departure_hour"].dropna() for c in sorted(sub["cluster"].unique())]
        plt.figure(figsize=(4.8, 3.6))
        plt.boxplot(data, tick_labels=sorted(sub["cluster"].unique()), showfliers=False)
        plt.xlabel("Cluster")
        plt.ylabel("Scheduled departure hour")
        plt.grid(axis="y", alpha=0.25)
        savefig(f"figure21{suffix}.pdf")


def enhanced_figures(df: pd.DataFrame, canc_main: pd.DataFrame) -> None:
    by_year = df[df["is_br_origin"]].groupby("year").agg(total=("status", "count"), cancelled=("is_cancelled", "sum"))
    by_year["rate"] = by_year["cancelled"] / by_year["total"] * 100
    fig, ax1 = plt.subplots(figsize=(10, 5.4))
    ax1.bar(by_year.index, by_year["cancelled"], color="#9d4d3f", alpha=0.75, label="Cancellations")
    ax1.set_ylabel("Number of cancellations")
    ax1.set_xlabel("Year")
    ax2 = ax1.twinx()
    ax2.plot(by_year.index, by_year["rate"], color="#1f5a8a", marker="o", linewidth=2, label="Cancellation rate")
    ax2.set_ylabel("Cancellation rate (%)")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, frameon=False, loc="upper right")
    savefig("figure25.pdf")

    airport_rate = (
        df[df["is_br_origin"]]
        .groupby("origin")
        .agg(total=("status", "count"), cancelled=("is_cancelled", "sum"))
        .assign(rate=lambda x: x["cancelled"] / x["total"] * 100)
    )
    airport_rate = airport_rate[airport_rate["total"] >= 5_000].sort_values("rate", ascending=False).head(15).sort_values("rate")
    plt.figure(figsize=(9, 5.8))
    plt.barh(airport_rate.index.astype(str), airport_rate["rate"], color="#5d4b8a")
    plt.xlabel("Cancellation rate (%)")
    plt.grid(axis="x", alpha=0.25)
    savefig("figure26.pdf")

    airline_year = canc_main.groupby(["year", "airline_grouped"]).size().reset_index(name="cancellations")
    pivot = airline_year.pivot(index="year", columns="airline_grouped", values="cancellations").fillna(0)
    pivot = pivot.reindex(columns=[c for c in ["Azul", "Gol", "TAM", "Others"] if c in pivot.columns])
    ax = pivot.plot(kind="area", stacked=True, figsize=(10, 5.4), colormap="viridis")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of cancellations")
    ax.legend(title="Airline group")
    savefig("figure27.pdf")


def silhouette_for_frame(data: pd.DataFrame, maps: dict[str, pd.Series] | None = None, k: int = 3) -> float:
    if len(data) < max(20, k + 1):
        return float("nan")
    x = encoded_matrix(data, "freq", maps)
    xs = sample_for_metric(x)
    labels = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(xs)
    return float(silhouette_score(xs, labels))


def cluster_size_summary(clustered: pd.DataFrame) -> None:
    sizes = clustered.groupby(["period", "cluster"]).size().reset_index(name="cancellations")
    totals = sizes.groupby("period")["cancellations"].transform("sum")
    sizes["share"] = sizes["cancellations"] / totals
    sizes.to_csv(OUT_DIR / "cluster_size_summary.csv", index=False)

    rows = []
    for (period, cluster), group in clustered.groupby(["period", "cluster"]):
        rows.append(
            {
                "period": period,
                "cluster": cluster,
                "cancellations": len(group),
                "share": len(group) / len(clustered[clustered["period"] == period]),
                "dominant_airline_group": group["airline_grouped"].mode(dropna=True).iat[0],
                "dominant_day_type": group["day_type"].mode(dropna=True).iat[0],
                "dominant_day_of_week": group["day_of_week"].mode(dropna=True).iat[0],
                "mean_departure_hour": group["departure_hour"].mean(),
            }
        )
    pd.DataFrame(rows).to_csv(OUT_DIR / "cluster_dominant_characteristics.csv", index=False)


def kmeans_stability(canc_main: pd.DataFrame, k: int = 3, sample_rows: int = 10_000) -> None:
    rows = []
    maps = make_frequency_maps(canc_main[FINAL_FEATURES].dropna())
    rng = np.random.default_rng(RANDOM_STATE)
    for period in PERIOD_ORDER:
        source = period_frame(canc_main, period)
        data = source[FINAL_FEATURES].dropna()
        if len(data) < sample_rows:
            sampled = data
        else:
            sampled = data.iloc[rng.choice(len(data), size=sample_rows, replace=False)]
        x = encoded_matrix(sampled, "freq", maps)
        labels_by_seed = []
        for seed in range(10):
            labels_by_seed.append(KMeans(n_clusters=k, n_init=1, random_state=seed).fit_predict(x))
        aris = [adjusted_rand_score(a, b) for a, b in combinations(labels_by_seed, 2)]
        rows.append(
            {
                "period": period,
                "k": k,
                "sample_rows": len(sampled),
                "mean_pairwise_ari": float(np.mean(aris)),
                "min_pairwise_ari": float(np.min(aris)),
                "max_pairwise_ari": float(np.max(aris)),
            }
        )
    pd.DataFrame(rows).to_csv(OUT_DIR / "kmeans_stability_k3.csv", index=False)


def k_sensitivity_profiles(canc_main: pd.DataFrame, k_values=(2, 3, 4)) -> None:
    rows = []
    maps = make_frequency_maps(canc_main[FINAL_FEATURES].dropna())
    for period in PERIOD_ORDER:
        source = period_frame(canc_main, period)
        data = source[FINAL_FEATURES].dropna()
        if len(data) < max(k_values):
            continue
        x = encoded_matrix(data, "freq", maps)
        for k in k_values:
            labels = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit_predict(x)
            assigned = source.loc[data.index].copy()
            assigned["cluster"] = labels + 1
            for cluster, group in assigned.groupby("cluster"):
                rows.append(
                    {
                        "period": period,
                        "k": k,
                        "cluster": int(cluster),
                        "cancellations": len(group),
                        "share": len(group) / len(assigned),
                        "dominant_airline_group": group["airline_grouped"].mode(dropna=True).iat[0],
                        "dominant_day_type": group["day_type"].mode(dropna=True).iat[0],
                        "dominant_day_of_week": group["day_of_week"].mode(dropna=True).iat[0],
                    }
                )
    pd.DataFrame(rows).to_csv(OUT_DIR / "k_sensitivity_profiles.csv", index=False)


def select_by_share(counts: pd.Series, share: float) -> list[str]:
    ordered = counts.sort_values(ascending=False)
    cumulative = ordered.cumsum() / ordered.sum()
    selected = ordered.loc[cumulative <= share].index.tolist()
    if not selected or (cumulative.loc[selected[-1]] < share and len(selected) < len(ordered)):
        selected = ordered.index[: len(selected) + 1].tolist()
    return selected


def airport_selection_sensitivity(df: pd.DataFrame, canc: pd.DataFrame) -> None:
    rows = []
    df_sb = df[df["is_br_origin"]].copy()
    selections = {
        "all_airports": sorted(canc["origin"].dropna().unique()),
        "cancellation_share_80": select_by_share(canc["origin"].value_counts(), 0.80),
        "cancellation_share_90": select_by_share(canc["origin"].value_counts(), 0.90),
        "scheduled_exposure_share_80": select_by_share(df_sb["origin"].value_counts(), 0.80),
    }
    for name, airports in selections.items():
        sub = canc[canc["origin"].isin(airports)].copy()
        data = sub[FINAL_FEATURES].dropna()
        maps = make_frequency_maps(data) if len(data) else None
        rows.append(
            {
                "selection": name,
                "airports": len(airports),
                "cancelled_records": len(sub),
                "share_of_cancelled_sb_origin": len(sub) / len(canc),
                "silhouette_k3": silhouette_for_frame(data, maps, k=3) if len(data) else float("nan"),
            }
        )
    pd.DataFrame(rows).to_csv(OUT_DIR / "airport_selection_sensitivity.csv", index=False)


def time_representation_sensitivity(canc_main: pd.DataFrame) -> None:
    rows = []
    linear_cols = ["day_of_week", "departure_hour", "airline_grouped"]
    circular_cols = FINAL_FEATURES
    for label, cols in [("linear_hour", linear_cols), ("circular_sine_cosine", circular_cols)]:
        maps = make_frequency_maps(canc_main[cols].dropna())
        for period in PERIOD_ORDER:
            data = period_frame(canc_main, period)[cols].dropna()
            rows.append(
                {
                    "representation": label,
                    "period": period,
                    "records": len(data),
                    "silhouette_k3": silhouette_for_frame(data, maps, k=3),
                }
            )
    pd.DataFrame(rows).to_csv(OUT_DIR / "time_representation_sensitivity.csv", index=False)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    mask_p = p > 0
    mask_q = q > 0
    return float(0.5 * np.sum(p[mask_p] * np.log2(p[mask_p] / m[mask_p])) + 0.5 * np.sum(q[mask_q] * np.log2(q[mask_q] / m[mask_q])))


def baseline_cluster_assignment(canc_main: pd.DataFrame, k: int = 3) -> None:
    maps = make_frequency_maps(canc_main[FINAL_FEATURES].dropna())
    pre_source = period_frame(canc_main, "Pre-pandemic")
    pre_data = pre_source[FINAL_FEATURES].dropna()
    pre_encoded = frequency_encoded_frame(pre_data, maps)
    scaler = StandardScaler().fit(pre_encoded)
    model = KMeans(n_clusters=k, n_init=30, random_state=RANDOM_STATE).fit(scaler.transform(pre_encoded))

    share_rows = []
    divergence_rows = []
    base_counts = np.bincount(model.labels_, minlength=k).astype(float)
    base_share = base_counts / base_counts.sum()
    for period in PERIOD_ORDER:
        source = period_frame(canc_main, period)
        data = source[FINAL_FEATURES].dropna()
        labels = model.predict(scaler.transform(frequency_encoded_frame(data, maps)))
        counts = np.bincount(labels, minlength=k).astype(float)
        shares = counts / counts.sum()
        for idx, (count, share) in enumerate(zip(counts, shares), start=1):
            share_rows.append({"period": period, "baseline_cluster": idx, "cancellations": int(count), "share": share})
        divergence_rows.append({"period": period, "js_divergence_from_pre_pandemic": js_divergence(base_share, shares)})
    pd.DataFrame(share_rows).to_csv(OUT_DIR / "baseline_cluster_assignment_shares.csv", index=False)
    pd.DataFrame(divergence_rows).to_csv(OUT_DIR / "baseline_cluster_assignment_divergence.csv", index=False)


def post_cutoff_sensitivity(canc_main: pd.DataFrame) -> None:
    rows = []
    maps = make_frequency_maps(canc_main[FINAL_FEATURES].dropna())
    cutoffs = ["2023-01-01", "2023-03-01", "2023-05-01", "2023-07-01", "2023-10-01"]
    for cutoff in cutoffs:
        cutoff_ts = pd.Timestamp(cutoff)
        temp = canc_main.copy()
        temp["period_alt"] = np.select(
            [temp["sched_dep"] < pd.Timestamp("2020-03-01"), temp["sched_dep"] < cutoff_ts],
            ["Pre-pandemic", "Pandemic"],
            default="Post-pandemic",
        )
        for period in ["Pandemic", "Post-pandemic"]:
            sub = temp[temp["period_alt"] == period]
            data = sub[FINAL_FEATURES].dropna()
            rows.append(
                {
                    "post_period_start": cutoff,
                    "period": period,
                    "cancelled_records": len(sub),
                    "silhouette_k3": silhouette_for_frame(data, maps, k=3),
                }
            )
    pd.DataFrame(rows).to_csv(OUT_DIR / "post_cutoff_sensitivity.csv", index=False)


def kmodes_fit(df: pd.DataFrame, columns: list[str], k: int = 3, max_iter: int = 30, seed: int = RANDOM_STATE) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], float]:
    rng = np.random.default_rng(seed)
    categories = [pd.Categorical(df[col]) for col in columns]
    levels = [np.asarray(cat.categories.astype(str)) for cat in categories]
    codes = np.vstack([cat.codes for cat in categories]).T.astype(np.int16)
    modes = codes[rng.choice(len(codes), size=k, replace=False)].copy()
    labels = np.full(len(codes), -1, dtype=np.int16)
    for _ in range(max_iter):
        distances = (codes[:, None, :] != modes[None, :, :]).sum(axis=2)
        new_labels = distances.argmin(axis=1).astype(np.int16)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for cluster in range(k):
            members = codes[labels == cluster]
            if len(members) == 0:
                modes[cluster] = codes[rng.integers(0, len(codes))]
                continue
            for col_idx in range(codes.shape[1]):
                values, counts = np.unique(members[:, col_idx], return_counts=True)
                modes[cluster, col_idx] = values[counts.argmax()]
    distances = (codes[:, None, :] != modes[None, :, :]).sum(axis=2)
    cost = float(distances[np.arange(len(codes)), labels].sum())
    return labels, modes, levels, cost


def categorical_kmodes_sensitivity(canc_main: pd.DataFrame, sample_rows: int = 30_000) -> None:
    rows = []
    summary_rows = []
    columns = ["airline_grouped", "line_type", "origin", "day_type", "departure_period"]
    rng = np.random.default_rng(RANDOM_STATE)
    for period in PERIOD_ORDER:
        source = period_frame(canc_main, period)[columns].dropna()
        if len(source) > sample_rows:
            source = source.iloc[rng.choice(len(source), size=sample_rows, replace=False)].copy()
        labels, modes, levels, cost = kmodes_fit(source, columns, k=3)
        summary_rows.append({"period": period, "sample_rows": len(source), "matching_dissimilarity_cost": cost, "mean_mismatches_per_record": cost / len(source)})
        assigned = source.copy()
        assigned["cluster"] = labels + 1
        for cluster in sorted(assigned["cluster"].unique()):
            group = assigned[assigned["cluster"] == cluster]
            mode_values = {columns[idx]: levels[idx][modes[cluster - 1, idx]] for idx in range(len(columns))}
            rows.append(
                {
                    "period": period,
                    "cluster": int(cluster),
                    "sample_records": len(group),
                    "share": len(group) / len(assigned),
                    **{f"mode_{key}": value for key, value in mode_values.items()},
                }
            )
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "categorical_kmodes_summary.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT_DIR / "categorical_kmodes_profiles.csv", index=False)


def cancellation_reason_coverage(canc: pd.DataFrame) -> None:
    valid = canc["justification"].notna() & canc["justification"].astype("string").str.len().gt(0)
    pd.DataFrame(
        [
            {"metric": "cancelled_records_sb_origin", "value": len(canc)},
            {"metric": "records_with_justification_code", "value": int(valid.sum())},
            {"metric": "justification_coverage_share", "value": float(valid.mean())},
            {"metric": "unique_justification_codes", "value": int(canc.loc[valid, "justification"].nunique())},
        ]
    ).to_csv(OUT_DIR / "cancellation_justification_coverage.csv", index=False)
    canc.loc[valid, "justification"].value_counts().head(30).rename_axis("justification").reset_index(name="cancelled_records").to_csv(
        OUT_DIR / "cancellation_justification_top_codes.csv", index=False
    )


def airline_grouping_summary(df: pd.DataFrame) -> None:
    (
        df[df["is_br_origin"]]
        .groupby(["airline_grouped", "airline"])
        .agg(scheduled_records=("status", "count"), cancelled_records=("is_cancelled", "sum"))
        .assign(cancellation_rate_pct=lambda x: x["cancelled_records"] / x["scheduled_records"] * 100)
        .reset_index()
        .sort_values(["airline_grouped", "scheduled_records"], ascending=[True, False])
        .to_csv(OUT_DIR / "airline_grouping_summary.csv", index=False)
    )
def rate_based_validation(df: pd.DataFrame) -> None:
    """Cancellation rates by period and airline group for rate-based profile validation."""
    df_sb = df[df["is_br_origin"]].copy()
    period_order = ["Pre-pandemic", "Pandemic", "Post-pandemic"]

    by_period = (
        df_sb.groupby("period")
        .agg(total=("status", "count"), cancelled=("is_cancelled", "sum"))
        .assign(rate_pct=lambda x: x["cancelled"] / x["total"] * 100)
        .reindex(period_order)
    )
    by_period.to_csv(OUT_DIR / "cancellation_rate_by_period.csv")

    by_period_airline = (
        df_sb.groupby(["period", "airline_grouped"])
        .agg(total=("status", "count"), cancelled=("is_cancelled", "sum"))
        .assign(rate_pct=lambda x: x["cancelled"] / x["total"] * 100)
        .reset_index()
    )
    by_period_airline["period"] = pd.Categorical(by_period_airline["period"], categories=period_order, ordered=True)
    by_period_airline.sort_values(["period", "airline_grouped"]).to_csv(
        OUT_DIR / "cancellation_rate_by_period_airline.csv", index=False
    )

    by_period_airline_day = (
        df_sb.groupby(["period", "airline_grouped", "day_type"])
        .agg(total=("status", "count"), cancelled=("is_cancelled", "sum"))
        .assign(rate_pct=lambda x: x["cancelled"] / x["total"] * 100)
        .reset_index()
    )
    by_period_airline_day["period"] = pd.Categorical(by_period_airline_day["period"], categories=period_order, ordered=True)
    by_period_airline_day.sort_values(["period", "airline_grouped", "day_type"]).to_csv(
        OUT_DIR / "cancellation_rate_by_period_airline_daytype.csv", index=False
    )

    cell_rates = (
        df_sb.groupby(["period", "airline_grouped", "origin", "dest", "day_type", "departure_period"])
        .agg(total=("status", "count"), cancelled=("is_cancelled", "sum"))
        .assign(rate_pct=lambda x: x["cancelled"] / x["total"] * 100)
        .reset_index()
    )
    cell_rates[cell_rates["total"] >= 500].sort_values("rate_pct", ascending=False).to_csv(
        OUT_DIR / "operational_cell_cancellation_rates_min500.csv", index=False
    )


def airline_daytype_crosstab(canc_main: pd.DataFrame, df: pd.DataFrame) -> None:
    """Benchmark the compact clusters against a simple airline x day-of-week table."""
    order_air = ["Azul", "Gol", "TAM", "Others"]
    order_day = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    counts = canc_main.pivot_table(index="airline_grouped", columns="day_of_week", values="status", aggfunc="count", fill_value=0)
    counts = counts.reindex(index=[a for a in order_air if a in counts.index], columns=[d for d in order_day if d in counts.columns], fill_value=0)
    counts.to_csv(OUT_DIR / "airline_daytype_crosstab_counts.csv")

    df_sb = df[df["is_br_origin"]]
    sched = df_sb.pivot_table(index="airline_grouped", columns="day_of_week", values="status", aggfunc="count", fill_value=0)
    canc_ct = df_sb[df_sb["is_cancelled"]].pivot_table(index="airline_grouped", columns="day_of_week", values="status", aggfunc="count", fill_value=0)
    rate = (canc_ct / sched * 100).reindex(index=[a for a in order_air if a in sched.index], columns=[d for d in order_day if d in sched.columns])
    rate.round(2).to_csv(OUT_DIR / "airline_daytype_crosstab_rate_pct.csv")


def rate_cell_clustering(df: pd.DataFrame, k: int = 3, min_sched: int = 500) -> None:
    """Exposure-based clustering on operational cells using cancellation rates.

    The unit of analysis is the airline-airport-day type-departure period cell with at
    least ``min_sched`` scheduled flights. Each cell is described by its cancellation rate
    (exposure normalized) together with cyclic departure period, weekday/weekend status, and
    prevalence-encoded airline group and origin. This addresses the recommendation to
    incorporate scheduled-flight exposure into the clustering procedure rather than only into
    descriptive rates.
    """
    df_sb = df[df["is_br_origin"]].copy()
    period_hour = {"Night": 3, "Morning": 9, "Afternoon": 15, "Evening": 21}
    profile_rows = []
    sil_rows = []
    fig_data = {}
    for period in PERIOD_ORDER:
        sub = period_frame(df_sb, period)
        cells = (
            sub.groupby(["airline_grouped", "origin", "day_type", "departure_period"])
            .agg(scheduled=("status", "count"), cancelled=("is_cancelled", "sum"))
            .reset_index()
        )
        cells = cells[cells["scheduled"] >= min_sched].copy()
        if len(cells) < k + 1:
            continue
        cells["rate_pct"] = cells["cancelled"] / cells["scheduled"] * 100
        ang = 2 * np.pi * cells["departure_period"].map(period_hour) / 24
        feat = pd.DataFrame(
            {
                "rate_pct": cells["rate_pct"].to_numpy(),
                "dep_sin": np.sin(ang).to_numpy(),
                "dep_cos": np.cos(ang).to_numpy(),
                "is_weekend": (cells["day_type"] == "Weekend").astype(float).to_numpy(),
            }
        )
        for col in ["airline_grouped", "origin"]:
            feat[f"{col}_freq"] = cells[col].map(cells[col].value_counts(normalize=True)).to_numpy()
        x = StandardScaler().fit_transform(feat)
        labels = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit_predict(x)
        cells["cluster"] = labels + 1
        sil = float(silhouette_score(x, labels)) if len(np.unique(labels)) > 1 else float("nan")
        sil_rows.append({"period": period, "k": k, "n_cells": int(len(cells)), "silhouette": sil})
        weighted = cells.groupby("cluster").apply(lambda g: g["cancelled"].sum() / g["scheduled"].sum() * 100, include_groups=False)
        order = weighted.sort_values().index.tolist()
        bars = []
        for rank, cluster in enumerate(order, start=1):
            g = cells[cells["cluster"] == cluster]
            wr = g["cancelled"].sum() / g["scheduled"].sum() * 100
            profile_rows.append(
                {
                    "period": period,
                    "rate_cluster": rank,
                    "n_cells": int(len(g)),
                    "scheduled": int(g["scheduled"].sum()),
                    "cancelled": int(g["cancelled"].sum()),
                    "weighted_rate_pct": wr,
                    "mean_cell_rate_pct": float(g["rate_pct"].mean()),
                    "dominant_airline_group": g.groupby("airline_grouped")["scheduled"].sum().idxmax(),
                    "dominant_day_type": g.groupby("day_type")["scheduled"].sum().idxmax(),
                    "dominant_departure_period": g.groupby("departure_period")["scheduled"].sum().idxmax(),
                }
            )
            bars.append(wr)
        fig_data[period] = (list(range(1, len(bars) + 1)), bars)
    pd.DataFrame(sil_rows).to_csv(OUT_DIR / "rate_cell_cluster_silhouette.csv", index=False)
    pd.DataFrame(profile_rows).to_csv(OUT_DIR / "rate_cell_cluster_profiles.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    for ax, period in zip(axes.ravel(), PERIOD_ORDER):
        if period not in fig_data:
            ax.axis("off")
            continue
        ranks, bars = fig_data[period]
        ax.bar(ranks, bars, color="#5d4b8a")
        ax.set_xlabel("Rate cluster (low to high)")
        ax.set_ylabel("Weighted cancellation rate (%)")
        ax.set_xticks(ranks)
        ax.grid(axis="y", alpha=0.25)
    savefig("figure28.pdf")

    for period, suffix in zip(PERIOD_ORDER, ["a", "b", "c", "d"]):
        if period not in fig_data:
            continue
        ranks, bars = fig_data[period]
        plt.figure(figsize=(4.8, 3.6))
        plt.bar(ranks, bars, color="#5d4b8a")
        plt.xlabel("Rate cluster (low to high)")
        plt.ylabel("Weighted cancellation rate (%)")
        plt.xticks(ranks)
        plt.grid(axis="y", alpha=0.25)
        savefig(f"figure28{suffix}.pdf")


def write_outputs(df: pd.DataFrame, canc: pd.DataFrame, canc_main: pd.DataFrame, selected_airports: list[str], clustered: pd.DataFrame, sils: dict[str, pd.DataFrame]) -> None:
    summary = pd.DataFrame(
        [
            {"metric": "all_vra_records", "value": len(df)},
            {"metric": "cancelled_records_sb_origin", "value": len(canc)},
            {"metric": "selected_main_airports", "value": len(selected_airports)},
            {"metric": "cancelled_records_main_airports", "value": len(canc_main)},
            {"metric": "main_airport_share", "value": len(canc_main) / len(canc) if len(canc) else np.nan},
        ]
    )
    summary.to_csv(OUT_DIR / "vra_summary.csv", index=False)
    pd.Series(selected_airports, name="origin").to_csv(OUT_DIR / "selected_main_airports.csv", index=False)
    df.groupby(["year", "status"]).size().reset_index(name="flights").to_csv(OUT_DIR / "status_by_year.csv", index=False)
    (
        df[df["is_br_origin"]]
        .groupby("origin")
        .agg(total=("status", "count"), cancelled=("is_cancelled", "sum"))
        .assign(rate=lambda x: x["cancelled"] / x["total"] * 100)
        .sort_values("rate", ascending=False)
        .to_csv(OUT_DIR / "origin_airport_cancellation_rates.csv")
    )
    canc_main.groupby(["year", "airline_grouped"]).size().reset_index(name="cancellations").to_csv(OUT_DIR / "cancellations_by_year_airline_group.csv", index=False)
    clustered.groupby(["period", "cluster", "airline_grouped"]).size().reset_index(name="cancellations").to_csv(OUT_DIR / "cluster_airline_profiles.csv", index=False)
    clustered.groupby(["period", "cluster", "day_of_week"]).size().reset_index(name="cancellations").to_csv(OUT_DIR / "cluster_weekday_profiles.csv", index=False)
    clustered.groupby(["period", "cluster", "day_type"]).size().reset_index(name="cancellations").to_csv(OUT_DIR / "cluster_daytype_profiles.csv", index=False)
    clustered.groupby(["period", "cluster", "departure_period"]).size().reset_index(name="cancellations").to_csv(OUT_DIR / "cluster_departure_period_profiles.csv", index=False)
    for name, table in sils.items():
        table.to_csv(OUT_DIR / f"silhouette_{name}.csv", index=False)


def main() -> None:
    df = load_vra()
    canc = filtered_cancellations(df)
    selected_airports = main_airports(canc)
    canc_main = canc[canc["origin"].isin(selected_airports)].copy()

    tables = feature_tables(canc_main)
    sils = {}
    for spec, fig in [
        ("onehot", "figure04.pdf"),
        ("freq", "figure05.pdf"),
        ("manip1", "figure10.pdf"),
        ("manip2", "figure11.pdf"),
        ("manip3", "figure12.pdf"),
    ]:
        sils[spec] = silhouette_table(tables, spec)
        plot_silhouettes(sils[spec], fig)
        for period, suffix in zip(PERIOD_ORDER, ["a", "b", "c", "d"]):
            plot_silhouette_single(sils[spec], period, fig.replace(".pdf", f"{suffix}.pdf"))

    clustered = cluster_profiles(canc_main)

    cluster_size_summary(clustered)
    kmeans_stability(canc_main)
    k_sensitivity_profiles(canc_main)
    airport_selection_sensitivity(df, canc)
    time_representation_sensitivity(canc_main)
    baseline_cluster_assignment(canc_main)
    post_cutoff_sensitivity(canc_main)
    categorical_kmodes_sensitivity(canc_main)
    cancellation_reason_coverage(canc)
    airline_grouping_summary(df)

    figure02_flow(df, canc, canc_main, selected_airports)
    figure03_airport_elbow(canc)
    figure06_year_weekday(canc_main)
    figure07_airlines(canc_main)
    figure08_airports_2024(canc_main)
    figure09_departure_time(canc_main)
    for period, fig_air, fig_day in [
        ("Overall", "figure13.pdf", "figure15.pdf"),
        ("Pre-pandemic", "figure14.pdf", "figure16.pdf"),
        ("Pandemic", "figure17.pdf", "figure19.pdf"),
        ("Post-pandemic", "figure18.pdf", "figure20.pdf"),
    ]:
        plot_cluster_bars(clustered, "airline_grouped", period, fig_air, f"Grouped airline distribution by cluster - {PERIOD_LABELS[period]}")
        plot_cluster_bars(clustered, "day_of_week", period, fig_day, f"Day-of-week distribution by cluster - {PERIOD_LABELS[period]}")
    figure21_departure_box(clustered)
    enhanced_figures(df, canc_main)
    rate_based_validation(df)
    airline_daytype_crosstab(canc_main, df)
    rate_cell_clustering(df)
    write_outputs(df, canc, canc_main, selected_airports, clustered, sils)

    print(f"Loaded VRA records: {len(df):,}")
    print(f"Canceled Brazilian-origin records: {len(canc):,}")
    print(f"Selected main airports: {len(selected_airports)}")
    print(f"Main-airport cancellations: {len(canc_main):,}")
    print(f"Wrote figures to: {', '.join(str(p) for p in FIG_DIRS)}")
    print(f"Wrote output tables to: {OUT_DIR}")


if __name__ == "__main__":
    main()








