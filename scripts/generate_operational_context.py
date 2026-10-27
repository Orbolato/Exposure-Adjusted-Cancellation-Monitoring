"""Market context from the ANAC air transport statistics (domestic regular services).

Draws the domestic regular activity indices relative to the 2019 monthly average
(Fig. A.3) and the airline concentration of passenger shares (Fig. A.1), and writes
annual and phase summaries of domestic regular activity.
"""

import re
import unicodedata

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paths import ANALYSIS_DIR, FIGURE_DIR, STATS_DIR

DATA_DIR = STATS_DIR
FIG_DIR = FIGURE_DIR
OUT_DIR = ANALYSIS_DIR
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def slug(text):
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    keep = [ch if ch.isalnum() else "_" for ch in text.upper()]
    return re.sub(r"_+", "_", "".join(keep)).strip("_")


def load_data():
    frames = []
    for path in sorted(DATA_DIR.glob("*.csv")):
        df = pd.read_csv(path, sep=";", encoding="latin1")
        df.columns = [slug(c) for c in df.columns]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    for col in ["NATUREZA", "GRUPO_DE_VOO"]:
        df[col] = df[col].astype(str).map(slug)

    numeric = [
        "ANO",
        "MES",
        "PASSAGEIROS_PAGOS",
        "PASSAGEIROS_GRATIS",
        "ASK",
        "RPK",
        "DECOLAGENS",
        "ASSENTOS",
        "DISTANCIA_VOADA_KM",
        "HORAS_VOADAS",
        "COMBUSTIVEL_LITROS",
    ]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["date"] = pd.to_datetime(
        {"year": df["ANO"].astype(int), "month": df["MES"].astype(int), "day": 1}
    )
    df["passengers"] = df["PASSAGEIROS_PAGOS"] + df["PASSAGEIROS_GRATIS"]
    df["load_factor"] = np.where(df["ASK"] > 0, df["RPK"] / df["ASK"], np.nan)
    df["avg_seats_per_departure"] = np.where(df["DECOLAGENS"] > 0, df["ASSENTOS"] / df["DECOLAGENS"], np.nan)
    df["passengers_per_departure"] = np.where(df["DECOLAGENS"] > 0, df["passengers"] / df["DECOLAGENS"], np.nan)
    df["airline_group"] = df["EMPRESA_SIGLA"].map({"AZU": "Azul", "GLO": "Gol", "TAM": "TAM"}).fillna("Others")
    df["period"] = np.select(
        [
            df["date"] < pd.Timestamp("2020-03-01"),
            df["date"] <= pd.Timestamp("2023-04-01"),
        ],
        ["Pre-pandemic", "Pandemic"],
        default="Post-pandemic",
    )
    return df


def domestic_regular(df):
    return df[(df["NATUREZA"] == "DOMESTICA") & (df["GRUPO_DE_VOO"] == "REGULAR")].copy()


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, bbox_inches="tight")
    plt.close()


def weighted_mean(values, weights):
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return np.nan
    return np.average(values[mask], weights=weights[mask])


def concentration_hhi(group):
    totals = group.groupby("EMPRESA_SIGLA")["passengers"].sum()
    shares = totals / totals.sum() if totals.sum() else totals * np.nan
    return float((shares**2).sum())


def figure22_activity_indices(df):
    monthly = (
        df.groupby("date", as_index=False)
        .agg(passengers=("passengers", "sum"), departures=("DECOLAGENS", "sum"), seats=("ASSENTOS", "sum"), ask=("ASK", "sum"))
        .sort_values("date")
    )
    base = monthly[(monthly["date"] >= "2019-01-01") & (monthly["date"] <= "2019-12-01")]
    for col in ["passengers", "departures", "seats", "ask"]:
        monthly[f"{col}_idx"] = monthly[col] / base[col].mean() * 100

    plt.figure(figsize=(10, 5))
    colors = {"passengers_idx": "#1f5a8a", "departures_idx": "#6f7d3c", "seats_idx": "#b4553b", "ask_idx": "#5d4b8a"}
    labels = {
        "passengers_idx": "Passengers",
        "departures_idx": "Departures",
        "seats_idx": "Seats",
        "ask_idx": "ASK",
    }
    for col in colors:
        plt.plot(monthly["date"], monthly[col], label=labels[col], color=colors[col], linewidth=2)
    plt.axhline(100, color="black", linewidth=0.8, linestyle="--", alpha=0.45)
    plt.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2023-04-30"), color="#d9a441", alpha=0.15)
    plt.ylabel("Index (2019 monthly average = 100)")
    plt.xlabel("Month")
    plt.title("Domestic regular activity indicators relative to 2019")
    plt.legend(ncol=2, frameon=False)
    plt.grid(axis="y", alpha=0.25)
    savefig("figure22.pdf")


def figure24_market_concentration(df):
    airline_year = df.groupby(["ANO", "EMPRESA_SIGLA"], as_index=False)["passengers"].sum()
    annual = (
        airline_year.groupby("ANO")
        .apply(lambda g: ((g["passengers"] / g["passengers"].sum()) ** 2).sum(), include_groups=False)
        .reset_index(name="hhi")
    )
    annual["hhi"] = annual["hhi"] * 10_000

    top3 = (
        df[df["airline_group"].isin(["Azul", "Gol", "TAM"])]
        .groupby(["ANO", "airline_group"])["passengers"]
        .sum()
        .groupby(level=0)
        .sum()
        / df.groupby("ANO")["passengers"].sum()
        * 100
    ).reset_index(name="top3_share")

    merged = annual.merge(top3, on="ANO", how="left")
    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    ax1.plot(merged["ANO"], merged["hhi"], color="#1f5a8a", marker="o", linewidth=2, label="Airline HHI")
    ax1.set_ylabel("HHI (passenger shares)")
    ax1.set_xlabel("Year")
    ax1.grid(axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(merged["ANO"], merged["top3_share"], color="#b4553b", marker="s", linewidth=2, label="Azul/Gol/TAM share")
    ax2.set_ylabel("Top-three passenger share (%)")
    ax1.axvspan(2020, 2023, color="#d9a441", alpha=0.12)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, frameon=False, loc="upper left")
    plt.title("Domestic regular airline concentration")
    savefig("figure24.pdf")


def write_tables(df_domestic):
    annual = (
        df_domestic.groupby("ANO")
        .agg(
            passengers=("passengers", "sum"),
            departures=("DECOLAGENS", "sum"),
            seats=("ASSENTOS", "sum"),
            ask=("ASK", "sum"),
            rpk=("RPK", "sum"),
            airlines=("EMPRESA_SIGLA", "nunique"),
            origin_airports=("AEROPORTO_DE_ORIGEM_SIGLA", "nunique"),
            destination_airports=("AEROPORTO_DE_DESTINO_SIGLA", "nunique"),
        )
        .reset_index()
    )
    annual["load_factor"] = annual["rpk"] / annual["ask"]
    annual["passengers_per_departure"] = annual["passengers"] / annual["departures"]
    annual["avg_seats_per_departure"] = annual["seats"] / annual["departures"]
    annual.to_csv(OUT_DIR / "domestic_regular_annual_summary.csv", index=False)

    period = (
        df_domestic.groupby("period")
        .agg(
            passengers=("passengers", "sum"),
            departures=("DECOLAGENS", "sum"),
            seats=("ASSENTOS", "sum"),
            ask=("ASK", "sum"),
            rpk=("RPK", "sum"),
            airlines=("EMPRESA_SIGLA", "nunique"),
            origin_airports=("AEROPORTO_DE_ORIGEM_SIGLA", "nunique"),
        )
        .reset_index()
    )
    period["load_factor"] = period["rpk"] / period["ask"]
    period["passengers_per_departure"] = period["passengers"] / period["departures"]
    period.to_csv(OUT_DIR / "domestic_regular_period_summary.csv", index=False)


def main():
    df_all = load_data()
    df_domestic = domestic_regular(df_all)
    figure22_activity_indices(df_domestic)
    figure24_market_concentration(df_domestic)
    write_tables(df_domestic)
    print(f"Loaded {len(df_all):,} rows from {DATA_DIR}")
    print(f"Domestic regular rows: {len(df_domestic):,}")
    print(f"Wrote figures to {FIG_DIR}")
    print(f"Wrote summaries to {OUT_DIR}")


if __name__ == "__main__":
    main()
