"""Brazilian monitoring frame shared by the analysis scripts.

Builds, from the harmonized SIROS-archive VRA (``build_vra_siros.py``), the
flight-level table used by ``cancellation_monitoring``: Brazilian-origin regular legs
with a final status, the operating carrier, the full-period carrier grouping kept for
descriptive tables, operator categories, reason groups, and alternative departure
windows and Brasilia-clock variables for sensitivity analyses.
"""

from __future__ import annotations

import glob
import re
import unicodedata

import numpy as np
import pandas as pd

from generate_vra_results import OUT_DIR, load_vra
from paths import STATS_DIR

CARGO_LINES = {"C", "G", "L"}
CODE_GROUPS = {
    "XA": "Administrative or commercial", "XB": "Administrative or commercial", "XL": "Administrative or commercial",
    "XR": "Administrative or commercial", "XN": "Technical or operational", "XO": "Weather", "XT": "Weather", "XS": "Weather",
    "XI": "Airport closure", "XJ": "Airport closure", "XM": "Airport closure",
}
GROUP_ORDER = ["Administrative or commercial", "Technical or operational", "Weather", "Airport closure"]


def _slug(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^A-Za-z0-9]+", "_", text.upper()).strip("_")


def carrier_reference() -> pd.DataFrame:
    """Latest carrier name/nationality: a retrospective descriptive taxonomy.

    Used for composition summaries and the conditional category sensitivity,
    not for the baseline prospective carrier-retention rule.
    """
    frames = []
    for path in sorted(glob.glob(str(STATS_DIR / "*.csv"))):
        data = pd.read_csv(path, sep=";", encoding="latin-1", low_memory=False, usecols=[0, 1, 2, 3])
        data.columns = [_slug(c) for c in data.columns]
        frames.append(data.drop_duplicates())
    ref = pd.concat(frames, ignore_index=True).rename(
        columns={"EMPRESA_SIGLA": "airline", "EMPRESA_NOME": "carrier_name", "EMPRESA_NACIONALIDADE": "nationality", "ANO": "stat_year"})
    ref["airline"] = ref["airline"].astype(str).str.upper().str.strip()
    ref = ref.sort_values("stat_year").groupby("airline").agg(carrier_name=("carrier_name", "last"), nationality=("nationality", "last"))
    ref["nationality"] = ref["nationality"].astype(str).str.upper().map({"BRASILEIRA": "Brazilian", "ESTRANGEIRA": "Foreign"})
    return ref


def window_labels(hour: pd.Series, width: int, offset: int = 0) -> tuple[pd.Series, dict[str, float]]:
    shifted = (hour - offset) % 24
    idx = np.floor(shifted / width).astype(int)
    names = np.array([f"W{(offset + i * width) % 24:02d}" for i in range(24 // width)])
    labels = pd.Series(names[idx.to_numpy()], index=hour.index, dtype="string")
    mids = {f"W{(offset + i * width) % 24:02d}": (offset + i * width + width / 2) % 24 for i in range(24 // width)}
    return labels, mids


def brazil_frame(save_reference: bool = True) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    df = load_vra()
    br = df[df["is_br_origin"]].copy()
    del df
    br["carrier_group_fixed"] = br["airline_grouped"]
    br["carrier_group"] = br["airline_grouped"]
    br["is_domestic"] = br["is_br_dest"].astype(bool)
    br["is_cargo_line"] = br["line_type"].isin(CARGO_LINES).astype(bool)
    br["reason_code"] = br["justification"].where(br["is_cancelled"])
    br["reason_group"] = br["reason_code"].map(CODE_GROUPS)
    br["quarter"] = ((br["month"].astype(int) - 1) // 3 + 1).astype(int)
    br["year"] = br["year"].astype(int)
    br["ym"] = br["year"] * 100 + br["month"].astype(int)

    windows = {"base": {"Night": 3, "Morning": 9, "Afternoon": 15, "Evening": 21}}
    br["win_shift6"], windows["shift6"] = window_labels(br["departure_hour"], 6, 3)
    br["win_3h"], windows["3h"] = window_labels(br["departure_hour"], 3, 0)
    br["win_2h"], windows["2h"] = window_labels(br["departure_hour"], 2, 0)

    ref = carrier_reference()
    per = br.groupby("airline").agg(flights=("is_cancelled", "size"), domestic_share=("is_domestic", "mean"),
                                    cargo_share=("is_cargo_line", "mean")).join(ref, how="left")
    fallback = per["nationality"].isna()
    per.loc[fallback, "nationality"] = np.where(per.loc[fallback, "domestic_share"] > 0.5, "Brazilian", "Foreign")
    per["nationality_source"] = np.where(fallback, "Domestic-share rule", "ANAC statistics")
    per["service"] = np.where(per["cargo_share"] >= 0.5, "cargo", "passenger")
    per["category"] = per["nationality"] + " " + per["service"]
    if save_reference:
        (OUT_DIR / "robustness").mkdir(parents=True, exist_ok=True)
        per.to_csv(OUT_DIR / "robustness" / "carrier_reference.csv")
    br["operator_category"] = br["airline"].map(per["category"])
    return br.reset_index(drop=True), windows
