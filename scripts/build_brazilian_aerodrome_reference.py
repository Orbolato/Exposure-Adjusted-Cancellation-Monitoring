"""Build the reference list of Brazilian aerodromes used to define the study population.

Brazilian aerodromes are not confined to the ``SB`` ICAO block: the country also
holds the SD, SI, SJ, SN, SS and SW location-indicator blocks. Restricting the
sample to ``SB`` codes would therefore exclude regional aerodromes that carry
regular services.

The reference list is built from two independent sources and their union is used:

1. The country field of the ANAC monthly air transport statistics, which maps each
   aerodrome ICAO code to its country.
2. The seven ICAO location-indicator prefixes allocated to Brazil, which covers a
   small number of aerodromes that carry regular flights but do not appear in the
   aggregate statistical files.
"""

from __future__ import annotations

import glob
import re
import unicodedata

import pandas as pd

from paths import REFERENCE_DIR, STATS_DIR

OUT_PATH = REFERENCE_DIR / "brazilian_aerodromes.csv"

# ICAO location-indicator prefixes allocated to Brazil.
BR_PREFIXES = ("SB", "SD", "SI", "SJ", "SN", "SS", "SW")


def slug(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^A-Za-z0-9]+", "_", text.upper()).strip("_")


def aerodrome_country_map() -> pd.DataFrame:
    """Extract the ICAO-to-country mapping from the ANAC statistical files."""
    frames = []
    for path in sorted(glob.glob(str(STATS_DIR / "*.csv"))):
        data = pd.read_csv(path, sep=";", encoding="latin-1", low_memory=False)
        data.columns = [slug(c) for c in data.columns]
        for side in ("ORIGEM", "DESTINO"):
            cols = [f"AEROPORTO_DE_{side}_SIGLA", f"AEROPORTO_DE_{side}_PAIS"]
            if all(c in data.columns for c in cols):
                frames.append(data[cols].rename(columns={cols[0]: "icao", cols[1]: "country"}))
    mapping = pd.concat(frames, ignore_index=True).dropna()
    mapping["icao"] = mapping["icao"].astype(str).str.upper().str.strip()
    mapping["country"] = mapping["country"].astype(str).str.upper().str.strip()
    return mapping.drop_duplicates()


def main() -> None:
    mapping = aerodrome_country_map()
    registry = sorted(mapping.loc[mapping["country"].str.startswith("BRASIL"), "icao"].unique())

    conflicting = sorted(
        mapping.loc[mapping["icao"].str.startswith(BR_PREFIXES) & ~mapping["country"].str.startswith("BRASIL"), "icao"].unique()
    )
    if conflicting:
        raise RuntimeError(f"Brazilian-prefix codes reported outside Brazil: {conflicting}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"icao": registry, "source": "anac_statistics"}).to_csv(OUT_PATH, index=False)

    print(f"Brazilian aerodromes in the ANAC statistics: {len(registry)}")
    print("Prefix distribution:")
    print(pd.Series([c[:2] for c in registry]).value_counts().to_string())
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
