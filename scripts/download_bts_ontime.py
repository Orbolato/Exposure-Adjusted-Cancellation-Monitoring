"""Download US DOT/BTS On-Time Reporting Carrier On-Time Performance monthly files.

The files are used for the external validation of the cancellation-monitoring
workflow on a second national registry. Only the columns required by the
workflow (carrier, origin, day of week, scheduled departure time, cancellation
flag) are retained; the compact extracts are stored as Parquet files.
"""

from __future__ import annotations

import io
import sys
import time
import urllib.request
import zipfile

import pandas as pd

from paths import BTS_DIR

RAW_DIR = BTS_DIR
RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)
USE_COLS = [
    "Year",
    "Month",
    "FlightDate",
    "DayOfWeek",
    "Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "Cancelled",
    "CancellationCode",
]
# Years analyzed in the article: three year pairs, 2018-2019, 2020-2021 and 2023-2024.
YEARS = [2018, 2019, 2020, 2021, 2023, 2024]


def fetch_month(year: int, month: int, retries: int = 3) -> pd.DataFrame | None:
    out = RAW_DIR / f"bts_{year}_{month:02d}.parquet"
    if out.exists():
        return None
    url = BASE.format(year=year, month=month)
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=300) as response:
                payload = response.read()
            break
        except Exception as exc:  # noqa: BLE001 - transient network issues
            print(f"  attempt {attempt} failed for {year}-{month:02d}: {exc}", flush=True)
            if attempt == retries:
                return None
            time.sleep(10)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        with archive.open(name) as handle:
            frame = pd.read_csv(handle, usecols=USE_COLS, low_memory=False)
    frame.to_parquet(out, index=False)
    print(f"  saved {out.name}: {len(frame):,} rows", flush=True)
    return frame


def main() -> None:
    years = [int(a) for a in sys.argv[1:]] or YEARS
    for year in years:
        print(f"Year {year}", flush=True)
        for month in range(1, 13):
            fetch_month(year, month)
    files = sorted(RAW_DIR.glob("*.parquet"))
    print(f"Stored {len(files)} monthly extracts in {RAW_DIR}")


if __name__ == "__main__":
    main()
