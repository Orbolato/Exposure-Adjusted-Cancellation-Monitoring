"""Download the monthly VRA files from ANAC's SIROS archive with a provenance manifest.

The SIROS archive (https://siros.anac.gov.br/siros/registros/diversos/vra/) publishes
every month of the Active Regular Flight registry in one format. It is used as the
single Brazilian source because the corresponding legacy open-data files for February
and March 2020 are unreadable. December 2014 and January 2025 are also retrieved so
that legs published in an adjacent month's file are not lost at the window edges.
"""

from __future__ import annotations

import concurrent.futures
import datetime as dt
import hashlib
import json

import requests

from paths import SIROS_DIR

OUT = SIROS_DIR
OUT.mkdir(parents=True, exist_ok=True)
BASE = "https://siros.anac.gov.br/siros/registros/diversos/vra/{y}/VRA_{y}_{m:02d}.csv"
MONTHS = [(2014, 12)] + [(y, m) for y in range(2015, 2025) for m in range(1, 13)] + [(2025, 1)]


def fetch(ym: tuple[int, int]) -> dict:
    y, m = ym
    url = BASE.format(y=y, m=m)
    path = OUT / f"VRA_{y}_{m:02d}.csv"
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=300, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            path.write_bytes(r.content)
            return {"file": path.name, "url": url, "bytes": len(r.content), "sha256": hashlib.sha256(r.content).hexdigest(),
                    "last_modified": r.headers.get("Last-Modified"), "retrieved": dt.date.today().isoformat()}
        except Exception as exc:  # noqa: BLE001
            err = str(exc)
    return {"file": path.name, "url": url, "error": err}


def main() -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(fetch, MONTHS))
    (OUT / "manifest.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    bad = [r for r in rows if "error" in r]
    print(f"downloaded {len(rows) - len(bad)} of {len(rows)} files; failures: {[r['file'] for r in bad]}")


if __name__ == "__main__":
    main()
