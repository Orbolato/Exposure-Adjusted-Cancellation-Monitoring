"""Airport-level IANA time zones for Brazilian aerodromes.

A state-level mapping is not sufficient in Brazil: western Amazonas follows the
UTC-5 rules of ``America/Eirunepe`` while Manaus follows ``America/Manaus``, and
Fernando de Noronha has its own zone. Each Brazilian aerodrome in the OurAirports
table (public domain, https://ourairports.com/data/) is therefore located by its
coordinates and assigned the IANA zone of the polygon that contains it
(``timezonefinder``, built on the timezone-boundary-builder data derived from
OpenStreetMap and the IANA database).

Output: ``reference/airport_timezones.csv`` with the ICAO code, coordinates, state,
zone and source, and ``reference/airport_timezones_manifest.json`` with the
retrieval date and checksum of the coordinate file. ``build_vra_siros.py`` uses the
table only when the airport state agrees with the historical flight record.
DECEA ROTAER validates the Amazonas split; state jurisdictions preserve historical
daylight-saving rules elsewhere. Coordinates alone are not treated as definitive.
"""

from __future__ import annotations

import hashlib
import sys
import json
from datetime import date
from importlib.metadata import version

import pandas as pd

from paths import REFERENCE_DIR

REF = REFERENCE_DIR
SOURCE = REF / "ourairports_airports.csv"
URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"


def main() -> None:
    if not SOURCE.exists():
        import urllib.request

        urllib.request.urlretrieve(URL, SOURCE)
        retrieved = date.today().isoformat()
    else:
        manifest = REF / "airport_timezones_manifest.json"
        retrieved = json.loads(manifest.read_text())["retrieved"] if manifest.exists() else date.today().isoformat()
    cached = REF / "airport_timezones.csv"
    manifest_path = REF / "airport_timezones_manifest.json"
    valid_cache = (cached.exists() and manifest_path.exists() and
                   json.loads(manifest_path.read_text())["sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    if valid_cache and "--refresh-coordinates" not in sys.argv:
        table = pd.read_csv(cached)
        if "coordinate_zone" in table:
            table["zone"] = table["coordinate_zone"]
    else:
        a = pd.read_csv(SOURCE, low_memory=False)
        a = a[a["iso_country"].eq("BR")].copy()
        from timezonefinder import TimezoneFinder
        tf = TimezoneFinder()
        a["zone"] = [tf.timezone_at(lng=lon, lat=lat) for lat, lon in zip(a["latitude_deg"], a["longitude_deg"])]
        a["uf"] = a["iso_region"].str.replace("BR-", "", regex=False)

        rows = []
        for _, r in a.iterrows():
            for code in {r["ident"], r["icao_code"], r["gps_code"]}:
                if isinstance(code, str) and len(code) == 4 and code.isalnum():
                    rows.append({"icao": code.upper(), "name": r["name"], "municipality": r["municipality"], "uf": r["uf"],
                                 "latitude": r["latitude_deg"], "longitude": r["longitude_deg"], "zone": r["zone"], "source": "coordinates"})
        table = pd.DataFrame(rows).drop_duplicates("icao").sort_values("icao")
    # Current coordinate polygons do not reliably encode every airport's civil
    # clock (notably Tabatinga). Use the official ROTAER offset for Amazonas.
    # For other states retain the historical IANA jurisdiction; this also
    # avoids offshore/border polygons that lack the state's former DST rules.
    from build_vra_siros import UF_ZONE
    official = pd.read_csv(REF / "rotaer_timezones.csv").set_index("icao")
    table["coordinate_zone"] = table["zone"]
    state_zone = table["uf"].map(UF_ZONE)
    non_am = table["uf"].ne("AM") & state_zone.notna() & table["icao"].ne("SBFN")
    table.loc[non_am, "zone"] = state_zone[non_am]
    table.loc[non_am, "source"] = "state jurisdiction; coordinates retained for audit"
    am = table["uf"].eq("AM") & table["icao"].map(official["uf"]).eq("AM")
    offset = table["icao"].map(official["utc_offset"])
    table.loc[am & offset.eq(-5), "zone"] = "America/Eirunepe"
    table.loc[am & offset.eq(-4), "zone"] = "America/Manaus"
    table.loc[am & offset.isin([-4, -5]), "source"] = "DECEA ROTAER 2026-09-23; IANA historical rules"
    table["rotaer_page"] = table["icao"].map(official["page"]).astype("Int64")
    table.to_csv(REF / "airport_timezones.csv", index=False)
    (REF / "airport_timezones_manifest.json").write_text(json.dumps({
        "coordinates_source": URL, "retrieved": retrieved,
        "sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "zone_lookup": "coordinates checked against state jurisdiction and DECEA ROTAER; IANA historical rules",
        "timezonefinder_version": version("timezonefinder"),
        "rotaer_source": "https://aisweb.decea.mil.br/downloads/rotaer/rotaer_completo.pdf",
        "rotaer_retrieved": "2026-09-23", "rotaer_sha256": hashlib.sha256((REF / "rotaer_20260923.pdf").read_bytes()).hexdigest(),
        "aerodromes": len(table),
    }, indent=1), encoding="utf-8")
    print(len(table), "aerodromes;", table["zone"].value_counts().to_dict())


if __name__ == "__main__":
    main()
