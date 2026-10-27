# Source data

The analysis reads four public sources. None of the source files is stored in this
repository: together they exceed 4 GB, and each remains subject to the terms of the
agency that publishes it. Each folder below holds a `manifest.json` that records, for
every file used in the study, its name, size, SHA-256 checksum and source, so that a
new download can be checked against the files that produced the published results.

| Folder | Source | Files used | Size | Obtained |
|---|---|---|---|---|
| `siros/` | ANAC, Active Regular Flight (VRA) registry, file directory of the air-services registration system (SIROS) | 122 monthly files, December 2014 to January 2025 | 2.79 GB | September 23, 2026 |
| `legacy_portal/` | ANAC, VRA files on the open-data portal | 118 files, 2015 to 2024 | 1.42 GB | May 27, 2026 |
| `bts/` | US Bureau of Transportation Statistics, Reporting Carrier On-Time Performance | 72 monthly extracts, 2018 to 2021, 2023 and 2024 (39,238,943 flights) | 0.10 GB | August 18, 2026 |
| `anac_statistics/` | ANAC, air transport statistics (*Dados estatísticos*) | 10 annual files, 2015 to 2024 | 0.09 GB | May 27, 2026 |

The scripts read the files from these folders by default. To keep them elsewhere, set
`CANCELLATIONS_DATA_DIR` to a folder with the same four subfolders.

## ANAC VRA, SIROS archive (`siros/`)

The analysis population is built from this archive.

```bash
python scripts/download_vra_siros.py
```

Files are taken from `https://siros.anac.gov.br/siros/registros/diversos/vra/<year>/VRA_<year>_<month>.csv`.
The download script rewrites `manifest.json` with the checksums and the server's
`Last-Modified` dates of the files it receives; compare them with the committed
manifest. December 2014 and January 2025 are included because a monthly file can
contain legs scheduled in the adjacent month. The SIROS file for March 2018 ends on
March 24; the legs scheduled from March 25 to 31, 2018 are taken from the open-data
portal file (below).

## ANAC VRA, open-data portal (`legacy_portal/`)

Used for two purposes only: the legs scheduled from March 25 to 31, 2018, and the
reconciliation of the two archives (Table B.1).

```bash
python scripts/download_vra_anac.py --start-year 2015 --end-year 2024
```

Files are stored as `legacy_portal/<year>/VRA_<year><month>.csv`, the names used on
the portal. In the portal version, every record of the February and March 2020 files
carries a placeholder string instead of dates and status; this is a defect of that
export, documented in the article, and those two months are unusable from this
source. When the files were obtained, the portal's 2024 folder held only April to
December; January to March 2024 come from a manual export from ANAC
(`2024/VRA_20260527133655.csv`, covering January 1 to May 1, 2024), of which only legs
scheduled before April 1, 2024 are read. Its checksum is in the manifest.

## BTS on-time performance (`bts/`)

Used for the United States replication.

```bash
python scripts/download_bts_ontime.py 2018 2019 2020 2021 2023 2024
```

The script downloads the monthly archives from
`https://transtats.bts.gov/PREZIP/` and keeps only the columns the workflow needs
(year, month, flight date, day of week, reporting airline, origin, destination,
scheduled departure time, cancellation flag and cancellation code), stored as one
Parquet file per month. The manifest records the row count of each extract. A new
download produces the same rows, but the Parquet files may not be byte-identical,
because the file metadata depend on library versions; compare row counts and the
results instead.

## ANAC air transport statistics (`anac_statistics/`)

Used for four purposes: the country of each aerodrome, which together with the ICAO
location-indicator blocks defines Brazilian origins; the state of aerodromes whose
state is missing from the VRA records, for the conversion to local time; the
nationality of each carrier, for the operator categories of the Others group; and the
market context shown in Figs. A.1 and A.3.

The study used one semicolon-separated file per year, 2015 to 2024, exported from
ANAC's statistics page (https://www.gov.br/anac/pt-br/assuntos/dados-e-estatisticas/dados-estatisticos)
and saved as `anac_statistics/<year>.csv` (Latin-1 encoding, monthly records by
airline, origin and destination). This export is not scripted. ANAC's open-data
directory also publishes the full series as a single file
(https://sistemas.anac.gov.br/dadosabertos/Voos%20e%20opera%C3%A7%C3%B5es%20a%C3%A9reas/Dados%20Estat%C3%ADsticos%20do%20Transporte%20A%C3%A9reo/);
it was not used, and its equivalence to the annual exports has not been verified.

## Reference tables (`../reference/`)

These small tables are derived from public sources and are versioned, so the analysis
does not need their raw inputs:

- `brazilian_aerodromes.csv`: ICAO codes reported in Brazil by the ANAC statistics,
  built by `scripts/build_brazilian_aerodrome_reference.py`.
- `airport_timezones.csv`: the IANA time zone of each Brazilian aerodrome, used to
  convert scheduled times from Brasília time to the local time of the origin, built
  by `scripts/build_airport_timezones.py` from OurAirports coordinates (public
  domain, https://ourairports.com/data/) and checked against DECEA's ROTAER
  aerodrome directory.
- `rotaer_timezones.csv`: state and UTC offset of each aerodrome, extracted from the
  ROTAER edition of September 23, 2026 by `scripts/extract_rotaer_timezones.py`.

The checksums of the two raw inputs (the OurAirports file and the ROTAER PDF) are in
`reference/airport_timezones_manifest.json`. To rebuild the tables, place them at
`reference/ourairports_airports.csv` and `reference/rotaer_20260923.pdf`. ROTAER is
updated regularly, so a current edition may differ from the one used.
