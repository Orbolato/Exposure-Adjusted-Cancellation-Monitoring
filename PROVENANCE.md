# Provenance

This repository was initialized on October 27, 2026 from the authors' analysis
project for the article

> Carvalho, L.O., Almeida, D.S., Pamplona, D.A. Exposure-adjusted monitoring of flight
> cancellations for service-reliability governance: A Brazilian case study with a
> cross-country replication. *Case Studies on Transport Policy* 26 (2026), article
> 102001. https://doi.org/10.1016/j.cstp.2026.102001

- Manuscript: CSTP-D-26-01481, accepted after two rounds of review.
- Random seed used for every reported result: `42`.
- Environment of the reported results: Windows 10, Python 3.12.3 and the package
  versions pinned in `requirements.txt`.

## What was copied

| Repository content | Origin in the analysis project |
|---|---|
| `scripts/` | The analysis scripts, run in the order of `reproduce.sh`, with the changes listed below. |
| `results/vra/`, `results/vra/robustness/`, `results/bts/`, `results/domestic_regular_*.csv` | The CSV outputs of the analysis behind the accepted manuscript, copied byte for byte. |
| `results/tables/` | The 23 table bodies extracted from the LaTeX source of the accepted manuscript, where they had been written by the table script. |
| `figures/` | The 17 figure files included by the LaTeX source of the accepted manuscript. |
| `reference/` | The derived aerodrome and time-zone reference tables used by the analysis. |
| `data/*/manifest.json` | Checksums of the source files used by the analysis (the SIROS manifest was written by the download script when the files were retrieved; the others were computed from the files used). |

`ARTIFACTS.json` records the size and SHA-256 checksum of every file in `results/`,
`figures/` and `reference/`, and the integrity tests check them on every change.

## Changes made to the code for publication

The analytical code is unchanged. The changes concern where files are read and
written, and outputs the article does not use:

1. **Paths.** Every script resolves its inputs and outputs through `scripts/paths.py`.
   Source files are read from `data/` (or `CANCELLATIONS_DATA_DIR`), and everything
   the pipeline writes goes to `output/` (or `CANCELLATIONS_OUTPUT_DIR`), including
   the harmonized flight-level caches, which the analysis project kept next to the
   source files. In the analysis project the source folders had other names
   (`data_vra/siros_archive`, `data_vra/raw`, `data_bts/raw` and `data`).
2. **`make_tables.py`** is the script that wrote the table bodies into the
   manuscript's LaTeX source. The code that builds each table is unchanged; the
   script now writes each body to a file and can compare it with the archived one.
3. **`generate_operational_context.py`** no longer draws two figures and writes a
   data note that belonged to an earlier version of the manuscript and do not appear
   in the article, and no longer writes PNG copies. The code that draws Figs. A.1
   and A.3 is unchanged.
4. **`download_bts_ontime.py`** downloads by default the six years used in the article
   (2018 to 2021, 2023 and 2024).
5. **`download_vra_anac.py`** gained a module docstring.
6. **New scripts:** `paths.py`, `verify_outputs.py` (comparison of a rerun with the
   archived results) and `write_artifact_manifest.py`.
7. Comments and docstrings were edited for clarity and consistent US English. Column
   names of the CSV outputs and a few code labels keep the spelling "cancelled" with
   which the archived results were produced, so that a rerun can be compared with
   them directly.

A script that translated the figures of an earlier version of the manuscript was not
published, because none of its figures appear in the article.

## Data not redistributed

The source files (ANAC VRA files from the SIROS archive and the open-data portal, ANAC
air transport statistics, and BTS on-time performance extracts) are not included; see
`data/README.md`. Their checksums allow a new download to be compared with the files
used. Files published by the agencies can change after retrieval; the retrieval dates
are recorded in the manifests.

## Verification record

On October 27, 2026, the complete pipeline (`reproduce.sh`) was run with the
published scripts from the source files listed in the data manifests, into an empty
output folder, on Windows 10 with the pinned environment. The run took 43 minutes.

- **CSV results.** The rerun wrote 106 CSV files, and every one matched its archived
  copy in `results/`: 102 byte for byte and 4 within the tolerance of
  `verify_outputs.py`. In those four files the only differences are in K-Means
  inertia, gap statistics and one lift, with a largest relative difference of
  4.5e-13, which comes from the order of floating-point summation.
- **Tables.** All 23 table bodies rebuilt from the rerun are identical to those
  printed in the article.
- **Figures.** All 17 figure files of the article were written. They were not
  compared pixel by pixel.
- **Reference table.** `reference/brazilian_aerodromes.csv`, rebuilt from the ANAC
  statistics, is byte-identical to the one used.

Seven CSV files in the analysis project's output folder were not produced by the
final scripts. They were left over from earlier versions of the analysis, written
before the final run, and no script or table reads them, so they were not archived.

After the verification run, only comments and docstrings of the scripts were edited.
