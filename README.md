# Exposure-adjusted monitoring of flight cancellations

[![Reproducibility checks](https://github.com/Orbolato/Exposure-Adjusted-Cancellation-Monitoring/actions/workflows/reproducibility.yml/badge.svg)](https://github.com/Orbolato/Exposure-Adjusted-Cancellation-Monitoring/actions/workflows/reproducibility.yml)

Code, archived results, and reproducibility materials for:

> Carvalho, L.O., Almeida, D.S., Pamplona, D.A. (2026).
> **Exposure-adjusted monitoring of flight cancellations for service-reliability governance: A Brazilian case study with a cross-country replication.** *Case Studies on Transport Policy*, 26, 102001.
> https://doi.org/10.1016/j.cstp.2026.102001

Project page: https://orbolato.github.io/publications/flight-cancellation-monitoring/

The study asks whether a public flight registry can support exposure-adjusted
monitoring of flight cancellations by a passenger-rights regulator. Regular flight
legs departing Brazilian aerodromes in 2015–2024 (8,262,859 scheduled legs, of which
543,899 were canceled), taken from the Active Regular Flight (VRA) registry of
Brazil's National Civil Aviation Agency (ANAC), are aggregated into operational cells
by carrier group, origin airport, day type, and departure period, each described by
its cancellation rate. Screening lists, a ranking of cells by rate and an optional
three-segment K-Means partition, are fixed with one year of data and scored against
the cancellations of the following year. The same workflow is applied to the US
Bureau of Transportation Statistics (BTS) on-time performance registry (39.2 million
scheduled domestic flights) as a cross-country replication.

## Reproducibility levels

These levels differ in what they require and in what they establish.

1. **Tables from the archived results (no source data).** `python scripts/make_tables.py --check`
   rebuilds the bodies of all 23 data tables of the article (Tables 4–13 and B.1–B.13)
   from the CSV results archived in `results/` and confirms that each is identical,
   character for character, to the table printed in the article. Continuous
   integration runs this check on every change, together with checksum tests of all
   archived files.
2. **Full rerun from the public sources.** After the source files are downloaded
   (see [`data/README.md`](data/README.md)), `bash reproduce.sh` (or
   `.\reproduce.ps1` on Windows) runs the whole analysis, writes every CSV result and
   figure to `output/`, rebuilds the tables from the rerun, and compares every CSV
   with the archived one using `scripts/verify_outputs.py`.

The source files are not redistributed (more than 4 GB, published by ANAC and BTS
under their own terms). Their checksums are recorded in the manifests under `data/`,
so a new download can be compared with the files that produced the published
results. See [`PROVENANCE.md`](PROVENANCE.md) for what was changed in the code to
publish it and for the verification record.

## Repository layout

```text
.
├── README.md
├── CITATION.cff
├── PROVENANCE.md
├── ARTIFACTS.json            # size and SHA-256 of every archived result, figure, and reference table
├── reproduce.sh, reproduce.ps1
├── requirements.txt          # pinned packages used for the reported results
├── requirements-ci.txt       # packages for the data-free checks
├── scripts/                  # analysis pipeline (run order in reproduce.sh)
├── data/                     # source-file manifests and download instructions; files not versioned
│   ├── siros/                # ANAC VRA, SIROS archive (analysis source)
│   ├── legacy_portal/        # ANAC VRA, open-data portal (March 25-31, 2018 and reconciliation)
│   ├── bts/                  # BTS on-time performance extracts (replication)
│   └── anac_statistics/      # ANAC air transport statistics
├── reference/                # aerodrome and time-zone reference tables derived from public sources
├── results/                  # archived CSV results of the published analysis
│   ├── vra/, vra/robustness/, bts/
│   └── tables/               # the 23 table bodies as printed in the article
├── figures/                  # the 17 figure files used in the article
├── tests/                    # data-free integrity tests
└── output/                   # written by the scripts; not versioned
```

## Environment

The reported results were produced on Windows 10 with Python 3.12.3, pandas 2.3.3,
NumPy 2.0.1, SciPy 1.16.0, scikit-learn 1.8.0, Matplotlib 3.9.2, and PyArrow 21.0.0. A
single random seed (42) governs every stochastic step. timezonefinder 9.0.0 and
PyMuPDF 1.28.0 are needed only to rebuild the time-zone reference tables.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduction

### 1. Tables from the archived results

```bash
python scripts/make_tables.py --check
```

The table bodies are written to `output/tables/`, together with `numbers.json`, which
holds key numbers quoted in the text.

### 2. Full rerun

```bash
bash reproduce.sh --download    # downloads SIROS, open-data portal and BTS files first
bash reproduce.sh               # if the source files are already in data/
```

The ANAC statistics files must be placed in `data/anac_statistics/` by hand (see
`data/README.md`). The verification rerun took 43 minutes on a desktop computer with
an Intel Core i7-9700KF processor and 32 GB of memory, most of it in
`generate_robustness_analyses.py`; single steps used up to about 14 GB of memory.
Two environment variables relocate the large folders:

- `CANCELLATIONS_DATA_DIR`: folder with the source files (default `data/`).
- `CANCELLATIONS_OUTPUT_DIR`: folder for all outputs (default `output/`).

The scripts can also be run one at a time in the order of `reproduce.sh`; each reads
the outputs of the scripts before it.

## Where each result comes from

Tables 1–3 (literature, cell dimensions, and documented registry changes) and
Tables 14–16 (monitoring protocol, operating cycle, and data requirements) contain
no computed values. Every other table is written by `scripts/make_tables.py` from the
CSV files listed below, and collects key numbers quoted in the text from the same
files in `output/tables/numbers.json`.

| Article element | CSV files in `results/` | Written by |
|---|---|---|
| Table 4, annual cancellation record | `vra/robustness/registry_monthly.csv` | `generate_robustness_analyses.py` |
| Table 5, carrier-group rates by registry window | `vra/robustness/regime_segments.csv` | `generate_robustness_analyses.py` |
| Table 6, composition of the Others group | `vra/robustness/others_composition_by_phase.csv`, `watchlist_composition_by_operator.csv` | `generate_robustness_analyses.py` |
| Table 7, number of segments by validity criterion | `vra/cell_cluster_k_selection.csv` | `generate_policy_analyses.py` |
| Table 8, exposure-based segments | `vra/rate_cell_cluster_profiles.csv`, `rate_cell_cluster_silhouette.csv`; `vra/watchlist_policy_accounting.csv` | `generate_vra_results.py`; `generate_policy_analyses.py` |
| Table 9, prospective validation for Brazil | `vra/watchlist_out_of_sample_summary.csv`; `vra/robustness/bootstrap_brazil.csv` | `generate_policy_analyses.py`; `generate_robustness_analyses.py` |
| Table 10, sensitivity analyses | `vra/robustness/sensitivity_out_of_sample_summary.csv` | `generate_robustness_analyses.py` |
| Table 11, comparable-period analysis | `vra/robustness/regime_out_of_sample.csv` | `generate_robustness_analyses.py` |
| Table 12, BTS registry by year | `bts/bts_yearly_summary.csv` | `generate_bts_validation.py` |
| Table 13, year-ahead validation across registries | `bts/bts_watchlist_out_of_sample_summary.csv`, `bts/cross_country_watchlist_comparison.csv`; `vra/watchlist_out_of_sample_summary.csv`; `vra/robustness/sensitivity_out_of_sample_pairs.csv` | `generate_bts_validation.py`; `generate_policy_analyses.py`; `generate_robustness_analyses.py` |
| Table B.1, reconciliation of the two VRA archives | `vra/archive_reconciliation_yearly.csv` | `reconcile_vra_archives.py` |
| Table B.2, construction of the analysis population | `vra/inclusion_flow.csv`, `vra/missing_schedule_status.csv` | `build_vra_siros.py` |
| Table B.3, reason codes of flagged cells | `vra/robustness/reason_groups_watchlist.csv` | `generate_robustness_analyses.py` |
| Table B.4, clustering algorithms | `vra/cell_algorithm_comparison.csv`, `vra/cell_algorithm_high_rate_segment.csv` | `generate_policy_analyses.py` |
| Table B.5, evaluation population | `vra/watchlist_out_of_sample_summary.csv` | `generate_policy_analyses.py` |
| Table B.6, airport-cluster bootstrap | `vra/robustness/bootstrap_brazil.csv`, `bootstrap_us.csv` | `generate_robustness_analyses.py` |
| Table B.7, stability of the flagged sets | `vra/robustness/membership_stability.csv` | `generate_robustness_analyses.py` |
| Table B.8, review workload of the screening lists | `vra/robustness/operational_workload.csv`; `annual_watchlist_lift_by_quarter.csv` | `generate_operational_implications.py`; `generate_robustness_analyses.py` |
| Table B.9, number of segments and minimum exposure | `vra/watchlist_k_exposure_sensitivity.csv` | `generate_watchlist_sensitivity.py` |
| Table B.10, prevalence and one-hot encoding | `vra/encoding_sensitivity_segments.csv` | `generate_encoding_sensitivity.py` |
| Table B.11, exclusions of the minimum-exposure rule | `vra/robustness/minimum_exposure_exclusions.csv` | `generate_robustness_analyses.py` |
| Table B.12, canceled-record benchmark profiles | `vra/cluster_dominant_characteristics.csv` | `generate_vra_results.py` |
| Table B.13, segmentation and benchmark diagnostics | seven files in `vra/` (see `diagnostics_table` in `make_tables.py`) | `generate_vra_results.py` |

| Figure | File in `figures/` | Drawn by |
|---|---|---|
| Fig. 1, monthly scheduled legs and cancellation rate by reason group | `figure32.pdf` | `generate_robustness_analyses.py` |
| Fig. 2, highest origin-airport cancellation rates | `figure26.pdf` | `generate_vra_results.py` |
| Fig. 3, internal cluster-validity criteria | `figure29a.pdf` to `figure29d.pdf` | `generate_policy_analyses.py` |
| Fig. 4, prospective validation for Brazil | `figure30a.pdf`, `figure30b.pdf` | `generate_policy_analyses.py` |
| Fig. 5, high-rate segment in Brazil and the United States | `figure31a.pdf`, `figure31b.pdf` | `generate_bts_validation.py` |
| Fig. A.1, domestic airline concentration | `figure24.pdf` | `generate_operational_context.py` |
| Fig. A.2, construction of the analysis population and analysis paths | `figure02.pdf` | `generate_vra_results.py` |
| Fig. A.3, domestic activity relative to 2019 | `figure22.pdf` | `generate_operational_context.py` |
| Figs. A.4–A.7, aerodrome selection and count-based distributions | `figure03.pdf`, `figure07.pdf`, `figure08.pdf`, `figure09.pdf` | `generate_vra_results.py` |

`generate_vra_results.py` also writes diagnostic figures of the canceled-record
benchmark that the article summarizes in tables; only the files listed above are
figures of the article. Table and figure numbers follow the accepted manuscript
(CSTP-D-26-01481), from which the published version was typeset.

### Verification

On October 27, 2026, a full rerun from the source files in the manifests reproduced
all 106 archived CSV results (102 byte for byte, 4 within a relative tolerance of
1e-9), rebuilt all 23 table bodies identically, and wrote all 17 figures. Details are
in [`PROVENANCE.md`](PROVENANCE.md).

## Data and code availability

The source data are public: the VRA monthly files in ANAC's SIROS file directory and
on its open-data portal, the ANAC air transport statistics, and the BTS Reporting
Carrier On-Time Performance files. [`data/README.md`](data/README.md) gives the
download commands, the exact files used, and their checksums. The aerodrome and
time-zone reference tables in `reference/` are derived from ANAC statistics,
OurAirports (public domain), and DECEA's ROTAER aerodrome directory.

## Citation

If you use this repository, please cite the published article:
https://doi.org/10.1016/j.cstp.2026.102001. Machine-readable citation metadata are in
[`CITATION.cff`](CITATION.cff).

The article is open access under a Creative Commons Attribution 4.0 license. No
software or data license is asserted by this repository; the terms of the
third-party source data remain applicable.
