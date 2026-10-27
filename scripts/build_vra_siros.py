"""Harmonize the SIROS-archive VRA files into the analysis dataset.

Source. Every month from December 2014 to January 2025 is read from ANAC's SIROS
archive (see ``download_vra_siros.py`` and ``data/siros/manifest.json``).
The legacy open-data files support reconciliation (``reconcile_vra_archives.py``)
and supply March 25-31, 2018, which is absent from the SIROS file.

Record identity and duplicates. A leg is identified by operating carrier, flight
number, authorization digit (DI), origin, destination and scheduled departure.
Rows that are identical in every field except the file they come from are kept once.
Where rows share the identity but differ in content, the row from the file of the
leg's own scheduled month is kept; if the conflict remains, the row with a final
status is preferred, then an operated record, then the actual departure closest to the schedule. Every step is
counted in ``output/analysis_outputs/vra/inclusion_flow.csv``.

Scope. The analysis population is regular legs (DI 0) with a scheduled departure and
a final status of operated or canceled, scheduled between January 1, 2015 and
December 31, 2024. Records without a usable scheduled departure cannot be assigned
to the analysis calendar; these include canceled records and are not assumed to
be exclusively unplanned operations. Other DI categories and unknown final
statuses are excluded and counted separately.

Clock. The registry reports scheduled times in Brasilia civil time. Departure period
and day type are derived from the origin's local civil time, obtained from the
historical origin state, an airport-level lookup checked against DECEA ROTAER,
and IANA time-zone rules (including daylight saving time until 2019). Current
airport codes are used only when their state agrees with the historical record,
because airport codes can be reassigned. The Brasilia-clock versions are retained for sensitivity analysis.

Reasons. Cancellation reasons are published as the descriptions of the IAC 1504
Annex 2 codes; they are mapped back to the codes by their normalized text.
"""

from __future__ import annotations

import glob
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from paths import ANALYSIS_DIR, INTERIM_DIR, LEGACY_DIR, REFERENCE_DIR, SIROS_DIR, STATS_DIR

SRC = SIROS_DIR
CACHE = INTERIM_DIR / "vra_siros_harmonized.parquet"
OUT = ANALYSIS_DIR / "vra"
REF = REFERENCE_DIR
OUT.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

BR_PREFIXES = ("SB", "SD", "SI", "SJ", "SN", "SS", "SW")
WINDOW = (pd.Timestamp("2015-01-01"), pd.Timestamp("2025-01-01"))

RENAME = {
    "Sigla ICAO Empresa Aérea": "airline", "Empresa Aérea": "carrier_name", "Número Voo": "flight_no",
    "Código DI": "di", "Código Tipo Linha": "line_type", "Modelo Equipamento": "aircraft_model",
    "Número de Assentos": "seats", "Sigla ICAO Aeroporto Origem": "origin", "Descrição Aeroporto Origem": "origin_desc",
    "Partida Prevista": "sched_dep", "Partida Real": "real_dep", "Sigla ICAO Aeroporto Destino": "dest",
    "Descrição Aeroporto Destino": "dest_desc", "Chegada Prevista": "sched_arr", "Chegada Real": "real_arr",
    "Situação Voo": "status", "Justificativa": "just_text", "Referência": "reference_date",
    "Situação Partida": "dep_situation", "Situação Chegada": "arr_situation",
}

# IAC 1504, Annex 2, part B: cancellation codes and their published descriptions.
IAC_CANCEL = {
    "XA": "PROGRAMADO - FERIADO NACIONAL",
    "XB": "AUTORIZADO",
    "XI": "DEVIDO AEROPORTO DE ORIGEM INTERDITADO",
    "XJ": "DEVIDO AEROPORTO DE DESTINO INTERDITADO",
    "XL": "FALTA PAX COM PASSAGEM MARCADA - (APENAS PARA AS LINHAS AEREAS DOMESTICAS REGIONAIS)",
    "XM": "CANCELAMENTO - CONEXAO AERONAVE/VOLTA - VOO DE IDA CANCELADO - AEROPORTO INTERDITADO",
    "XN": "CANCELAMENTO POR MOTIVOS TECNICOS - OPERACIONAIS",
    "XO": "CANCELAMENTO - AEROPORTO ORIGEM ABAIXO LIMITES",
    "XT": "CANCELAMENTO - AEROPORTO DESTINO ABAIXO LIMITES",
    "XR": "CANCELAMENTO DE VOOS OPERADOS EM CODE SHARING",
    "XS": "CANCELAMENTO - CONEXAO AERONAVE/VOLTA - VOO DE IDA CANCELADO - CONDICOES METEOROLOGICAS",
}

UF_ZONE = {
    "AC": "America/Rio_Branco", "AL": "America/Maceio", "AP": "America/Belem", "AM": "America/Manaus",
    "BA": "America/Bahia", "CE": "America/Fortaleza", "DF": "America/Sao_Paulo", "ES": "America/Sao_Paulo",
    "GO": "America/Sao_Paulo", "MA": "America/Fortaleza", "MT": "America/Cuiaba", "MS": "America/Campo_Grande",
    "MG": "America/Sao_Paulo", "PA": "America/Belem", "PB": "America/Fortaleza", "PR": "America/Sao_Paulo",
    "PE": "America/Recife", "PI": "America/Fortaleza", "RJ": "America/Sao_Paulo", "RN": "America/Fortaleza",
    "RS": "America/Sao_Paulo", "RO": "America/Porto_Velho", "RR": "America/Boa_Vista", "SC": "America/Sao_Paulo",
    "SP": "America/Sao_Paulo", "SE": "America/Maceio", "TO": "America/Araguaina",
}
AIRPORT_ZONE = {"SBFN": "America/Noronha"}


def norm_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).upper()
    text = re.sub(r"[‐-―−-]", "-", text)
    text = re.sub(r"[\"'“”‘’]", "", text)
    text = re.sub(r"\s*-\s*", " - ", text)
    text = re.sub(r"\(\s*", "(", text)
    text = re.sub(r"\s*\)", ")", text)
    text = re.sub(r"\s+\d+\s+A\d\s*-\s*\d\s*$", "", text)  # page footer carried over from the IAC document
    return re.sub(r"\s+", " ", text).strip()


CODE_BY_TEXT = {norm_text(v): k for k, v in IAC_CANCEL.items()}


def read_month(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str, keep_default_na=False, na_values=[""])
    d = d.rename(columns=RENAME)
    d["source_file"] = path.name
    m = re.match(r"VRA_(\d{4})_(\d{2})\.csv", path.name)
    d["file_month"] = int(m.group(1)) * 100 + int(m.group(2))
    return d


def parse_times(d: pd.DataFrame) -> pd.DataFrame:
    for col in ("sched_dep", "real_dep"):
        raw = d[col]
        d[col] = pd.to_datetime(raw, format="%d/%m/%Y %H:%M", errors="coerce")
        bad = raw.notna() & d[col].isna()
        if bad.any():  # tolerate an occasional seconds field
            d.loc[bad, col] = pd.to_datetime(raw[bad], format="mixed", dayfirst=True, errors="coerce")
    return d


def origin_uf(desc: pd.Series) -> pd.Series:
    parts = desc.fillna("").str.split(" - ")
    uf = parts.map(lambda p: p[-2].strip() if len(p) >= 3 and p[-1].strip().upper() in ("BRASIL", "BRAZIL") else None)
    return uf.where(uf.isin(UF_ZONE.keys()))


def to_local(sched: pd.Series, zones: pd.Series) -> pd.Series:
    out = pd.Series(pd.NaT, index=sched.index, dtype="datetime64[ns]")
    # Ambiguous Brasilia times use standard time; nonexistent times shift to
    # the first valid instant. This convention is deterministic.
    for zone, idx in zones.groupby(zones).groups.items():
        s = sched.loc[idx]
        if zone == "America/Sao_Paulo":
            out.loc[idx] = s.to_numpy()
            continue
        loc = s.dt.tz_localize("America/Sao_Paulo", ambiguous=np.zeros(len(s), dtype=bool), nonexistent="shift_forward")
        out.loc[idx] = loc.dt.tz_convert(zone).dt.tz_localize(None).to_numpy()
    return out


def derive_calendar(df: pd.DataFrame, clock: pd.Series, suffix: str) -> None:
    hour = clock.dt.hour + clock.dt.minute / 60
    df[f"departure_hour{suffix}"] = hour
    df[f"departure_period{suffix}"] = pd.cut(hour, bins=[0, 6, 12, 18, 24], labels=["Night", "Morning", "Afternoon", "Evening"],
                                             include_lowest=True, right=False).astype("string")
    dow = clock.dt.day_name()
    df[f"day_of_week{suffix}"] = dow
    df[f"day_type{suffix}"] = np.where(dow.isin(["Saturday", "Sunday"]), "Weekend", "Weekday")


LEGACY_GAP_FILE = LEGACY_DIR / "2018" / "VRA_20183.csv"
LEGACY_GAP = (pd.Timestamp("2018-03-25"), pd.Timestamp("2018-04-01"))
LEGACY_COLUMNS = {
    "ICAO Empresa Aérea": "airline", "Número Voo": "flight_no", "Código Autorização (DI)": "di", "Código Tipo Linha": "line_type",
    "ICAO Aeródromo Origem": "origin", "ICAO Aeródromo Destino": "dest", "Partida Prevista": "sched_dep", "Partida Real": "real_dep",
    "Chegada Prevista": "sched_arr", "Chegada Real": "real_arr", "Situação Voo": "status", "Código Justificativa": "code",
}


def legacy_gap_records() -> pd.DataFrame:
    """Legs scheduled 25-31 March 2018, absent from the SIROS archive, from the legacy file."""
    d = pd.read_csv(LEGACY_GAP_FILE, sep=";", encoding="utf-8-sig", skiprows=1, dtype=str, keep_default_na=False, na_values=[""])
    d = d.rename(columns=LEGACY_COLUMNS)
    when = pd.to_datetime(d["sched_dep"], format="mixed", dayfirst=True, errors="coerce")
    d = d[when.between(LEGACY_GAP[0], LEGACY_GAP[1], inclusive="left")].copy()
    d["sched_dep"] = when[d.index].dt.strftime("%d/%m/%Y %H:%M")
    real = pd.to_datetime(d["real_dep"], format="mixed", dayfirst=True, errors="coerce")
    d["real_dep"] = real.dt.strftime("%d/%m/%Y %H:%M")
    d["di"] = d["di"].str.replace(r"\.0$", "", regex=True)
    d["just_text"] = d["code"].map(IAC_CANCEL)
    d["source_file"] = "legacy:" + LEGACY_GAP_FILE.name
    d["file_month"] = 201803
    return d.drop(columns=["code"])


def main() -> None:
    files = sorted(SRC.glob("VRA_*.csv"))
    raw = pd.concat([read_month(f) for f in files] + [legacy_gap_records()], ignore_index=True)
    n_gap = int(raw["source_file"].str.startswith("legacy:").sum())
    flow = [("Records in the 122 SIROS monthly files (Dec 2014 to Jan 2025)", len(raw) - n_gap),
            ("Records for 25-31 March 2018 taken from the legacy file (absent from SIROS)", n_gap)]

    for col in ("airline", "origin", "dest", "status", "line_type", "di"):
        raw[col] = raw[col].astype("string").str.upper().str.strip()
    raw["flight_no"] = raw["flight_no"].astype("string").str.strip().str.zfill(4)
    raw = parse_times(raw)

    # 1. Exact duplicates (identical in every published field).
    fields = [c for c in raw.columns if c not in ("source_file", "file_month")]
    dup_exact = raw.duplicated(subset=fields, keep="first")
    exact_by_year = raw.loc[dup_exact].groupby(raw.loc[dup_exact, "sched_dep"].dt.year).size()
    raw = raw.loc[~dup_exact].copy()
    flow.append(("Exact duplicates removed", int(dup_exact.sum())))

    # 2. Scheduled dates contradicted by the leg's reference date by more than one day
    #    (concentrated in the March 2016 file) are restored from the reference date,
    #    keeping the published clock time.
    ref_raw = raw["reference_date"].astype("string")
    ref = pd.to_datetime(ref_raw.where(ref_raw.str.match(r"^\d{4}-\d{2}-\d{2}")), format="%Y-%m-%d", errors="coerce")
    ref = ref.fillna(pd.to_datetime(ref_raw.where(ref_raw.str.match(r"^\d{2}/\d{2}/\d{4}")).str.slice(0, 10), format="%d/%m/%Y", errors="coerce"))
    gap = (ref - raw["sched_dep"].dt.normalize()).dt.days
    fix = raw["sched_dep"].notna() & ref.notna() & gap.abs().gt(1)
    raw["sched_date_restored"] = fix
    fixed_by_file = raw.loc[fix, "source_file"].value_counts()
    raw.loc[fix, "sched_dep"] = ref[fix] + (raw.loc[fix, "sched_dep"] - raw.loc[fix, "sched_dep"].dt.normalize())
    flow.append(("Scheduled dates restored from the reference date", int(fix.sum())))

    # 3. One outcome per planned leg: prefer the home-month file, then a final status,
    #    then an operated record, then the actual departure closest to the schedule.
    key = ["airline", "flight_no", "di", "origin", "dest", "sched_dep"]
    has_key = raw["sched_dep"].notna()
    keyed = raw.loc[has_key].copy()
    keyed["home"] = (keyed["sched_dep"].dt.year * 100 + keyed["sched_dep"].dt.month) == keyed["file_month"]
    keyed["final"] = keyed["status"].isin(["REALIZADO", "CANCELADO"])
    keyed["operated"] = keyed["status"].eq("REALIZADO")
    keyed["distance"] = (keyed["real_dep"] - keyed["sched_dep"]).abs()
    keyed = keyed.sort_values(key + ["home", "final", "operated", "distance"], ascending=[True] * len(key) + [False, False, False, True],
                              na_position="last", kind="mergesort")
    conflict = keyed.duplicated(subset=key, keep="first")
    conflict_rows = keyed.loc[keyed.duplicated(subset=key, keep=False)]
    status_conflicts = int(conflict_rows.groupby(key)["status"].nunique().gt(1).sum())
    conflicts_by_year = keyed.loc[conflict].groupby(keyed.loc[conflict, "sched_dep"].dt.year).size()
    flow.append(("Additional outcome records for the same planned leg (dropped)", int(conflict.sum())))
    raw = pd.concat([keyed.loc[~conflict].drop(columns=["home", "final", "operated", "distance"]), raw.loc[~has_key]], ignore_index=True)

    # 3. Scope.
    raw["in_window"] = raw["sched_dep"].between(WINDOW[0], WINDOW[1], inclusive="left")
    missing_schedule = raw["sched_dep"].isna()
    aerodromes = set(pd.read_csv(REF / "brazilian_aerodromes.csv")["icao"].astype(str).str.upper())
    missing = raw.loc[missing_schedule].copy()
    missing["brazilian_origin"] = missing["origin"].isin(aerodromes) | missing["origin"].str.startswith(BR_PREFIXES).fillna(False)
    missing.groupby(["status", "di", "brazilian_origin"], dropna=False).size().reset_index(name="records").to_csv(
        OUT / "missing_schedule_status.csv", index=False)
    flow.append(("Records without usable scheduled departure", int(missing_schedule.sum())))
    missing_cancelled = int(missing["status"].eq("CANCELADO").sum())
    missing_br_regular_cancelled = int((missing["status"].eq("CANCELADO") & missing["di"].eq("0") & missing["brazilian_origin"]).sum())
    raw = raw.loc[~missing_schedule & raw["in_window"]].copy()
    flow.append(("Planned legs scheduled in 2015-2024", len(raw)))

    aerodromes = set(pd.read_csv(REF / "brazilian_aerodromes.csv")["icao"].astype(str).str.upper())
    raw["is_br_origin"] = raw["origin"].isin(aerodromes) | raw["origin"].str.startswith(BR_PREFIXES).fillna(False)
    raw["is_br_dest"] = raw["dest"].isin(aerodromes) | raw["dest"].str.startswith(BR_PREFIXES).fillna(False)
    raw["is_regular"] = raw["di"].eq("0")
    raw["known_status"] = raw["status"].isin(["REALIZADO", "CANCELADO"])
    raw["is_cancelled"] = raw["status"].eq("CANCELADO")

    # 4. Reasons.
    norm = raw["just_text"].map(lambda t: norm_text(t) if isinstance(t, str) else None)
    raw["justification"] = norm.map(CODE_BY_TEXT).astype("string")
    unmapped = raw.loc[raw["is_cancelled"] & raw["just_text"].notna() & raw["justification"].isna(), "just_text"].value_counts()
    unmapped.rename_axis("text").reset_index(name="cancelled_records").to_csv(OUT / "unmapped_cancellation_texts.csv", index=False)

    # 5. Clock.
    uf = origin_uf(raw["origin_desc"])
    # SIROS sometimes labels Olhos d'Agua / Agua Boa as MG; the named
    # airport is SWHP in MT (DECEA ROTAER p. 1050), not a reassigned code.
    uf = uf.mask(raw["origin"].eq("SWHP"), "MT")
    # Resolve legacy/missing descriptions from this archive before using the
    # auxiliary statistical files; never let a newly reassigned code overwrite
    # the state explicitly supplied with a historical record.
    observed = pd.DataFrame({"origin": raw["origin"], "uf": uf}).dropna().drop_duplicates()
    unique = observed.groupby("origin")["uf"].nunique()
    archive_uf = observed[observed["origin"].isin(unique[unique.eq(1)].index)].set_index("origin")["uf"]
    uf = uf.fillna(raw["origin"].map(archive_uf))
    stats_uf = {}
    for path in sorted(glob.glob(str(STATS_DIR / "*.csv"))):
        s = pd.read_csv(path, sep=";", encoding="latin-1", usecols=[5, 7], low_memory=False)
        s.columns = ["icao", "uf"]
        stats_uf.update(s.dropna().drop_duplicates("icao").set_index("icao")["uf"].to_dict())
    uf = uf.fillna(raw["origin"].map(stats_uf))
    airport = pd.read_csv(REF / "airport_timezones.csv").set_index("icao")
    airport_uf = raw["origin"].map(airport["uf"])
    compatible = airport_uf.eq(uf)
    zone = uf.map(UF_ZONE)
    airport_zone = raw["origin"].map(airport["zone"])
    zone = airport_zone.where(compatible).fillna(zone)
    zone = raw["origin"].map(AIRPORT_ZONE).fillna(zone)
    br = raw["is_br_origin"]
    raw["origin_zone"] = zone.where(br)
    missing_zone = int((br & raw["origin_zone"].isna()).sum())
    if missing_zone:
        raise ValueError(f"{missing_zone} Brazilian-origin records have no validated time zone")
    # Each Amazonas origin must have an airport lookup, since state alone is
    # insufficient. Refuse a silent Manaus fallback for an unknown AM airport.
    unresolved_am = br & uf.eq("AM") & (~compatible | airport_zone.isna())
    if unresolved_am.any():
        raise ValueError("Unresolved Amazonas airports: " + str(raw.loc[unresolved_am, "origin"].unique()))
    provenance = pd.DataFrame({"origin": raw.loc[br, "origin"], "historical_uf": uf[br],
        "zone": zone[br], "lookup_state_agrees": compatible[br],
        "in_analysis": (raw["is_regular"] & raw["known_status"])[br]})
    provenance.groupby(list(provenance.columns), dropna=False).size().reset_index(name="records").to_csv(
        OUT / "airport_timezone_usage.csv", index=False)
    local = raw["sched_dep"].copy()
    local.loc[br] = to_local(raw.loc[br, "sched_dep"], raw.loc[br, "origin_zone"])
    raw["sched_dep_local"] = local
    derive_calendar(raw, raw["sched_dep_local"], "")
    derive_calendar(raw, raw["sched_dep"], "_brt")
    angle = 2 * np.pi * raw["departure_hour"] / 24
    raw["departure_sin"], raw["departure_cos"] = np.sin(angle), np.cos(angle)

    # 6. Registry calendar (Brasilia date) and phases.
    raw["year"] = raw["sched_dep"].dt.year.astype("int16")
    raw["month"] = raw["sched_dep"].dt.month.astype("int8")
    raw["date"] = raw["sched_dep"].dt.normalize()
    raw["period"] = np.select(
        [raw["sched_dep"] < pd.Timestamp("2020-03-01"), raw["sched_dep"] < pd.Timestamp("2023-05-01")],
        ["Pre-pandemic", "Pandemic"], default="Post-pandemic",
    )
    raw["airline_grouped"] = raw["airline"].map({"AZU": "Azul", "GLO": "Gol", "TAM": "TAM"}).fillna("Others")
    raw["seats"] = pd.to_numeric(raw["seats"], errors="coerce")

    scope = raw["is_br_origin"]
    flow.append(("Brazilian-origin planned legs", int(scope.sum())))
    flow.append(("  of which non-regular DI (excluded)", int((scope & ~raw["is_regular"]).sum())))
    flow.append(("  of which regular with status not informed (excluded)", int((scope & raw["is_regular"] & ~raw["known_status"]).sum())))
    final = scope & raw["is_regular"] & raw["known_status"]
    flow.append(("Analysis population: Brazilian-origin regular legs with a final status", int(final.sum())))
    flow.append(("  cancelled", int((final & raw["is_cancelled"]).sum())))
    pd.DataFrame(flow, columns=["step", "records"]).to_csv(OUT / "inclusion_flow.csv", index=False)
    pd.DataFrame({"exact_duplicates_removed": exact_by_year, "extra_outcome_records_dropped": conflicts_by_year}).fillna(0).astype(int).to_csv(
        OUT / "duplicates_by_year.csv")
    fixed_by_file.rename_axis("source_file").reset_index(name="scheduled_dates_restored").to_csv(OUT / "scheduled_dates_restored_by_file.csv", index=False)
    (OUT / "harmonization_notes.json").write_text(json.dumps({
        "identity_groups_with_status_conflict": status_conflicts,
        "brazilian_origin_rows_without_zone_defaulted_to_sao_paulo": missing_zone,
        "unmapped_cancellation_text_rows": int(unmapped.sum()),
        "missing_schedule_cancelled_records": missing_cancelled,
        "missing_schedule_brazilian_regular_cancelled_records": missing_br_regular_cancelled,
    }, indent=1), encoding="utf-8")

    keep = ["airline", "carrier_name", "flight_no", "di", "line_type", "aircraft_model", "seats", "origin", "dest", "sched_dep", "sched_date_restored",
            "sched_dep_local", "real_dep", "status", "just_text", "justification", "source_file", "file_month", "is_br_origin",
            "is_br_dest", "is_regular", "known_status", "is_cancelled", "origin_zone", "year", "month", "date", "day_of_week",
            "day_type", "departure_hour", "departure_sin", "departure_cos", "departure_period", "day_of_week_brt", "day_type_brt",
            "departure_hour_brt", "departure_period_brt", "airline_grouped", "period"]
    out = raw[keep].copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype("string")
    out.to_parquet(CACHE, index=False)
    for step, n in flow:
        print(f"{n:>12,}  {step}")
    print(json.loads((OUT / "harmonization_notes.json").read_text()))


if __name__ == "__main__":
    main()
