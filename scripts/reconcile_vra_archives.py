"""Reconcile the SIROS-archive VRA with the legacy open-data VRA files.

Both archives are compared on Brazilian-origin regular legs (DI 0) with a scheduled
departure, month by month: record counts, cancellations, the share of legs matched on
carrier, flight number, origin, destination and scheduled departure, and the status
agreement among matched legs. February and March 2020 exist only in the SIROS archive.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from generate_vra_results import OUT_DIR, load_legacy_vra, load_vra

KEY = ["airline", "flight_no", "origin", "dest", "sched_dep"]


def main() -> None:
    new = load_vra(scope="all")
    new = new[new["is_br_origin"] & new["is_regular"]]
    old = load_legacy_vra()
    old = old[old["is_br_origin"] & old["di"].astype(str).isin(["0", "0.0"])]
    old = old.drop_duplicates(subset=[c for c in old.columns if c != "source_file"])
    for d in (new, old):
        d["ym"] = d["sched_dep"].dt.year * 100 + d["sched_dep"].dt.month
    rows = []
    for ym in sorted(set(new["ym"]) | set(old["ym"])):
        n, o = new[new["ym"] == ym], old[old["ym"] == ym]
        # Legacy rows repeating a key are counted, then matched once.
        o_unique = o.drop_duplicates(subset=KEY)
        m = n.drop_duplicates(subset=KEY)[KEY + ["status"]].merge(o_unique[KEY + ["status"]], on=KEY, how="inner", suffixes=("_new", "_old"))
        rows.append({
            "ym": ym, "siros_legs": len(n), "legacy_legs": len(o), "legacy_repeated_keys": len(o) - len(o_unique),
            "siros_final_status_legs": int(n["known_status"].sum()),
            "siros_cancelled": int(n["is_cancelled"].sum()), "legacy_cancelled": int(o["is_cancelled"].sum()),
            "matched_legs": len(m), "matched_share_of_siros_pct": len(m) / len(n) * 100 if len(n) else float("nan"),
            "status_agreement_pct": (m["status_new"] == m["status_old"]).mean() * 100 if len(m) else float("nan"),
        })
    t = pd.DataFrame(rows)
    expected = {y * 100 + m for y in range(2015, 2025) for m in range(1, 13)}
    empty = sorted(expected - set(t.loc[t["siros_legs"] > 1_000, "ym"]))
    assert not empty, f"SIROS months with no or very few legs: {empty}"
    t.to_csv(OUT_DIR / "archive_reconciliation_monthly.csv", index=False)
    t["year"] = t["ym"] // 100
    y = t.groupby("year")[["siros_legs", "legacy_legs", "legacy_repeated_keys", "siros_cancelled", "legacy_cancelled", "matched_legs"]].sum()
    y["matched_share_of_siros_pct"] = y["matched_legs"] / y["siros_legs"] * 100
    agree = t.dropna(subset=["status_agreement_pct"])
    y["status_agreement_pct"] = agree.assign(w=agree["status_agreement_pct"] * agree["matched_legs"]).groupby("year")["w"].sum() / y["matched_legs"]
    y.to_csv(OUT_DIR / "archive_reconciliation_yearly.csv")
    print(y.round(2).to_string())


if __name__ == "__main__":
    main()
