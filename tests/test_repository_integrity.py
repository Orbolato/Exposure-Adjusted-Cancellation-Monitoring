"""Repository checks that need no source data (run in continuous integration)."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOI = "10.1016/j.cstp.2026.102001"
CANONICAL = "https://github.com/Orbolato/Exposure-Adjusted-Cancellation-Monitoring"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_archived_artifacts_are_exact() -> None:
    """Archived results, figures and reference tables match ARTIFACTS.json byte for byte."""
    artifacts = json.loads((ROOT / "ARTIFACTS.json").read_text(encoding="utf-8"))
    assert artifacts, "ARTIFACTS.json is empty"
    for relative, record in artifacts.items():
        path = ROOT / relative
        assert path.is_file(), f"Missing artifact: {relative}"
        assert path.stat().st_size == record["bytes"], f"Unexpected size: {relative}"
        assert sha256(path) == record["sha256"], f"Unexpected contents: {relative}"


def test_every_archived_artifact_is_listed() -> None:
    artifacts = set(json.loads((ROOT / "ARTIFACTS.json").read_text(encoding="utf-8")))
    on_disk = {
        p.relative_to(ROOT).as_posix()
        for folder in ("results", "figures", "reference")
        for p in (ROOT / folder).rglob("*")
        if p.is_file() and p.suffix in {".csv", ".tex", ".pdf", ".json"}
        and not p.name.startswith(("ourairports_", "rotaer_2"))
    }
    assert on_disk == artifacts, f"Unlisted: {sorted(on_disk - artifacts)}; missing: {sorted(artifacts - on_disk)}"


def test_tables_regenerate_exactly(tmp_path: Path) -> None:
    """All table bodies of the article are rebuilt from results/ without any source data."""
    env = dict(os.environ, CANCELLATIONS_OUTPUT_DIR=str(tmp_path))
    run = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_tables.py"), "--check"],
                         cwd=ROOT, env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert len(list((tmp_path / "tables").glob("*.tex"))) == len(list((ROOT / "results" / "tables").glob("*.tex"))) == 23


def test_source_data_are_not_versioned() -> None:
    """Only the README and the manifests are versioned under data/."""
    try:
        tracked = subprocess.run(["git", "ls-files", "data"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        return  # not a Git checkout
    allowed = {"data/README.md"} | {f"data/{d}/manifest.json" for d in ("siros", "legacy_portal", "bts", "anac_statistics")}
    assert set(tracked) <= allowed, f"Source files must not be committed: {sorted(set(tracked) - allowed)}"


def test_manifests_cover_the_study_inputs() -> None:
    def load(folder: str) -> list[dict]:
        return json.loads((ROOT / "data" / folder / "manifest.json").read_text(encoding="utf-8"))

    siros = load("siros")
    assert len(siros) == 122 and not any("error" in r for r in siros)
    assert {r["file"] for r in siros} >= {"VRA_2014_12.csv", "VRA_2018_03.csv", "VRA_2020_03.csv", "VRA_2025_01.csv"}
    assert len(load("legacy_portal")) == 118
    bts = load("bts")
    assert len(bts) == 72 and sum(r["rows"] for r in bts) == 39_238_943
    assert [r["file"] for r in load("anac_statistics")] == [f"{y}.csv" for y in range(2015, 2025)]


def test_citation_metadata() -> None:
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert CANONICAL in citation and DOI in citation
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"https://doi.org/{DOI}" in readme
