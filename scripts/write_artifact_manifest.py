"""Write ARTIFACTS.json: size and SHA-256 of every archived result, figure and reference table.

Run only when an archived artifact is deliberately replaced; the integrity tests check
the committed files against this manifest.
"""

from __future__ import annotations

import hashlib
import json

from paths import ROOT

FOLDERS = ("results", "figures", "reference")
SUFFIXES = {".csv", ".tex", ".pdf", ".json"}
RAW_REFERENCE_PREFIXES = ("ourairports_", "rotaer_2")  # raw inputs, not versioned


def main() -> None:
    records = {}
    for folder in FOLDERS:
        for path in sorted((ROOT / folder).rglob("*")):
            if not path.is_file() or path.suffix not in SUFFIXES or path.name.startswith(RAW_REFERENCE_PREFIXES):
                continue
            data = path.read_bytes()
            records[path.relative_to(ROOT).as_posix()] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    (ROOT / "ARTIFACTS.json").write_text(json.dumps(records, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"listed {len(records)} artifacts")


if __name__ == "__main__":
    main()
