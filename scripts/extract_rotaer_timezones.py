"""Extract airport/state/UTC offset from the retained official DECEA directory.

Retrieve reference/rotaer_20260923.pdf from
https://aisweb.decea.mil.br/downloads/rotaer/rotaer_completo.pdf.
The PDF checksum is retained in airport_timezones_manifest.json.
"""
import re

import fitz
import pandas as pd

from paths import REFERENCE_DIR

rows = []
for i, page in enumerate(fitz.open(REFERENCE_DIR / "rotaer_20260923.pdf")):
    text = page.get_text()
    for match in re.finditer(r"\(\s*([A-Z0-9]{4})\s*\)\s*/", text):
        chunk = text[match.end():match.end()+200]
        info = re.search(r",\s*([A-Z]{2})\b.*?UTC\s*([-+]\d+)", chunk, re.S)
        if info:
            rows.append(dict(icao=match[1], uf=info[1], utc_offset=int(info[2]),
                             page=i+1, excerpt=chunk[:info.end()].replace("\n", " ")))
pd.DataFrame(rows).drop_duplicates("icao").to_csv(REFERENCE_DIR / "rotaer_timezones.csv", index=False)
