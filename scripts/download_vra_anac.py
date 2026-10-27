"""Download the VRA files published on ANAC's open-data portal (legacy archive).

The analysis uses the SIROS archive (``download_vra_siros.py``). The portal files are
needed for two purposes only: the legs scheduled from 25 to 31 March 2018, which the
SIROS file for March 2018 omits, and the reconciliation of the two archives
(``reconcile_vra_archives.py``). Files are written to ``<data>/legacy_portal/<year>/``.
"""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
from time import sleep
from urllib.parse import unquote, urljoin, urlparse
from urllib.parse import quote
from urllib.request import Request, urlopen

from paths import LEGACY_DIR


BASE_URL = (
    "https://sistemas.anac.gov.br/dadosabertos/"
    "Voos%20e%20opera%C3%A7%C3%B5es%20a%C3%A9reas/"
    "Voo%20Regular%20Ativo%20%28VRA%29/"
)

MONTH_DIRS = [
    "01 - Janeiro",
    "02 - Fevereiro",
    "03 - Março",
    "04 - Abril",
    "05 - Maio",
    "06 - Junho",
    "07 - Julho",
    "08 - Agosto",
    "09 - Setembro",
    "10 - Outubro",
    "11 - Novembro",
    "12 - Dezembro",
]


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.links.append(href)


def fetch_text(url: str, retries: int = 4) -> str:
    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=60) as response:
                return response.read().decode("utf-8", errors="replace")
        except Exception as exc:
            if attempt == retries:
                raise
            print(f"retry listing {attempt}/{retries} for {url}: {exc}")
            sleep(2 * attempt)
    raise RuntimeError(f"failed to fetch listing: {url}")


def links_at(url: str) -> list[str]:
    parser = LinkParser()
    parser.feed(fetch_text(url))
    return [urljoin(url, href) for href in parser.links if href != "../"]


def remote_size(url: str) -> int | None:
    req = Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(req, timeout=60) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except Exception:
        return None


def download(url: str, target: Path, retries: int = 3) -> None:
    expected = remote_size(url)
    if target.exists() and expected is not None and target.stat().st_size == expected:
        print(f"skip {target} ({expected:,} bytes)")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=120) as response, tmp.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            if expected is not None and tmp.stat().st_size != expected:
                raise OSError(f"expected {expected} bytes, got {tmp.stat().st_size}")
            tmp.replace(target)
            print(f"downloaded {target} ({target.stat().st_size:,} bytes)")
            return
        except Exception as exc:
            if attempt == retries:
                raise
            print(f"retry {attempt}/{retries} for {url}: {exc}")
            sleep(2 * attempt)


def file_name_from_url(url: str) -> str:
    return unquote(Path(urlparse(url).path).name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ANAC VRA monthly CSV files.")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--output", type=Path, default=LEGACY_DIR)
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Build known monthly VRA CSV URLs directly instead of crawling listings.",
    )
    args = parser.parse_args()

    selected = list(range(args.start_year, args.end_year + 1))
    if not selected:
        raise SystemExit("No years selected.")

    total = 0
    if args.direct:
        for year in selected:
            print(f"year {year}")
            for month, month_dir in enumerate(MONTH_DIRS, start=1):
                name = f"VRA_{year}{month}.csv"
                month_path = quote(month_dir, safe="")
                csv_url = f"{BASE_URL}{year}/{month_path}/{name}"
                target = args.output / str(year) / name
                download(csv_url, target)
                total += 1
    else:
        year_urls = {
            int(file_name_from_url(u).strip("/")): u
            for u in links_at(BASE_URL)
            if file_name_from_url(u).strip("/").isdigit()
        }
        selected = [year for year in selected if year in year_urls]
        if not selected:
            raise SystemExit("No matching year directories found.")
        for year in selected:
            print(f"year {year}")
            for month_url in links_at(year_urls[year]):
                csv_urls = [u for u in links_at(month_url) if file_name_from_url(u).lower().endswith(".csv")]
                for csv_url in csv_urls:
                    name = file_name_from_url(csv_url)
                    target = args.output / str(year) / name
                    download(csv_url, target)
                    total += 1

    print(f"done: processed {total} CSV links into {args.output}")


if __name__ == "__main__":
    main()
