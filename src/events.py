"""Turn raw EDGAR search hits into one row per cybersecurity incident.

For each incident we keep the first Item 1.05 disclosure, its SEC acceptance time,
the ticker, follow-up filings, and the text of the Item 1.05 section.

Run from the project root (after src.edgar):
    python -m src.events
"""
import re
import time
import warnings
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from src.edgar import HEADERS

HITS_PATH = "data/raw/edgar_item105_hits.csv"
FILINGS_DIR = Path("data/raw/filings")
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
OUTPUT_PATH = "data/processed/events.csv"

FOLLOWUP_WINDOW_DAYS = 60  # a new 8-K within this window is a follow-up, not a new incident
AMENDMENT_WINDOW_DAYS = 365  # an 8-K/A must follow an original within this window
# The SEC list sometimes puts a note or preferred ticker before the common stock.
TICKER_OVERRIDES = {"0001593538": "NAVI"}  # Navient: JSM is a senior note
# Phrases showing the incident was already public before this Item 1.05 filing.
PRIOR_DISCLOSURE = re.compile(
    r"previously (?:disclosed|reported|announced)|as disclosed in|voluntary disclosure|prior 8-K",
    flags=re.IGNORECASE,
)

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


def get(url: str) -> requests.Response:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    time.sleep(0.15)  # stay under the SEC's 10 requests/second limit
    return response


def cached_get(url: str, path: Path) -> str:
    """Download `url` once and reuse the local copy afterwards."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(get(url).text, encoding="utf-8")
    return path.read_text(encoding="utf-8")


def filing_folder(cik: str, accession: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}"


def acceptance_time(cik: str, accession: str) -> pd.Timestamp:
    """SEC acceptance timestamp (US Eastern), read from the filing index page."""
    html = cached_get(
        f"{filing_folder(cik, accession)}/{accession}-index.htm",
        FILINGS_DIR / f"{accession}-index.htm",
    )
    match = re.search(r"Accepted</div>\s*<div class=\"info\">([\d\- :]+)</div>", html)
    return pd.Timestamp(match.group(1)) if match else pd.NaT


def item_105_text(cik: str, accession: str, document: str) -> str:
    """Plain text of the Item 1.05 section of the filing's main document."""
    html = cached_get(f"{filing_folder(cik, accession)}/{document}", FILINGS_DIR / f"{accession}.htm")
    text = BeautifulSoup(html, "lxml").get_text(" ")
    text = re.sub(r"\s+", " ", text.replace("\xa0", " "))
    start = re.search(r"Item\s*1\.05", text, flags=re.IGNORECASE)
    if not start:
        return text
    rest = text[start.start():]
    # The section ends at the next item heading ("Item 7.01 Regulation FD ..."), not at a
    # cross-reference inside a sentence ("as disclosed in Item 8.01 of ..."). Legal
    # boilerplate on forward-looking statements is cut off as well.
    end = re.search(
        r"Item\s*(?!1\.05)\d\.\d\d\.?\s+[A-Z]|SIGNATURES?\b|Forward[- ]Looking Statements|Cautionary (?:Note|Statement)",
        rest[10:],
    )
    return rest[: end.start() + 10] if end else rest


def ticker_map() -> dict[str, str]:
    """CIK -> current ticker from the SEC's company list."""
    data = get(TICKERS_URL).json()
    mapping = {}
    for row in data.values():
        mapping.setdefault(str(row["cik_str"]).zfill(10), row["ticker"])
    return mapping


def ticker_from_name(display_name: str) -> str | None:
    """Fallback: 'ACME CORP  (ACME, ACME-PA)  (CIK ...)' -> 'ACME'."""
    match = re.search(r"\(([A-Z0-9.\-]+)(?:,[^)]*)?\)\s*\(CIK", display_name)
    return match.group(1) if match else None


def assign_incidents(filings: pd.DataFrame) -> pd.DataFrame:
    """Label each filing with an incident id; originals start incidents, others are follow-ups."""
    filings = filings.sort_values(["cik", "file_date", "form"]).copy()
    incident_ids, current = [], {}
    for row in filings.itertuples():
        start = current.get(row.cik)
        gap = (row.file_date - start[1]).days if start else None
        if row.form == "8-K" and (start is None or gap > FOLLOWUP_WINDOW_DAYS):
            current[row.cik] = (row.accession, row.file_date)
            incident_ids.append(row.accession)
        elif start is not None and gap <= AMENDMENT_WINDOW_DAYS:
            incident_ids.append(start[0])
        else:
            incident_ids.append(None)  # amendment whose original we don't observe
    filings["incident_id"] = incident_ids
    return filings


def build_events(hits: pd.DataFrame) -> pd.DataFrame:
    filings = hits[hits["is_item_105"]].copy()
    filings["file_date"] = pd.to_datetime(filings["file_date"])
    filings = assign_incidents(filings)
    orphans = filings["incident_id"].isna().sum()
    filings = filings.dropna(subset=["incident_id"])

    followups = filings[filings["accession"] != filings["incident_id"]]
    followup_stats = followups.groupby("incident_id").agg(
        n_followups=("accession", "size"), first_followup=("file_date", "min")
    )

    events = filings[filings["accession"] == filings["incident_id"]].copy()
    events = events.join(followup_stats, on="incident_id")
    events["n_followups"] = events["n_followups"].fillna(0).astype(int)
    events["days_to_followup"] = (events["first_followup"] - events["file_date"]).dt.days
    events = events.drop(columns=["first_followup", "incident_id", "is_item_105"])

    tickers = ticker_map()
    events["ticker"] = [
        TICKER_OVERRIDES.get(c) or tickers.get(c) or ticker_from_name(n)
        for c, n in zip(events["cik"], events["company"])
    ]
    events["company"] = events["company"].str.replace(r"\s*\(.*$", "", regex=True)
    events["confounded"] = events["items"].str.contains(r"2\.02")  # earnings in the same filing

    print(f"Downloading {len(events)} filings from EDGAR (cached after the first run)...")
    events["accepted"] = [acceptance_time(c, a) for c, a in zip(events["cik"], events["accession"])]
    events["after_close"] = events["accepted"].dt.hour >= 16
    events["text"] = [item_105_text(c, a, d) for c, a, d in zip(events["cik"], events["accession"], events["document"])]
    events["n_words"] = events["text"].str.split().str.len()
    # Firms say "As previously disclosed..." in the opening sentence; later matches are boilerplate.
    events["prior_disclosure"] = events["text"].str[:400].str.contains(PRIOR_DISCLOSURE)

    print(f"Dropped {orphans} amendments without an observed original filing")
    return events.sort_values("file_date").reset_index(drop=True)


def main() -> None:
    hits = pd.read_csv(HITS_PATH, dtype={"cik": str, "sic": "Int64"})
    events = build_events(hits)
    events.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(events)} incidents to {OUTPUT_PATH}")
    print(f"After market close: {events['after_close'].sum()}, with follow-ups: {(events['n_followups'] > 0).sum()}")
    print(f"Already disclosed earlier: {events['prior_disclosure'].sum()}, missing ticker: {events['ticker'].isna().sum()}")


if __name__ == "__main__":
    main()
