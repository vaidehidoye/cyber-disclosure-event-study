"""Fetch 8-K filings that mention Item 1.05 (material cybersecurity incidents) from SEC EDGAR.

Run from the project root:
    python -m src.edgar
"""
import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
# The SEC blocks requests without a User-Agent that identifies you (name + email).
HEADERS = {"User-Agent": os.environ["SEC_USER_AGENT"]}
START_DATE = "2023-12-18"  # Item 1.05 became mandatory for most firms on this date
OUTPUT_PATH = "data/raw/edgar_item105_hits.csv"


def search_page(start: int, end_date: str) -> dict:
    """Return one page of full-text search results, beginning at result number `start`."""
    params = {
        "q": '"Item 1.05"',
        "forms": "8-K",
        "dateRange": "custom",
        "startdt": START_DATE,
        "enddt": end_date,
        "from": start,
    }
    response = requests.get(SEARCH_URL, params=params, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_all(end_date: str) -> list[dict]:
    """Collect every search hit by walking through the result pages."""
    hits = []
    start = 0
    while True:
        data = search_page(start, end_date)
        page = data["hits"]["hits"]
        total = data["hits"]["total"]["value"]
        hits.extend(page)
        if not page or len(hits) >= total:
            break
        start += len(page)
        time.sleep(0.2)  # SEC allows at most 10 requests per second
    return hits


def to_row(hit: dict) -> dict:
    """Flatten one search hit into the columns we care about."""
    s = hit["_source"]
    return {
        "accession": s["adsh"],  # unique ID of the filing
        "cik": s["ciks"][0],  # SEC company ID
        "company": s["display_names"][0],
        "form": s["form"],  # "8-K" or "8-K/A" (amendment)
        "file_date": s["file_date"],
        "items": ",".join(s["items"]),
        "is_item_105": "1.05" in s["items"],
        "sic": (s.get("sics") or [None])[0],  # industry code
        "document": hit["_id"].split(":")[1],
    }


def main() -> None:
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    hits = fetch_all(end_date)
    # One filing can contain several matching documents (e.g. the 8-K and an exhibit),
    # so keep one row per accession number.
    df = pd.DataFrame([to_row(h) for h in hits]).drop_duplicates("accession")
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} filings to {OUTPUT_PATH}")
    print(df["form"].value_counts())
    print(f"Filed under Item 1.05: {df['is_item_105'].sum()}")


if __name__ == "__main__":
    main()
