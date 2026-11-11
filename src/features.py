"""Measure disclosure content and firm characteristics for each incident.

Text features come from two transparent methods:
- keyword flags for incident characteristics (ransomware, data theft, disruption, ...)
- Loughran-McDonald finance dictionary word shares (negative, uncertainty, litigious)

LLM-extracted features (notebooks/llm_extraction.ipynb, run on Google Colab) are merged
separately in src.analysis when data/processed/llm_features.csv exists.

Run from the project root (after src.events and src.prices):
    python -m src.features
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pysentiment2

from src.event_study import ESTIMATION_WINDOW, event_day_position

EVENTS_PATH = "data/processed/events.csv"
DOLLAR_VOLUME_PATH = "data/processed/dollar_volume.parquet"
OUTPUT_PATH = "data/processed/features.csv"
LM_PATH = Path(pysentiment2.__file__).parent / "static" / "LM.csv"

KEYWORDS = {
    "ransomware": r"ransomware|encrypt(?:ed|ion) (?:of )?(?:certain )?(?:systems|files|data)|ransom",
    "data_exfiltration": r"exfiltrat|personal(?:ly identifiable)? information|customer data|personal data|data (?:was|were) (?:accessed|acquired|stolen)",
    "operations_disrupted": r"disrupt|offline|shut(?: ?down)|took (?:certain )?systems|impaired|interruption|outage",
    "nation_state": r"nation[- ]state|state[- ]sponsored|foreign government|threat actor (?:affiliated|associated) with",
    "third_party": r"third[- ]party (?:vendor|service provider|provider|software)|vendor",
    "law_enforcement": r"law enforcement|FBI|Federal Bureau",
    # Firm says it cannot yet judge the financial impact (the "not yet determined" language).
    "impact_undetermined": r"not yet determined|has not determined|not (?:yet )?(?:able|possible) to (?:determine|estimate|assess)|unable to (?:determine|estimate|assess)|too early to",
    # Firm explicitly says the impact is not (reasonably likely to be) material.
    "impact_not_material": r"(?:not|no longer) (?:had|have|has had|believe[sd]?|reasonably likely)[^.]{0,80}material(?:ly)? (?:impact|adverse)",
}


def keyword_flags(text: str) -> dict[str, bool]:
    return {name: bool(re.search(pattern, text, flags=re.IGNORECASE)) for name, pattern in KEYWORDS.items()}


def load_lm_dictionary() -> dict[str, set[str]]:
    lm = pd.read_csv(LM_PATH, usecols=["Word", "Negative", "Uncertainty", "Litigious"])
    return {cat.lower(): set(lm.loc[lm[cat] != 0, "Word"].str.lower()) for cat in ["Negative", "Uncertainty", "Litigious"]}


def lm_shares(text: str, dictionary: dict[str, set[str]]) -> dict[str, float]:
    """Share of words in each Loughran-McDonald category, in percent."""
    words = re.findall(r"[a-z]+", text.lower())
    n = max(len(words), 1)
    return {f"lm_{cat}": 100 * sum(w in vocab for w in words) / n for cat, vocab in dictionary.items()}


def pre_event_dollar_volume(events: pd.DataFrame, dollar_volume: pd.DataFrame) -> pd.Series:
    """Log average daily dollar volume over the estimation window (a size/liquidity proxy)."""
    values = {}
    for ev in events.itertuples():
        if ev.ticker not in dollar_volume:
            continue
        day0 = event_day_position(dollar_volume.index, ev.accepted)
        if day0 is None:
            continue
        window = dollar_volume[ev.ticker].iloc[max(day0 + ESTIMATION_WINDOW[0], 0) : day0 + ESTIMATION_WINDOW[1] + 1]
        if window.count() > 0 and window.mean() > 0:
            values[ev.accession] = np.log(window.mean())
    return pd.Series(values, name="log_dollar_volume")


def industry(sic) -> str:
    if pd.isna(sic):
        return "other"
    sic = int(sic)
    if 6000 <= sic <= 6799:
        return "finance"
    if 8000 <= sic <= 8099 or 2830 <= sic <= 2836 or 3841 <= sic <= 3851:
        return "health"
    if 3570 <= sic <= 3579 or 3660 <= sic <= 3679 or 7370 <= sic <= 7379 or 4800 <= sic <= 4899:
        return "tech_telecom"
    if 2000 <= sic <= 3999:
        return "manufacturing"
    return "other"


def build_features(events: pd.DataFrame, dollar_volume: pd.DataFrame) -> pd.DataFrame:
    dictionary = load_lm_dictionary()
    feats = pd.DataFrame(
        [{**keyword_flags(t), **lm_shares(t, dictionary)} for t in events["text"]],
        index=events["accession"],
    )
    feats["log_words"] = np.log(events.set_index("accession")["n_words"])
    feats["industry"] = events.set_index("accession")["sic"].map(industry)
    feats = feats.join(pre_event_dollar_volume(events, dollar_volume))
    return feats.rename_axis("accession").reset_index()


def main() -> None:
    events = pd.read_csv(EVENTS_PATH, parse_dates=["accepted"], dtype={"cik": str})
    dollar_volume = pd.read_parquet(DOLLAR_VOLUME_PATH)
    feats = build_features(events, dollar_volume)
    feats.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved features for {len(feats)} incidents to {OUTPUT_PATH}")
    print(feats.select_dtypes("bool").mean().round(2).to_string())
    print(feats.filter(like="lm_").describe().round(2).loc[["mean", "std"]].to_string())
    print(feats["industry"].value_counts().to_string())


if __name__ == "__main__":
    main()
