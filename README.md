# Cyber Breach Disclosures and Market Reaction

How do stock markets react when firms disclose cybersecurity incidents under the SEC's
Form 8-K Item 1.05 rule (effective December 2023)? And does the *content* of a
disclosure (breach type, severity, vagueness) explain the size of the reaction?

## Approach
1. Collect all 8-K filings reporting Item 1.05 from SEC EDGAR.
2. Build an event dataset (firm, ticker, disclosure date).
3. Estimate abnormal returns with a market model / Fama-French event study.
4. Extract disclosure features with an LLM (Google Colab) and a finance dictionary.
5. Explain cross-sectional variation in abnormal returns with regressions.

## Structure
```
src/         pipeline code
tests/       unit tests
notebooks/   exploration + Colab LLM notebook
data/        raw and processed data (not committed; re-created by the pipeline)
figures/     output charts
paper/       research note
```

## Setup
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

*Work in progress.*
