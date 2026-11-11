"""Download daily returns for the event firms, the market, and the Fama-French factors.

Run from the project root (after src.events):
    python -m src.prices
"""
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf

EVENTS_PATH = "data/processed/events.csv"
RETURNS_PATH = "data/processed/returns.parquet"
FACTORS_PATH = "data/processed/ff_factors.parquet"

MARKET_TICKER = "SPY"  # S&P 500 ETF, adjusted for dividends
HISTORY_DAYS = 450  # calendar days before the first event: covers a 250-trading-day estimation window


def download_returns(tickers: list[str], start: pd.Timestamp) -> pd.DataFrame:
    """Daily simple returns from dividend- and split-adjusted closing prices."""
    prices = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    return prices.pct_change(fill_method=None).iloc[1:]


def download_ff_factors(start: pd.Timestamp) -> pd.DataFrame:
    """Daily Fama-French 3 factors (in decimals) from Kenneth French's data library."""
    factors = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench", start=start)[0]
    if isinstance(factors.index, pd.PeriodIndex):
        factors.index = factors.index.to_timestamp()
    return factors / 100


def main() -> None:
    events = pd.read_csv(EVENTS_PATH, parse_dates=["file_date"])
    tickers = sorted(events["ticker"].dropna().unique())
    start = events["file_date"].min() - pd.Timedelta(days=HISTORY_DAYS)

    returns = download_returns(tickers + [MARKET_TICKER], start)
    missing = [t for t in tickers if t not in returns or returns[t].count() == 0]
    returns = returns.drop(columns=missing, errors="ignore")
    returns.to_parquet(RETURNS_PATH)
    print(f"Saved returns for {returns.shape[1] - 1} firms ({returns.index.min():%Y-%m-%d} to {returns.index.max():%Y-%m-%d})")
    print(f"No price data: {missing}")

    factors = download_ff_factors(start)
    factors.to_parquet(FACTORS_PATH)
    print(f"Saved Fama-French factors up to {factors.index.max():%Y-%m-%d}")


if __name__ == "__main__":
    main()
