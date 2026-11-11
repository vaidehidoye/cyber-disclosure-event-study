"""Event study engine: abnormal returns, cumulative abnormal returns (CARs) and significance tests.

Conventions
- Day 0 is the first trading day on which the market could react to the filing: the
  acceptance date if the filing was accepted before the 4pm close, otherwise the next trading day.
- Normal returns are estimated over trading days [-250, -30] relative to day 0.
- Models: "market" (R_i = a + b R_m) or "ff3" (Fama-French three-factor, on excess returns).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

ESTIMATION_WINDOW = (-250, -30)
MIN_ESTIMATION_DAYS = 120
EVENT_WINDOW = (-10, 10)  # days for which abnormal returns are stored
MARKET_CLOSE_HOUR = 16


def event_day_position(trading_days: pd.DatetimeIndex, accepted: pd.Timestamp) -> int | None:
    """Index into `trading_days` of day 0 for a filing accepted at `accepted` (US Eastern)."""
    date = accepted.normalize()
    if accepted.hour >= MARKET_CLOSE_HOUR:
        date += pd.Timedelta(days=1)
    pos = trading_days.searchsorted(date)
    return int(pos) if pos < len(trading_days) else None


@dataclass
class Fit:
    params: np.ndarray
    sigma: float  # residual standard deviation
    X_est: np.ndarray  # estimation-window regressors (with constant)

    def prediction_variance(self, X_event: np.ndarray) -> np.ndarray:
        """Variance of each abnormal return, including estimation error in the parameters."""
        xtx_inv = np.linalg.inv(self.X_est.T @ self.X_est)
        leverage = np.einsum("ij,jk,ik->i", X_event, xtx_inv, X_event)
        return self.sigma**2 * (1 + leverage)


def fit_ols(y: np.ndarray, X: np.ndarray) -> Fit:
    X = np.column_stack([np.ones(len(X)), X])
    params, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ params
    sigma = np.sqrt(resid @ resid / (len(y) - X.shape[1]))
    return Fit(params, sigma, X)


def model_inputs(stock: pd.Series, market: pd.Series, factors: pd.DataFrame | None, model: str):
    """Dependent variable and regressors for the chosen normal-return model."""
    if model == "market":
        return stock, market.to_frame("mkt")
    if model == "ff3":
        rf = factors["RF"].reindex(stock.index)
        return stock - rf, factors[["Mkt-RF", "SMB", "HML"]].reindex(stock.index)
    raise ValueError(f"Unknown model: {model}")


def abnormal_returns(
    events: pd.DataFrame,
    returns: pd.DataFrame,
    market_col: str = "SPY",
    factors: pd.DataFrame | None = None,
    model: str = "market",
) -> pd.DataFrame:
    """Abnormal returns per event and relative day, in long format.

    `events` needs columns: event_id, ticker, accepted (Timestamp).
    Returns columns: event_id, rel_day, date, ar, ar_var.
    """
    trading_days = returns.index
    rows = []
    for ev in events.itertuples():
        if ev.ticker not in returns:
            continue
        day0 = event_day_position(trading_days, ev.accepted)
        if day0 is None or day0 + ESTIMATION_WINDOW[0] < 0:
            continue
        y, X = model_inputs(returns[ev.ticker], returns[market_col], factors, model)
        est = slice(day0 + ESTIMATION_WINDOW[0], day0 + ESTIMATION_WINDOW[1] + 1)
        est_data = pd.concat([y.iloc[est], X.iloc[est]], axis=1).dropna()
        if len(est_data) < MIN_ESTIMATION_DAYS:
            continue
        fit = fit_ols(est_data.iloc[:, 0].to_numpy(), est_data.iloc[:, 1:].to_numpy())

        win = slice(max(day0 + EVENT_WINDOW[0], 0), day0 + EVENT_WINDOW[1] + 1)
        win_data = pd.concat([y.iloc[win], X.iloc[win]], axis=1).dropna()
        X_event = np.column_stack([np.ones(len(win_data)), win_data.iloc[:, 1:].to_numpy()])
        ar = win_data.iloc[:, 0].to_numpy() - X_event @ fit.params
        rel_day = trading_days.get_indexer(win_data.index) - day0
        rows.append(
            pd.DataFrame(
                {
                    "event_id": ev.event_id,
                    "rel_day": rel_day,
                    "date": win_data.index,
                    "ar": ar,
                    "ar_var": fit.prediction_variance(X_event),
                }
            )
        )
    if not rows:
        return pd.DataFrame(columns=["event_id", "rel_day", "date", "ar", "ar_var"])
    return pd.concat(rows, ignore_index=True)


def cars(ar: pd.DataFrame, window: tuple[int, int]) -> pd.DataFrame:
    """CAR and standardized CAR per event over an inclusive window of relative days.

    Events missing any day in the window are dropped.
    """
    lo, hi = window
    in_window = ar[ar["rel_day"].between(lo, hi)]
    grouped = in_window.groupby("event_id").agg(car=("ar", "sum"), var=("ar_var", "sum"), days=("ar", "size"))
    grouped = grouped[grouped["days"] == hi - lo + 1]
    grouped["scar"] = grouped["car"] / np.sqrt(grouped["var"])
    return grouped[["car", "scar"]]


def significance_tests(c: pd.DataFrame) -> dict:
    """Is the average CAR different from zero? Parametric and non-parametric tests."""
    n = len(c)
    t_stat, t_p = stats.ttest_1samp(c["car"], 0)
    # Boehmer, Musumeci & Poulsen (1991): cross-sectional test on standardized CARs,
    # robust to event-induced changes in volatility.
    bmp_t = c["scar"].mean() / (c["scar"].std(ddof=1) / np.sqrt(n))
    bmp_p = 2 * stats.t.sf(abs(bmp_t), df=n - 1)
    wilcoxon_p = stats.wilcoxon(c["car"]).pvalue
    return {
        "n": n,
        "mean_car": c["car"].mean(),
        "median_car": c["car"].median(),
        "pct_negative": (c["car"] < 0).mean(),
        "t": t_stat,
        "p_t": t_p,
        "bmp_t": bmp_t,
        "p_bmp": bmp_p,
        "p_wilcoxon": wilcoxon_p,
    }
