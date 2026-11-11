import numpy as np
import pandas as pd
import pytest

from src.event_study import abnormal_returns, cars, event_day_position, fit_ols, significance_tests


@pytest.fixture
def market_data():
    """400 trading days where the stock follows a known market model."""
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2024-01-01", periods=400)
    mkt = rng.normal(0.0005, 0.01, len(days))
    stock = 0.0002 + 1.5 * mkt + rng.normal(0, 0.002, len(days))
    return pd.DataFrame({"ACME": stock, "SPY": mkt}, index=days)


def test_event_day_before_close_is_same_day():
    days = pd.bdate_range("2024-03-01", periods=10)
    assert days[event_day_position(days, pd.Timestamp("2024-03-05 09:15"))] == pd.Timestamp("2024-03-05")


def test_event_day_after_close_is_next_trading_day():
    days = pd.bdate_range("2024-03-01", periods=10)  # 2024-03-08 is a Friday
    assert days[event_day_position(days, pd.Timestamp("2024-03-08 16:30"))] == pd.Timestamp("2024-03-11")


def test_event_day_beyond_data_is_none():
    days = pd.bdate_range("2024-03-01", periods=5)
    assert event_day_position(days, pd.Timestamp("2025-01-01 10:00")) is None


def test_fit_ols_recovers_parameters(market_data):
    fit = fit_ols(market_data["ACME"].to_numpy(), market_data[["SPY"]].to_numpy())
    assert fit.params[1] == pytest.approx(1.5, abs=0.05)
    assert fit.sigma == pytest.approx(0.002, rel=0.15)


def test_injected_shock_shows_up_as_car(market_data):
    day0 = market_data.index[300]
    market_data.loc[day0, "ACME"] -= 0.05
    events = pd.DataFrame({"event_id": ["e1"], "ticker": ["ACME"], "accepted": [day0 + pd.Timedelta(hours=9)]})

    ar = abnormal_returns(events, market_data)
    assert set(ar["rel_day"]) == set(range(-10, 11))
    car_01 = cars(ar, (0, 1)).loc["e1", "car"]
    assert car_01 == pytest.approx(-0.05, abs=0.01)
    car_pre = cars(ar, (-5, -1)).loc["e1", "car"]
    assert abs(car_pre) < 0.01


def test_events_without_enough_history_are_skipped(market_data):
    events = pd.DataFrame(
        {"event_id": ["early", "unknown"], "ticker": ["ACME", "NOPE"], "accepted": [market_data.index[50]] * 2}
    )
    assert abnormal_returns(events, market_data).empty


def test_cars_drops_events_with_missing_days():
    ar = pd.DataFrame({"event_id": ["a", "a", "b"], "rel_day": [0, 1, 0], "ar": [0.01, 0.02, 0.03], "ar_var": [1e-4] * 3})
    result = cars(ar, (0, 1))
    assert list(result.index) == ["a"]
    assert result.loc["a", "car"] == pytest.approx(0.03)


def test_significance_tests_detect_negative_mean():
    rng = np.random.default_rng(1)
    c = pd.DataFrame({"car": rng.normal(-0.03, 0.02, 40)})
    c["scar"] = c["car"] / 0.02
    result = significance_tests(c)
    assert result["mean_car"] < 0
    assert result["p_t"] < 0.01 and result["p_bmp"] < 0.01 and result["p_wilcoxon"] < 0.01
