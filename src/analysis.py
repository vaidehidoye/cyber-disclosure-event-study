"""Main results: average market reaction, figures, cross-sectional regressions and robustness checks.

Run from the project root (after src.events, src.prices and src.features):
    python -m src.analysis
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col

from src.event_study import ESTIMATION_WINDOW, abnormal_returns, cars, event_day_position, significance_tests

EVENTS_PATH = "data/processed/events.csv"
FEATURES_PATH = "data/processed/features.csv"
LLM_FEATURES_PATH = Path("data/processed/llm_features.csv")
RETURNS_PATH = "data/processed/returns.parquet"
FACTORS_PATH = "data/processed/ff_factors.parquet"
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

WINDOWS = [(-5, -2), (-1, 1), (0, 0), (0, 1), (0, 5), (0, 10)]
MAIN_WINDOW = (0, 1)
PLACEBO_DRAWS = 200
# Data-quality filter: a stock with many daily moves beyond +/-50% has a corrupted price series
# (typically reverse splits missing from Yahoo's adjustments), so its abnormal returns are noise.
EXTREME_RETURN, MAX_EXTREME_DAYS = 0.5, 3
SEED = 42

# Default dataviz palette (light surface): categorical slot 1 blue, plus text/grid inks.
BLUE = "#2a78d6"
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load_data():
    events = pd.read_csv(EVENTS_PATH, parse_dates=["accepted", "file_date"], dtype={"cik": str})
    events = events.merge(pd.read_csv(FEATURES_PATH), on="accession")
    if LLM_FEATURES_PATH.exists():
        llm = pd.read_csv(LLM_FEATURES_PATH).add_prefix("llm_").rename(columns={"llm_accession": "accession"})
        events = events.merge(llm, on="accession", how="left")
    events["event_id"] = events["accession"]
    returns = pd.read_parquet(RETURNS_PATH)
    factors = pd.read_parquet(FACTORS_PATH)
    return events, returns, factors


def unreliable_tickers(returns: pd.DataFrame) -> list[str]:
    extreme_days = (returns.abs() > EXTREME_RETURN).sum()
    return sorted(extreme_days[extreme_days > MAX_EXTREME_DAYS].index)


def window_label(w: tuple[int, int]) -> str:
    return f"[{w[0]:+d}, {w[1]:+d}]"


def format_tests(rows: dict) -> pd.DataFrame:
    table = pd.DataFrame(rows).T
    table["n"] = table["n"].astype(int)
    return table


def to_markdown(df: pd.DataFrame, path: Path, pct_cols=("mean_car", "median_car", "pct_negative")) -> None:
    out = df.copy()
    for col in out.columns:
        if col in pct_cols:
            out[col] = (100 * out[col]).map("{:.2f}%".format)
        elif col.startswith("p_"):
            out[col] = out[col].map("{:.4f}".format)
        elif out[col].dtype.kind == "f":
            out[col] = out[col].map("{:.2f}".format)
    path.write_text(out.to_markdown() + "\n")


# --- Figures ---------------------------------------------------------------------------


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def caar_path(ar: pd.DataFrame) -> pd.DataFrame:
    """Cumulative average abnormal return from day -10, with a 95% cross-sectional band."""
    wide = ar.pivot(index="event_id", columns="rel_day", values="ar").dropna()
    cum = wide.cumsum(axis=1)
    se = cum.std(ddof=1) / np.sqrt(len(cum))
    return pd.DataFrame({"caar": cum.mean(), "lo": cum.mean() - 1.96 * se, "hi": cum.mean() + 1.96 * se, "n": len(cum)})


def plot_caar(data: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    style_axes(ax)
    ax.axhline(0, color=INK_MUTED, linewidth=0.8)
    ax.axvline(0, color=INK_MUTED, linewidth=0.8, linestyle=":")
    ax.fill_between(data.index, 100 * data["lo"], 100 * data["hi"], color=BLUE, alpha=0.12, linewidth=0)
    ax.plot(data.index, 100 * data["caar"], color=BLUE, linewidth=2)
    for day in [-1, data.index[-1]]:
        value = 100 * data.loc[day, "caar"]
        ax.plot(day, value, "o", color=BLUE, markersize=6, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f"{value:.1f}%", (day, value), xytext=(8, 4), textcoords="offset points", fontsize=9, color=INK)
    ax.set_xlabel("Trading days relative to disclosure (day 0)", color=INK, fontsize=10)
    ax.set_ylabel("Cumulative abnormal return (%)", color=INK, fontsize=10)
    ax.set_xlim(-10, 12)
    ax.set_xticks(range(-10, 11, 2))
    ax.set_title("Stock prices fall after a cybersecurity incident disclosure", loc="left", color=INK, fontsize=12)
    fig.text(0.01, 0.01, f"First Item 1.05 disclosures, n={data['n'].iloc[0]}. Market model, SPY benchmark. "
             "Shaded: 95% confidence band.", fontsize=8, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_car_distribution(c: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor=SURFACE)
    style_axes(ax)
    values = 100 * c["car"].clip(-20, 20)
    bins = np.arange(2 * np.floor(values.min() / 2), 2 * np.ceil(values.max() / 2) + 2, 2)
    ax.hist(values, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=2)
    ax.set_xticks(bins[::2])
    ax.xaxis.set_major_formatter("{x:.0f}")
    ax.axvline(0, color=INK_MUTED, linewidth=0.8)
    ax.axvline(values.median(), color=INK, linewidth=1, linestyle="--")
    ax.annotate(f"median {values.median():.1f}%", (values.median(), ax.get_ylim()[1] * 0.92),
                xytext=(-6, 0), textcoords="offset points", ha="right", fontsize=9, color=INK)
    ax.set_xlabel("CAR over days [0, +1] (%), clipped at ±20%", color=INK, fontsize=10)
    ax.set_ylabel("Incidents", color=INK, fontsize=10)
    ax.set_title(f"{(c['car'] < 0).mean():.0%} of firms lose value in the two days after disclosure",
                 loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# --- Cross-sectional regressions -------------------------------------------------------


def winsorize(s: pd.Series, q: float = 0.05) -> pd.Series:
    return s.clip(s.quantile(q), s.quantile(1 - q))


def regressions(data: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """What explains the size of the reaction? OLS with HC3 standard errors (small sample)."""
    data = data.copy()
    data["car_pct"] = 100 * winsorize(data["car"])
    for col in data.select_dtypes("bool"):
        data[col] = data[col].astype(int)
    specs = {
        "(1) Size": "car_pct ~ log_dollar_volume",
        "(2) Incident type": "car_pct ~ log_dollar_volume + operations_disrupted + data_exfiltration + ransomware",
        "(3) + Disclosure language": (
            "car_pct ~ log_dollar_volume + operations_disrupted + data_exfiltration + ransomware"
            " + impact_undetermined + lm_uncertainty + log_words"
        ),
    }
    if "llm_severity" in data and data["llm_severity"].notna().sum() > 20:
        data["llm_material_or_undetermined"] = data["llm_materiality"].isin(["material", "undetermined"]).astype(int)
        specs["(4) LLM features"] = (
            "car_pct ~ log_dollar_volume + llm_severity + llm_vagueness + llm_material_or_undetermined"
        )
    fits = {name: smf.ols(formula, data=data).fit(cov_type="HC3") for name, formula in specs.items()}
    table = summary_col(list(fits.values()), model_names=list(fits), stars=True, float_format="%.2f",
                        info_dict={"N": lambda r: f"{int(r.nobs)}", "R2": lambda r: f"{r.rsquared:.2f}"})
    coefs = pd.concat({name: pd.DataFrame({"coef": f.params, "p": f.pvalues}) for name, f in fits.items()})
    return table.as_text(), coefs


# --- Robustness ------------------------------------------------------------------------


def placebo_distribution(events: pd.DataFrame, returns: pd.DataFrame, draws: int) -> np.ndarray:
    """Mean CAR[0,+1] when each firm's event date is replaced by a random non-event date."""
    rng = np.random.default_rng(SEED)
    days = returns.index
    candidates = {}
    for ev in events.itertuples():
        day0 = event_day_position(days, ev.accepted)
        if day0 is None:
            continue
        valid = np.arange(-ESTIMATION_WINDOW[0], len(days) - 2)
        candidates[ev.event_id] = (ev.ticker, valid[np.abs(valid - day0) > 20])
    means = []
    for _ in range(draws):
        pseudo = pd.DataFrame(
            [{"event_id": eid, "ticker": t, "accepted": days[rng.choice(pool)] + pd.Timedelta(hours=10)}
             for eid, (t, pool) in candidates.items()]
        )
        means.append(cars(abnormal_returns(pseudo, returns), MAIN_WINDOW)["car"].mean())
    return np.array(means)


def robustness(events, main, returns, factors, ar_main) -> pd.DataFrame:
    rows = {}
    base = cars(ar_main, MAIN_WINDOW)
    rows["Baseline: market model"] = significance_tests(base)

    unfiltered = events[~events["prior_disclosure"]]
    rows["Including unreliable price series"] = significance_tests(cars(abnormal_returns(unfiltered, returns), MAIN_WINDOW))

    covered = main[main["accepted"] + pd.Timedelta(days=20) < factors.index.max()]
    ar_ff = abnormal_returns(covered, returns, factors=factors, model="ff3")
    rows["Fama-French 3-factor model"] = significance_tests(cars(ar_ff, MAIN_WINDOW))

    for lo, hi in [(-1, 1), (0, 2)]:
        rows[f"Window {window_label((lo, hi))}"] = significance_tests(cars(ar_main, (lo, hi)))

    w = base.assign(car=winsorize(base["car"]))
    rows["Winsorized at 5% / 95%"] = significance_tests(w)

    size = main.set_index("event_id")["log_dollar_volume"]
    large = base[base.index.isin(size[size > size.quantile(1 / 3)].index)]
    rows["Excluding smallest third (dollar volume)"] = significance_tests(large)

    clean = base[base.index.isin(main.loc[~main["confounded"], "event_id"])]
    rows["Excluding earnings-confounded filing"] = significance_tests(clean)

    dropped = unreliable_tickers(returns.drop(columns="SPY"))
    ar_prior = abnormal_returns(events[events["prior_disclosure"] & ~events["ticker"].isin(dropped)], returns)
    if ar_prior["event_id"].nunique() >= 3:
        rows["Comparison: already disclosed earlier"] = significance_tests(cars(ar_prior, MAIN_WINDOW))
    return format_tests(rows)


def llm_validation(data: pd.DataFrame) -> pd.DataFrame | None:
    """Agreement between LLM answers and the keyword rules on overlapping concepts."""
    if "llm_breach_type" not in data:
        return None
    pairs = {
        "ransomware": data["llm_breach_type"].eq("ransomware"),
        "data_exfiltration": data["llm_personal_data_involved"].astype(str).str.lower().eq("true"),
        "operations_disrupted": data["llm_operations_disrupted"].astype(str).str.lower().eq("true"),
        "impact_undetermined": data["llm_materiality"].eq("undetermined"),
    }
    rows = {}
    for keyword, llm_flag in pairs.items():
        kw = data[keyword].astype(bool)
        rows[keyword] = {"keyword_share": kw.mean(), "llm_share": llm_flag.mean(), "agreement": (kw == llm_flag).mean()}
    return pd.DataFrame(rows).T


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    events, returns, factors = load_data()
    dropped = unreliable_tickers(returns.drop(columns="SPY"))
    print(f"Excluded for unreliable price data: {dropped}")
    main_sample = events[~events["prior_disclosure"] & ~events["ticker"].isin(dropped)]

    ar = abnormal_returns(main_sample, returns)
    ar.to_csv("data/processed/abnormal_returns.csv", index=False)
    print(f"Main sample: {ar['event_id'].nunique()} incidents with price data (of {len(main_sample)})")

    tests = format_tests({window_label(w): significance_tests(cars(ar, w)) for w in WINDOWS})
    to_markdown(tests, RESULTS_DIR / "car_tests.md")
    print("\nAverage market reaction\n", tests.round(4).to_string())

    plot_caar(caar_path(ar), FIGURES_DIR / "caar.png")
    car_main = cars(ar, MAIN_WINDOW)
    plot_car_distribution(car_main, FIGURES_DIR / "car_distribution.png")

    data = main_sample.set_index("event_id").join(car_main, how="inner")
    data[["company", "ticker", "file_date", "car", "scar"]].to_csv(RESULTS_DIR / "event_cars.csv")
    table, coefs = regressions(data)
    (RESULTS_DIR / "regressions.txt").write_text(table)
    coefs.to_csv(RESULTS_DIR / "regression_coefficients.csv")
    print("\nCross-sectional regressions (CAR[0,+1] in %)\n", table)

    robust = robustness(events, main_sample, returns, factors, ar)
    placebo = placebo_distribution(main_sample, returns, PLACEBO_DRAWS)
    actual = car_main["car"].mean()
    placebo_note = (
        f"\nPlacebo: replacing each firm's disclosure date with a random non-event date ({PLACEBO_DRAWS} draws) "
        f"gives a mean CAR[0,+1] of {100 * placebo.mean():.2f}% (5th-95th percentile: "
        f"{100 * np.percentile(placebo, 5):.2f}% to {100 * np.percentile(placebo, 95):.2f}%). "
        f"{(placebo <= actual).sum()} of {PLACEBO_DRAWS} draws are as negative as the actual {100 * actual:.2f}%.\n"
    )
    to_markdown(robust, RESULTS_DIR / "robustness.md")
    with open(RESULTS_DIR / "robustness.md", "a") as f:
        f.write(placebo_note)
    print("\nRobustness (CAR[0,+1])\n", robust.round(4).to_string(), placebo_note)

    validation = llm_validation(data)
    if validation is not None:
        to_markdown(validation, RESULTS_DIR / "llm_validation.md", pct_cols=validation.columns)
        print("\nLLM vs keyword features\n", validation.round(2).to_string())


if __name__ == "__main__":
    main()
