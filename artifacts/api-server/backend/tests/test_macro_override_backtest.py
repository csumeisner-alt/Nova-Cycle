"""
Offline replay of scripts/backtest_macro_override.py.
=====================================================
Uses real VOO / ^VIX daily data cached in
tests/fixtures/macro_backtest_voo_vix.csv (built by
scripts/build_macro_backtest_fixture.py, which needs network) so the
macro shock-flag threshold assertions run in the normal, network-free
test suite. If volatility-regime, VIX-regime, or macro-override logic
drifts, these tests fail.

Acceptance criteria (same as the online backtest):
  - COVID crash 2020:        flag fires on >= 10% of trading days
  - CPI shocks Sep-Oct 2022: flag fires at least once
  - Calm 2017 / 2023 H2:     flag fires on <= 5% of trading days
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from indicators.technical import TechnicalIndicators
from ml.features import compute_volatility_regime, macro_override_flag

# Reuse the live backtest's periods and acceptance thresholds so the
# offline test can never drift out of sync with the online script.
from scripts.backtest_macro_override import (
    CALM_MAX_FIRE_RATE,
    PERIODS as SCRIPT_PERIODS,
    SHOCK_MIN_FIRE_RATE,
)

FIXTURE = Path(__file__).parent / "fixtures" / "macro_backtest_voo_vix.csv"

PERIODS = {name: (start, end) for name, start, end, _kind in SCRIPT_PERIODS}


@pytest.fixture(scope="module")
def fixture_df() -> pd.DataFrame:
    assert FIXTURE.exists(), (
        f"Missing fixture {FIXTURE}. Rebuild with "
        "`python scripts/build_macro_backtest_fixture.py` (needs network)."
    )
    df = pd.read_csv(FIXTURE, parse_dates=["date"])
    return df


def _fire_stats(df: pd.DataFrame, period: str) -> tuple[int, int, list[str]]:
    sub = df[df["period"] == period].set_index("date").sort_index()
    assert not sub.empty, f"No fixture rows for period {period!r}"

    close = sub["voo_close"]
    open_ = sub["voo_open"]
    vix_close = sub["vix_close"]

    ti = TechnicalIndicators()
    vix_regime = ti.compute_vix_regime(vix_close)
    vol_regime = compute_volatility_regime(close)

    flag = macro_override_flag(
        close.index,
        close=close,
        open_=open_,
        vix_regime=vix_regime,
        volatility_regime=vol_regime,
    )

    start, end = PERIODS[period]
    mask = (close.index >= pd.Timestamp(start)) & (close.index <= pd.Timestamp(end))
    flag_w = flag[mask]
    days = int(mask.sum())
    fired = int(flag_w.sum())
    dates = list(flag_w[flag_w > 0].index.strftime("%Y-%m-%d"))
    assert days > 0, f"Evaluation window empty for {period!r}"
    return days, fired, dates


def test_fixture_covers_all_periods(fixture_df):
    assert set(fixture_df["period"].unique()) == set(PERIODS)


def test_covid_crash_fires_often(fixture_df):
    days, fired, dates = _fire_stats(fixture_df, "COVID crash 2020")
    rate = fired / days
    assert rate >= SHOCK_MIN_FIRE_RATE, (
        f"COVID crash fire rate {rate:.1%} < {SHOCK_MIN_FIRE_RATE:.0%} "
        f"(too tight); fired on {dates}"
    )


def test_cpi_shocks_fire_at_least_once(fixture_df):
    _, fired, _ = _fire_stats(fixture_df, "CPI shocks Sep-Oct 2022")
    assert fired >= 1, "Flag never fired on known CPI-shock days (too tight)"


@pytest.mark.parametrize("period", ["Calm 2017", "Calm 2023 H2"])
def test_calm_periods_stay_quiet(fixture_df, period):
    days, fired, dates = _fire_stats(fixture_df, period)
    rate = fired / days
    assert rate <= CALM_MAX_FIRE_RATE, (
        f"{period} fire rate {rate:.1%} > {CALM_MAX_FIRE_RATE:.0%} "
        f"(too loose); fired on {dates}"
    )
