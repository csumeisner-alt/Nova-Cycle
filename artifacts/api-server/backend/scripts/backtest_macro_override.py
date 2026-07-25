"""
Backtest the macro_override_flag against real historical VOO / VIX data.
========================================================================
Confirms the configured thresholds (MACRO_OVERRIDE_VIX_REGIME,
MACRO_OVERRIDE_OVERNIGHT_MOVE_PCT in config.py) are neither too tight
(never fires during genuine shocks) nor too loose (fires constantly in
calm markets).

Periods checked:
  - COVID crash     (2020-02-15 .. 2020-04-15) — flag SHOULD fire often
  - 2022 CPI shocks (2022-09-01 .. 2022-10-31) — flag should fire at least once
  - Calm 2017       (2017-03-01 .. 2017-11-30) — flag should (almost) NEVER fire
  - Calm 2023 H2    (2023-06-01 .. 2023-11-30) — flag should (almost) NEVER fire

Usage:
    cd artifacts/api-server/backend
    python scripts/backtest_macro_override.py

Exit code is non-zero when acceptance criteria fail:
  - shock-period fire rate must be  >= 10% of trading days (COVID)
  - calm-period fire rate must be   <=  5% of trading days
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from the backend dir (imports use flat module names)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yfinance as yf

from indicators.technical import TechnicalIndicators
from ml.features import compute_volatility_regime, macro_override_flag

PERIODS = [
    # (name, start, end, kind)  kind: "shock" | "calm" | "info"
    ("COVID crash 2020", "2020-02-15", "2020-04-15", "shock"),
    ("CPI shocks Sep-Oct 2022", "2022-09-01", "2022-10-31", "shock_light"),
    ("Calm 2017", "2017-03-01", "2017-11-30", "calm"),
    ("Calm 2023 H2", "2023-06-01", "2023-11-30", "calm"),
]

SHOCK_MIN_FIRE_RATE = 0.10  # >=10% of COVID-crash days should fire
CALM_MAX_FIRE_RATE = 0.05   # <=5% of calm days may fire

# Fetch with warm-up padding so rolling baselines (100 bars) are seeded.
WARMUP_DAYS = 200


def fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=(pd.Timestamp(start) - pd.Timedelta(days=WARMUP_DAYS * 1.6)).date().isoformat(),
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {ticker} {start}..{end}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    return df


def evaluate(name: str, start: str, end: str) -> dict:
    voo = fetch("VOO", start, end)
    vix = fetch("^VIX", start, end)

    close, open_ = voo["Close"], voo["Open"]
    vix_close = vix["Close"].reindex(close.index, method="ffill")

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

    # Restrict to the evaluation window (drop warm-up)
    mask = (close.index >= pd.Timestamp(start)) & (close.index <= pd.Timestamp(end))
    flag_w = flag[mask]
    days = int(mask.sum())
    fired = int(flag_w.sum())
    fired_dates = list(flag_w[flag_w > 0].index.strftime("%Y-%m-%d"))

    return {
        "name": name,
        "days": days,
        "fired": fired,
        "rate": fired / days if days else 0.0,
        "dates": fired_dates,
    }


def main() -> int:
    failures = []
    print(
        f"Thresholds: MACRO_OVERRIDE_VIX_REGIME={__import__('config').settings.MACRO_OVERRIDE_VIX_REGIME}, "
        f"MACRO_OVERRIDE_OVERNIGHT_MOVE_PCT={__import__('config').settings.MACRO_OVERRIDE_OVERNIGHT_MOVE_PCT}%\n"
    )
    for name, start, end, kind in PERIODS:
        r = evaluate(name, start, end)
        print(f"{name} ({start}..{end}): fired {r['fired']}/{r['days']} days "
              f"({r['rate']:.1%})")
        if r["fired"]:
            print(f"  fire dates: {', '.join(r['dates'])}")
        if kind == "shock" and r["rate"] < SHOCK_MIN_FIRE_RATE:
            failures.append(f"{name}: fire rate {r['rate']:.1%} < {SHOCK_MIN_FIRE_RATE:.0%} (too tight)")
        if kind == "shock_light" and r["fired"] < 1:
            failures.append(f"{name}: flag never fired on known CPI-shock days (too tight)")
        if kind == "calm" and r["rate"] > CALM_MAX_FIRE_RATE:
            failures.append(f"{name}: fire rate {r['rate']:.1%} > {CALM_MAX_FIRE_RATE:.0%} (too loose)")
        print()

    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS: thresholds fire during shocks and stay quiet in calm markets.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
