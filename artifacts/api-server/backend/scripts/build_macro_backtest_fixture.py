"""
Build the offline fixture used by tests/test_macro_override_backtest.py.
=======================================================================
Downloads real VOO / ^VIX daily data (with warm-up padding) for the
shock/calm windows checked by scripts/backtest_macro_override.py and
stores them in a single small CSV so the backtest assertions can run
as a network-free pytest.

Usage (needs network for yfinance):
    cd artifacts/api-server/backend
    python scripts/build_macro_backtest_fixture.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from scripts.backtest_macro_override import PERIODS, fetch

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "macro_backtest_voo_vix.csv"


def main() -> int:
    frames = []
    for name, start, end, kind in PERIODS:
        voo = fetch("VOO", start, end)
        vix = fetch("^VIX", start, end)
        vix_close = vix["Close"].reindex(voo.index, method="ffill")
        df = pd.DataFrame(
            {
                "period": name,
                "date": voo.index.strftime("%Y-%m-%d"),
                "voo_open": voo["Open"].round(4).values,
                "voo_close": voo["Close"].round(4).values,
                "vix_close": vix_close.round(4).values,
            }
        )
        frames.append(df)
        print(f"{name}: {len(df)} rows (incl. warm-up)")

    out = pd.concat(frames, ignore_index=True)
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(FIXTURE_PATH, index=False)
    print(f"Wrote {len(out)} rows to {FIXTURE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
