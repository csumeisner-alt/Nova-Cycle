---
name: Macro override threshold backtest
description: Why MACRO_OVERRIDE_VIX_REGIME defaults to HIGH, validated against real VOO/VIX history
---

Rule: keep `MACRO_OVERRIDE_VIX_REGIME=HIGH` (VIX ≥ 25) unless a new backtest says otherwise.

**Why:** Backtest against real yfinance VOO/VIX data showed EXTREME (VIX ≥ 35) missed all 2022 CPI-surprise shock days (e.g. 2022-09-13, -4.3% gap day, VIX ~27). HIGH catches those 3 CPI days and 90% of COVID-crash days while firing 0 times across ~317 calm trading days (2017, 2023 H2) — no looseness penalty.

**How to apply:** Re-run `artifacts/api-server/backend/scripts/backtest_macro_override.py` (needs network for yfinance) after changing macro-override thresholds or the volatility-regime/VIX-regime logic; it exits non-zero if shock periods stop firing or calm periods exceed 5%.
