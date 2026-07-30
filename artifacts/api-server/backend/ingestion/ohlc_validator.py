"""
OHLC Integrity Validator
========================
Detects and quarantines candles whose OHLC values are internally impossible.

A valid OHLC bar must satisfy:
  low  <= open  <= high
  low  <= close <= high
  low  <= high

Any violation means the data is internally inconsistent — e.g. a high that is
below the open (as occurred on 2024-07-30 for VOO where open=680.12 but
high=676.71).  Such a candle would feed wrong features into the prediction
engine and produce a misleading signal.

Typical callers:
  - DataFetcher._normalise_columns()  — filter before DB storage
  - predictions._load_daily_candles() — filter already-stored bad rows at
    prediction time and mark the result as data_quality_degraded

The module is intentionally side-effect-free: it never logs or raises on its
own; callers decide what to do with the quarantined rows.
"""

from __future__ import annotations

import pandas as pd


# ── Single-row check ─────────────────────────────────────────────────────────

def validate_ohlc_row(
    open_p: float,
    high_p: float,
    low_p: float,
    close_p: float,
) -> tuple[bool, str]:
    """
    Check whether a single OHLC tuple is internally self-consistent.

    Returns:
        (True, "")              — candle is valid
        (False, reason_string)  — candle is malformed; reason explains why

    Rules applied in priority order:
      1. high < low            → high_below_low
      2. high < open           → high_below_open
      3. high < close          → high_below_close
      4. low > open            → low_above_open
      5. low > close           → low_above_close

    Tolerates floating-point noise up to 0.001 (i.e. violations smaller
    than 0.1 cents are ignored so sub-penny rounding artefacts don't
    quarantine otherwise fine bars).
    """
    EPSILON = 0.001

    if high_p < low_p - EPSILON:
        return False, f"high_below_low (high={high_p:.4f} low={low_p:.4f})"
    if high_p < open_p - EPSILON:
        return False, f"high_below_open (high={high_p:.4f} open={open_p:.4f})"
    if high_p < close_p - EPSILON:
        return False, f"high_below_close (high={high_p:.4f} close={close_p:.4f})"
    if low_p > open_p + EPSILON:
        return False, f"low_above_open (low={low_p:.4f} open={open_p:.4f})"
    if low_p > close_p + EPSILON:
        return False, f"low_above_close (low={low_p:.4f} close={close_p:.4f})"
    return True, ""


# ── DataFrame filter ─────────────────────────────────────────────────────────

def filter_valid_ohlc(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split *df* into two DataFrames: (valid, quarantined).

    *df* must have lower-cased columns: open, high, low, close.
    Rows missing any of those columns are treated as valid (no-op) so the
    function gracefully handles DataFrames that lack OHLC (e.g. pure-VIX
    frames that were erroneously passed in).

    Returns:
        valid       — rows that pass all OHLC consistency checks
        quarantined — rows that failed at least one check; each row has an
                      extra column ``ohlc_invalid_reason`` explaining why

    Neither frame is a copy when the input is all-valid or all-invalid; a
    copy is only made for the quarantined slice (to add the reason column
    without SettingWithCopyWarning).
    """
    required = {"open", "high", "low", "close"}
    if df.empty or not required.issubset(df.columns):
        return df, pd.DataFrame(columns=list(df.columns) + ["ohlc_invalid_reason"])

    bad_mask: list[bool] = []
    reasons: list[str] = []

    for _, row in df.iterrows():
        ok, reason = validate_ohlc_row(
            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
        )
        bad_mask.append(not ok)
        reasons.append(reason)

    if not any(bad_mask):
        return df, pd.DataFrame(columns=list(df.columns) + ["ohlc_invalid_reason"])

    bad_series = pd.Series(bad_mask, index=df.index)

    quarantined = df[bad_series].copy()
    quarantined["ohlc_invalid_reason"] = [r for r, b in zip(reasons, bad_mask) if b]
    valid = df[~bad_series]
    return valid, quarantined
