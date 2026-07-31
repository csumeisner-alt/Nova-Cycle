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

import logging
from datetime import date
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ── Per-session spike-quarantine counter ──────────────────────────────────────

class _SpikeQuarantineTracker:
    """
    In-memory counter of cross-bar spike quarantines within the current trading
    session.  The counter resets automatically when the calendar date changes
    (i.e. at the start of each new trading day).

    When the cumulative count reaches or exceeds
    ``settings.SPIKE_QUARANTINE_ALERT_COUNT`` a WARN-level structured log is
    emitted so operators can investigate a possible systematic feed problem.
    The alert is emitted on *every* quarantine once the threshold is reached,
    so a runaway feed does not produce just one warning and then go silent.
    """

    def __init__(self) -> None:
        self._session_date: Optional[date] = None
        self._count: int = 0

    # Exposed for tests only — do not use in production code.
    def _reset(self) -> None:
        self._session_date = None
        self._count = 0

    def record(self, n_spikes: int) -> None:
        """
        Register *n_spikes* new cross-bar spike quarantines.

        Emits a WARNING when the running session total reaches or exceeds the
        configured alert threshold.  Silently returns when *n_spikes* is 0.
        """
        if n_spikes <= 0:
            return

        today = date.today()
        if self._session_date != today:
            # New trading day — start fresh.
            self._session_date = today
            self._count = 0

        self._count += n_spikes

        from config import settings  # local import to avoid circular deps
        threshold = settings.SPIKE_QUARANTINE_ALERT_COUNT

        if self._count >= threshold:
            logger.warning(
                "spike_quarantine_alert: %d cross-bar spike quarantines in "
                "the current trading session (threshold=%d). "
                "Possible systematic feed problem — check data source.",
                self._count,
                threshold,
                extra={
                    "event": "spike_quarantine_alert",
                    "session_spike_quarantine_count": self._count,
                    "alert_threshold": threshold,
                },
            )


# Module-level singleton — reset at process startup; lives for the process lifetime.
_spike_tracker = _SpikeQuarantineTracker()


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


# ── Cross-bar spike detection ─────────────────────────────────────────────────

def flag_cross_bar_spikes(
    df: pd.DataFrame,
    threshold: Optional[float] = None,
    window: int = 5,
) -> "pd.Series[bool]":
    """
    Identify bars whose close deviates more than *threshold* from the rolling
    median of their neighbours.

    Uses a *centered* rolling window so both preceding and following bars inform
    the comparison.  Edge bars with fewer than 3 valid neighbours inside the
    window receive a NaN median and are never flagged (insufficient context to
    distinguish a real move from a spike at the boundary).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``close`` column.  The index is assumed to be sorted in
        chronological order; callers should sort before passing.
    threshold : float, optional
        Maximum allowed fractional deviation, e.g. 0.10 for 10 %.
        When *None* (default) the value is read from ``settings.SPIKE_CLOSE_THRESHOLD``.
    window : int
        Size of the centered rolling window (default 5 bars → ±2 neighbours).

    Returns
    -------
    pd.Series[bool]
        True for bars that are flagged as cross-bar spikes, aligned to *df.index*.
        All-False when *df* is empty, lacks a ``close`` column, or *threshold* <= 0.
    """
    false_series: "pd.Series[bool]" = pd.Series(False, index=df.index, dtype=bool)

    if df.empty or "close" not in df.columns:
        return false_series

    if threshold is None:
        from config import settings
        threshold = settings.SPIKE_CLOSE_THRESHOLD

    if threshold <= 0:
        return false_series

    closes = pd.to_numeric(df["close"], errors="coerce")
    rolling_median = closes.rolling(window=window, center=True, min_periods=3).median()

    # Only flag bars where the rolling median is well-defined and non-zero
    has_context = rolling_median.notna() & (rolling_median != 0)
    spike_mask = false_series.copy()
    spike_mask[has_context] = (
        (closes[has_context] - rolling_median[has_context]).abs()
        / rolling_median[has_context].abs()
        > threshold
    )
    return spike_mask


def filter_valid_ohlc(
    df: pd.DataFrame,
    spike_threshold: Optional[float] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split *df* into two DataFrames: (valid, quarantined).

    Two complementary checks are applied in order:

    1. **Intra-bar consistency** — each candle must satisfy
       ``low <= open <= high`` and ``low <= close <= high`` (see
       ``validate_ohlc_row``).

    2. **Cross-bar spike detection** — a bar whose close deviates more than
       *spike_threshold* (default: ``settings.SPIKE_CLOSE_THRESHOLD``) from the
       rolling median of its neighbours (centered window of 5, min 3 valid
       neighbours) is flagged as a probable data glitch.  The check is skipped
       when *spike_threshold* is 0 or negative.

    *df* must have lower-cased columns: open, high, low, close.
    Rows missing any of those columns are treated as valid (no-op) so the
    function gracefully handles DataFrames that lack OHLC (e.g. pure-VIX
    frames that were erroneously passed in).

    Returns:
        valid       — rows that pass all checks
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

    # ── Pass 1: intra-bar consistency ─────────────────────────────────────────
    for _, row in df.iterrows():
        ok, reason = validate_ohlc_row(
            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
        )
        bad_mask.append(not ok)
        reasons.append(reason)

    # ── Pass 2: cross-bar spike detection ─────────────────────────────────────
    # Run on the full frame so the rolling window sees all neighbours, then
    # merge results with the intra-bar mask.
    spike_flags = flag_cross_bar_spikes(df, threshold=spike_threshold)
    n_new_spikes = 0
    for i, (already_bad, is_spike) in enumerate(zip(bad_mask, spike_flags)):
        if is_spike and not already_bad:
            bad_mask[i] = True
            close_val = float(df.iloc[i]["close"])
            reasons[i] = f"cross_bar_spike (close={close_val:.4f})"
            n_new_spikes += 1

    # Notify the session tracker so operators are alerted when quarantines
    # accumulate within a single trading day (possible systematic feed issue).
    _spike_tracker.record(n_new_spikes)

    if not any(bad_mask):
        return df, pd.DataFrame(columns=list(df.columns) + ["ohlc_invalid_reason"])

    bad_series = pd.Series(bad_mask, index=df.index)

    quarantined = df[bad_series].copy()
    quarantined["ohlc_invalid_reason"] = [r for r, b in zip(reasons, bad_mask) if b]
    valid = df[~bad_series]
    return valid, quarantined
