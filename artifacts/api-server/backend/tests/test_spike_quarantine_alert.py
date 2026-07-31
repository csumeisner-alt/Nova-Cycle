"""
Tests for the cross-bar spike quarantine session counter and operator alert.

Verifies that:
  - The alert fires exactly at the configured threshold, not before.
  - Each subsequent quarantine beyond the threshold also triggers a warning.
  - Quarantines that arise from intra-bar violations (not cross-bar spikes)
    do not count toward the session total.
  - The tracker counter increments correctly across multiple filter_valid_ohlc
    calls within a session.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pandas as pd
import pytest

from ingestion.ohlc_validator import _spike_tracker, filter_valid_ohlc


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_flat_df(n: int, base_close: float = 100.0) -> pd.DataFrame:
    """Return a DataFrame of *n* perfectly flat, valid candles."""
    return pd.DataFrame(
        {
            "open": [base_close] * n,
            "high": [base_close] * n,
            "low": [base_close] * n,
            "close": [base_close] * n,
            "volume": [1_000] * n,
        }
    )


def _make_spike_df(spike_index: int = 2, n: int = 7, base: float = 100.0) -> pd.DataFrame:
    """
    Return a DataFrame of *n* candles where row *spike_index* is a clear
    cross-bar spike (close 5× the base — well above the 10 % threshold).
    """
    closes = [base] * n
    closes[spike_index] = base * 5.0  # 400 % deviation → certain spike
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1_000] * n,
        }
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_tracker():
    """Ensure the module-level singleton starts from zero for every test."""
    _spike_tracker._reset()
    yield
    _spike_tracker._reset()


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestSpikeQuarantineAlert:

    def test_no_alert_below_threshold(self, caplog):
        """Calls that produce fewer quarantines than the threshold must be silent."""
        with patch("config.settings") as mock_settings:
            mock_settings.SPIKE_CLOSE_THRESHOLD = 0.10
            mock_settings.SPIKE_QUARANTINE_ALERT_COUNT = 3

            with caplog.at_level(logging.WARNING, logger="ingestion.ohlc_validator"):
                # One spike per call — 2 total, threshold is 3
                filter_valid_ohlc(_make_spike_df(), spike_threshold=0.10)
                filter_valid_ohlc(_make_spike_df(), spike_threshold=0.10)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings == [], (
            "Expected no warning before threshold is reached, "
            f"but got: {[r.message for r in warnings]}"
        )

    def test_alert_fires_exactly_at_threshold(self, caplog):
        """The alert must fire on the call that pushes the count to the threshold."""
        with patch("config.settings") as mock_settings:
            mock_settings.SPIKE_CLOSE_THRESHOLD = 0.10
            mock_settings.SPIKE_QUARANTINE_ALERT_COUNT = 3

            with caplog.at_level(logging.WARNING, logger="ingestion.ohlc_validator"):
                filter_valid_ohlc(_make_spike_df(), spike_threshold=0.10)  # count → 1
                filter_valid_ohlc(_make_spike_df(), spike_threshold=0.10)  # count → 2
                filter_valid_ohlc(_make_spike_df(), spike_threshold=0.10)  # count → 3 ← alert

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, f"Expected exactly 1 warning, got {len(warnings)}"
        assert "spike_quarantine_alert" in warnings[0].message

    def test_alert_fires_on_each_quarantine_after_threshold(self, caplog):
        """Every quarantine once the threshold is exceeded must emit a warning."""
        with patch("config.settings") as mock_settings:
            mock_settings.SPIKE_CLOSE_THRESHOLD = 0.10
            mock_settings.SPIKE_QUARANTINE_ALERT_COUNT = 2

            with caplog.at_level(logging.WARNING, logger="ingestion.ohlc_validator"):
                filter_valid_ohlc(_make_spike_df(), spike_threshold=0.10)  # count → 1
                filter_valid_ohlc(_make_spike_df(), spike_threshold=0.10)  # count → 2 ← alert
                filter_valid_ohlc(_make_spike_df(), spike_threshold=0.10)  # count → 3 ← alert
                filter_valid_ohlc(_make_spike_df(), spike_threshold=0.10)  # count → 4 ← alert

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 3, f"Expected 3 warnings, got {len(warnings)}"

    def test_alert_message_contains_count_and_threshold(self, caplog):
        """The WARN message must include the running count and the threshold."""
        with patch("config.settings") as mock_settings:
            mock_settings.SPIKE_CLOSE_THRESHOLD = 0.10
            mock_settings.SPIKE_QUARANTINE_ALERT_COUNT = 1

            with caplog.at_level(logging.WARNING, logger="ingestion.ohlc_validator"):
                filter_valid_ohlc(_make_spike_df(), spike_threshold=0.10)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "Expected at least one warning"
        msg = warnings[0].message
        # The count (1) and threshold (1) must both appear in the message.
        assert "1" in msg
        assert "threshold=1" in msg

    def test_intra_bar_violations_do_not_count(self, caplog):
        """
        Rows quarantined for intra-bar inconsistency (not cross-bar spikes)
        must not increment the spike session counter.
        """
        # Build a DataFrame where high < open — intra-bar violation only.
        df = pd.DataFrame(
            {
                "open": [100.0, 100.0, 100.0],
                "high": [90.0, 90.0, 90.0],   # high < open → intra-bar fail
                "low": [80.0, 80.0, 80.0],
                "close": [95.0, 95.0, 95.0],
                "volume": [1_000, 1_000, 1_000],
            }
        )

        with patch("config.settings") as mock_settings:
            mock_settings.SPIKE_CLOSE_THRESHOLD = 0.10
            mock_settings.SPIKE_QUARANTINE_ALERT_COUNT = 1  # very low threshold

            with caplog.at_level(logging.WARNING, logger="ingestion.ohlc_validator"):
                filter_valid_ohlc(df, spike_threshold=0.10)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings == [], (
            "Intra-bar violations should not trigger the spike quarantine alert"
        )

    def test_no_alert_when_no_spikes(self, caplog):
        """Entirely clean data must never trigger an alert."""
        with patch("config.settings") as mock_settings:
            mock_settings.SPIKE_CLOSE_THRESHOLD = 0.10
            mock_settings.SPIKE_QUARANTINE_ALERT_COUNT = 1  # low threshold

            with caplog.at_level(logging.WARNING, logger="ingestion.ohlc_validator"):
                for _ in range(5):
                    filter_valid_ohlc(_make_flat_df(10), spike_threshold=0.10)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings == [], "Clean data should produce no spike quarantine alerts"

    def test_counter_accumulates_across_multiple_calls(self):
        """The session total must be the sum of spikes across all calls."""
        with patch("config.settings") as mock_settings:
            mock_settings.SPIKE_CLOSE_THRESHOLD = 0.10
            mock_settings.SPIKE_QUARANTINE_ALERT_COUNT = 999  # never alert

            filter_valid_ohlc(_make_spike_df(), spike_threshold=0.10)  # +1
            filter_valid_ohlc(_make_spike_df(), spike_threshold=0.10)  # +1
            filter_valid_ohlc(_make_spike_df(), spike_threshold=0.10)  # +1

        assert _spike_tracker._count == 3, (
            f"Expected session count of 3, got {_spike_tracker._count}"
        )

    def test_threshold_of_one_alerts_on_first_spike(self, caplog):
        """With threshold=1 the very first spike must trigger the alert."""
        with patch("config.settings") as mock_settings:
            mock_settings.SPIKE_CLOSE_THRESHOLD = 0.10
            mock_settings.SPIKE_QUARANTINE_ALERT_COUNT = 1

            with caplog.at_level(logging.WARNING, logger="ingestion.ohlc_validator"):
                filter_valid_ohlc(_make_spike_df(), spike_threshold=0.10)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
