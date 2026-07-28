"""
Tests for _parse_window: new short (3h/6h/12h) and month (3mo/6mo) windows,
existing windows unchanged, and lenient fallback preserved.
"""

from datetime import timedelta

import pytest

from routers.predictions import _parse_window


@pytest.mark.parametrize("window,expected", [
    ("3h",  timedelta(hours=3)),
    ("6h",  timedelta(hours=6)),
    ("12h", timedelta(hours=12)),
    ("24h", timedelta(hours=24)),
    ("7d",  timedelta(days=7)),
    ("30d", timedelta(days=30)),
    ("3mo", timedelta(days=90)),
    ("6mo", timedelta(days=180)),
])
def test_parse_window_supported_values(window, expected):
    assert _parse_window(window) == expected


@pytest.mark.parametrize("window", ["banana", "", "mo", "xd", "zzh", "1.5d"])
def test_parse_window_lenient_fallback_to_30d(window):
    # Unparseable values must fall back to 30 days, never raise.
    assert _parse_window(window) == timedelta(days=30)
