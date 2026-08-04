"""Regression tests for the read-only long-trend strategy benchmark."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "long_trend_dry_run.py"
)


def _load_dry_run_module():
    spec = importlib.util.spec_from_file_location("long_trend_dry_run_test", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_strategy_performance_reports_expected_metrics():
    module = _load_dry_run_module()
    positions = np.ones(4)
    returns = np.array([0.01, -0.01, 0.02, -0.02])

    result = module._strategy_performance(positions, returns, returns)

    assert result["evaluated"] is True
    assert result["n_days"] == 4
    assert result["total_return"] == np.prod(1.0 + returns) - 1.0
    assert result["max_drawdown"] < 0
    assert result["annualized_turnover"] > 0
    assert result["downside_capture"] == 1.0


def test_benchmark_positions_use_only_information_available_at_decision_close():
    module = _load_dry_run_module()
    dates = pd.date_range("2024-01-01", periods=220, freq="B")
    close = pd.Series(np.linspace(100.0, 200.0, len(dates)), index=dates)

    positions = module._benchmark_strategy_positions(
        close,
        dates[-3:],
        np.array([1.0, 0.0, 1.0]),
    )

    assert positions["current_long_model"].tolist() == [1.0, 0.0, 1.0]
    assert np.allclose(positions["buy_and_hold"], 1.0)
    assert positions["_next_returns"][0] > 0
    # The first 200-day moving-average value is available at the decision
    # close, while volatility targeting starts only after its warm-up window.
    assert positions["sma200_filter"].tolist() == [1.0, 1.0, 1.0]
    assert np.all(np.isfinite(positions["volatility_targeted"]))


def test_benchmark_labels_do_not_trade_on_future_return_filter():
    module = _load_dry_run_module()
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    close = pd.Series([100.0, 100.5, 101.0], index=dates)
    forward_returns = np.array([0.001, 0.001, np.nan])
    predictions = np.array([0, 1, 0])

    accuracy = module._classification_accuracy(
        predictions, forward_returns, threshold=0.02
    )

    # No ±2% future event exists, so classification is undefined rather than
    # silently treating filtered-out days as negative labels.
    assert accuracy is None