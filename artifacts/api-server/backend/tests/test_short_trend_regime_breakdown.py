"""Confirm that the per-regime OOS breakdown is persisted to the short-trend
walk-forward report when VIX regime labels are supplied during training.

Done looks like:
- walk_forward_evaluate is called with regime_labels for the short-trend model
- The resulting regime_breakdown is stored in ml/models/short_trend_walkforward.json
- /api/healthz surfaces it under models.short_trend.walk_forward.regime_breakdown
"""

import json
import numpy as np
import pandas as pd
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Unit test: walk_forward_evaluate emits regime_breakdown when regime_labels
# are supplied (validates the calibration module directly).
# ─────────────────────────────────────────────────────────────────────────────

class _TrivialClassifier:
    """Minimal classifier that always predicts 0.6 BUY probability."""

    def fit(self, X, y, sample_weight=None, verbose=False):
        return self

    def predict_proba(self, X):
        return np.column_stack([
            np.full(len(X), 0.4),
            np.full(len(X), 0.6),
        ])


class TestWalkForwardRegimeBreakdown:
    def test_regime_breakdown_present_when_labels_supplied(self):
        from ml.calibration import walk_forward_evaluate

        rng = np.random.default_rng(0)
        n = 500
        X = rng.standard_normal((n, 4)).astype(np.float32)
        y = (rng.random(n) > 0.5).astype(int)
        # Cycle through 4 VIX regimes so all are represented in OOS folds
        regime_labels = np.tile([0, 1, 2, 3], n // 4 + 1)[:n].astype(np.int32)

        metrics, oos_probs, oos_labels = walk_forward_evaluate(
            X, y, weights=None,
            model_factory=_TrivialClassifier,
            n_splits=4,
            embargo=5,
            regime_labels=regime_labels,
        )

        assert metrics.get("evaluated") is True
        assert "regime_breakdown" in metrics, (
            "regime_breakdown must be present in walk-forward metrics when "
            "regime_labels are supplied"
        )
        breakdown = metrics["regime_breakdown"]
        assert isinstance(breakdown, list) and len(breakdown) > 0

        regimes_seen = {row["regime"] for row in breakdown}
        # At least two regimes must appear in OOS folds (the distribution is
        # uniform across LOW/NORMAL/HIGH/EXTREME so all four should appear)
        assert len(regimes_seen) >= 2

        # Each entry must have the expected keys
        required_keys = {
            "regime", "regime_code", "oos_samples",
            "oos_accuracy", "majority_baseline_accuracy",
            "accuracy_lift_vs_majority", "oos_brier_score",
        }
        for row in breakdown:
            assert required_keys.issubset(row.keys()), (
                f"Missing keys in regime row {row['regime']}: "
                f"{required_keys - set(row.keys())}"
            )

    def test_regime_breakdown_absent_when_no_labels(self):
        from ml.calibration import walk_forward_evaluate

        rng = np.random.default_rng(1)
        n = 500
        X = rng.standard_normal((n, 4)).astype(np.float32)
        y = (rng.random(n) > 0.5).astype(int)

        metrics, _, _ = walk_forward_evaluate(
            X, y, weights=None,
            model_factory=_TrivialClassifier,
            n_splits=4,
            embargo=5,
            regime_labels=None,
        )

        assert metrics.get("evaluated") is True
        assert "regime_breakdown" not in metrics


# ─────────────────────────────────────────────────────────────────────────────
# Integration test: ShortTrendModel.train() with a string vix_regime Series
# produces a persisted short_trend_walkforward.json that includes
# regime_breakdown. This exercises the real training path, including the
# string→int conversion that was the source of the original regression.
# ─────────────────────────────────────────────────────────────────────────────

class TestShortTrendTrainRegimeBreakdown:
    def test_train_with_string_vix_regime_persists_regime_breakdown(
        self, tmp_path, monkeypatch
    ):
        """Training with a string vix_regime Series (the real indicator format)
        must result in a short_trend_walkforward.json that contains
        regime_breakdown — not an empty/absent field caused by a conversion error.
        """
        import ml.calibration as cal
        from ml.short_trend import ShortTrendModel, MODEL_DIR, MODEL_PATH

        # Redirect model output to tmp_path so the test is isolated
        monkeypatch.setattr("ml.short_trend.MODEL_DIR", tmp_path)
        monkeypatch.setattr("ml.short_trend.MODEL_PATH", tmp_path / "short_trend_model.pkl")
        monkeypatch.setattr(cal, "MODEL_DIR", tmp_path)

        rng = np.random.default_rng(42)
        # Build a minimal but realistic 5-min DataFrame (needs > ~120 rows for
        # the walk-forward to produce at least one valid fold; 600 rows gives
        # comfortable headroom for the 12-bar embargo and 5-fold split).
        n = 600
        base_price = 450.0
        prices = base_price + np.cumsum(rng.normal(0, 0.05, n))
        prices = np.clip(prices, 1.0, None)

        idx = pd.date_range("2026-01-01 09:30", periods=n, freq="5min")
        df = pd.DataFrame({
            "open":   prices * (1 + rng.normal(0, 0.001, n)),
            "high":   prices * (1 + np.abs(rng.normal(0, 0.002, n))),
            "low":    prices * (1 - np.abs(rng.normal(0, 0.002, n))),
            "close":  prices,
            "volume": rng.integers(10_000, 100_000, n).astype(float),
            "is_extended_hours": np.zeros(n, dtype=int),
        }, index=idx)

        # Build minimal indicator dict with the realistic string vix_regime
        # format that TechnicalIndicators.compute_all() actually produces.
        regimes = ["LOW", "NORMAL", "HIGH", "EXTREME"]
        vix_regime_series = pd.Series(
            np.array(regimes)[np.tile([0, 1, 2, 3], n // 4 + 1)[:n]],
            index=idx,
        )

        rsi_s = pd.Series(50.0 + rng.normal(0, 10, n), index=idx).clip(0, 100)
        stoch_k = pd.Series(50.0 + rng.normal(0, 15, n), index=idx).clip(0, 100)
        stoch_d = pd.Series(50.0 + rng.normal(0, 10, n), index=idx).clip(0, 100)
        bb_pct = pd.Series(0.5 + rng.normal(0, 0.15, n), index=idx)
        bb_bw  = pd.Series(0.02 + np.abs(rng.normal(0, 0.005, n)), index=idx)
        atr_s  = pd.Series(0.5 + np.abs(rng.normal(0, 0.1, n)), index=idx)

        indicators = {
            "rsi": rsi_s,
            "stoch": {"k": stoch_k, "d": stoch_d},
            "stoch_rsi": {"k": stoch_k, "d": stoch_d},
            "bollinger": {"pct_b": bb_pct, "bandwidth": bb_bw},
            "atr_all": atr_s,
            "vix_regime": vix_regime_series,   # ← string Series, real format
            "spx_futures_close": None,
        }

        model = ShortTrendModel()
        result = model.train(df, indicators)

        # Training must succeed
        assert result.get("accuracy", 0) > 0 or result.get("val_accuracy", 0) >= 0, (
            "Training must return a result dict even with synthetic data"
        )

        # The persisted walk-forward report must contain regime_breakdown
        loaded = cal.get_walkforward_report("short_trend")
        assert loaded is not None, (
            "short_trend_walkforward.json must be written after training"
        )
        assert loaded.get("evaluated") is True, (
            "Walk-forward evaluation must have run successfully with 600 rows"
        )
        assert "regime_breakdown" in loaded, (
            "regime_breakdown must be persisted when a string vix_regime Series "
            "is supplied — string→int conversion must not silently fail"
        )
        breakdown = loaded["regime_breakdown"]
        assert len(breakdown) >= 2, (
            "At least two VIX regimes must appear in the OOS folds given the "
            "uniform LOW/NORMAL/HIGH/EXTREME distribution in training data"
        )
        # Verify the regime names are the expected string labels, not raw ints
        regime_names = {row["regime"] for row in breakdown}
        assert regime_names.issubset({"LOW", "NORMAL", "HIGH", "EXTREME"}), (
            f"Unexpected regime names in breakdown: {regime_names}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Integration test: save_walkforward_report persists regime_breakdown and
# get_walkforward_report can read it back.
# ─────────────────────────────────────────────────────────────────────────────

class TestWalkForwardReportPersistence:
    def test_regime_breakdown_round_trips_through_json(self, tmp_path, monkeypatch):
        import ml.calibration as cal

        monkeypatch.setattr(cal, "MODEL_DIR", tmp_path)

        sample_metrics = {
            "evaluated": True,
            "oos_accuracy": 0.55,
            "oos_brier_score": 0.24,
            "regime_breakdown": [
                {
                    "regime": "LOW",
                    "regime_code": 0,
                    "oos_samples": 120,
                    "oos_accuracy": 0.58,
                    "majority_baseline_accuracy": 0.52,
                    "accuracy_lift_vs_majority": 0.06,
                    "oos_brier_score": 0.22,
                    "positive_rate": 0.48,
                },
                {
                    "regime": "HIGH",
                    "regime_code": 2,
                    "oos_samples": 40,
                    "oos_accuracy": 0.47,
                    "majority_baseline_accuracy": 0.53,
                    "accuracy_lift_vs_majority": -0.06,
                    "oos_brier_score": 0.26,
                    "positive_rate": 0.47,
                },
            ],
        }

        cal.save_walkforward_report("short_trend", sample_metrics)

        loaded = cal.get_walkforward_report("short_trend")
        assert loaded is not None
        assert loaded.get("evaluated") is True
        assert "regime_breakdown" in loaded

        breakdown = loaded["regime_breakdown"]
        assert len(breakdown) == 2
        regimes = {r["regime"] for r in breakdown}
        assert regimes == {"LOW", "HIGH"}
        low_row = next(r for r in breakdown if r["regime"] == "LOW")
        assert low_row["oos_samples"] == 120
        assert abs(low_row["accuracy_lift_vs_majority"] - 0.06) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Integration test: /api/healthz surfaces regime_breakdown under
# models.short_trend.walk_forward when the report file contains it.
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthzShortTrendRegimeBreakdown:
    @pytest.mark.asyncio
    async def test_healthz_surfaces_short_trend_regime_breakdown(
        self, tmp_path, monkeypatch
    ):
        import ml.calibration as cal

        monkeypatch.setattr(cal, "MODEL_DIR", tmp_path)

        # Write a pre-built short_trend walkforward report with regime_breakdown
        report = {
            "evaluated": True,
            "oos_accuracy": 0.54,
            "oos_brier_score": 0.25,
            "regime_breakdown": [
                {
                    "regime": "NORMAL",
                    "regime_code": 1,
                    "oos_samples": 200,
                    "oos_accuracy": 0.56,
                    "majority_baseline_accuracy": 0.51,
                    "accuracy_lift_vs_majority": 0.05,
                    "oos_brier_score": 0.24,
                    "positive_rate": 0.49,
                },
                {
                    "regime": "EXTREME",
                    "regime_code": 3,
                    "oos_samples": 30,
                    "oos_accuracy": 0.43,
                    "majority_baseline_accuracy": 0.57,
                    "accuracy_lift_vs_majority": -0.14,
                    "oos_brier_score": 0.28,
                    "positive_rate": 0.43,
                },
            ],
            "generated_at": "2026-08-01T12:00:00+00:00",
        }
        report_path = tmp_path / "short_trend_walkforward.json"
        report_path.write_text(json.dumps(report))

        from httpx import AsyncClient, ASGITransport
        from main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/healthz")

        assert resp.status_code == 200
        body = resp.json()

        short = body.get("models", {}).get("short_trend", {})
        wf = short.get("walk_forward")
        assert wf is not None, (
            "/api/healthz must include models.short_trend.walk_forward"
        )
        assert "regime_breakdown" in wf, (
            "regime_breakdown must be present in models.short_trend.walk_forward "
            "when the persisted report contains it"
        )
        breakdown = wf["regime_breakdown"]
        assert len(breakdown) == 2
        extreme_row = next(
            (r for r in breakdown if r["regime"] == "EXTREME"), None
        )
        assert extreme_row is not None
        assert extreme_row["accuracy_lift_vs_majority"] < 0, (
            "EXTREME regime row must show negative lift (model degrades under stress)"
        )
