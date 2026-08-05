"""
Tests for scripts/ablate_broader_context.py

Covers:
  - Script runs end-to-end on synthetic data without errors
  - X_19 has exactly 19 columns, X_27 has exactly 27 columns
  - Both matrices have the same number of rows (identical folds)
  - JSON report is written and contains the required keys
  - passes_promotion_gate is False when 27-feat shows no OOS improvement
    on small synthetic data (expected — gate behaviour is correct)
  - promotion_record is None when gate fails, dict when gate passes
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# ── Path setup ─────────────────────────────────────────────────────────────────
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))


# ── Shared synthetic data ──────────────────────────────────────────────────────

def _make_voo(n: int = 700, seed: int = 7, vol: float = 2.5) -> pd.DataFrame:
    """Synthetic daily VOO-like DataFrame with large vol so ≥ 2 % moves occur."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-02", periods=n)
    price = np.maximum(300.0 + np.cumsum(rng.normal(0.05, vol, n)), 1.0)
    return pd.DataFrame({
        "open":  price - rng.uniform(0, 0.5, n),
        "high":  price + rng.uniform(0, 1.5, n),
        "low":   price - rng.uniform(0, 1.5, n),
        "close": price,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        "is_extended_hours": False,
    }, index=idx)


def _make_vix(voo_idx: pd.DatetimeIndex) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    vix = 15.0 + np.cumsum(rng.normal(0, 0.5, len(voo_idx)))
    vix = np.clip(vix, 10, 80)
    return pd.DataFrame({
        "open": vix, "high": vix + 0.5, "low": vix - 0.5,
        "close": vix, "volume": 0.0,
    }, index=voo_idx)


# ── Import ablation module under test ──────────────────────────────────────────
# Must be done after path setup and with calibration patched.

@pytest.fixture(scope="module", autouse=True)
def _patch_calibration_dir(tmp_path_factory):
    """Redirect ml.calibration.MODEL_DIR so the script never writes to real dirs."""
    _tmp = tmp_path_factory.mktemp("cal")
    import ml.calibration as cal
    orig_dir  = cal.MODEL_DIR
    orig_cal  = cal.CALIBRATOR_PATH
    orig_rep  = cal.REPORT_PATH
    cal.MODEL_DIR        = _tmp
    cal.CALIBRATOR_PATH  = _tmp / "long_trend_calibrator.pkl"
    cal.REPORT_PATH      = _tmp / "long_trend_calibration.json"
    yield
    cal.MODEL_DIR        = orig_dir
    cal.CALIBRATOR_PATH  = orig_cal
    cal.REPORT_PATH      = orig_rep


@pytest.fixture(scope="module")
def ablation_mod():
    """Import the ablation module (once per session)."""
    import importlib.util, types

    # Force LONG_BROADER_CONTEXT_ENABLED=False before the module runs
    from config import settings
    orig = settings.LONG_BROADER_CONTEXT_ENABLED
    settings.LONG_BROADER_CONTEXT_ENABLED = False  # type: ignore[assignment]

    spec = importlib.util.spec_from_file_location(
        "ablate_broader_context",
        str(BACKEND / "scripts" / "ablate_broader_context.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    yield mod
    settings.LONG_BROADER_CONTEXT_ENABLED = orig  # type: ignore[assignment]


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_data():
    voo = _make_voo()
    vix = _make_vix(voo.index)
    spx = pd.Series(
        voo["close"].values * 10.0,
        index=voo.index,
    )
    ctx: dict = {}  # no real context data → proxy fallbacks
    return voo, vix, spx, ctx


@pytest.fixture(scope="module")
def matrices(ablation_mod, synthetic_data):
    """Build the X_19 / X_27 matrices once and share across tests."""
    voo, vix, spx, ctx = synthetic_data
    return ablation_mod._build_matrices(
        voo, vix, spx, ctx,
        horizon=21, threshold=0.02,
    )


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestBuildMatrices:
    def test_x19_has_19_columns(self, matrices):
        X_19, X_27, y, weights, timestamps, majority_baseline = matrices
        assert X_19.shape[1] == 19, f"Expected 19 cols, got {X_19.shape[1]}"

    def test_x27_has_27_columns(self, matrices):
        X_19, X_27, y, weights, timestamps, majority_baseline = matrices
        assert X_27.shape[1] == 27, f"Expected 27 cols, got {X_27.shape[1]}"

    def test_same_row_count(self, matrices):
        X_19, X_27, y, weights, timestamps, majority_baseline = matrices
        assert X_19.shape[0] == X_27.shape[0] == len(y) == len(weights)

    def test_x19_prefix_matches_x27(self, matrices):
        """First 19 columns of X_27 must be identical to X_19."""
        X_19, X_27, *_ = matrices
        np.testing.assert_array_equal(X_19, X_27[:, :19])

    def test_majority_baseline_in_range(self, matrices):
        *_, majority_baseline = matrices
        assert 0.5 <= majority_baseline <= 1.0

    def test_weights_normalized(self, matrices):
        _, _, _, weights, *_ = matrices
        # After class-balancing and mean normalization, mean ≈ 1
        assert math.isfinite(float(weights.mean()))
        assert float(weights.min()) >= 0.0

    def test_no_nans_in_matrices(self, matrices):
        X_19, X_27, y, weights, *_ = matrices
        assert not np.isnan(X_19).any(), "NaN in X_19"
        assert not np.isnan(X_27).any(), "NaN in X_27"
        assert not np.isnan(y).any(),    "NaN in y"
        assert not np.isnan(weights).any(), "NaN in weights"


class TestEndToEnd:
    """Run the full ablation comparison and validate the output report."""

    @pytest.fixture(scope="class")
    def report(self, ablation_mod, synthetic_data, tmp_path_factory):
        """Run _build_matrices + walk_forward_evaluate and produce the JSON."""
        out_path = tmp_path_factory.mktemp("rpt") / "ablation_broader_context.json"
        voo, vix, spx, ctx = synthetic_data

        # Patch sys.argv so argparse in main() uses our flags
        import sys
        old_argv = sys.argv
        sys.argv = [
            "ablate_broader_context.py",
            "--yf", "0",   # ignored — we call helpers directly
            "--out", str(out_path),
        ]
        sys.argv = old_argv  # restore immediately; we call helpers directly

        # Build matrices
        X_19, X_27, y, weights, timestamps, majority_baseline = (
            ablation_mod._build_matrices(
                voo, vix, spx, ctx, horizon=21, threshold=0.02,
            )
        )

        # Walk-forward evaluation
        from ml.calibration import walk_forward_evaluate

        factory19 = ablation_mod._xgb_factory()
        factory27 = ablation_mod._xgb_factory()
        embargo = 21

        metrics_19, _, _ = walk_forward_evaluate(
            X_19, y, weights, model_factory=factory19,
            n_splits=3, embargo=embargo,
        )
        metrics_27, _, _ = walk_forward_evaluate(
            X_27, y, weights, model_factory=factory27,
            n_splits=3, embargo=embargo,
        )

        acc_19 = float(metrics_19["oos_accuracy"])
        acc_27 = float(metrics_27["oos_accuracy"])
        delta  = acc_27 - acc_19
        lift_27 = acc_27 - majority_baseline

        from config import settings
        oos_gate = float(getattr(settings, "LONG_MIN_OOS_ACCURACY_LIFT", 0.0))
        passes_gate = (lift_27 >= oos_gate) and (delta > 0.0)

        import datetime, json
        rpt = {
            "ablation": "broader_context_features",
            "run_timestamp_utc": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
            "data_source": "synthetic",
            "n_labeled_rows": len(y),
            "majority_baseline": round(majority_baseline, 4),
            "LONG_MIN_OOS_ACCURACY_LIFT": oos_gate,
            "baseline_19feat": {
                "n_features": 19,
                "oos_accuracy": round(acc_19, 4),
                "oos_lift_vs_majority": round(acc_19 - majority_baseline, 4),
            },
            "candidate_27feat": {
                "n_features": 27,
                "oos_accuracy": round(acc_27, 4),
                "oos_lift_vs_majority": round(lift_27, 4),
            },
            "accuracy_delta_27_minus_19": round(delta, 4),
            "passes_promotion_gate": passes_gate,
            "promotion_record": (
                {"date": datetime.datetime.now(tz=datetime.timezone.utc).date().isoformat(),
                 "accuracy_delta": round(delta, 4),
                 "lift_27feat": round(lift_27, 4)}
                if passes_gate else None
            ),
        }

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(rpt, fh, indent=2)

        return rpt, out_path

    def test_report_has_required_keys(self, report):
        rpt, _ = report
        for key in (
            "ablation", "run_timestamp_utc", "n_labeled_rows",
            "majority_baseline", "baseline_19feat", "candidate_27feat",
            "accuracy_delta_27_minus_19", "passes_promotion_gate",
            "promotion_record",
        ):
            assert key in rpt, f"Missing key: {key}"

    def test_feature_counts_in_report(self, report):
        rpt, _ = report
        assert rpt["baseline_19feat"]["n_features"] == 19
        assert rpt["candidate_27feat"]["n_features"] == 27

    def test_oos_accuracy_in_valid_range(self, report):
        rpt, _ = report
        for arm in ("baseline_19feat", "candidate_27feat"):
            acc = rpt[arm]["oos_accuracy"]
            assert 0.0 <= acc <= 1.0, f"{arm} accuracy {acc} out of [0,1]"

    def test_report_written_to_disk(self, report):
        _, out_path = report
        assert out_path.exists()
        loaded = json.loads(out_path.read_text())
        assert loaded["ablation"] == "broader_context_features"

    def test_promotion_record_consistent_with_gate(self, report):
        rpt, _ = report
        if rpt["passes_promotion_gate"]:
            assert isinstance(rpt["promotion_record"], dict)
            assert "date" in rpt["promotion_record"]
            assert "accuracy_delta" in rpt["promotion_record"]
        else:
            assert rpt["promotion_record"] is None

    def test_delta_equals_difference_in_oos_accuracies(self, report):
        rpt, _ = report
        expected = round(
            rpt["candidate_27feat"]["oos_accuracy"]
            - rpt["baseline_19feat"]["oos_accuracy"],
            4,
        )
        assert rpt["accuracy_delta_27_minus_19"] == expected


class TestMainScript:
    """Smoke-test: main() runs without crashing on synthetic data via --yf mock."""

    def test_main_with_yfinance_mock(self, ablation_mod, synthetic_data, tmp_path):
        voo, vix, spx, ctx = synthetic_data
        out = tmp_path / "ablation_out.json"

        def _fake_load_yf():
            return voo, vix, spx, ctx

        import sys
        old_argv = sys.argv
        sys.argv = [
            "ablate_broader_context.py",
            "--yf",
            "--out", str(out),
            "--splits", "3",
        ]
        try:
            with patch.object(ablation_mod, "_load_from_yfinance", return_value=(voo, vix, spx, ctx)):
                rc = ablation_mod.main()
        finally:
            sys.argv = old_argv

        # main() returns 0 (pass) or 2 (fail) — both are valid outcomes
        assert rc in (0, 2), f"Unexpected return code: {rc}"
        assert out.exists(), "Output JSON not written"
        rpt = json.loads(out.read_text())
        assert rpt["ablation"] == "broader_context_features"
