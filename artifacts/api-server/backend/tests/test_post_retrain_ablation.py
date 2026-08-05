"""
Tests for ml/post_retrain_ablation.py

Covers:
  - _build_matrices returns X_19 (19 cols), X_27 (27 cols), aligned rows
  - X_19 is an exact prefix of X_27 (first 19 columns are identical)
  - Feature matrices contain no NaN values
  - Sample weights are finite, non-negative, and mean-normalized to ≈ 1
  - _append_to_json creates a list on first write and appends on subsequent calls
  - _append_to_json promotes a legacy single-object file to a list
  - run_broader_context_ablation returns a dict with the required top-level keys
  - run_broader_context_ablation appends a record to the JSON file
  - run_broader_context_ablation never raises, returning {} on failure
  - The structured gate-pass log line is emitted when the 27-feat model wins
  - LONG_BROADER_CONTEXT_ENABLED is restored to its original value after the call
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))


# ─────────────────────────────────────────────────────────────────────────────
# Shared synthetic data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_voo(n: int = 700, seed: int = 7, vol: float = 2.5) -> pd.DataFrame:
    """Synthetic daily VOO-like DataFrame with large vol so ≥ 2% moves occur."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-02", periods=n)
    price = np.maximum(300.0 + np.cumsum(rng.normal(0.05, vol, n)), 1.0)
    return pd.DataFrame(
        {
            "open":  price - rng.uniform(0, 0.5, n),
            "high":  price + rng.uniform(0, 1.5, n),
            "low":   price - rng.uniform(0, 1.5, n),
            "close": price,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
            "is_extended_hours": False,
        },
        index=idx,
    )


def _make_vix(voo_idx: pd.DatetimeIndex) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    vix = np.clip(15.0 + np.cumsum(rng.normal(0, 0.5, len(voo_idx))), 10, 80)
    return pd.DataFrame(
        {"open": vix, "high": vix + 0.5, "low": vix - 0.5, "close": vix, "volume": 0.0},
        index=voo_idx,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Module-level fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ablation_mod():
    """Import ml.post_retrain_ablation once per module."""
    import ml.post_retrain_ablation as mod
    return mod


@pytest.fixture(scope="module")
def synthetic_voo():
    return _make_voo()


@pytest.fixture(scope="module")
def synthetic_vix(synthetic_voo):
    return _make_vix(synthetic_voo.index)


@pytest.fixture(scope="module")
def empty_spx():
    return pd.Series(dtype=float)


@pytest.fixture(scope="module")
def empty_ctx():
    return {}


@pytest.fixture(scope="module")
def matrices(ablation_mod, synthetic_voo, synthetic_vix, empty_spx, empty_ctx):
    """Build the feature matrices once and share across tests in this module."""
    return ablation_mod._build_matrices(
        synthetic_voo, synthetic_vix, empty_spx, empty_ctx,
        horizon=21, threshold=0.02,
    )


# ─────────────────────────────────────────────────────────────────────────────
# _build_matrices tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildMatrices:
    def test_x19_has_19_columns(self, matrices):
        X_19 = matrices[0]
        assert X_19.shape[1] == 19, f"Expected 19 cols, got {X_19.shape[1]}"

    def test_x27_has_27_columns(self, matrices):
        X_27 = matrices[1]
        assert X_27.shape[1] == 27, f"Expected 27 cols, got {X_27.shape[1]}"

    def test_same_row_count(self, matrices):
        X_19, X_27, y, weights, timestamps, majority_baseline, *_ = matrices
        assert X_19.shape[0] == X_27.shape[0] == len(y) == len(weights)

    def test_x19_is_prefix_of_x27(self, matrices):
        """The first 19 columns of X_27 must be identical to X_19."""
        X_19, X_27 = matrices[0], matrices[1]
        np.testing.assert_array_equal(
            X_19, X_27[:, :19],
            err_msg="First 19 columns of X_27 differ from X_19",
        )

    def test_no_nans(self, matrices):
        X_19, X_27, y, weights = matrices[0], matrices[1], matrices[2], matrices[3]
        assert not np.isnan(X_19).any(),     "NaN found in X_19"
        assert not np.isnan(X_27).any(),     "NaN found in X_27"
        assert not np.isnan(y).any(),        "NaN found in y"
        assert not np.isnan(weights).any(),  "NaN found in weights"

    def test_weights_normalized(self, matrices):
        weights = matrices[3]
        assert math.isfinite(float(weights.mean()))
        assert float(weights.min()) >= 0.0
        # Mean should be approximately 1 (class-balanced + mean-normalized)
        assert 0.5 <= float(weights.mean()) <= 2.0

    def test_majority_baseline_in_range(self, matrices):
        majority_baseline = matrices[5]
        assert 0.5 <= majority_baseline <= 1.0

    def test_raises_on_insufficient_data(self, ablation_mod, empty_spx, empty_ctx):
        """Fewer than 80 labeled rows should raise ValueError."""
        tiny_voo = _make_voo(n=30, seed=42, vol=0.01)  # very low vol → few 2% moves
        tiny_vix = _make_vix(tiny_voo.index)
        with pytest.raises(ValueError, match="Too few labeled rows"):
            ablation_mod._build_matrices(
                tiny_voo, tiny_vix, empty_spx, empty_ctx,
                horizon=21, threshold=0.02,
            )

    def test_flag_not_mutated_by_build_matrices(self, ablation_mod, synthetic_voo,
                                                synthetic_vix, empty_spx, empty_ctx):
        """_build_matrices must never write to settings.LONG_BROADER_CONTEXT_ENABLED."""
        from config import settings
        before = settings.LONG_BROADER_CONTEXT_ENABLED
        ablation_mod._build_matrices(
            synthetic_voo, synthetic_vix, empty_spx, empty_ctx,
            horizon=21, threshold=0.02,
        )
        after = settings.LONG_BROADER_CONTEXT_ENABLED
        assert after == before, (
            f"_build_matrices mutated LONG_BROADER_CONTEXT_ENABLED "
            f"({before!r} → {after!r})"
        )

    def test_x19_correct_when_flag_true(self, ablation_mod, synthetic_voo,
                                        synthetic_vix, empty_spx, empty_ctx):
        """X_19 should have exactly 19 columns regardless of the live flag value."""
        from config import settings
        original = settings.LONG_BROADER_CONTEXT_ENABLED
        try:
            settings.LONG_BROADER_CONTEXT_ENABLED = True  # type: ignore[assignment]
            result = ablation_mod._build_matrices(
                synthetic_voo, synthetic_vix, empty_spx, empty_ctx,
                horizon=21, threshold=0.02,
            )
            X_19 = result[0]
            assert X_19.shape[1] == 19, (
                f"X_19 should have 19 cols even when flag=True, got {X_19.shape[1]}"
            )
        finally:
            settings.LONG_BROADER_CONTEXT_ENABLED = original  # type: ignore[assignment]


# ─────────────────────────────────────────────────────────────────────────────
# _append_to_json tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAppendToJson:
    def test_creates_list_on_first_write(self, ablation_mod, tmp_path):
        out = tmp_path / "abl.json"
        ablation_mod._append_to_json({"foo": 1}, out)
        data = json.loads(out.read_text())
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["foo"] == 1

    def test_appends_on_second_write(self, ablation_mod, tmp_path):
        out = tmp_path / "abl.json"
        ablation_mod._append_to_json({"run": 1}, out)
        ablation_mod._append_to_json({"run": 2}, out)
        data = json.loads(out.read_text())
        assert len(data) == 2
        assert data[0]["run"] == 1
        assert data[1]["run"] == 2

    def test_promotes_legacy_single_object(self, ablation_mod, tmp_path):
        """A file that already contains a single JSON object is promoted to a list."""
        out = tmp_path / "legacy.json"
        out.write_text(json.dumps({"old": True, "passes_promotion_gate": False}))
        ablation_mod._append_to_json({"new": True}, out)
        data = json.loads(out.read_text())
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["old"] is True
        assert data[1]["new"] is True

    def test_handles_corrupt_file_gracefully(self, ablation_mod, tmp_path):
        """A corrupted JSON file is replaced with a fresh single-element list."""
        out = tmp_path / "corrupt.json"
        out.write_text("NOT VALID JSON {{{{")
        ablation_mod._append_to_json({"salvaged": True}, out)
        data = json.loads(out.read_text())
        assert isinstance(data, list)
        assert data[0]["salvaged"] is True

    def test_creates_parent_dirs(self, ablation_mod, tmp_path):
        out = tmp_path / "nested" / "dir" / "abl.json"
        ablation_mod._append_to_json({"x": 9}, out)
        assert out.exists()


# ─────────────────────────────────────────────────────────────────────────────
# run_broader_context_ablation integration tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRunBroaderContextAblation:
    """End-to-end tests using synthetic data and a temp output path."""

    @pytest.fixture(scope="class")
    def ablation_result(self, ablation_mod, synthetic_voo, synthetic_vix,
                        empty_spx, empty_ctx, tmp_path_factory):
        out = tmp_path_factory.mktemp("abl") / "ablation_broader_context.json"
        result = ablation_mod.run_broader_context_ablation(
            synthetic_voo, synthetic_vix, empty_spx, empty_ctx,
            horizon=21, threshold=0.02, n_splits=3,
            out_path=out,
        )
        return result, out

    def test_returns_dict(self, ablation_result):
        result, _ = ablation_result
        assert isinstance(result, dict)
        assert result  # non-empty

    def test_required_top_level_keys(self, ablation_result):
        result, _ = ablation_result
        required = {
            "ablation", "run_timestamp_utc", "trigger", "data_source",
            "horizon_days", "meaningful_move_threshold", "n_splits",
            "n_labeled_rows", "date_range_start", "date_range_end",
            "positive_rate", "majority_baseline", "LONG_MIN_OOS_ACCURACY_LIFT",
            "baseline_19feat", "candidate_27feat",
            "accuracy_delta_27_minus_19", "passes_promotion_gate",
            "recommendation",
        }
        missing = required - result.keys()
        assert not missing, f"Missing keys: {missing}"

    def test_trigger_is_post_retrain(self, ablation_result):
        result, _ = ablation_result
        assert result["trigger"] == "post_retrain"

    def test_feature_counts(self, ablation_result):
        result, _ = ablation_result
        assert result["baseline_19feat"]["n_features"] == 19
        assert result["candidate_27feat"]["n_features"] == 27

    def test_passes_promotion_gate_is_bool(self, ablation_result):
        result, _ = ablation_result
        assert isinstance(result["passes_promotion_gate"], bool)

    def test_recommendation_string(self, ablation_result):
        result, _ = ablation_result
        passes = result["passes_promotion_gate"]
        rec = result["recommendation"]
        if passes:
            assert "LONG_BROADER_CONTEXT_ENABLED=True" in rec
        else:
            assert "LONG_BROADER_CONTEXT_ENABLED=False" in rec

    def test_appended_to_json_file(self, ablation_result):
        _, out = ablation_result
        assert out.exists(), "Output JSON file was not created"
        data = json.loads(out.read_text())
        assert isinstance(data, list)
        assert len(data) >= 1
        assert data[-1]["trigger"] == "post_retrain"

    def test_accuracies_in_range(self, ablation_result):
        result, _ = ablation_result
        acc_19 = result["baseline_19feat"]["oos_accuracy"]
        acc_27 = result["candidate_27feat"]["oos_accuracy"]
        assert 0.0 <= acc_19 <= 1.0
        assert 0.0 <= acc_27 <= 1.0

    def test_second_call_appends_not_overwrites(self, ablation_mod, ablation_result,
                                                 synthetic_voo, synthetic_vix,
                                                 empty_spx, empty_ctx):
        _, out = ablation_result
        ablation_mod.run_broader_context_ablation(
            synthetic_voo, synthetic_vix, empty_spx, empty_ctx,
            horizon=21, threshold=0.02, n_splits=3, out_path=out,
        )
        data = json.loads(out.read_text())
        assert len(data) >= 2, "Second call should append, not overwrite"

    def test_flag_not_mutated_after_ablation(self, ablation_mod, synthetic_voo,
                                            synthetic_vix, empty_spx, empty_ctx,
                                            tmp_path):
        """run_broader_context_ablation must never write to LONG_BROADER_CONTEXT_ENABLED."""
        from config import settings
        before = settings.LONG_BROADER_CONTEXT_ENABLED
        ablation_mod.run_broader_context_ablation(
            synthetic_voo, synthetic_vix, empty_spx, empty_ctx,
            out_path=tmp_path / "abl.json",
        )
        after = settings.LONG_BROADER_CONTEXT_ENABLED
        assert after == before, (
            f"run_broader_context_ablation mutated LONG_BROADER_CONTEXT_ENABLED "
            f"({before!r} → {after!r})"
        )

    def test_never_raises_on_empty_dataframe(self, ablation_mod, tmp_path):
        """An empty daily_df must not raise — must return {} and log an error."""
        result = ablation_mod.run_broader_context_ablation(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.Series(dtype=float),
            {},
            out_path=tmp_path / "abl_empty.json",
        )
        assert result == {}

    def test_never_raises_on_bad_out_path(self, ablation_mod, synthetic_voo,
                                           synthetic_vix, empty_spx, empty_ctx):
        """A read-only or non-writable out_path should not raise."""
        result = ablation_mod.run_broader_context_ablation(
            synthetic_voo, synthetic_vix, empty_spx, empty_ctx,
            out_path=Path("/dev/null/cannot_write_here/abl.json"),
        )
        # Result may be {} (write failed) or a valid dict — either is acceptable
        # as long as no exception propagated.
        assert isinstance(result, dict)


class TestGatePassLogLine:
    """Verify the structured WARNING log is emitted when the gate passes.

    walk_forward_evaluate and settings are imported lazily inside the function,
    so they cannot be patched as module-level attributes of post_retrain_ablation.
    Instead:
      - walk_forward_evaluate is patched at ml.calibration (the source module)
      - _build_matrices is patched directly on the module object (it IS a module
        attribute because it lives in post_retrain_ablation.py)
      - _append_to_json is patched the same way
      - caplog captures the WARNING record emitted by the module logger
    """

    def test_gate_pass_emits_warning_log(self, ablation_mod, synthetic_voo,
                                          synthetic_vix, empty_spx, empty_ctx,
                                          tmp_path, caplog):
        """When passes_promotion_gate=True, a WARNING containing 'gate_pass' is logged."""
        call_count = {"n": 0}

        def _mock_wf(X, y, weights, model_factory, n_splits, embargo):
            # First call = 19-feat baseline (low accuracy)
            # Second call = 27-feat candidate (high accuracy → gate passes)
            call_count["n"] += 1
            acc = 0.52 if call_count["n"] == 1 else 0.75
            return (
                {"evaluated": True, "oos_accuracy": acc,
                 "oos_balanced_accuracy": acc - 0.01, "folds": []},
                None, None,
            )

        n = 100
        rng = np.random.default_rng(0)
        fake_X19 = rng.random((n, 19)).astype(np.float32)
        fake_X27 = np.hstack([fake_X19, rng.random((n, 8)).astype(np.float32)])
        fake_y   = rng.integers(0, 2, n)
        fake_w   = np.ones(n, dtype=np.float32)
        fake_ts  = pd.date_range("2020-01-01", periods=n, freq="B")
        base_names = [f"f{i}" for i in range(19)]
        ctx_names  = [f"c{i}" for i in range(8)]

        with patch("ml.calibration.walk_forward_evaluate", side_effect=_mock_wf), \
             patch.object(ablation_mod, "_build_matrices",
                          return_value=(fake_X19, fake_X27, fake_y, fake_w,
                                       fake_ts, 0.55, base_names, ctx_names)), \
             patch.object(ablation_mod, "_append_to_json"), \
             caplog.at_level(logging.WARNING, logger="ml.post_retrain_ablation"):
            ablation_mod.run_broader_context_ablation(
                synthetic_voo, synthetic_vix, empty_spx, empty_ctx,
                out_path=tmp_path / "abl.json",
            )

        gate_pass_lines = [r.message for r in caplog.records
                           if "gate_pass" in r.message]
        assert gate_pass_lines, (
            "Expected at least one WARNING log record containing 'gate_pass'. "
            f"Captured records: {[r.message for r in caplog.records]}"
        )
