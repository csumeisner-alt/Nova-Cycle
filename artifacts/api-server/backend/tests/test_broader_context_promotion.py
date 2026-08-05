"""
Tests for broader-context automatic promotion after a gate-passing retrain.

Covers:
  - record_broader_context_promotion writes and is idempotent on the same day
  - get_broader_context_promotion returns None when no file exists
  - should_send_broader_context_promotion_alert respects alert_sent flag
  - mark_broader_context_promotion_alert_sent marks the alert as sent
  - post_retrain_ablation.run_broader_context_ablation writes a promotion record on gate pass
  - LONG_BROADER_CONTEXT_AUTO_ENABLE=True flips the in-memory flag
  - LONG_BROADER_CONTEXT_AUTO_ENABLE=False does not flip the flag
  - FCMNotifier.send_broader_context_promotion_alert is a no-op when no FCM key
  - healthz returns broader_context_promotion field (null when absent, dict when present)
  - healthz adds an operator alert when promotion exists but flag is still False
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_voo(n: int = 700, seed: int = 7, vol: float = 2.5) -> pd.DataFrame:
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


# ─────────────────────────────────────────────────────────────────────────────
# Training-status promotion tracking
# ─────────────────────────────────────────────────────────────────────────────

class TestBroaderContextPromotion:
    """Unit tests for the promotion tracking functions in training_status.py."""

    @pytest.fixture(autouse=True)
    def _patch_promo_path(self, tmp_path):
        """Redirect PROMOTION_PATH to a temp directory for each test."""
        import ml.training_status as ts
        original = ts.PROMOTION_PATH
        ts.PROMOTION_PATH = tmp_path / "broader_context_promotion.json"
        yield
        ts.PROMOTION_PATH = original

    def test_get_returns_none_when_no_file(self):
        from ml.training_status import get_broader_context_promotion
        assert get_broader_context_promotion() is None

    def test_record_writes_file(self):
        from ml.training_status import record_broader_context_promotion, get_broader_context_promotion
        record_broader_context_promotion(delta=0.03, lift=0.04, acc_27=0.62)
        promo = get_broader_context_promotion()
        assert promo is not None
        assert promo["accuracy_delta_27_minus_19"] == pytest.approx(0.03)
        assert promo["oos_lift_27feat"] == pytest.approx(0.04)
        assert promo["oos_accuracy_27feat"] == pytest.approx(0.62)
        assert promo["auto_enabled"] is False
        assert promo["alert_sent"] is False
        assert "promoted_at_utc" in promo

    def test_record_auto_enabled_flag(self):
        from ml.training_status import record_broader_context_promotion, get_broader_context_promotion
        record_broader_context_promotion(delta=0.02, lift=0.03, acc_27=0.60, auto_enabled=True)
        promo = get_broader_context_promotion()
        assert promo["auto_enabled"] is True

    def test_record_idempotent_same_day(self):
        """Second call on the same day preserves original timestamp."""
        from ml.training_status import record_broader_context_promotion, get_broader_context_promotion
        record_broader_context_promotion(delta=0.02, lift=0.03, acc_27=0.60)
        promo1 = get_broader_context_promotion()
        ts1 = promo1["promoted_at_utc"]

        # Second call with different values — same-day, timestamp must be preserved
        record_broader_context_promotion(delta=0.05, lift=0.06, acc_27=0.65)
        promo2 = get_broader_context_promotion()
        assert promo2["promoted_at_utc"] == ts1, (
            "Second call on the same day must not overwrite the original timestamp"
        )

    def test_record_auto_enabled_upgrades_false_to_true(self):
        """auto_enabled upgrading False→True on second call is persisted."""
        from ml.training_status import record_broader_context_promotion, get_broader_context_promotion
        record_broader_context_promotion(delta=0.02, lift=0.03, acc_27=0.60, auto_enabled=False)
        assert get_broader_context_promotion()["auto_enabled"] is False

        record_broader_context_promotion(delta=0.02, lift=0.03, acc_27=0.60, auto_enabled=True)
        assert get_broader_context_promotion()["auto_enabled"] is True

    def test_should_send_alert_true_initially(self):
        from ml.training_status import (
            record_broader_context_promotion,
            should_send_broader_context_promotion_alert,
        )
        record_broader_context_promotion(delta=0.02, lift=0.03, acc_27=0.60)
        assert should_send_broader_context_promotion_alert() is True

    def test_should_send_alert_false_when_no_promotion(self):
        from ml.training_status import should_send_broader_context_promotion_alert
        assert should_send_broader_context_promotion_alert() is False

    def test_should_send_alert_false_after_sent(self):
        from ml.training_status import (
            record_broader_context_promotion,
            should_send_broader_context_promotion_alert,
            mark_broader_context_promotion_alert_sent,
        )
        record_broader_context_promotion(delta=0.02, lift=0.03, acc_27=0.60)
        assert should_send_broader_context_promotion_alert() is True
        mark_broader_context_promotion_alert_sent()
        assert should_send_broader_context_promotion_alert() is False

    def test_mark_alert_sent_persists(self):
        from ml.training_status import (
            record_broader_context_promotion,
            mark_broader_context_promotion_alert_sent,
            get_broader_context_promotion,
        )
        record_broader_context_promotion(delta=0.02, lift=0.03, acc_27=0.60)
        mark_broader_context_promotion_alert_sent()
        promo = get_broader_context_promotion()
        assert promo["alert_sent"] is True

    def test_mark_alert_sent_no_op_when_no_file(self):
        """mark_broader_context_promotion_alert_sent must not raise when file absent."""
        from ml.training_status import mark_broader_context_promotion_alert_sent
        mark_broader_context_promotion_alert_sent()  # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# post_retrain_ablation: promotion record written on gate pass
# ─────────────────────────────────────────────────────────────────────────────

class TestAblationWritesPromotionRecord:
    """Verify run_broader_context_ablation calls record_broader_context_promotion."""

    def test_promotion_record_written_on_gate_pass(self, tmp_path):
        """When passes_gate=True, record_broader_context_promotion is called."""
        import ml.post_retrain_ablation as mod

        promo_calls = []

        n = 80
        rng = np.random.default_rng(0)
        fake_X19 = rng.random((n, 19)).astype(np.float32)
        fake_X27 = np.hstack([fake_X19, rng.random((n, 8)).astype(np.float32)])
        fake_y   = rng.integers(0, 2, n)
        fake_w   = np.ones(n, dtype=np.float32)
        fake_ts  = pd.date_range("2020-01-01", periods=n, freq="B")

        def _mock_wf(X, y, weights, model_factory, n_splits, embargo):
            # 19-feat: below baseline; 27-feat: beats gate → passes_gate=True
            acc = 0.52 if X.shape[1] == 19 else 0.75
            return (
                {"evaluated": True, "oos_accuracy": acc,
                 "oos_balanced_accuracy": acc - 0.01, "folds": []},
                None, None,
            )

        with patch("ml.calibration.walk_forward_evaluate", side_effect=_mock_wf), \
             patch.object(mod, "_build_matrices",
                          return_value=(fake_X19, fake_X27, fake_y, fake_w,
                                        fake_ts, 0.50,
                                        [f"f{i}" for i in range(19)],
                                        [f"c{i}" for i in range(8)])), \
             patch.object(mod, "_append_to_json"), \
             patch("ml.training_status.record_broader_context_promotion",
                   side_effect=lambda **kw: promo_calls.append(kw)):
            mod.run_broader_context_ablation(
                pd.DataFrame(), pd.DataFrame(),
                pd.Series(dtype=float), {},
                out_path=tmp_path / "abl.json",
            )

        assert promo_calls, "record_broader_context_promotion was not called"
        kw = promo_calls[0]
        assert kw["delta"] > 0.0
        assert kw["lift"] > 0.0
        assert kw["acc_27"] > 0.0

    def test_promotion_record_not_written_on_gate_fail(self, tmp_path):
        """When passes_gate=False, record_broader_context_promotion is NOT called."""
        import ml.post_retrain_ablation as mod

        promo_calls = []

        n = 80
        rng = np.random.default_rng(0)
        fake_X19 = rng.random((n, 19)).astype(np.float32)
        fake_X27 = np.hstack([fake_X19, rng.random((n, 8)).astype(np.float32)])
        fake_y   = rng.integers(0, 2, n)
        fake_w   = np.ones(n, dtype=np.float32)
        fake_ts  = pd.date_range("2020-01-01", periods=n, freq="B")

        def _mock_wf(X, y, weights, model_factory, n_splits, embargo):
            # Both arms identical → delta=0 → fails gate
            return (
                {"evaluated": True, "oos_accuracy": 0.55,
                 "oos_balanced_accuracy": 0.54, "folds": []},
                None, None,
            )

        with patch("ml.calibration.walk_forward_evaluate", side_effect=_mock_wf), \
             patch.object(mod, "_build_matrices",
                          return_value=(fake_X19, fake_X27, fake_y, fake_w,
                                        fake_ts, 0.50,
                                        [f"f{i}" for i in range(19)],
                                        [f"c{i}" for i in range(8)])), \
             patch.object(mod, "_append_to_json"), \
             patch("ml.training_status.record_broader_context_promotion",
                   side_effect=lambda **kw: promo_calls.append(kw)):
            mod.run_broader_context_ablation(
                pd.DataFrame(), pd.DataFrame(),
                pd.Series(dtype=float), {},
                out_path=tmp_path / "abl.json",
            )

        assert not promo_calls, "record_broader_context_promotion should not be called on gate fail"


# ─────────────────────────────────────────────────────────────────────────────
# Auto-enable: in-memory flag flip
# ─────────────────────────────────────────────────────────────────────────────

class TestAutoEnable:
    """Verify LONG_BROADER_CONTEXT_AUTO_ENABLE flips the in-memory flag."""

    @pytest.fixture(autouse=True)
    def _restore_settings(self):
        from config import settings
        orig_flag     = settings.LONG_BROADER_CONTEXT_ENABLED
        orig_auto     = settings.LONG_BROADER_CONTEXT_AUTO_ENABLE
        yield
        settings.LONG_BROADER_CONTEXT_ENABLED   = orig_flag   # type: ignore[assignment]
        settings.LONG_BROADER_CONTEXT_AUTO_ENABLE = orig_auto  # type: ignore[assignment]

    def _run_gate_pass(self, tmp_path):
        """Trigger a gate-passing ablation run with mocked walk-forward."""
        import ml.post_retrain_ablation as mod

        n = 80
        rng = np.random.default_rng(0)
        fake_X19 = rng.random((n, 19)).astype(np.float32)
        fake_X27 = np.hstack([fake_X19, rng.random((n, 8)).astype(np.float32)])
        fake_y   = rng.integers(0, 2, n)
        fake_w   = np.ones(n, dtype=np.float32)
        fake_ts  = pd.date_range("2020-01-01", periods=n, freq="B")

        def _mock_wf(X, y, weights, model_factory, n_splits, embargo):
            acc = 0.52 if X.shape[1] == 19 else 0.75
            return (
                {"evaluated": True, "oos_accuracy": acc,
                 "oos_balanced_accuracy": acc - 0.01, "folds": []},
                None, None,
            )

        with patch("ml.calibration.walk_forward_evaluate", side_effect=_mock_wf), \
             patch.object(mod, "_build_matrices",
                          return_value=(fake_X19, fake_X27, fake_y, fake_w,
                                        fake_ts, 0.50,
                                        [f"f{i}" for i in range(19)],
                                        [f"c{i}" for i in range(8)])), \
             patch.object(mod, "_append_to_json"), \
             patch("ml.training_status.record_broader_context_promotion"):
            return mod.run_broader_context_ablation(
                pd.DataFrame(), pd.DataFrame(),
                pd.Series(dtype=float), {},
                out_path=tmp_path / "abl.json",
            )

    def test_auto_enable_true_flips_flag(self, tmp_path):
        from config import settings
        settings.LONG_BROADER_CONTEXT_ENABLED    = False  # type: ignore[assignment]
        settings.LONG_BROADER_CONTEXT_AUTO_ENABLE = True   # type: ignore[assignment]
        self._run_gate_pass(tmp_path)
        assert settings.LONG_BROADER_CONTEXT_ENABLED is True

    def test_auto_enable_false_does_not_flip_flag(self, tmp_path):
        from config import settings
        settings.LONG_BROADER_CONTEXT_ENABLED    = False  # type: ignore[assignment]
        settings.LONG_BROADER_CONTEXT_AUTO_ENABLE = False  # type: ignore[assignment]
        self._run_gate_pass(tmp_path)
        assert settings.LONG_BROADER_CONTEXT_ENABLED is False

    def test_auto_enable_does_not_raise_when_flag_already_true(self, tmp_path):
        from config import settings
        settings.LONG_BROADER_CONTEXT_ENABLED    = True   # type: ignore[assignment]
        settings.LONG_BROADER_CONTEXT_AUTO_ENABLE = True   # type: ignore[assignment]
        result = self._run_gate_pass(tmp_path)
        assert result  # must return a valid dict, not {}


# ─────────────────────────────────────────────────────────────────────────────
# FCMNotifier.send_broader_context_promotion_alert
# ─────────────────────────────────────────────────────────────────────────────

class TestFCMPromotionAlert:
    def test_no_op_when_fcm_key_missing(self):
        """Returns False without raising when FCM_SERVER_KEY is not set."""
        from notifications.fcm import FCMNotifier
        from config import settings
        orig = settings.FCM_SERVER_KEY
        try:
            settings.FCM_SERVER_KEY = ""  # type: ignore[assignment]
            import asyncio
            notifier = FCMNotifier()
            result = asyncio.get_event_loop().run_until_complete(
                notifier.send_broader_context_promotion_alert(
                    device_token="tok", delta=0.03, lift=0.04, acc_27=0.62,
                )
            )
            assert result is False
        finally:
            settings.FCM_SERVER_KEY = orig  # type: ignore[assignment]

    def test_no_op_when_no_token(self):
        """Returns False without raising when device_token is empty."""
        from notifications.fcm import FCMNotifier
        from config import settings
        orig = settings.FCM_SERVER_KEY
        try:
            settings.FCM_SERVER_KEY = "fake_key"  # type: ignore[assignment]
            import asyncio
            notifier = FCMNotifier()
            result = asyncio.get_event_loop().run_until_complete(
                notifier.send_broader_context_promotion_alert(
                    device_token="", delta=0.03, lift=0.04, acc_27=0.62,
                )
            )
            assert result is False
        finally:
            settings.FCM_SERVER_KEY = orig  # type: ignore[assignment]


# ─────────────────────────────────────────────────────────────────────────────
# healthz: broader_context_promotion field
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthzBroaderContextPromotion:
    """Verify /api/healthz surfaces the promotion field and operator alert."""

    @pytest.mark.asyncio
    async def test_field_is_null_when_no_promotion(self, tmp_path):
        """broader_context_promotion is null when no promotion record exists."""
        import ml.training_status as ts
        orig = ts.PROMOTION_PATH
        ts.PROMOTION_PATH = tmp_path / "promo.json"
        try:
            from main import app
            from httpx import AsyncClient, ASGITransport
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/api/healthz")
            assert resp.status_code == 200
            body = resp.json()
            assert "broader_context_promotion" in body
            assert body["broader_context_promotion"] is None
        finally:
            ts.PROMOTION_PATH = orig

    @pytest.mark.asyncio
    async def test_field_carries_promotion_when_file_exists(self, tmp_path):
        """broader_context_promotion mirrors the record when present."""
        import ml.training_status as ts
        orig = ts.PROMOTION_PATH
        promo_file = tmp_path / "promo.json"
        ts.PROMOTION_PATH = promo_file
        promo_data = {
            "promoted_at_utc": "2026-08-01T10:00:00+00:00",
            "accuracy_delta_27_minus_19": 0.032,
            "oos_lift_27feat": 0.045,
            "oos_accuracy_27feat": 0.615,
            "auto_enabled": False,
            "alert_sent": False,
        }
        promo_file.write_text(json.dumps(promo_data))
        try:
            from main import app
            from httpx import AsyncClient, ASGITransport
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/api/healthz")
            assert resp.status_code == 200
            body = resp.json()
            promo = body.get("broader_context_promotion")
            assert promo is not None
            assert promo["promoted_at_utc"] == promo_data["promoted_at_utc"]
            assert promo["accuracy_delta_27_minus_19"] == pytest.approx(0.032)
        finally:
            ts.PROMOTION_PATH = orig

    @pytest.mark.asyncio
    async def test_operator_alert_added_when_flag_off_and_not_auto_enabled(self, tmp_path):
        """An operator alert is added to healthz.alerts when promoted but flag is still False."""
        import ml.training_status as ts
        from config import settings
        orig_promo = ts.PROMOTION_PATH
        orig_flag  = settings.LONG_BROADER_CONTEXT_ENABLED
        promo_file = tmp_path / "promo.json"
        ts.PROMOTION_PATH = promo_file
        promo_data = {
            "promoted_at_utc": "2026-08-01T10:00:00+00:00",
            "accuracy_delta_27_minus_19": 0.03,
            "oos_lift_27feat": 0.04,
            "oos_accuracy_27feat": 0.61,
            "auto_enabled": False,
            "alert_sent": False,
        }
        promo_file.write_text(json.dumps(promo_data))
        settings.LONG_BROADER_CONTEXT_ENABLED = False  # type: ignore[assignment]
        try:
            from main import app
            from httpx import AsyncClient, ASGITransport
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/api/healthz")
            assert resp.status_code == 200
            body = resp.json()
            alerts = body.get("alerts", [])
            assert any("broader_context_promoted" in a for a in alerts), (
                f"Expected a 'broader_context_promoted' alert in: {alerts}"
            )
        finally:
            ts.PROMOTION_PATH = orig_promo
            settings.LONG_BROADER_CONTEXT_ENABLED = orig_flag  # type: ignore[assignment]

    @pytest.mark.asyncio
    async def test_no_operator_alert_when_auto_enabled(self, tmp_path):
        """No operator alert when auto_enabled=True (flag already flipped in-memory)."""
        import ml.training_status as ts
        from config import settings
        orig_promo = ts.PROMOTION_PATH
        orig_flag  = settings.LONG_BROADER_CONTEXT_ENABLED
        promo_file = tmp_path / "promo.json"
        ts.PROMOTION_PATH = promo_file
        promo_data = {
            "promoted_at_utc": "2026-08-01T10:00:00+00:00",
            "accuracy_delta_27_minus_19": 0.03,
            "oos_lift_27feat": 0.04,
            "oos_accuracy_27feat": 0.61,
            "auto_enabled": True,
            "alert_sent": True,
        }
        promo_file.write_text(json.dumps(promo_data))
        settings.LONG_BROADER_CONTEXT_ENABLED = False  # type: ignore[assignment]
        try:
            from main import app
            from httpx import AsyncClient, ASGITransport
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/api/healthz")
            assert resp.status_code == 200
            body = resp.json()
            alerts = body.get("alerts", [])
            assert not any("broader_context_promoted" in a for a in alerts), (
                f"Unexpected operator alert when auto_enabled=True: {alerts}"
            )
        finally:
            ts.PROMOTION_PATH = orig_promo
            settings.LONG_BROADER_CONTEXT_ENABLED = orig_flag  # type: ignore[assignment]
