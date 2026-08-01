"""
NovaCycle Model Trainer
========================
Orchestrates training of both the long-trend (XGBoost) and
short-trend (Keras) models from data stored in the database.

Schedule: retrain weekly (checked on each startup).
"""

import json
import logging
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from database.models import ModelMetadata, VooCandle, VixCandle, SpxCandle
from indicators.technical import TechnicalIndicators
from ml.long_trend import LongTrendModel
from ml.short_trend import ShortTrendModel
from ml.model_health import check_accuracy_regression
from ml.training_status import (
    any_model_failed_last_attempt,
    get_consecutive_failures,
    get_last_successful_accuracy,
    get_last_successful_accuracy_metric,
    get_training_status,
    mark_stuck_alert_sent,
    record_training_result,
    should_send_stuck_alert,
)

logger = logging.getLogger(__name__)

_RETRAIN_INTERVAL_DAYS = 7
# When the last training attempt failed (e.g. a regressed retrain rolled back
# to the previous model), retry much sooner instead of waiting a full week —
# the fresh ModelMetadata row would otherwise mask the failure for 7 days.
_FAILED_RETRAIN_INTERVAL_DAYS = 1


def _sidecar_files(model_path: Path) -> list:
    """Sidecar files that must roll back together with a model file.

    The long-trend model carries a probability calibrator and its
    walk-forward calibration report; restoring the model without them would
    apply a flagged retrain's calibration to the last known-good model.
    """
    try:
        if model_path.name == "long_trend_model.pkl":
            from ml.calibration import CALIBRATOR_PATH, REPORT_PATH
            return [CALIBRATOR_PATH, REPORT_PATH]
        if model_path.name == "short_trend_model.pkl":
            from ml.calibration import (
                _walkforward_report_path,
                calibration_report_path,
                calibrator_path,
            )
            return [
                _walkforward_report_path("short_trend"),
                calibrator_path("short_trend"),
                calibration_report_path("short_trend"),
            ]
    except Exception as exc:
        logger.error("_sidecar_files error for %s: %s", model_path, exc)
    return []


def _is_metric_upgrade_transition(prev_metric, new_metric) -> bool:
    """True only for the one-time accuracy-semantics upgrade from the legacy
    leakage-inflated train accuracy (or pre-tracking runs, prev_metric=None)
    to honest purged walk-forward OOS accuracy. Any other metric mismatch —
    notably a retrain falling back from "purged_walk_forward_oos" to "train"
    because walk-forward could not be evaluated — must NOT be exempted from
    the accuracy-regression check.
    """
    return new_metric == "purged_walk_forward_oos" and prev_metric in (None, "train")


def _backup_model_file(model_path: Path) -> Optional[Path]:
    """Copy the current model file aside before retraining.

    Returns the backup path when a backup was made, None when there was no
    existing model file to back up. Never raises.
    """
    try:
        if not model_path.exists():
            return None
        backup_path = model_path.with_suffix(model_path.suffix + ".bak")
        shutil.copy2(model_path, backup_path)
        # Back up sidecar files (e.g. the long-trend probability calibrator)
        # so a rollback restores the model together with its calibration.
        for sidecar in _sidecar_files(model_path):
            try:
                sidecar_backup = sidecar.with_suffix(sidecar.suffix + ".bak")
                if sidecar.exists():
                    shutil.copy2(sidecar, sidecar_backup)
                elif sidecar_backup.exists():
                    # No current sidecar: drop any stale backup so restore
                    # correctly deletes the sidecar instead of resurrecting it.
                    sidecar_backup.unlink()
            except Exception as exc:
                logger.error("sidecar backup error for %s: %s", sidecar, exc)
        logger.info(
            "ml_model_backup model_file=%s backup=%s", model_path.name, backup_path.name
        )
        return backup_path
    except Exception as exc:
        logger.error("_backup_model_file error for %s: %s", model_path, exc)
        return None


def _restore_model_file(model_path: Path, backup_path: Optional[Path], model_name: str) -> bool:
    """Restore the last known-good model file after a flagged retrain.

    Returns True when the previous model was restored. Never raises.
    """
    try:
        if backup_path is None or not backup_path.exists():
            logger.warning(
                "ml_model_rollback_unavailable model=%s reason=no_backup", model_name
            )
            return False
        # Plain copy (not copy2): the restored file must get a *fresh* mtime so
        # the models' _maybe_reload() detects the change and drops the
        # regressed in-memory model in favour of the restored one.
        shutil.copy(backup_path, model_path)
        # Restore sidecar files (e.g. calibrator) to their pre-retrain state:
        # copy back the backup when one existed, otherwise remove the sidecar
        # the flagged retrain just wrote.
        for sidecar in _sidecar_files(model_path):
            try:
                sidecar_backup = sidecar.with_suffix(sidecar.suffix + ".bak")
                if sidecar_backup.exists():
                    shutil.copy(sidecar_backup, sidecar)
                elif sidecar.exists():
                    sidecar.unlink()
            except Exception as exc:
                logger.error("sidecar restore error for %s: %s", sidecar, exc)
        logger.warning(
            "ml_model_rollback model=%s restored=%s reason=flagged_retrain "
            "— predictions continue on last known-good model",
            model_name,
            model_path.name,
        )
        return True
    except Exception as exc:
        logger.error("_restore_model_file error for %s: %s", model_path, exc)
        return False


class ModelTrainer:
    """Trains and periodically retrains the NovaCycle ML models."""

    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.long_model = LongTrendModel()
        self.short_model = ShortTrendModel()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    async def run_initial_training(self, db_session: AsyncSession) -> None:
        """
        Load all historical data from DB, train both models, and log results
        to the ModelMetadata table.

        Steps:
          1. Load daily VOO candles (regular-hours only)
          2. Load VIX candles
          3. Compute all indicators
          4. Train long-trend XGBoost model
          5. Load 5-min VOO candles
          6. Train short-trend Keras model
          7. Persist ModelMetadata rows for both models
        """
        logger.info("Starting initial model training…")
        ticker = settings.TICKER

        # ── Load daily VOO candles ─────────────────────────────────────────────
        daily_df = await self._load_daily_voo(db_session)
        if daily_df.empty:
            logger.warning("No daily VOO data available for training. Skipping.")
            record_training_result(
                "long_trend", success=False, error="No daily VOO data available"
            )
            record_training_result(
                "short_trend", success=False, error="No daily VOO data available"
            )
            await self._maybe_send_stuck_alert(db_session, "long_trend")
            await self._maybe_send_stuck_alert(db_session, "short_trend")
            return

        # ── Load VIX candles ───────────────────────────────────────────────────
        vix_df = await self._load_vix(db_session)

        # ── Load SPX futures candles (real macro series; empty → fallback) ────
        spx_close = await self._load_spx_close(db_session)

        # ── Compute indicators ─────────────────────────────────────────────────
        logger.info("Computing indicators for training data…")
        try:
            indicators = self.indicators.compute_all(
                daily_df, vix_df, exclude_extended=True
            )
        except Exception as exc:
            logger.error("Indicator computation failed: %s", exc)
            indicators = {}

        # Attach the SPX futures close series so models can feed the real
        # series into compute_macro_sensitivity (fallback preserved when empty).
        if not spx_close.empty:
            indicators["spx_futures_close"] = spx_close

        # ── Train long-trend model ─────────────────────────────────────────────
        logger.info("Training long-trend model…")
        from ml.long_trend import MODEL_PATH as LONG_MODEL_PATH

        long_backup = _backup_model_file(LONG_MODEL_PATH)
        try:
            long_flagged = True
            long_result = self.long_model.train(daily_df, indicators)
            if self.long_model.model is None:
                # train() swallows exceptions internally and returns zeros.
                restored = _restore_model_file(LONG_MODEL_PATH, long_backup, "long_trend")
                record_training_result(
                    "long_trend",
                    success=False,
                    error="Training produced no model (see server logs)",
                    rolled_back=restored,
                )
            elif long_result.get("degenerate"):
                restored = _restore_model_file(LONG_MODEL_PATH, long_backup, "long_trend")
                record_training_result(
                    "long_trend",
                    success=False,
                    error="Degenerate model: "
                    + (long_result.get("degeneracy_reason") or "predictions do not vary"),
                    rolled_back=restored,
                )
            else:
                new_acc = long_result.get("accuracy", 0.0)
                prev_acc = get_last_successful_accuracy("long_trend")
                new_metric = long_result.get("accuracy_metric")
                prev_metric = get_last_successful_accuracy_metric("long_trend")
                walk_forward = long_result.get("calibration") or {}
                oos_lift = walk_forward.get("accuracy_lift_vs_majority")
                regressed = False
                reason = None
                if (
                    new_metric == "purged_walk_forward_oos"
                    and oos_lift is not None
                    and float(oos_lift)
                    <= float(settings.LONG_MIN_OOS_ACCURACY_LIFT)
                ):
                    regressed = True
                    reason = (
                        "OOS quality gate: accuracy lift versus majority baseline "
                        f"is {float(oos_lift):.4f}, required > "
                        f"{float(settings.LONG_MIN_OOS_ACCURACY_LIFT):.4f}"
                    )
                    logger.error("Long-trend %s", reason)
                # The existing 0.56 value is a legacy train-set metric. It is
                # not comparable to a new purged OOS result and must not block
                # migration to the honest metric.
                if not regressed:
                    upgrade_transition = _is_metric_upgrade_transition(
                        prev_metric, new_metric
                    )
                    if upgrade_transition:
                        logger.info(
                            "long_trend accuracy metric upgraded (%s → %s); "
                            "skipping regression comparison against %.4f",
                            prev_metric, new_metric, prev_acc or 0.0,
                        )
                        regressed, reason = False, None
                    else:
                        if prev_metric != new_metric:
                            logger.warning(
                                "long_trend accuracy metric mismatch (%s → %s); "
                                "regression check still applies",
                                prev_metric, new_metric,
                            )
                        regressed, reason = check_accuracy_regression(new_acc, prev_acc)
                if regressed:
                    logger.error("Long-trend %s", reason)
                    restored = _restore_model_file(LONG_MODEL_PATH, long_backup, "long_trend")
                    record_training_result(
                        "long_trend",
                        success=False,
                        error=reason,
                        accuracy=new_acc,
                        accuracy_metric=new_metric,
                        rolled_back=restored,
                    )
                else:
                    record_training_result(
                        "long_trend",
                        success=True,
                        accuracy=new_acc,
                        accuracy_metric=long_result.get("accuracy_metric"),
                    )
                    long_flagged = False
            if long_flagged:
                # Flagged retrain (regressed/degenerate/no model): the model
                # file was rolled back, so do NOT persist a metadata row with
                # the discarded accuracy — health surfaces read the latest row
                # and must keep reflecting the restored last-good model.
                logger.warning(
                    "ml_metadata_skipped model=long_trend reason=flagged_retrain "
                    "— keeping last-good metadata visible to health endpoints"
                )
            else:
                await self._save_metadata(
                    db_session,
                    model_name="long_trend",
                    ticker=ticker,
                    accuracy=long_result.get("accuracy", 0.0),
                    feature_importances=long_result.get("feature_importances", {}),
                )
            logger.info(
                "Long-trend training complete: accuracy=%.4f",
                long_result.get("accuracy", 0.0),
            )
        except Exception as exc:
            logger.error("Long-trend training failed: %s", exc)
            restored = _restore_model_file(LONG_MODEL_PATH, long_backup, "long_trend")
            record_training_result(
                "long_trend", success=False, error=str(exc), rolled_back=restored
            )
        await self._maybe_send_stuck_alert(db_session, "long_trend")

        # ── Load 5-min VOO candles ─────────────────────────────────────────────
        fivemin_df = await self._load_fivemin_voo(db_session)

        if fivemin_df.empty:
            logger.warning("No 5-min VOO data available. Skipping short-trend training.")
            record_training_result(
                "short_trend", success=False, error="No 5-min VOO data available"
            )
            await self._maybe_send_stuck_alert(db_session, "short_trend")
            return

        # ── Compute short indicators ───────────────────────────────────────────
        try:
            short_indicators = self.indicators.compute_all(
                fivemin_df, vix_df, exclude_extended=False
            )
        except Exception as exc:
            logger.error("Short indicator computation failed: %s", exc)
            short_indicators = {}

        if not spx_close.empty:
            short_indicators["spx_futures_close"] = spx_close

        # ── Train short-trend model ────────────────────────────────────────────
        logger.info("Training short-trend model…")
        from ml.short_trend import MODEL_PATH as SHORT_MODEL_PATH

        short_backup = _backup_model_file(SHORT_MODEL_PATH)
        try:
            short_flagged = True
            short_result = self.short_model.train(fivemin_df, short_indicators)
            if self.short_model.model is None:
                restored = _restore_model_file(SHORT_MODEL_PATH, short_backup, "short_trend")
                record_training_result(
                    "short_trend",
                    success=False,
                    error="Training produced no model (see server logs)",
                    rolled_back=restored,
                )
            elif short_result.get("degenerate"):
                restored = _restore_model_file(SHORT_MODEL_PATH, short_backup, "short_trend")
                record_training_result(
                    "short_trend",
                    success=False,
                    error="Degenerate model: "
                    + (short_result.get("degeneracy_reason") or "predictions do not vary"),
                    rolled_back=restored,
                )
            else:
                new_acc = short_result.get("accuracy", 0.0)
                new_metric = short_result.get("accuracy_metric")
                prev_acc = get_last_successful_accuracy("short_trend")
                prev_metric = get_last_successful_accuracy_metric("short_trend")
                # Skip the regression comparison ONLY for the one-time
                # semantics upgrade from the legacy leakage-inflated train
                # accuracy (or pre-tracking runs) to honest purged
                # walk-forward OOS accuracy — the honest number is expected
                # to be far lower and is not a regression. Any other metric
                # mismatch (e.g. a retrain falling back to "train" because
                # walk-forward could not be evaluated) still goes through the
                # regression check so a degraded retrain is not silently
                # exempted.
                upgrade_transition = _is_metric_upgrade_transition(prev_metric, new_metric)
                if upgrade_transition:
                    if prev_acc is not None:
                        logger.info(
                            "short_trend accuracy metric upgraded (%s → %s); "
                            "skipping regression comparison against %.4f",
                            prev_metric, new_metric, prev_acc,
                        )
                    regressed, reason = False, None
                else:
                    if prev_metric != new_metric:
                        logger.warning(
                            "short_trend accuracy metric mismatch (%s → %s); "
                            "regression check still applies",
                            prev_metric, new_metric,
                        )
                    regressed, reason = check_accuracy_regression(new_acc, prev_acc)
                if regressed:
                    logger.error("Short-trend %s", reason)
                    restored = _restore_model_file(SHORT_MODEL_PATH, short_backup, "short_trend")
                    record_training_result(
                        "short_trend",
                        success=False,
                        error=reason,
                        accuracy=new_acc,
                        rolled_back=restored,
                    )
                else:
                    record_training_result(
                        "short_trend", success=True, accuracy=new_acc,
                        accuracy_metric=new_metric,
                    )
                    short_flagged = False
            if short_flagged:
                logger.warning(
                    "ml_metadata_skipped model=short_trend reason=flagged_retrain "
                    "— keeping last-good metadata visible to health endpoints"
                )
            else:
                await self._save_metadata(
                    db_session,
                    model_name="short_trend",
                    ticker=ticker,
                    accuracy=short_result.get("accuracy", 0.0),
                    feature_importances={},
                )
            logger.info(
                "Short-trend training complete: accuracy=%.4f  val_accuracy=%.4f",
                short_result.get("accuracy", 0.0),
                short_result.get("val_accuracy", 0.0),
            )
        except Exception as exc:
            logger.error("Short-trend training failed: %s", exc)
            restored = _restore_model_file(SHORT_MODEL_PATH, short_backup, "short_trend")
            record_training_result(
                "short_trend", success=False, error=str(exc), rolled_back=restored
            )
        await self._maybe_send_stuck_alert(db_session, "short_trend")

        logger.info("Initial model training complete.")

    async def retrain_if_needed(self, db_session: AsyncSession) -> bool:
        """
        Check whether models were last trained more than RETRAIN_INTERVAL_DAYS ago.
        If so, kick off retraining.

        Returns:
            True if retraining was performed, False otherwise.
        """
        missing_files = self._missing_model_files()
        if missing_files:
            logger.warning(
                "Model file(s) missing on disk (%s). Training regardless of metadata.",
                ", ".join(missing_files),
            )
            await self.run_initial_training(db_session)
            return True

        last_trained = await self._get_last_trained(db_session)

        if last_trained is None:
            logger.info("No previous training found. Running initial training.")
            await self.run_initial_training(db_session)
            return True

        now = datetime.utcnow()
        age_days = (now - last_trained).days

        if any_model_failed_last_attempt():
            interval_days = _FAILED_RETRAIN_INTERVAL_DAYS
            logger.warning(
                "Last training attempt failed for at least one model — using "
                "shortened retry interval (%d day(s) instead of %d).",
                _FAILED_RETRAIN_INTERVAL_DAYS,
                _RETRAIN_INTERVAL_DAYS,
            )
        else:
            interval_days = _RETRAIN_INTERVAL_DAYS

        if age_days >= interval_days:
            logger.info(
                "Models last trained %d days ago (threshold=%d). Retraining…",
                age_days,
                interval_days,
            )
            await self.run_initial_training(db_session)
            return True
        else:
            logger.info(
                "Models are up-to-date (last trained %d days ago). No retraining needed.",
                age_days,
            )
            return False

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    async def _maybe_send_stuck_alert(
        self, db_session: AsyncSession, model_name: str
    ) -> None:
        """Push a one-time "training stuck" alert to registered devices when a
        model first crosses CONSECUTIVE_FAILURE_ALERT_THRESHOLD consecutive
        failed retrains. A successful retrain re-arms the alert.

        Never raises — a notification failure must not break training.
        """
        try:
            if not should_send_stuck_alert(model_name):
                return

            from sqlalchemy import select as _select
            from database.models import DeviceToken
            from notifications.fcm import FCMNotifier

            result = await db_session.execute(_select(DeviceToken))
            tokens = result.scalars().all()
            if not tokens:
                logger.warning(
                    "training_stuck_alert model=%s — no device tokens registered; "
                    "alert stays armed until a device is available",
                    model_name,
                )
                return

            failures = get_consecutive_failures(model_name)
            last_error = get_training_status().get(model_name, {}).get("error")

            notifier = FCMNotifier()
            any_sent = False
            for device in tokens:
                ok = await notifier.send_training_stuck_alert(
                    device_token=device.token,
                    model_name=model_name,
                    consecutive_failures=failures,
                    last_error=last_error,
                )
                any_sent = any_sent or ok

            if any_sent:
                mark_stuck_alert_sent(model_name)
                logger.warning(
                    "training_stuck_alert_sent model=%s consecutive_failures=%d",
                    model_name,
                    failures,
                )
            else:
                logger.error(
                    "training_stuck_alert_failed model=%s — will retry on next "
                    "failed retrain attempt",
                    model_name,
                )
        except Exception as exc:
            logger.error("_maybe_send_stuck_alert error for %s: %s", model_name, exc)

    @staticmethod
    def _missing_model_files() -> list:
        """Return names of expected model files that are absent on disk.

        The deployment image may ship the SQLite DB (with recent trained_at
        metadata) but not the trained .pkl files, so the metadata check alone
        can wrongly skip training on a fresh instance.
        """
        from ml.long_trend import MODEL_PATH as LONG_MODEL_PATH
        from ml.short_trend import MODEL_PATH as SHORT_MODEL_PATH

        missing = []
        if not LONG_MODEL_PATH.exists():
            missing.append(LONG_MODEL_PATH.name)
        if not SHORT_MODEL_PATH.exists():
            missing.append(SHORT_MODEL_PATH.name)
        return missing

    @staticmethod
    async def _load_daily_voo(db_session: AsyncSession) -> pd.DataFrame:
        """Load all daily regular-hours VOO candles from the database."""
        try:
            result = await db_session.execute(
                select(VooCandle).where(
                    VooCandle.ticker == settings.TICKER,
                    VooCandle.timeframe == "daily",
                    VooCandle.is_extended_hours == False,
                ).order_by(VooCandle.timestamp.asc())
            )
            rows = result.scalars().all()
            if not rows:
                return pd.DataFrame()

            records = [
                {
                    "timestamp": r.timestamp,
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "volume": r.volume,
                    "is_extended_hours": r.is_extended_hours,
                    "session_type": r.session_type,
                }
                for r in rows
            ]
            df = pd.DataFrame(records)
            df.set_index("timestamp", inplace=True)
            df.index = pd.to_datetime(df.index)
            df = df[~df.index.duplicated(keep="last")]
            return df
        except Exception as exc:
            logger.error("_load_daily_voo error: %s", exc)
            return pd.DataFrame()

    @staticmethod
    async def _load_fivemin_voo(db_session: AsyncSession) -> pd.DataFrame:
        """Load all 5-min VOO candles (including extended hours) from the database."""
        try:
            result = await db_session.execute(
                select(VooCandle).where(
                    VooCandle.ticker == settings.TICKER,
                    VooCandle.timeframe == "5min",
                ).order_by(VooCandle.timestamp.asc())
            )
            rows = result.scalars().all()
            if not rows:
                return pd.DataFrame()

            records = [
                {
                    "timestamp": r.timestamp,
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "volume": r.volume,
                    "is_extended_hours": r.is_extended_hours,
                    "session_type": r.session_type,
                    "gap_percent": r.gap_percent or 0.0,
                    "gap_type": r.gap_type,
                }
                for r in rows
            ]
            df = pd.DataFrame(records)
            df.set_index("timestamp", inplace=True)
            df.index = pd.to_datetime(df.index)
            df = df[~df.index.duplicated(keep="last")]
            return df
        except Exception as exc:
            logger.error("_load_fivemin_voo error: %s", exc)
            return pd.DataFrame()

    @staticmethod
    async def _load_vix(db_session: AsyncSession) -> pd.DataFrame:
        """Load VIX daily candles from the database."""
        try:
            result = await db_session.execute(
                select(VixCandle).where(
                    VixCandle.ticker == settings.VIX_TICKER,
                    VixCandle.timeframe == "daily",
                ).order_by(VixCandle.timestamp.asc())
            )
            rows = result.scalars().all()
            if not rows:
                return pd.DataFrame()

            records = [
                {
                    "timestamp": r.timestamp,
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "volume": r.volume or 0.0,
                }
                for r in rows
            ]
            df = pd.DataFrame(records)
            df.set_index("timestamp", inplace=True)
            df.index = pd.to_datetime(df.index)
            df = df[~df.index.duplicated(keep="last")]
            return df
        except Exception as exc:
            logger.error("_load_vix error: %s", exc)
            return pd.DataFrame()

    @staticmethod
    async def _load_spx_close(db_session: AsyncSession) -> pd.Series:
        """
        Load the daily SPX futures close series from the database.

        Returns an empty Series when no data exists so callers keep the
        VOO overnight-return fallback in compute_macro_sensitivity.
        """
        try:
            result = await db_session.execute(
                select(SpxCandle).where(
                    SpxCandle.ticker == settings.SPX_FUTURES_TICKER,
                    SpxCandle.timeframe == "daily",
                ).order_by(SpxCandle.timestamp.asc())
            )
            rows = result.scalars().all()
            if not rows:
                logger.info(
                    "ml_trainer_spx_unavailable — macro sensitivity will use fallback"
                )
                return pd.Series(dtype=float)

            series = pd.Series(
                [r.close for r in rows],
                index=pd.to_datetime([r.timestamp for r in rows]),
                dtype=float,
            )
            series = series[~series.index.duplicated(keep="last")]
            logger.info("Loaded %d daily SPX futures closes", len(series))
            return series
        except Exception as exc:
            logger.error("_load_spx_close error: %s", exc)
            return pd.Series(dtype=float)

    @staticmethod
    async def _save_metadata(
        db_session: AsyncSession,
        model_name: str,
        ticker: str,
        accuracy: float,
        feature_importances: dict,
    ) -> None:
        """Persist a ModelMetadata training record."""
        try:
            record = ModelMetadata(
                model_name=model_name,
                ticker=ticker,
                trained_at=datetime.utcnow(),
                accuracy=accuracy,
                feature_importances=json.dumps(feature_importances),
            )
            db_session.add(record)
            await db_session.flush()
        except Exception as exc:
            logger.error("_save_metadata error: %s", exc)

    @staticmethod
    async def _get_last_trained(db_session: AsyncSession):
        """Return the most recent trained_at datetime across all models."""
        try:
            result = await db_session.execute(
                select(func.max(ModelMetadata.trained_at))
            )
            return result.scalar()
        except Exception as exc:
            logger.error("_get_last_trained error: %s", exc)
            return None
