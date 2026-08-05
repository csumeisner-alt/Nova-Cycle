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
    clear_baseline_mode_tracking,
    get_baseline_mode_days,
    get_baseline_mode_since,
    get_consecutive_failures,
    get_last_successful_accuracy,
    get_last_successful_accuracy_metric,
    get_training_status,
    mark_baseline_duration_alert_sent,
    mark_stuck_alert_sent,
    record_baseline_mode_onset,
    record_training_result,
    should_send_baseline_duration_alert,
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

    Implementation note: paths are resolved via the path-computing functions
    (calibrator_path, calibration_report_path) rather than by importing the
    module-level constants (CALIBRATOR_PATH, REPORT_PATH).  The functions
    always look up the current value of ml.calibration.MODEL_DIR, so
    monkeypatching MODEL_DIR in tests automatically redirects sidecar
    operations to the test's tmp_path — no separate patch of CALIBRATOR_PATH
    or REPORT_PATH is required (though patching them is also harmless).
    """
    try:
        if model_path.name == "long_trend_model.pkl":
            from ml.calibration import calibrator_path, calibration_report_path
            from ml.long_trend import _META_PATH
            return [
                calibrator_path("long_trend"),
                calibration_report_path("long_trend"),
                _META_PATH,
            ]
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


async def _fetch_daily_candle_meta(db_session: AsyncSession) -> Optional[dict]:
    """Return current daily VOO candle counts and labeled-row estimate.

    Queries the DB the same way long_trend.train() does so the numbers in the
    calibration report are directly comparable to what training would see.
    Never raises.
    """
    try:
        count_result = await db_session.execute(
            select(
                func.count(VooCandle.id),
                func.min(VooCandle.timestamp),
                func.max(VooCandle.timestamp),
            ).where(
                VooCandle.ticker == settings.TICKER,
                VooCandle.timeframe == "daily",
                VooCandle.is_extended_hours == False,  # noqa: E712
            )
        )
        total_candles, ts_min, ts_max = count_result.one()
        if not total_candles:
            return None

        closes_result = await db_session.execute(
            select(VooCandle.close).where(
                VooCandle.ticker == settings.TICKER,
                VooCandle.timeframe == "daily",
                VooCandle.is_extended_hours == False,  # noqa: E712
            ).order_by(VooCandle.timestamp.asc())
        )
        closes = [row[0] for row in closes_result]

        horizon, threshold = 21, 0.02
        labeled = sum(
            1 for i in range(len(closes) - horizon)
            if closes[i] and closes[i] > 0 and closes[i + horizon]
            and abs(closes[i + horizon] / closes[i] - 1.0) >= threshold
        )

        date_start = str(ts_min)[:10] if ts_min else None
        date_end = str(ts_max)[:10] if ts_max else None
        return {
            "total_candles": int(total_candles),
            "labeled_rows": labeled,
            "date_start": date_start,
            "date_end": date_end,
            "note": (
                "VOO daily regular-hours candles; "
                "labeled = rows where |21-day return| >= 2%"
            ),
        }
    except Exception as exc:
        logger.error("_fetch_daily_candle_meta error: %s", exc)
        return None


def _extract_report_labeled_rows(report: dict) -> int:
    """Extract the labeled-row count recorded in a calibration report.

    Prefers ``report["dataset"]["labeled_rows"]``; falls back to parsing
    the "not enough rows (N)" reason string.  Returns 0 when neither is
    available.
    """
    import re

    dataset = report.get("dataset") or {}
    try:
        v = dataset.get("labeled_rows")
        if v is not None:
            return int(v)
    except (TypeError, ValueError):
        pass
    m = re.search(r"\((\d+)\)", report.get("reason", ""))
    return int(m.group(1)) if m else 0


async def audit_calibration_report_staleness(
    db_session: AsyncSession,
    model_name: str = "long_trend",
    stale_row_multiple: float = 2.0,
    min_db_rows_to_flag: int = 500,
) -> bool:
    """Detect and repair a misleading calibration report.

    When the DB now holds at least ``stale_row_multiple`` × as many labeled
    rows as the report was produced with, and the report represents a skipped
    or failed evaluation (``evaluated=False``), this function calls
    ``mark_calibration_report_stale`` with current DB counts so operators see
    accurate metadata.

    Idempotent: if the report is already marked stale with the current DB
    counts, the function returns False without re-writing the file.

    Always called with ``stale_row_multiple=2.0``, meaning "the DB now has at
    least twice as many rows as the report claimed" — conservative enough to
    avoid spurious marks.

    Args:
        db_session:            Active async SQLAlchemy session.
        model_name:            Which model's report to audit (default
                               ``"long_trend"``).
        stale_row_multiple:    Minimum ratio DB-labeled / report-labeled that
                               triggers a stale mark.
        min_db_rows_to_flag:   Only flag when the DB has this many labeled
                               rows — avoids marking a report stale on a
                               nearly-empty fresh deployment.

    Returns:
        True if the report was (re-)marked stale, False otherwise.

    Never raises.
    """
    try:
        from ml.calibration import (
            get_calibration_report,
            mark_calibration_report_stale,
        )

        report = get_calibration_report(model_name)
        if report is None:
            return False

        # Only audit reports that represent a skipped/failed evaluation.
        # A fresh evaluated=True report is intentional — leave it alone.
        if report.get("evaluated") is True and not report.get("stale"):
            return False

        current_meta = await _fetch_daily_candle_meta(db_session)
        if current_meta is None:
            return False

        current_labeled = current_meta.get("labeled_rows", 0)
        if current_labeled < min_db_rows_to_flag:
            return False  # DB itself too small to justify flagging

        report_labeled = _extract_report_labeled_rows(report)

        if report_labeled > 0 and current_labeled < report_labeled * stale_row_multiple:
            return False  # DB not significantly larger than what the report claimed

        # Idempotent: already stale with these exact counts — nothing to do.
        if report.get("stale") is True:
            existing = (report.get("dataset") or {})
            if (
                existing.get("labeled_rows") == current_labeled
                and existing.get("total_candles") == current_meta.get("total_candles")
            ):
                return False

        note = (
            f"Report was produced when the DB had only ~{report_labeled} labeled "
            f"rows. DB now holds {current_meta['total_candles']} daily candles "
            f"({current_meta['date_start']} to {current_meta['date_end']}) "
            f"yielding {current_labeled} meaningful-move labeled rows. "
            f"Re-run a full retrain to replace this report."
        )
        mark_calibration_report_stale(model_name, note=note, dataset_meta=current_meta)
        logger.info(
            "audit_calibration_report_staleness: marked %s stale "
            "(report_labeled=%d → db_labeled=%d)",
            model_name,
            report_labeled,
            current_labeled,
        )
        return True
    except Exception as exc:
        logger.error("audit_calibration_report_staleness error: %s", exc)
        return False


def _short_event_gate_failed(short_result: dict) -> Optional[str]:
    """Return a failure reason when the short-model candidate fails the
    rare-event quality gate, or None when it passes.

    Gate: on purged walk-forward OOS data, PR-AUC (average precision) must
    exceed the event base rate — a random scorer's PR-AUC equals the base
    rate, so anything at or below it has no ranking power for the rally
    event.  When walk-forward could not be evaluated (not enough rows) the
    gate does not apply; the regression comparison still runs.
    """
    wf = short_result.get("walk_forward") or {}
    if not wf.get("evaluated"):
        return None
    pr_auc = wf.get("pr_auc")
    base_rate = wf.get("positive_rate")
    if pr_auc is None or base_rate is None or base_rate <= 0:
        return None
    if float(pr_auc) <= float(base_rate):
        return (
            f"OOS PR-AUC {float(pr_auc):.4f} does not beat the event base "
            f"rate {float(base_rate):.4f} (no ranking power for rally events)"
        )
    return None


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
            logger.error(
                "retrain_no_data ticker=%s timeframe=daily — weekly retrain "
                "aborted because the candle table is empty; models may be stale. "
                "Check that the ingestion pipeline is writing to the correct "
                "database file and that DATABASE_URL is not overriding the "
                "SQLite path.",
                settings.TICKER,
            )
            record_training_result(
                "long_trend", success=False, error="No daily VOO data available"
            )
            record_training_result(
                "short_trend", success=False, error="No daily VOO data available"
            )
            await self._maybe_send_stuck_alert(db_session, "long_trend")
            await self._maybe_send_stuck_alert(db_session, "short_trend")
            await self._track_and_alert_baseline_duration(db_session, "long_trend")
            return

        # ── Load VIX candles ───────────────────────────────────────────────────
        vix_df = await self._load_vix(db_session)

        # ── Load SPX futures candles (real macro series; empty → fallback) ────
        spx_close = await self._load_spx_close(db_session)

        # ── Load broader market context series (empty → neutral fallback) ─────
        # These series feed the new context features when
        # settings.LONG_BROADER_CONTEXT_ENABLED=True.  When the DB tables for
        # these tickers do not yet exist, each loader returns an empty Series
        # and the corresponding _missing indicator fires during training — the
        # model learns that absent context carries no signal.  No error is
        # raised; the existing 19-feature baseline is unaffected when the flag
        # is False.
        broader_context = await self._load_broader_context(db_session)

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

        # Attach broader context series (empty Series are harmless — feature
        # functions check emptiness and fall back to neutral values + missing=1).
        indicators.update(broader_context)

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
                target_type = long_result.get("target_type", settings.LONG_TARGET_TYPE)
                regressed = False
                reason = None

                if target_type == "drawdown_event":
                    # Gate: PR-AUC lift ≥ 2× event prevalence AND precision lift ≥ 2×
                    pr_auc_lift = long_result.get("pr_auc_lift_vs_prevalence")
                    wf_precision_lift = walk_forward.get("precision_lift_vs_base_rate")
                    if new_metric == "purged_walk_forward_oos" and (
                        pr_auc_lift is None
                        or float(pr_auc_lift) < 2.0
                        or wf_precision_lift is None
                        or float(wf_precision_lift) < 2.0
                    ):
                        regressed = True
                        reason = (
                            "Drawdown-event quality gate: "
                            f"PR-AUC lift={pr_auc_lift} (required ≥ 2.0), "
                            f"precision lift={wf_precision_lift} (required ≥ 2.0)"
                        )
                        logger.error("Long-trend %s", reason)
                elif target_type == "three_state":
                    # Gate: macro-F1 > 0.40 AND each class F1 > 0.25
                    macro_f1 = long_result.get("macro_f1") or walk_forward.get("macro_f1") or 0.0
                    per_class = long_result.get("per_class") or walk_forward.get("per_class") or []
                    per_class_f1s = [pc.get("f1", 0.0) for pc in per_class]
                    if new_metric == "purged_walk_forward_multiclass" and (
                        float(macro_f1) <= 0.40
                        or not all(f > 0.25 for f in per_class_f1s)
                    ):
                        regressed = True
                        reason = (
                            "Three-state quality gate: "
                            f"macro_F1={macro_f1:.4f} (required > 0.40), "
                            f"per-class F1s={[round(f, 4) for f in per_class_f1s]} "
                            "(each required > 0.25)"
                        )
                        logger.error("Long-trend %s", reason)
                else:
                    # direction: existing OOS accuracy lift gate
                    oos_lift = walk_forward.get("accuracy_lift_vs_majority")
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
                        # Only run the accuracy-regression comparison for
                        # direction models; drawdown and three-state use their
                        # own gate metrics (PR-AUC lift / macro-F1).
                        if target_type == "direction":
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
                    # Gate passed — write the target_type meta sidecar so
                    # load_model() can verify alignment on the next startup.
                    from ml.long_trend import LongTrendModel as _LTM
                    _LTM.save_promotion_meta(target_type)
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
        await self._track_and_alert_baseline_duration(db_session, "long_trend")

        # ── Post-retrain broader-context ablation (non-blocking) ──────────────
        # Runs only when the long-trend retrain succeeded (long_flagged=False).
        # A failed ablation is logged but never raises — it must not abort the
        # rest of training or delay the short-trend model.
        if not long_flagged:
            try:
                from ml.post_retrain_ablation import run_broader_context_ablation  # noqa: PLC0415
                run_broader_context_ablation(
                    daily_df, vix_df, spx_close, broader_context
                )
            except Exception as _abl_exc:
                logger.error("post_retrain_ablation_error error=%s", _abl_exc)

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
        # Broader context is long-model-specific; short model ignores unknown keys.

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
            elif _short_event_gate_failed(short_result):
                # Rare-event quality gate: the short model is an alert model
                # for a minority event, so majority-style accuracy is not
                # enough.  Require the purged walk-forward PR-AUC to beat the
                # event base rate (the PR-AUC of a random scorer).  A candidate
                # that cannot rank rally bars above non-rally bars OOS must
                # not replace the active model.
                reason = _short_event_gate_failed(short_result)
                logger.error("Short-trend event quality gate: %s", reason)
                restored = _restore_model_file(SHORT_MODEL_PATH, short_backup, "short_trend")
                record_training_result(
                    "short_trend",
                    success=False,
                    error=f"Event quality gate: {reason}",
                    accuracy=short_result.get("accuracy", 0.0),
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
            # When skipping a retrain, audit whether the on-disk calibration
            # report still accurately describes the current DB.  A stale
            # "not enough rows" report can persist indefinitely if retraining
            # never happens; this ensures operators see honest metadata.
            await audit_calibration_report_staleness(db_session)
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

    async def _track_and_alert_baseline_duration(
        self, db_session: AsyncSession, model_name: str
    ) -> None:
        """Update baseline-mode onset tracking and fire a one-time duration alert.

        Called after each long-trend training attempt.  Checks whether the
        model is currently in baseline mode (via a forced mtime-reload), updates
        the ``baseline_mode_since`` timestamp accordingly, and pushes a push
        alert the first time the model has been continuously in baseline mode
        past ``settings.LONG_BASELINE_MODE_ALERT_DAYS``.

        Never raises — a notification failure must not break training.
        """
        if model_name != "long_trend":
            # Baseline-mode duration tracking is only defined for the long model.
            return
        try:
            # Force mtime-reload so we see the model state *after* the retrain
            # (a gate-passing retrain updates the pkl mtime).
            in_baseline = self.long_model.is_baseline_mode()

            if in_baseline:
                record_baseline_mode_onset(model_name)
                logger.debug(
                    "baseline_mode_tracking model=%s status=in_baseline "
                    "since=%s days=%.1f",
                    model_name,
                    get_baseline_mode_since(model_name),
                    get_baseline_mode_days(model_name) or 0.0,
                )
            else:
                clear_baseline_mode_tracking(model_name)
                return  # Not in baseline — nothing to alert about.

            threshold_days = float(settings.LONG_BASELINE_MODE_ALERT_DAYS)
            if not should_send_baseline_duration_alert(model_name, threshold_days):
                return

            from database.models import DeviceToken
            from notifications.fcm import FCMNotifier

            result = await db_session.execute(select(DeviceToken))
            tokens = result.scalars().all()
            if not tokens:
                logger.warning(
                    "baseline_duration_alert model=%s — no device tokens registered; "
                    "alert stays armed until a device is available",
                    model_name,
                )
                return

            days = get_baseline_mode_days(model_name) or 0.0
            notifier = FCMNotifier()
            any_sent = False
            for device in tokens:
                ok = await notifier.send_baseline_duration_alert(
                    device_token=device.token,
                    model_name=model_name,
                    days_in_baseline=days,
                    threshold_days=threshold_days,
                )
                any_sent = any_sent or ok

            if any_sent:
                mark_baseline_duration_alert_sent(model_name)
                logger.warning(
                    "baseline_duration_alert_sent model=%s days_in_baseline=%.1f "
                    "threshold_days=%.0f",
                    model_name,
                    days,
                    threshold_days,
                )
            else:
                logger.error(
                    "baseline_duration_alert_failed model=%s — will retry on next "
                    "failed retrain attempt",
                    model_name,
                )
        except Exception as exc:
            logger.error(
                "_track_and_alert_baseline_duration error for %s: %s", model_name, exc
            )

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
    async def _load_broader_context(db_session: AsyncSession) -> dict:
        """
        Load broader market context series for the long-trend model.

        Returns a dict mapping indicator key → pd.Series (daily closes).
        Keys consumed by long_trend.build_features():
          "vix_short_close"  — VIX9D  (9-day VIX, term-structure numerator)
          "vix_long_close"   — VIX3M  (3-month VIX, denominator)
          "credit_hy_close"  — HYG    (high-yield bond ETF)
          "credit_ig_close"  — LQD    (investment-grade bond ETF)
          "breadth_close"    — NYAD   (NYSE advance-decline line)
          "rates_close"      — TNX    (10-year Treasury yield × 10)

        When settings.LONG_BROADER_CONTEXT_ENABLED=False the caller still
        calls this method but long_trend ignores the keys, so returning empty
        Series is always safe.

        Each source is loaded from its own DB table when it exists.  If the
        table is absent, the key maps to an empty Series and the corresponding
        _missing feature fires to 1.0 during training — the model learns that
        absent context carries no directional signal.

        NOTE: DB tables for these tickers are not yet created.  When ingestion
        is wired up for a new ticker, add the corresponding SQLAlchemy model
        import and loader here following the _load_spx_close pattern.  Until
        then, each loader returns an empty Series and the feature layer
        gracefully falls back.

        Never raises.
        """
        result: dict = {
            "vix_short_close": pd.Series(dtype=float),
            "vix_long_close":  pd.Series(dtype=float),
            "credit_hy_close": pd.Series(dtype=float),
            "credit_ig_close": pd.Series(dtype=float),
            "breadth_close":   pd.Series(dtype=float),
            "rates_close":     pd.Series(dtype=float),
        }
        if not settings.LONG_BROADER_CONTEXT_ENABLED:
            # Short-circuit: flag off → context ignored downstream anyway.
            return result

        from database.models import (
            VixShortCandle, VixLongCandle, RatesCandle,
            CreditHyCandle, CreditIgCandle, BreadthCandle,
        )

        async def _load_close_series(model, ticker: str, key: str) -> pd.Series:
            """Load a daily close series from a context candle table.

            Returns an empty Series (triggers missing=1.0 in the feature
            layer) when the table has no rows or the query fails.
            """
            try:
                res = await db_session.execute(
                    select(model).where(
                        model.ticker == ticker,
                        model.timeframe == "daily",
                    ).order_by(model.timestamp.asc())
                )
                rows = res.scalars().all()
                if not rows:
                    logger.info(
                        "ml_trainer_context_unavailable key=%s ticker=%s — "
                        "missing feature will fire 1.0",
                        key, ticker,
                    )
                    return pd.Series(dtype=float)
                series = pd.Series(
                    [r.close for r in rows],
                    index=pd.to_datetime([r.timestamp for r in rows]),
                    dtype=float,
                )
                series = series[~series.index.duplicated(keep="last")]
                logger.info(
                    "ml_trainer_context_loaded key=%s ticker=%s rows=%d",
                    key, ticker, len(series),
                )
                return series
            except Exception as exc:
                logger.error(
                    "ml_trainer_context_load_error key=%s ticker=%s error=%s",
                    key, ticker, exc,
                )
                return pd.Series(dtype=float)

        result["vix_short_close"] = await _load_close_series(
            VixShortCandle, settings.VIX_SHORT_TICKER, "vix_short_close"
        )
        result["vix_long_close"] = await _load_close_series(
            VixLongCandle, settings.VIX_LONG_TICKER, "vix_long_close"
        )
        result["rates_close"] = await _load_close_series(
            RatesCandle, settings.RATES_TICKER, "rates_close"
        )
        result["credit_hy_close"] = await _load_close_series(
            CreditHyCandle, settings.CREDIT_HY_TICKER, "credit_hy_close"
        )
        result["credit_ig_close"] = await _load_close_series(
            CreditIgCandle, settings.CREDIT_IG_TICKER, "credit_ig_close"
        )
        result["breadth_close"] = await _load_close_series(
            BreadthCandle, settings.BREADTH_TICKER, "breadth_close"
        )
        return result

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
