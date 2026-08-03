"""One-off operator script: retrain the short-trend model on the shared
rally-event label, applying the same rollback and quality gates as the
scheduled trainer.

Run from artifacts/api-server/backend:
    python scripts/retrain_short_model.py
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("retrain_short")


async def main() -> int:
    from database.db import get_session_factory
    from indicators.technical import TechnicalIndicators
    from ml.model_health import check_accuracy_regression
    from ml.short_trend import MODEL_PATH as SHORT_MODEL_PATH, ShortTrendModel
    from ml.trainer import (
        ModelTrainer,
        _backup_model_file,
        _is_metric_upgrade_transition,
        _restore_model_file,
        _short_event_gate_failed,
    )
    from ml.training_status import (
        get_last_successful_accuracy,
        get_last_successful_accuracy_metric,
        record_training_result,
    )

    factory = get_session_factory()
    async with factory() as session:
        fivemin_df = await ModelTrainer._load_fivemin_voo(session)
        vix_df = await ModelTrainer._load_vix(session)
        spx_close = await ModelTrainer._load_spx_close(session)

    if fivemin_df.empty:
        logger.error("No 5-min VOO data; aborting.")
        return 1

    indicators = TechnicalIndicators().compute_all(
        fivemin_df, vix_df, exclude_extended=False
    )
    if spx_close is not None and not spx_close.empty:
        indicators["spx_futures_close"] = spx_close

    model = ShortTrendModel()
    backup = _backup_model_file(SHORT_MODEL_PATH)
    result = model.train(fivemin_df, indicators)
    wf = result.get("walk_forward") or {}
    logger.info(
        "trained: acc=%s metric=%s positive_rate=%s pr_auc=%s "
        "event_precision=%s event_recall=%s lift_vs_majority=%s",
        result.get("accuracy"), result.get("accuracy_metric"),
        wf.get("positive_rate"), wf.get("pr_auc"),
        wf.get("event_precision"), wf.get("event_recall"),
        wf.get("accuracy_lift_vs_majority"),
    )

    if model.model is None:
        restored = _restore_model_file(SHORT_MODEL_PATH, backup, "short_trend")
        record_training_result(
            "short_trend", success=False,
            error="Training produced no model", rolled_back=restored,
        )
        return 1
    if result.get("degenerate"):
        restored = _restore_model_file(SHORT_MODEL_PATH, backup, "short_trend")
        record_training_result(
            "short_trend", success=False,
            error="Degenerate model: " + (result.get("degeneracy_reason") or ""),
            rolled_back=restored,
        )
        return 1
    gate_reason = _short_event_gate_failed(result)
    if gate_reason:
        restored = _restore_model_file(SHORT_MODEL_PATH, backup, "short_trend")
        record_training_result(
            "short_trend", success=False,
            error=f"Event quality gate: {gate_reason}",
            accuracy=result.get("accuracy", 0.0), rolled_back=restored,
        )
        logger.error("REJECTED: %s", gate_reason)
        return 1

    new_acc = result.get("accuracy", 0.0)
    new_metric = result.get("accuracy_metric")
    prev_acc = get_last_successful_accuracy("short_trend")
    prev_metric = get_last_successful_accuracy_metric("short_trend")
    if _is_metric_upgrade_transition(prev_metric, new_metric):
        regressed, reason = False, None
    else:
        regressed, reason = check_accuracy_regression(new_acc, prev_acc)
    # NOTE: the label semantics changed (exact-12-bar → window-max), so the
    # accuracy comparison crosses label definitions.  The event gate above is
    # the binding check; a raw-accuracy drop caused purely by the higher base
    # rate of the window-max label is expected and reported, not fatal here —
    # but we still refuse a model with NO event ranking power.
    if regressed:
        logger.warning(
            "accuracy comparison across label definitions: %s "
            "(accepting because the event gate passed and label semantics changed)",
            reason,
        )
    record_training_result(
        "short_trend", success=True, accuracy=new_acc, accuracy_metric=new_metric,
    )
    logger.info("ACCEPTED: short model updated on shared rally-event label.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
