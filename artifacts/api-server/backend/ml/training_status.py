"""
NovaCycle Training Status Tracker
==================================
Persists the outcome of the most recent training attempt for each model so
the API can surface a health flag when the weekly retrain fails silently and
predictions fall back to neutral 0.5.

Status is stored as JSON at ml/models/training_status.json so it survives
process restarts (the same directory that holds the pickled models).
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

STATUS_PATH = Path(__file__).parent / "models" / "training_status.json"

MODEL_NAMES = ("long_trend", "short_trend")

# Number of consecutive failed training attempts after which the health
# endpoint flags a model as "stuck" (retrying daily but never succeeding).
CONSECUTIVE_FAILURE_ALERT_THRESHOLD = 3


def _load_raw() -> dict:
    try:
        if STATUS_PATH.exists():
            with open(STATUS_PATH, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception as exc:
        logger.error("training_status read error: %s", exc)
    return {}


def record_training_result(
    model_name: str,
    success: bool,
    error: Optional[str] = None,
    accuracy: Optional[float] = None,
    rolled_back: bool = False,
    accuracy_metric: Optional[str] = None,
) -> None:
    """Persist the outcome of a training attempt for one model.

    Never raises — failure to record must not break training itself.
    """
    try:
        data = _load_raw()
        prev = data.get(model_name) if isinstance(data.get(model_name), dict) else {}
        if success and accuracy is not None:
            last_success_accuracy = float(accuracy)
            last_success_accuracy_metric = accuracy_metric
        else:
            last_success_accuracy_metric = prev.get("last_success_accuracy_metric")
            # Carry the last known good accuracy forward through failures so
            # a later retrain can still be compared against it.
            last_success_accuracy = prev.get("last_success_accuracy")
            if last_success_accuracy is None and prev.get("success") and prev.get("accuracy") is not None:
                last_success_accuracy = prev.get("accuracy")
        if success:
            consecutive_failures = 0
            # A successful retrain ends the stuck episode and re-arms the alert.
            stuck_alert_sent = False
        else:
            try:
                consecutive_failures = int(prev.get("consecutive_failures") or 0) + 1
            except (TypeError, ValueError):
                consecutive_failures = 1
            # Carry the flag forward through the episode so the alert fires
            # only once per stuck episode.
            stuck_alert_sent = bool(prev.get("stuck_alert_sent"))
        # Carry baseline-mode tracking fields forward so they survive across
        # training attempts.  These fields are managed exclusively by
        # record_baseline_mode_onset() and clear_baseline_mode_tracking();
        # record_training_result() must never reset them.
        data[model_name] = {
            "success": bool(success),
            "rolled_back": bool(rolled_back) and not success,
            "error": (str(error)[:500] if error else None),
            "accuracy": (float(accuracy) if accuracy is not None else None),
            "accuracy_metric": accuracy_metric,
            "last_success_accuracy": last_success_accuracy,
            "last_success_accuracy_metric": last_success_accuracy_metric,
            "consecutive_failures": consecutive_failures,
            "stuck_alert_sent": stuck_alert_sent,
            "attempted_at": datetime.now(timezone.utc).isoformat(),
            # Baseline-mode duration: preserve across training attempts.
            "baseline_mode_since": prev.get("baseline_mode_since"),
            "baseline_mode_alert_sent": bool(prev.get("baseline_mode_alert_sent")),
        }
        STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(STATUS_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as exc:
        logger.error("training_status write error: %s", exc)


def _safe_int(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def get_consecutive_failures(model_name: str) -> int:
    """Return the current consecutive-failure count for one model.

    Never raises.
    """
    try:
        entry = _load_raw().get(model_name)
        if isinstance(entry, dict):
            return _safe_int(entry.get("consecutive_failures"))
    except Exception as exc:
        logger.error("training_status get_consecutive_failures error: %s", exc)
    return 0


def should_send_stuck_alert(model_name: str) -> bool:
    """Return True when a "training stuck" push notification should be sent.

    True only when the model's consecutive-failure count has reached
    CONSECUTIVE_FAILURE_ALERT_THRESHOLD and no alert has been sent for the
    current stuck episode yet. Never raises.
    """
    try:
        entry = _load_raw().get(model_name)
        if not isinstance(entry, dict):
            return False
        if _safe_int(entry.get("consecutive_failures")) < CONSECUTIVE_FAILURE_ALERT_THRESHOLD:
            return False
        return not bool(entry.get("stuck_alert_sent"))
    except Exception as exc:
        logger.error("training_status should_send_stuck_alert error: %s", exc)
    return False


def mark_stuck_alert_sent(model_name: str) -> None:
    """Record that the stuck-training alert was sent for the current episode.

    Never raises.
    """
    try:
        data = _load_raw()
        entry = data.get(model_name)
        if not isinstance(entry, dict):
            return
        entry["stuck_alert_sent"] = True
        STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(STATUS_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as exc:
        logger.error("training_status mark_stuck_alert_sent error: %s", exc)


def get_training_status() -> dict:
    """Return last-attempt status per model.

    Models with no recorded attempt yet get {"success": None, ...} so the
    health surface can distinguish "never trained" from "trained OK".
    """
    data = _load_raw()
    out = {}
    for name in MODEL_NAMES:
        entry = data.get(name)
        if isinstance(entry, dict):
            out[name] = {
                "success": entry.get("success"),
                "rolled_back": bool(entry.get("rolled_back")),
                "error": entry.get("error"),
                "accuracy": entry.get("accuracy"),
                "accuracy_metric": entry.get("accuracy_metric"),
                "last_success_accuracy": entry.get("last_success_accuracy"),
                "last_success_accuracy_metric": entry.get(
                    "last_success_accuracy_metric"
                ),
                "attempted_at": entry.get("attempted_at"),
                "consecutive_failures": _safe_int(entry.get("consecutive_failures")),
            }
        else:
            out[name] = {
                "success": None,
                "rolled_back": False,
                "error": None,
                "accuracy": None,
                "accuracy_metric": None,
                "last_success_accuracy": None,
                "last_success_accuracy_metric": None,
                "attempted_at": None,
                "consecutive_failures": 0,
            }
    return out


def any_model_failed_last_attempt() -> bool:
    """Return True when the most recent recorded training attempt failed for
    any model (e.g. a regressed retrain that was rolled back).

    Models with no recorded attempt yet are not counted as failed.
    Never raises.
    """
    try:
        data = _load_raw()
        for name in MODEL_NAMES:
            entry = data.get(name)
            if isinstance(entry, dict) and entry.get("success") is False:
                return True
    except Exception as exc:
        logger.error("training_status any_model_failed_last_attempt error: %s", exc)
    return False


def get_last_successful_accuracy(model_name: str) -> Optional[float]:
    """Return the accuracy of the most recent *successful* training run.

    Falls back through the carried-forward ``last_success_accuracy`` so a
    failed attempt in between does not erase the reference point.
    Never raises.
    """
    try:
        entry = _load_raw().get(model_name)
        if not isinstance(entry, dict):
            return None
        if entry.get("success") and entry.get("accuracy") is not None:
            return float(entry["accuracy"])
        if entry.get("last_success_accuracy") is not None:
            return float(entry["last_success_accuracy"])
    except Exception as exc:
        logger.error("training_status get_last_successful_accuracy error: %s", exc)
    return None


def get_last_successful_accuracy_metric(model_name: str) -> Optional[str]:
    """Return the metric kind ("train", "purged_walk_forward_oos", …) of the
    most recent successful run's recorded accuracy, or None for runs recorded
    before metric tracking existed. Never raises.
    """
    try:
        entry = _load_raw().get(model_name)
        if not isinstance(entry, dict):
            return None
        if entry.get("success") and entry.get("accuracy") is not None:
            return entry.get("accuracy_metric")
        return entry.get("last_success_accuracy_metric")
    except Exception as exc:
        logger.error("training_status get_last_successful_accuracy_metric error: %s", exc)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Baseline-mode duration tracking
# ──────────────────────────────────────────────────────────────────────────────


def record_baseline_mode_onset(model_name: str) -> None:
    """Record that ``model_name`` has entered baseline mode (idempotent).

    Sets ``baseline_mode_since`` only on the first call for an episode; a
    second call during the same episode preserves the original timestamp so
    the elapsed-days calculation remains accurate.  Never raises.
    """
    try:
        data = _load_raw()
        entry = data.get(model_name)
        if not isinstance(entry, dict):
            entry = {}
        # Idempotent: preserve the original onset timestamp across retrain
        # cycles so we measure total continuous duration, not just since the
        # last failed retrain.
        if not entry.get("baseline_mode_since"):
            entry["baseline_mode_since"] = datetime.now(timezone.utc).isoformat()
            data[model_name] = entry
            STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(STATUS_PATH, "w") as f:
                json.dump(data, f, indent=2)
    except Exception as exc:
        logger.error("training_status record_baseline_mode_onset error: %s", exc)


def clear_baseline_mode_tracking(model_name: str) -> None:
    """Clear baseline-mode tracking when the model exits baseline mode.

    Resets ``baseline_mode_since`` and ``baseline_mode_alert_sent`` so a
    future re-entry starts a fresh duration counter.  Never raises.
    """
    try:
        data = _load_raw()
        entry = data.get(model_name)
        if not isinstance(entry, dict):
            return
        changed = False
        if entry.get("baseline_mode_since") is not None:
            entry["baseline_mode_since"] = None
            changed = True
        if entry.get("baseline_mode_alert_sent"):
            entry["baseline_mode_alert_sent"] = False
            changed = True
        if changed:
            data[model_name] = entry
            STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(STATUS_PATH, "w") as f:
                json.dump(data, f, indent=2)
    except Exception as exc:
        logger.error("training_status clear_baseline_mode_tracking error: %s", exc)


def get_baseline_mode_since(model_name: str) -> Optional[str]:
    """Return the ISO-8601 timestamp when baseline mode began, or None.

    Never raises.
    """
    try:
        entry = _load_raw().get(model_name)
        if isinstance(entry, dict):
            return entry.get("baseline_mode_since") or None
    except Exception as exc:
        logger.error("training_status get_baseline_mode_since error: %s", exc)
    return None


def get_baseline_mode_days(model_name: str) -> Optional[float]:
    """Return how many calendar days the model has been in baseline mode.

    Returns None when the model is not currently in baseline mode (i.e. no
    ``baseline_mode_since`` timestamp is recorded).  Never raises.
    """
    since_str = get_baseline_mode_since(model_name)
    if not since_str:
        return None
    try:
        since = datetime.fromisoformat(since_str)
        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - since
        return delta.total_seconds() / 86400.0
    except Exception as exc:
        logger.error("training_status get_baseline_mode_days error: %s", exc)
    return None


def should_send_baseline_duration_alert(model_name: str, threshold_days: float) -> bool:
    """Return True when a baseline-mode duration alert should be sent.

    True only when:
    - ``threshold_days`` > 0 (alert not disabled),
    - the model has been in baseline mode for at least ``threshold_days``, and
    - no alert has been sent for the current baseline-mode episode yet.
    Never raises.
    """
    if threshold_days <= 0:
        return False
    try:
        entry = _load_raw().get(model_name)
        if not isinstance(entry, dict):
            return False
        if entry.get("baseline_mode_alert_sent"):
            return False
        days = get_baseline_mode_days(model_name)
        return days is not None and days >= threshold_days
    except Exception as exc:
        logger.error("training_status should_send_baseline_duration_alert error: %s", exc)
    return False


def mark_baseline_duration_alert_sent(model_name: str) -> None:
    """Record that the baseline-duration alert was sent for the current episode.

    Never raises.
    """
    try:
        data = _load_raw()
        entry = data.get(model_name)
        if not isinstance(entry, dict):
            return
        entry["baseline_mode_alert_sent"] = True
        data[model_name] = entry
        STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(STATUS_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as exc:
        logger.error("training_status mark_baseline_duration_alert_sent error: %s", exc)
