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
) -> None:
    """Persist the outcome of a training attempt for one model.

    Never raises — failure to record must not break training itself.
    """
    try:
        data = _load_raw()
        prev = data.get(model_name) if isinstance(data.get(model_name), dict) else {}
        if success and accuracy is not None:
            last_success_accuracy = float(accuracy)
        else:
            # Carry the last known good accuracy forward through failures so
            # a later retrain can still be compared against it.
            last_success_accuracy = prev.get("last_success_accuracy")
            if last_success_accuracy is None and prev.get("success") and prev.get("accuracy") is not None:
                last_success_accuracy = prev.get("accuracy")
        data[model_name] = {
            "success": bool(success),
            "error": (str(error)[:500] if error else None),
            "accuracy": (float(accuracy) if accuracy is not None else None),
            "last_success_accuracy": last_success_accuracy,
            "attempted_at": datetime.now(timezone.utc).isoformat(),
        }
        STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(STATUS_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as exc:
        logger.error("training_status write error: %s", exc)


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
                "error": entry.get("error"),
                "accuracy": entry.get("accuracy"),
                "attempted_at": entry.get("attempted_at"),
            }
        else:
            out[name] = {
                "success": None,
                "error": None,
                "accuracy": None,
                "attempted_at": None,
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
