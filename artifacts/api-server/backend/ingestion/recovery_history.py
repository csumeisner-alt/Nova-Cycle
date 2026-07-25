"""
NovaCycle 5-min Feed Recovery History Persistence
=================================================
Persists the 5-min feed auto-recovery attempt record (last attempt plus a
small rolling history and cumulative failure count) so a backend restart
does not wipe the evidence that a recovery fired or failed shortly before
the restart.

Same pattern as ml/fallback_stats.py (JSON file, never raises on the write
path used while serving requests).
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

HISTORY_PATH = Path(__file__).parent / "recovery_history.json"

# Rolling history cap — small on purpose: this is operator evidence for
# /healthz, not an audit log.
MAX_HISTORY = 20


def _load_raw() -> dict:
    try:
        if HISTORY_PATH.exists():
            with open(HISTORY_PATH, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception as exc:
        logger.error("recovery_history read error: %s", exc)
    return {}


def record_recovery_attempt(
    outcome: str, at_iso: str, bars_fetched: Optional[int]
) -> None:
    """Persist one recovery attempt (never raises).

    Appends to the rolling history, updates the last-attempt record, and
    bumps the cumulative failure counter on "failed" outcomes.
    """
    try:
        data = _load_raw()
        entry = {
            "last_attempt_at": at_iso,
            "outcome": outcome,
            "bars_fetched": bars_fetched,
        }
        data["last_attempt"] = entry

        history = data.get("history")
        if not isinstance(history, list):
            history = []
        history.append(entry)
        data["history"] = history[-MAX_HISTORY:]

        try:
            failures = int(data.get("failure_count", 0))
        except (TypeError, ValueError):
            failures = 0
        if outcome == "failed":
            failures += 1
        data["failure_count"] = failures

        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as exc:
        logger.error("recovery_history write error: %s", exc)


def get_persisted_recovery_status() -> dict:
    """Return the persisted recovery record (survives restarts). Never raises.

    Shape:
        {"last_attempt": {...}|None, "history": [...], "failure_count": int}
    """
    data = _load_raw()
    last = data.get("last_attempt")
    if not isinstance(last, dict):
        last = None
    history = data.get("history")
    if not isinstance(history, list):
        history = []
    try:
        failures = int(data.get("failure_count", 0))
    except (TypeError, ValueError):
        failures = 0
    return {"last_attempt": last, "history": history, "failure_count": failures}
