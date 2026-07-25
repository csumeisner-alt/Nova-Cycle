"""
NovaCycle ML Fallback Stats Persistence
=======================================
Persists cumulative neutral-0.5 fallback counters so a backend restart does
not wipe the evidence that predictions had been repeatedly degrading.

Stored as JSON at ml/models/ml_fallback_stats.json (same directory / pattern
as training_status.json, which survives process restarts).
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

STATS_PATH = Path(__file__).parent / "models" / "ml_fallback_stats.json"

MODEL_NAMES = ("long_trend", "short_trend")


def _load_raw() -> dict:
    try:
        if STATS_PATH.exists():
            with open(STATS_PATH, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception as exc:
        logger.error("fallback_stats read error: %s", exc)
    return {}


def record_fallback(model_name: str, reason: str) -> None:
    """Increment the persisted cumulative fallback counter for one model.

    Never raises — failure to persist must not break serving a prediction.
    """
    try:
        data = _load_raw()
        prev = data.get(model_name) if isinstance(data.get(model_name), dict) else {}
        try:
            total = int(prev.get("total_count", 0))
        except (TypeError, ValueError):
            total = 0
        data[model_name] = {
            "total_count": total + 1,
            "last_at": datetime.now(timezone.utc).isoformat(),
            "last_reason": str(reason)[:300],
        }
        STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(STATS_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as exc:
        logger.error("fallback_stats write error: %s", exc)


def get_persisted_fallback_stats() -> dict:
    """Return cumulative fallback stats per model (all restarts included).

    Models with no recorded fallback yet get zeros/None so the health
    surface always has a consistent shape. Never raises.
    """
    data = _load_raw()
    out = {}
    for name in MODEL_NAMES:
        entry = data.get(name)
        if isinstance(entry, dict):
            try:
                total = int(entry.get("total_count", 0))
            except (TypeError, ValueError):
                total = 0
            out[name] = {
                "total_count": total,
                "last_at": entry.get("last_at"),
                "last_reason": entry.get("last_reason"),
            }
        else:
            out[name] = {"total_count": 0, "last_at": None, "last_reason": None}
    return out
