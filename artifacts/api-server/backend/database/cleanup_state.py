"""
Shared state for the background OHLC startup-cleanup task.

Imported by main.py (to set state) and routers/predictions.py (to read it)
without creating a circular import between the two modules.

State machine
-------------
  cleanup_pending=False, cleanup_done=False  → not started yet / not applicable
  cleanup_pending=True,  cleanup_done=False  → running
  cleanup_pending=False, cleanup_done=True   → finished (success or timeout)
"""
from __future__ import annotations

_cleanup_pending: bool = False
_cleanup_done: bool = False


def mark_cleanup_started() -> None:
    """Call before launching the background cleanup task."""
    global _cleanup_pending, _cleanup_done
    _cleanup_pending = True
    _cleanup_done = False


def mark_cleanup_finished() -> None:
    """Call in the finally block of the background cleanup task."""
    global _cleanup_pending, _cleanup_done
    _cleanup_pending = False
    _cleanup_done = True


def is_cleanup_pending() -> bool:
    """True while the background cleanup task is still running."""
    return _cleanup_pending


def is_cleanup_done() -> bool:
    """True once the background cleanup task has finished (success or timeout)."""
    return _cleanup_done


def reset_for_testing() -> None:
    """Reset to initial state — only intended for use in unit tests."""
    global _cleanup_pending, _cleanup_done
    _cleanup_pending = False
    _cleanup_done = False
