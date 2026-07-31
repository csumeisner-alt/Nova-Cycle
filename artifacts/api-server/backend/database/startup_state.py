"""
Shared state for pipeline startup status.

Imported by main.py (to set state) and routers/predictions.py (to read it)
without creating a circular import between the two modules.

State machine
-------------
  status="pending"  → startup not yet complete
  status="ok"       → _init_pipeline() completed without error
  status="degraded" → _init_pipeline()'s inner try caught an exception before
                      initialize() succeeded; jobs were unblocked but the
                      pipeline may be in a partial state
"""
from __future__ import annotations

_startup_status: str = "pending"      # "pending" | "ok" | "degraded"
_startup_error: str | None = None     # one-line summary of the caught exception


def mark_startup_ok() -> None:
    """Call when _init_pipeline() completes successfully."""
    global _startup_status, _startup_error
    _startup_status = "ok"
    _startup_error = None


def mark_startup_degraded(error: str) -> None:
    """Call when _init_pipeline()'s inner try catches an exception."""
    global _startup_status, _startup_error
    _startup_status = "degraded"
    _startup_error = error


def get_startup_status() -> str:
    """Return "pending" | "ok" | "degraded"."""
    return _startup_status


def get_startup_error() -> str | None:
    """Return the error summary if startup was degraded, else None."""
    return _startup_error


def reset_for_testing() -> None:
    """Reset to initial state — only intended for use in unit tests."""
    global _startup_status, _startup_error
    _startup_status = "pending"
    _startup_error = None
