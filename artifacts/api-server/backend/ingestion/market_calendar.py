"""
NovaCycle Market Calendar
=========================
US equity market calendar helpers for ingestion:

  - Proper US/Eastern timezone conversion (DST-correct, via zoneinfo)
  - NYSE full-day holiday detection (rule-based, no network)
  - Half-day (early close 13:00 ET) detection
  - Session classification with a fallback heuristic classifier

Session rules (ET):
  04:00 – 09:30  → pre_market   (extended)
  09:30 – 16:00  → regular      (13:00 close on half-days)
  16:00 – 20:00  → after_hours  (extended)

All timestamps stored in the DB remain UTC-naive; these helpers convert
UTC(-naive) → US/Eastern only for classification purposes.
"""

import logging
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

EASTERN = ZoneInfo("America/New_York")

# Session boundaries in fractional ET hours
PRE_MARKET_START = 4.0
REGULAR_START = 9.5
REGULAR_END = 16.0
HALF_DAY_END = 13.0
AFTER_HOURS_END = 20.0


# ─────────────────────────────────────────────────────────────────────────────
# Timezone conversion
# ─────────────────────────────────────────────────────────────────────────────

def to_eastern(ts: datetime) -> datetime:
    """
    Convert a timestamp to US/Eastern.

    Naive timestamps are assumed to be UTC (the pipeline stores UTC-naive).
    """
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(EASTERN)


# ─────────────────────────────────────────────────────────────────────────────
# Holiday rules (NYSE)
# ─────────────────────────────────────────────────────────────────────────────

def _easter(year: int) -> date:
    """Anonymous Gregorian algorithm for Easter Sunday."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th (1-based) `weekday` (Mon=0) of a month."""
    d = date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    return d + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """Last `weekday` (Mon=0) of a month."""
    if month == 12:
        d = date(year, 12, 31)
    else:
        d = date(year, month + 1, 1) - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d: date) -> date:
    """Weekend holidays are observed Friday (Sat) or Monday (Sun)."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


@lru_cache(maxsize=64)
def market_holidays(year: int) -> frozenset:
    """Full-day NYSE holidays for a given year."""
    days = {
        _observed(date(year, 1, 1)),                      # New Year's Day
        _nth_weekday(year, 1, 0, 3),                      # MLK Day
        _nth_weekday(year, 2, 0, 3),                      # Washington's Birthday
        _easter(year) - timedelta(days=2),                # Good Friday
        _last_weekday(year, 5, 0),                        # Memorial Day
        _observed(date(year, 7, 4)),                      # Independence Day
        _nth_weekday(year, 9, 0, 1),                      # Labor Day
        _nth_weekday(year, 11, 3, 4),                     # Thanksgiving
        _observed(date(year, 12, 25)),                    # Christmas
    }
    if year >= 2022:                                      # Juneteenth (NYSE since 2022)
        days.add(_observed(date(year, 6, 19)))
    return frozenset(days)


def is_market_holiday(d: date) -> bool:
    return d in market_holidays(d.year)


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and not is_market_holiday(d)


@lru_cache(maxsize=64)
def half_days(year: int) -> frozenset:
    """
    Early-close (13:00 ET) days:
      - July 3 when it is a trading day and July 4 is a weekday
      - Day after Thanksgiving
      - Christmas Eve when it is a trading day
    """
    days = set()
    jul3 = date(year, 7, 3)
    if jul3.weekday() < 5 and date(year, 7, 4).weekday() < 5 and not is_market_holiday(jul3):
        days.add(jul3)
    days.add(_nth_weekday(year, 11, 3, 4) + timedelta(days=1))  # day after Thanksgiving
    dec24 = date(year, 12, 24)
    if dec24.weekday() < 5 and not is_market_holiday(dec24):
        days.add(dec24)
    return frozenset(days)


def is_half_day(d: date) -> bool:
    return d in half_days(d.year)


# ─────────────────────────────────────────────────────────────────────────────
# Session classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_session(ts: datetime) -> tuple[bool, str, str]:
    """
    Classify a timestamp into (is_extended_hours, session_type, method).

    method is 'calendar' when the primary DST/holiday-aware classifier was
    used, 'fallback' when the fixed-offset heuristic had to be used.
    """
    try:
        ts_et = to_eastern(ts)
        d = ts_et.date()
        hour = ts_et.hour + ts_et.minute / 60.0

        if not is_trading_day(d):
            # Holiday / weekend candle – anything traded is extended-hours
            return True, "after_hours", "calendar"

        regular_end = HALF_DAY_END if is_half_day(d) else REGULAR_END

        if PRE_MARKET_START <= hour < REGULAR_START:
            return True, "pre_market", "calendar"
        if REGULAR_START <= hour < regular_end:
            return False, "regular", "calendar"
        if regular_end <= hour < AFTER_HOURS_END:
            return True, "after_hours", "calendar"
        return True, "after_hours", "calendar"
    except Exception as exc:
        logger.warning(
            "session_classify_fallback ts=%s error=%s", ts.isoformat(), exc
        )
        return _fallback_classify(ts)


def _fallback_utc_offset_hours(ts_utc: datetime) -> int:
    """
    Approximate US/Eastern UTC offset without zoneinfo.

    US DST (since 2007): starts second Sunday of March, ends first Sunday
    of November, transitions at 2:00 local (07:00 UTC start / 06:00 UTC end).
    Returns -4 (EDT) or -5 (EST).
    """
    year = ts_utc.year
    dst_start = datetime.combine(
        _nth_weekday(year, 3, 6, 2), datetime.min.time()
    ) + timedelta(hours=7)   # 2:00 EST == 07:00 UTC
    dst_end = datetime.combine(
        _nth_weekday(year, 11, 6, 1), datetime.min.time()
    ) + timedelta(hours=6)   # 2:00 EDT == 06:00 UTC
    return -4 if dst_start <= ts_utc < dst_end else -5


def _fallback_classify(ts: datetime) -> tuple[bool, str, str]:
    """DST-approximate fixed-offset heuristic used when the calendar classifier fails."""
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
    ts_local = ts + timedelta(hours=_fallback_utc_offset_hours(ts))
    hour = ts_local.hour + ts_local.minute / 60.0
    if 4.0 <= hour < 9.5:
        return True, "pre_market", "fallback"
    if 9.5 <= hour < 16.0:
        return False, "regular", "fallback"
    return True, "after_hours", "fallback"


# ─────────────────────────────────────────────────────────────────────────────
# Timestamp sanity checks
# ─────────────────────────────────────────────────────────────────────────────

MIN_PLAUSIBLE = datetime(2000, 1, 1)


def timestamp_sanity_issue(ts: datetime, now_utc: datetime | None = None) -> str | None:
    """
    Return a short issue tag when a UTC-naive timestamp is implausible,
    or None when it looks fine.
    """
    if now_utc is None:
        now_utc = datetime.utcnow()
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
    if ts < MIN_PLAUSIBLE:
        return "before_2000"
    if ts > now_utc + timedelta(days=1):
        return "in_future"
    return None
