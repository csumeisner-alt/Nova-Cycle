"""
NovaCycle Data Fetcher
======================
Wraps yfinance to provide async-compatible data access.

NOTE: "Pipeline currently fetches only VOO. Multi-ticker ingestion will be added later."

Session classification rules:
  04:00 – 09:30 ET  → pre_market
  09:30 – 16:00 ET  → regular
  16:00 – 20:00 ET  → after_hours

Gap formula:
  GapPercent = (PreMarketOpen - PreviousClose) / PreviousClose * 100
  gap_up   if GapPercent > +1.0 %
  gap_down if GapPercent < -1.0 %
  none     otherwise

Liquidity formula:
  LiquidityScore = Volume_extended / Volume_regular
  If < 0.15 → weights × 0.5, thresholds × 1.25, suppress weak signals
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import yfinance as yf

from config import settings
from ingestion import market_calendar

logger = logging.getLogger(__name__)


class DataFetcher:
    """Async-friendly data fetcher backed by yfinance."""

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _classify_session(ts: datetime) -> tuple[bool, str]:
        """
        Return (is_extended_hours, session_type) for a given UTC timestamp.

        Uses the DST/holiday/half-day aware calendar classifier; falls back
        to a fixed-offset heuristic when the calendar classifier fails
        (the fallback is logged inside market_calendar).
        """
        is_ext, session, _method = market_calendar.classify_session(ts)
        return is_ext, session

    @staticmethod
    def _run_sync(func, *args, **kwargs):
        """Run a synchronous yfinance call inside the event loop's thread pool."""
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(None, lambda: func(*args, **kwargs))

    # ── Public API ────────────────────────────────────────────────────────────

    async def fetch_historical_voo(self, years: int = 10) -> dict[str, pd.DataFrame]:
        """
        Fetch full historical VOO data:
          - Daily candles going back `years` years
          - 5-minute candles for the last 60 days (yfinance limit)

        Returns:
            {"daily": pd.DataFrame, "5min": pd.DataFrame}

        NOTE: "Pipeline currently fetches only VOO. Multi-ticker ingestion will be added later."
        """
        ticker = settings.TICKER
        logger.info("Fetching %d years of historical daily VOO data…", years)

        # ── Daily candles ──────────────────────────────────────────────────────
        try:
            daily_df = await self._run_sync(
                yf.download,
                ticker,
                period=f"{years}y",
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            daily_df = self._normalise_columns(daily_df)
            daily_df["is_extended_hours"] = False
            daily_df["session_type"] = "regular"
            logger.info("Fetched %d daily candles for VOO", len(daily_df))
        except Exception as exc:
            logger.error("Error fetching daily VOO data: %s", exc)
            daily_df = pd.DataFrame()

        # ── 5-minute candles ───────────────────────────────────────────────────
        try:
            fivemin_df = await self.fetch_5min_candles(period="60d")
            logger.info("Fetched %d 5-min candles for VOO", len(fivemin_df))
        except Exception as exc:
            logger.error("Error fetching 5-min VOO data: %s", exc)
            fivemin_df = pd.DataFrame()

        return {"daily": daily_df, "5min": fivemin_df}

    async def fetch_historical_vix(self, years: int = 10) -> pd.DataFrame:
        """
        Fetch `years` years of daily VIX data.

        Returns:
            pd.DataFrame with columns: open, high, low, close, volume, timestamp
        """
        vix_ticker = settings.VIX_TICKER
        logger.info("Fetching %d years of historical VIX data…", years)
        try:
            df = await self._run_sync(
                yf.download,
                vix_ticker,
                period=f"{years}y",
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            df = self._normalise_columns(df)
            logger.info("Fetched %d daily VIX candles", len(df))
            return df
        except Exception as exc:
            logger.error("Error fetching VIX data: %s", exc)
            return pd.DataFrame()

    async def fetch_incremental_voo(self, last_timestamp: datetime) -> dict[str, pd.DataFrame]:
        """
        Fetch only new candles since `last_timestamp`.
        Never re-fetches full history – uses a short rolling window.

        Returns:
            {"daily": pd.DataFrame, "5min": pd.DataFrame}
        """
        ticker = settings.TICKER
        start_str = last_timestamp.strftime("%Y-%m-%d")
        logger.info("Incremental fetch: VOO since %s", start_str)

        result = {"daily": pd.DataFrame(), "5min": pd.DataFrame()}

        # ── Daily incremental ──────────────────────────────────────────────────
        try:
            daily_df = await self._run_sync(
                yf.download,
                ticker,
                start=start_str,
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            daily_df = self._normalise_columns(daily_df)
            if not daily_df.empty:
                # Drop the row equal to last_timestamp to avoid duplicate
                daily_df = daily_df[daily_df.index > pd.Timestamp(last_timestamp)]
            daily_df["is_extended_hours"] = False
            daily_df["session_type"] = "regular"
            result["daily"] = daily_df
            logger.info("Incremental: %d new daily candles", len(daily_df))
        except Exception as exc:
            logger.error("Error in incremental daily fetch: %s", exc)

        # ── 5-min incremental (yfinance limit: 60d) ────────────────────────────
        try:
            fivemin_df = await self.fetch_5min_candles(period="5d")
            if not fivemin_df.empty:
                fivemin_df = fivemin_df[fivemin_df.index > pd.Timestamp(last_timestamp)]
            result["5min"] = fivemin_df
            logger.info("Incremental: %d new 5-min candles", len(fivemin_df))
        except Exception as exc:
            logger.error("Error in incremental 5-min fetch: %s", exc)

        return result

    async def fetch_5min_candles(self, period: str = "60d") -> pd.DataFrame:
        """
        Fetch 5-minute candles with extended-hours data (prepost=True).

        Session classification:
          04:00 – 09:30 ET → pre_market   (is_extended_hours=True)
          09:30 – 16:00 ET → regular       (is_extended_hours=False)
          16:00 – 20:00 ET → after_hours  (is_extended_hours=True)

        Returns:
            pd.DataFrame with extra columns: is_extended_hours, session_type
        """
        ticker = settings.TICKER
        logger.info("Fetching 5-min candles (period=%s, prepost=True)…", period)
        try:
            df = await self._run_sync(
                yf.download,
                ticker,
                period=period,
                interval="5m",
                prepost=True,
                auto_adjust=True,
                progress=False,
            )
            df = self._normalise_columns(df)

            if df.empty:
                return df

            # Classify each row's session
            sessions = [self._classify_session(ts) for ts in df.index]
            df["is_extended_hours"] = [s[0] for s in sessions]
            df["session_type"] = [s[1] for s in sessions]

            return df
        except Exception as exc:
            logger.error("Error fetching 5-min candles: %s", exc)
            return pd.DataFrame()

    async def detect_gap(
        self, prev_close: float, premarket_open: float
    ) -> dict:
        """
        Calculate the overnight gap.

        Formula:
          GapPercent = (PreMarketOpen - PreviousClose) / PreviousClose * 100

        Thresholds (from config):
          gap_up   : GapPercent > +GAP_UP_THRESHOLD   (default +1.0 %)
          gap_down : GapPercent < +GAP_DOWN_THRESHOLD  (default -1.0 %)
          none     : otherwise

        Additive classification (does not change gap_type semantics):
          gap_class:
            micro : 0 < |GapPercent| < MICRO_GAP_THRESHOLD (default 0.1 %)
            macro : |GapPercent| > MACRO_GAP_THRESHOLD     (default 1.0 %)
            minor : otherwise (between micro and macro)
            none  : GapPercent == 0
          gap_momentum: placeholder (None) — reserved for future
            follow-through measurement; always present in the dict.

        Returns:
            {"gap_percent": float, "gap_type": str,
             "gap_class": str, "gap_momentum": None}
        """
        try:
            if prev_close == 0:
                return {
                    "gap_percent": 0.0,
                    "gap_type": "none",
                    "gap_class": "none",
                    "gap_momentum": None,
                }

            gap_pct = (premarket_open - prev_close) / prev_close * 100.0

            if gap_pct > settings.GAP_UP_THRESHOLD:
                gap_type = "gap_up"
            elif gap_pct < settings.GAP_DOWN_THRESHOLD:
                gap_type = "gap_down"
            else:
                gap_type = "none"

            return {
                "gap_percent": round(gap_pct, 4),
                "gap_type": gap_type,
                "gap_class": self.classify_gap_magnitude(gap_pct),
                "gap_momentum": None,
            }
        except Exception as exc:
            logger.error("Error detecting gap: %s", exc)
            return {
                "gap_percent": 0.0,
                "gap_type": "none",
                "gap_class": "none",
                "gap_momentum": None,
            }

    @staticmethod
    def classify_gap_magnitude(gap_pct: float) -> str:
        """
        Classify a gap percentage by magnitude (additive, VOO only):
          none  : exactly 0
          micro : |gap| < MICRO_GAP_THRESHOLD (default 0.1 %)
          macro : |gap| > MACRO_GAP_THRESHOLD (default 1.0 %)
          minor : in between
        """
        mag = abs(gap_pct)
        if mag == 0.0:
            return "none"
        if mag < settings.MICRO_GAP_THRESHOLD:
            return "micro"
        if mag > settings.MACRO_GAP_THRESHOLD:
            return "macro"
        return "minor"

    def compute_liquidity_score(
        self, extended_volume: float, regular_volume: float
    ) -> float:
        """
        Compute the liquidity score for extended-hours sessions.

        Formula:
          LiquidityScore = Volume_extended / Volume_regular

        Interpretation:
          < LIQUIDITY_SCORE_THRESHOLD (0.15):
            → reduce indicator weights by 50 %
            → increase signal thresholds by 25 %
            → suppress weak signals

        Returns:
            float in [0, ∞)  (typically 0–1 for liquid markets)
        """
        try:
            if regular_volume <= 0:
                return 0.0
            return round(extended_volume / regular_volume, 6)
        except Exception as exc:
            logger.error("Error computing liquidity score: %s", exc)
            return 0.0

    # Cache of computed liquidity metrics keyed by rounded inputs so repeated
    # runs with the same volumes always return identical results.
    _liquidity_cache: dict[tuple[float, float], dict] = {}

    def compute_liquidity_metrics(
        self, extended_volume: float, regular_volume: float
    ) -> dict:
        """
        Deterministic liquidity metrics (additive; LiquidityScore semantics
        are unchanged and still come from compute_liquidity_score):

          liquidity_score       : Volume_extended / Volume_regular
          liquidity_class       : 'adequate' if score >= LIQUIDITY_SCORE_THRESHOLD
                                  'thin'     if 0 < score < threshold
                                  'none'     if score == 0
          liquidity_compression : min(score / threshold, 1.0) — how compressed
                                  extended-hours liquidity is vs the threshold

        Results are cached on rounded inputs so identical inputs always
        produce identical outputs within a process lifetime.
        """
        key = (round(float(extended_volume), 6), round(float(regular_volume), 6))
        cached = self._liquidity_cache.get(key)
        if cached is not None:
            return dict(cached)

        score = self.compute_liquidity_score(extended_volume, regular_volume)
        threshold = settings.LIQUIDITY_SCORE_THRESHOLD

        if score == 0.0:
            liquidity_class = "none"
        elif score < threshold:
            liquidity_class = "thin"
        else:
            liquidity_class = "adequate"

        compression = round(min(score / threshold, 1.0), 6) if threshold > 0 else 0.0

        metrics = {
            "liquidity_score": score,
            "liquidity_class": liquidity_class,
            "liquidity_compression": compression,
        }
        if len(self._liquidity_cache) > 4096:
            self._liquidity_cache.clear()
        self._liquidity_cache[key] = metrics
        return dict(metrics)

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardise yfinance DataFrame column names to lower-case.
        yfinance returns MultiIndex columns for single tickers in newer versions;
        flatten them here.
        """
        if df.empty:
            return df

        # Flatten MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() for col in df.columns]
        else:
            df.columns = [str(c).lower() for c in df.columns]

        # Ensure index is a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Normalize timezone-aware timestamps to UTC-naive so they match the DB
        # and so incremental comparisons between fetched data and last_timestamp
        # do not raise dtype mismatch errors.
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)

        # Drop rows with all-NaN OHLC
        df.dropna(subset=["open", "high", "low", "close"], inplace=True)

        # ── Timestamp sanity checks ────────────────────────────────────────────
        # Flag and drop implausible timestamps (before 2000 or in the future)
        # with structured log entries so ingestion runs are auditable.
        if not df.empty:
            try:
                now_utc = datetime.utcnow()
                bad_mask = []
                for ts in df.index:
                    issue = market_calendar.timestamp_sanity_issue(
                        ts.to_pydatetime(), now_utc=now_utc
                    )
                    bad_mask.append(issue is not None)
                    if issue is not None:
                        logger.warning(
                            "ingest_timestamp_anomaly issue=%s ts=%s",
                            issue, ts.isoformat(),
                        )
                if any(bad_mask):
                    dropped = sum(bad_mask)
                    df = df[[not b for b in bad_mask]]
                    logger.warning(
                        "ingest_timestamp_anomaly_summary dropped=%d remaining=%d",
                        dropped, len(df),
                    )
            except Exception as exc:
                # Sanity checking must never abort ingestion
                logger.error("ingest_timestamp_sanity_check_failed error=%s", exc)

        return df
