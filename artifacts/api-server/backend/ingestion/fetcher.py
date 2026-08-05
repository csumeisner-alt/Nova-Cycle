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


def ohlc_validation_issue(
    open_price: object,
    high: object,
    low: object,
    close: object,
) -> Optional[str]:
    """Return a reason when an OHLC row cannot represent a real candle.

    Vendor feeds can occasionally return partial or internally contradictory
    rows.  Letting one through is worse than dropping it: the row can become a
    feature in both models and produce a plausible-looking but wrong signal.
    """
    try:
        values = [float(open_price), float(high), float(low), float(close)]
    except (TypeError, ValueError):
        return "non_numeric_ohlc"

    if not all(pd.notna(value) for value in values):
        return "non_finite_ohlc"
    if not all(value > 0 for value in values):
        return "non_positive_ohlc"
    open_value, high_value, low_value, close_value = values
    if high_value < max(open_value, close_value):
        return "high_below_open_or_close"
    if low_value > min(open_value, close_value):
        return "low_above_open_or_close"
    return None


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

    async def fetch_live_quote(self) -> Optional[dict]:
        """Fetch the vendor's current/pre-market/after-hours quote.

        Candle history is useful for model features, but it can lag or omit
        the latest extended-hours quote.  Yahoo exposes the quote separately
        in ``Ticker.info``; choose the field for the session that is active
        now and return a small, validated snapshot for the price endpoint.
        """
        try:
            now = datetime.now(timezone.utc)
            _, session_type, _ = market_calendar.classify_session(now)

            def read_quote() -> dict:
                return dict(yf.Ticker(settings.TICKER).info)

            info = await self._run_sync(read_quote)
            field_by_session = {
                "pre_market": ("preMarketPrice", "preMarketTime"),
                "regular": ("regularMarketPrice", "regularMarketTime"),
                "after_hours": ("postMarketPrice", "postMarketTime"),
            }
            price_field, time_field = field_by_session.get(
                session_type, ("regularMarketPrice", "regularMarketTime")
            )
            price = info.get(price_field)
            quote_time = info.get(time_field)

            # If the session-specific quote is unavailable, a regular-market
            # quote is still safer than returning a malformed value.  Do not
            # use previousClose: that is explicitly a closing price, not a
            # live quote.
            if price is None:
                price = info.get("regularMarketPrice")
                quote_time = info.get("regularMarketTime")

            try:
                price = float(price)
            except (TypeError, ValueError):
                return None
            if not pd.notna(price) or price <= 0:
                return None

            timestamp = None
            try:
                if quote_time is not None:
                    timestamp = datetime.fromtimestamp(
                        float(quote_time), tz=timezone.utc
                    ).replace(tzinfo=None).isoformat()
            except (TypeError, ValueError, OSError, OverflowError):
                timestamp = None

            return {
                "price": price,
                "timestamp": timestamp,
                "session_type": session_type,
                "is_extended_hours": session_type != "regular",
                "source": "live_quote",
            }
        except Exception as exc:
            logger.warning("Live VOO quote fetch failed: %s", exc)
            return None

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
            # ^VIX is an index; Yahoo reports valid index candles with volume 0.
            # VIX is an index and legitimate regime jumps are common. Keep
            # intra-bar OHLC validation, but skip the equity-feed cross-bar
            # spike heuristic so real VIX observations are not quarantined.
            df = self._normalise_columns(
                df, drop_zero_volume=False, spike_threshold=0.0
            )
            logger.info("Fetched %d daily VIX candles", len(df))
            return df
        except Exception as exc:
            logger.error("Error fetching VIX data: %s", exc)
            return pd.DataFrame()

    async def fetch_historical_spx(self, years: int = 10) -> pd.DataFrame:
        """
        Fetch `years` years of daily SPX futures (ES=F) data.

        Returns:
            pd.DataFrame with columns: open, high, low, close, volume
            (empty on error — callers preserve fallback behavior).
        """
        spx_ticker = settings.SPX_FUTURES_TICKER
        logger.info("Fetching %d years of historical SPX futures data…", years)
        try:
            df = await self._run_sync(
                yf.download,
                spx_ticker,
                period=f"{years}y",
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            df = self._normalise_columns(df)
            logger.info("Fetched %d daily SPX futures candles", len(df))
            return df
        except Exception as exc:
            logger.error("Error fetching SPX futures data: %s", exc)
            return pd.DataFrame()

    async def fetch_historical_context_ticker(
        self,
        ticker: str,
        years: int = 10,
        *,
        is_index: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch `years` years of daily data for a broader-context ticker.

        Used for VIX9D, VIX3M, TNX, HYG, LQD, and NYAD ingestion.

        Args:
            ticker:   yfinance symbol (e.g. "^VIX9D", "HYG").
            years:    Number of calendar years of history to fetch.
            is_index: When True, zero-volume bars are accepted (indices such as
                      ^VIX9D and ^NYAD always report volume 0 from Yahoo).
                      When False, zero-volume bars are filtered out.

        Returns:
            pd.DataFrame (may be empty on error).
        """
        logger.info("Fetching %d years of daily context data: %s", years, ticker)
        try:
            df = await self._run_sync(
                yf.download,
                ticker,
                period=f"{years}y",
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            # Indices always report zero-volume; ETFs (HYG, LQD) have real
            # volume but may occasionally emit glitch zero-vol bars.
            df = self._normalise_columns(
                df,
                drop_zero_volume=not is_index,
                spike_threshold=0.0,  # disable cross-bar spike filter for context tickers
            )
            logger.info("Fetched %d daily candles for %s", len(df), ticker)
            return df
        except Exception as exc:
            logger.error("Error fetching context data for %s: %s", ticker, exc)
            return pd.DataFrame()

    async def fetch_context_ticker_range(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        *,
        is_index: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch daily candles for a broader-context ticker over a specific
        [start, end] date range.  Used for targeted backfill of missing
        context-feed trading days (VIX9D, VIX3M, TNX, HYG, LQD, NYAD).

        yfinance's ``end`` is exclusive, so one day is added to include it.

        Args:
            ticker:   yfinance symbol (e.g. "^VIX9D", "HYG").
            start:    First day of the desired range (inclusive).
            end:      Last day of the desired range (inclusive).
            is_index: When True, zero-volume bars are accepted.

        Returns:
            pd.DataFrame (may be empty on error).
        """
        start_str = start.strftime("%Y-%m-%d")
        end_str = (end + timedelta(days=1)).strftime("%Y-%m-%d")
        logger.info(
            "Backfill fetch: %s daily %s → %s", ticker, start_str, end_str
        )
        try:
            df = await self._run_sync(
                yf.download,
                ticker,
                start=start_str,
                end=end_str,
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            df = self._normalise_columns(
                df,
                drop_zero_volume=not is_index,
                spike_threshold=0.0,
            )
            logger.info(
                "Backfill fetch: %d daily candles for %s", len(df), ticker
            )
            return df
        except Exception as exc:
            logger.error(
                "Error in backfill context fetch %s %s→%s: %s",
                ticker, start_str, end_str, exc,
            )
            return pd.DataFrame()

    async def fetch_vix_daily_range(self, start: datetime, end: datetime) -> pd.DataFrame:
        """
        Fetch daily VIX candles for a specific [start, end] date range.
        Used for targeted backfill of missing VIX trading days.

        yfinance's `end` is exclusive, so one day is added to include it.

        Returns:
            pd.DataFrame (may be empty on error).
        """
        vix_ticker = settings.VIX_TICKER
        start_str = start.strftime("%Y-%m-%d")
        end_str = (end + timedelta(days=1)).strftime("%Y-%m-%d")
        logger.info("Backfill fetch: VIX daily %s → %s", start_str, end_str)
        try:
            df = await self._run_sync(
                yf.download,
                vix_ticker,
                start=start_str,
                end=end_str,
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            # ^VIX is an index; Yahoo reports valid index candles with volume 0.
            df = self._normalise_columns(
                df, drop_zero_volume=False, spike_threshold=0.0
            )
            logger.info("Backfill fetch: %d daily VIX candles", len(df))
            return df
        except Exception as exc:
            logger.error("Error in backfill VIX fetch %s→%s: %s", start_str, end_str, exc)
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
                # Keep the row equal to last_timestamp.  Storage skips an
                # unchanged valid row, but can replace an older malformed row
                # when Yahoo returns the corrected candle on a later run.
                daily_df = daily_df[daily_df.index >= pd.Timestamp(last_timestamp)]
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
                # As with daily candles, re-read the boundary bar so a
                # previously malformed vendor row can be repaired.
                fivemin_df = fivemin_df[fivemin_df.index >= pd.Timestamp(last_timestamp)]
            result["5min"] = fivemin_df
            logger.info("Incremental: %d new 5-min candles", len(fivemin_df))
        except Exception as exc:
            logger.error("Error in incremental 5-min fetch: %s", exc)

        return result

    async def fetch_daily_range(self, start: datetime, end: datetime) -> pd.DataFrame:
        """
        Fetch daily VOO candles for a specific [start, end] date range.
        Used for targeted backfill of missing trading days.

        yfinance's `end` is exclusive, so one day is added to include it.

        Returns:
            pd.DataFrame (may be empty on error) with is_extended_hours /
            session_type columns matching the daily ingestion path.
        """
        ticker = settings.TICKER
        start_str = start.strftime("%Y-%m-%d")
        end_str = (end + timedelta(days=1)).strftime("%Y-%m-%d")
        logger.info("Backfill fetch: VOO daily %s → %s", start_str, end_str)
        try:
            df = await self._run_sync(
                yf.download,
                ticker,
                start=start_str,
                end=end_str,
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            df = self._normalise_columns(df)
            if not df.empty:
                df["is_extended_hours"] = False
                df["session_type"] = "regular"
            logger.info("Backfill fetch: %d daily candles", len(df))
            return df
        except Exception as exc:
            logger.error("Error in backfill daily fetch %s→%s: %s", start_str, end_str, exc)
            return pd.DataFrame()

    async def fetch_5min_range(self, start: datetime, end: datetime) -> pd.DataFrame:
        """
        Fetch 5-minute VOO candles (prepost=True) for a specific [start, end]
        date range. Used for targeted backfill of missing intraday sessions
        inside yfinance's 60-day 5-min window.

        yfinance's `end` is exclusive, so one day is added to include it.

        Returns:
            pd.DataFrame (may be empty on error) with is_extended_hours /
            session_type columns matching the regular 5-min ingestion path.
        """
        ticker = settings.TICKER
        start_str = start.strftime("%Y-%m-%d")
        end_str = (end + timedelta(days=1)).strftime("%Y-%m-%d")
        logger.info("Backfill fetch: VOO 5min %s → %s", start_str, end_str)
        try:
            df = await self._run_sync(
                yf.download,
                ticker,
                start=start_str,
                end=end_str,
                interval="5m",
                prepost=True,
                auto_adjust=True,
                progress=False,
            )
            df = self._normalise_columns(df)
            if not df.empty:
                sessions = [self._classify_session(ts) for ts in df.index]
                df["is_extended_hours"] = [s[0] for s in sessions]
                df["session_type"] = [s[1] for s in sessions]
            logger.info("Backfill fetch: %d 5-min candles", len(df))
            return df
        except Exception as exc:
            logger.error("Error in backfill 5-min fetch %s→%s: %s", start_str, end_str, exc)
            return pd.DataFrame()

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
        self,
        prev_close: float,
        premarket_open: float,
        post_open_candles: Optional[pd.DataFrame] = None,
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
          gap_momentum: real follow-through measurement (additive).
            When `post_open_candles` (the day's regular-session 5-min
            candles, oldest first) contains at least
            GAP_MOMENTUM_CANDLES rows and the gap is non-zero, this is
            computed by compute_gap_momentum(); otherwise it is None.
            Always present in the dict.

        Returns:
            {"gap_percent": float, "gap_type": str,
             "gap_class": str, "gap_momentum": Optional[float]}
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
                "gap_momentum": self.compute_gap_momentum(gap_pct, post_open_candles),
            }
        except Exception as exc:
            logger.error("Error detecting gap: %s", exc)
            return {
                "gap_percent": 0.0,
                "gap_type": "none",
                "gap_class": "none",
                "gap_momentum": None,
            }

    # Number of post-open regular-session 5-min candles required to measure
    # gap follow-through (6 × 5 min = first 30 minutes of the regular session).
    GAP_MOMENTUM_CANDLES: int = 6

    @classmethod
    def compute_gap_momentum(
        cls, gap_pct: float, post_open_candles: Optional[pd.DataFrame]
    ) -> Optional[float]:
        """
        Compute overnight-gap follow-through momentum (additive metric).

        Formula:
          Momentum = sign(GapPercent) ×
                     (Close_N − Open_1) / Open_1 × 100

          where Open_1 is the open of the first regular-session 5-min candle
          and Close_N is the close of the GAP_MOMENTUM_CANDLES-th candle
          (i.e. price movement over the first 30 minutes after the open,
          signed relative to the gap direction).

        Interpretation:
          > 0 → price continued in the gap's direction (follow-through)
          < 0 → price moved against the gap (fade)

        Returns None (never raises) when:
          - gap_pct is 0 (no gap to follow through on)
          - post_open_candles is None/empty or has fewer than
            GAP_MOMENTUM_CANDLES rows
          - the first candle's open is 0 or data is malformed
        """
        try:
            if not gap_pct:
                return None
            if post_open_candles is None or len(post_open_candles) < cls.GAP_MOMENTUM_CANDLES:
                return None
            candles = post_open_candles.sort_index()
            open_1 = float(candles.iloc[0]["open"])
            close_n = float(candles.iloc[cls.GAP_MOMENTUM_CANDLES - 1]["close"])
            if open_1 == 0 or pd.isna(open_1) or pd.isna(close_n):
                return None
            direction = 1.0 if gap_pct > 0 else -1.0
            return round(direction * (close_n - open_1) / open_1 * 100.0, 4)
        except Exception as exc:
            logger.error("Error computing gap momentum: %s", exc)
            return None

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
    def _normalise_columns(
        df: pd.DataFrame,
        *,
        drop_zero_volume: bool = True,
        spike_threshold: Optional[float] = None,
    ) -> pd.DataFrame:
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

        # ── OHLC consistency + cross-bar spike checks ─────────────────────────
        # yfinance has returned contradictory daily rows in the past (for
        # example, a high below the open).  Drop those rows before they reach
        # the database or model.  The incremental path deliberately re-fetches
        # the latest timestamp so a later corrected vendor response can repair
        # an already-stored bad row.
        #
        # A second cross-bar pass flags bars whose close deviates more than
        # SPIKE_CLOSE_THRESHOLD (default 10 %) from the rolling median of
        # their neighbours — these are internally self-consistent but are
        # probable data glitches that would skew RSI / Bollinger features.
        if not df.empty:
            try:
                from ingestion.ohlc_validator import filter_valid_ohlc
                df_valid, df_quarantined = filter_valid_ohlc(
                    df, spike_threshold=spike_threshold
                )
                if not df_quarantined.empty:
                    for ts, row in df_quarantined.iterrows():
                        logger.warning(
                            "ingest_ohlc_anomaly issue=%s ts=%s",
                            row.get("ohlc_invalid_reason", "unknown"),
                            pd.Timestamp(ts).isoformat(),
                        )
                    logger.warning(
                        "ingest_ohlc_anomaly_summary dropped=%d remaining=%d",
                        len(df_quarantined),
                        len(df_valid),
                    )
                    df = df_valid
            except Exception as exc:
                # A validation implementation error must never turn a feed
                # outage into an unhandled ingestion crash.
                logger.error("ingest_ohlc_sanity_check_failed error=%s", exc)

        # ── Zero-volume bar filter ─────────────────────────────────────────────
        # yfinance occasionally emits zero-volume daily bars for glitch days.
        # Drop them before DB storage; the startup cleanup also removes any
        # that already exist (see IngestionPipeline.remove_invalid_voo_candles).
        if drop_zero_volume and not df.empty:
            try:
                from ingestion.ohlc_validator import filter_zero_volume_bars
                df_valid, df_zero_vol = filter_zero_volume_bars(df)
                if not df_zero_vol.empty:
                    for ts, row in df_zero_vol.iterrows():
                        logger.warning(
                            "ingest_zero_volume_bar_dropped ts=%s",
                            pd.Timestamp(ts).isoformat(),
                        )
                    logger.warning(
                        "ingest_zero_volume_bars_dropped count=%d remaining=%d",
                        len(df_zero_vol),
                        len(df_valid),
                    )
                    df = df_valid
            except Exception as exc:
                logger.error("ingest_zero_volume_check_failed error=%s", exc)

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
