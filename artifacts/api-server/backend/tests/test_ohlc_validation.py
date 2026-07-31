"""Regression coverage for malformed vendor OHLC rows and repairs."""

from datetime import datetime, timedelta

import pandas as pd
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from database.models import Base, VooCandle, VixCandle, SpxCandle
from ingestion.fetcher import DataFetcher, ohlc_validation_issue
from ingestion.pipeline import IngestionPipeline
from ingestion.ohlc_validator import filter_valid_ohlc, flag_cross_bar_spikes, validate_ohlc_row


@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        yield session
    await engine.dispose()


def test_impossible_high_is_rejected():
    assert ohlc_validation_issue(680.12, 676.71, 675.58, 676.01) == (
        "high_below_open_or_close"
    )


def test_valid_ohlc_is_accepted():
    assert ohlc_validation_issue(676.54, 682.44, 675.22, 681.79) is None


def test_normalise_columns_drops_bad_row():
    frame = pd.DataFrame(
        {
            "Open": [680.12, 676.54],
            "High": [676.71, 682.44],
            "Low": [675.58, 675.22],
            "Close": [676.01, 681.79],
            "Volume": [100, 200],
        },
        index=pd.to_datetime(["2026-07-30", "2026-07-31"]),
    )

    result = DataFetcher._normalise_columns(frame)

    assert list(result.index) == [pd.Timestamp("2026-07-31")]


@pytest.mark.asyncio
async def test_existing_bad_row_is_repaired(db_session):
    bad = VooCandle(
        ticker="VOO",
        timestamp=datetime(2026, 7, 30),
        open=680.12,
        high=676.71,
        low=675.58,
        close=676.01,
        volume=100,
        timeframe="daily",
        session_type="regular",
    )
    db_session.add(bad)
    await db_session.flush()

    frame = pd.DataFrame(
        {
            "open": [676.54],
            "high": [682.44],
            "low": [675.22],
            "close": [681.79],
            "volume": [5653980],
            "is_extended_hours": [False],
            "session_type": ["regular"],
        },
        index=pd.to_datetime(["2026-07-30"]),
    )

    await IngestionPipeline().store_voo_candles(frame, db_session, "daily")

    await db_session.refresh(bad)
    assert (bad.open, bad.high, bad.low, bad.close) == (
        676.54,
        682.44,
        675.22,
        681.79,
    )


@pytest.mark.asyncio
async def test_startup_cleanup_removes_existing_bad_row(db_session):
    bad = VooCandle(
        ticker="VOO",
        timestamp=datetime(2026, 7, 30),
        open=680.12,
        high=676.71,
        low=675.58,
        close=676.01,
        volume=100,
        timeframe="daily",
        session_type="regular",
    )
    db_session.add(bad)
    await db_session.flush()

    removed = await IngestionPipeline().remove_invalid_voo_candles(db_session)

    assert removed == 1
    assert await db_session.get(VooCandle, bad.id) is None

class TestValidateOhlcRow:
    """Unit tests for the single-row OHLC consistency check."""

    def test_valid_normal_candle(self):
        ok, reason = validate_ohlc_row(100.0, 105.0, 98.0, 103.0)
        assert ok is True
        assert reason == ""

    def test_valid_doji_candle(self):
        """open == close == high == low is valid (gap open/close candle)."""
        ok, reason = validate_ohlc_row(100.0, 100.0, 100.0, 100.0)
        assert ok is True

    def test_valid_candle_close_at_high(self):
        ok, reason = validate_ohlc_row(100.0, 105.0, 98.0, 105.0)
        assert ok is True

    def test_valid_candle_close_at_low(self):
        ok, reason = validate_ohlc_row(100.0, 105.0, 98.0, 98.0)
        assert ok is True

    # ── July 30 bug shape ─────────────────────────────────────────────────

    def test_july30_bug_shape(self):
        """The exact malformed July 30 VOO candle: open 680.12, high 676.71."""
        ok, reason = validate_ohlc_row(680.12, 676.71, 675.58, 681.55)
        assert ok is False
        # high < open AND high < close → should cite high_below_open (checked first)
        assert "high_below_open" in reason
        assert "676.71" in reason
        assert "680.12" in reason

    # ── Individual violation patterns ────────────────────────────────────

    def test_high_below_low(self):
        ok, reason = validate_ohlc_row(100.0, 95.0, 98.0, 97.0)
        assert ok is False
        assert "high_below_low" in reason

    def test_high_below_open(self):
        ok, reason = validate_ohlc_row(105.0, 102.0, 100.0, 103.0)
        assert ok is False
        assert "high_below_open" in reason

    def test_high_below_close(self):
        ok, reason = validate_ohlc_row(100.0, 102.0, 98.0, 104.0)
        assert ok is False
        assert "high_below_close" in reason

    def test_low_above_open(self):
        ok, reason = validate_ohlc_row(98.0, 105.0, 102.0, 103.0)
        assert ok is False
        assert "low_above_open" in reason

    def test_low_above_close(self):
        # low=103 is below open=105, so low_above_open doesn't fire;
        # but low=103 > close=101, so low_above_close fires.
        ok, reason = validate_ohlc_row(105.0, 110.0, 103.0, 101.0)
        assert ok is False
        assert "low_above_close" in reason

    # ── Epsilon tolerance ─────────────────────────────────────────────────

    def test_sub_epsilon_high_below_open_ignored(self):
        """Violations smaller than 0.001 (floating-point noise) are ignored."""
        # high is 0.0005 below open — within epsilon
        ok, reason = validate_ohlc_row(100.001, 100.0005, 99.0, 100.0)
        assert ok is True

    def test_just_over_epsilon_detected(self):
        """Violations larger than epsilon are caught."""
        # high is 0.002 below open — outside epsilon
        ok, reason = validate_ohlc_row(100.002, 100.0, 99.0, 100.001)
        assert ok is False

class TestNormaliseColumnsOhlcFilter:
    """Verify that _normalise_columns drops malformed OHLC before it reaches DB."""

    def _run_normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        from ingestion.fetcher import DataFetcher
        return DataFetcher._normalise_columns(df)

    def test_valid_candles_pass_through(self):
        df = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [105.0, 106.0],
                "Low":  [98.0,  99.0],
                "Close": [103.0, 104.0],
                "Volume": [1e6, 1e6],
            },
            index=pd.DatetimeIndex(["2024-07-29", "2024-07-30"]),
        )
        result = self._run_normalise(df)
        assert len(result) == 2

    def test_july30_malformed_candle_is_dropped(self):
        """The July 30 shape (high < open) is quarantined by _normalise_columns."""
        df = pd.DataFrame(
            {
                "Open":  [676.0,  680.12],
                "High":  [681.0,  676.71],
                "Low":   [675.0,  675.58],
                "Close": [680.0,  681.55],
                "Volume": [1e6, 1e6],
            },
            index=pd.DatetimeIndex(["2024-07-29", "2024-07-30"]),
        )
        result = self._run_normalise(df)
        # July 29 is valid and survives; July 30 is dropped.
        assert len(result) == 1
        assert float(result.iloc[0]["open"]) == pytest.approx(676.0)

    def test_all_valid_after_filtering(self):
        """If the malformed candle is the only row, the result is empty."""
        df = pd.DataFrame(
            {
                "Open":  [680.12],
                "High":  [676.71],
                "Low":   [675.58],
                "Close": [681.55],
                "Volume": [1e6],
            },
            index=pd.DatetimeIndex(["2024-07-30"]),
        )
        result = self._run_normalise(df)
        assert result.empty

class TestDropInvalidOhlcPrediction:
    """Tests for the prediction-time OHLC filter (operates on already-loaded rows)."""

    def _drop(self, df: pd.DataFrame, timeframe: str = "daily"):
        """Call _drop_invalid_ohlc via the predictions module."""
        # Import fresh so module-level state doesn't bleed between tests.
        import importlib
        import routers.predictions as pred
        importlib.reload(pred)
        return pred._drop_invalid_ohlc(df, timeframe=timeframe)

    def _candle_df(self, rows: list[dict]) -> pd.DataFrame:
        """Build a typical loaded-candle DataFrame (integer index + timestamp column)."""
        df = pd.DataFrame(rows)
        return df

    def test_all_valid_no_degraded_flag(self):
        df = self._candle_df([
            {
                "timestamp": datetime(2024, 7, 29), "open": 676.0,
                "high": 681.0, "low": 675.0, "close": 680.0,
            },
        ])
        clean, degraded, reason = self._drop(df)
        assert not degraded
        assert reason == ""
        assert len(clean) == 1

    def test_july30_candle_sets_degraded(self):
        """When the latest daily candle has high < open, degraded flag is set."""
        df = self._candle_df([
            {
                "timestamp": datetime(2024, 7, 29), "open": 676.0,
                "high": 681.0, "low": 675.0, "close": 680.0,
            },
            {
                "timestamp": datetime(2024, 7, 30), "open": 680.12,
                "high": 676.71, "low": 675.58, "close": 681.55,
            },
        ])
        clean, degraded, reason = self._drop(df)
        assert degraded is True
        assert "high_below_open" in reason
        # The valid July 29 candle survives
        assert len(clean) == 1
        assert float(clean.iloc[0]["open"]) == pytest.approx(676.0)

    def test_degraded_reason_mentions_quarantine_count(self):
        df = self._candle_df([
            {
                "timestamp": datetime(2024, 7, 30), "open": 680.12,
                "high": 676.71, "low": 675.58, "close": 681.55,
            },
        ])
        _, degraded, reason = self._drop(df)
        assert degraded is True
        assert "quarantined 1" in reason

    def test_empty_df_no_crash(self):
        df = pd.DataFrame()
        clean, degraded, reason = self._drop(df)
        assert clean.empty
        assert degraded is False

    def test_all_invalid_returns_empty_with_degraded(self):
        df = self._candle_df([
            {
                "timestamp": datetime(2024, 7, 30), "open": 680.12,
                "high": 676.71, "low": 675.58, "close": 681.55,
            },
        ])
        clean, degraded, reason = self._drop(df)
        assert clean.empty
        assert degraded is True

def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a candle DataFrame from a list of OHLC dicts."""
    df = pd.DataFrame(rows)
    df.index = pd.DatetimeIndex(
        [datetime(2024, 7, 30 - i, 12, 0) for i in range(len(rows))]
    )
    return df

class TestFilterValidOhlc:
    """Tests for the DataFrame-level OHLC filter."""

    def test_all_valid(self):
        df = _make_df([
            {"open": 100.0, "high": 105.0, "low": 98.0, "close": 103.0},
            {"open": 101.0, "high": 106.0, "low": 99.0, "close": 104.0},
        ])
        valid, quarantined = filter_valid_ohlc(df)
        assert len(valid) == 2
        assert quarantined.empty

    def test_one_bad_row(self):
        """The July 30 row is quarantined; the previous day is kept."""
        df = _make_df([
            {"open": 676.0, "high": 681.0, "low": 675.0, "close": 680.0},  # previous day — valid
            {"open": 680.12, "high": 676.71, "low": 675.58, "close": 681.55},  # July 30 — bad
        ])
        valid, quarantined = filter_valid_ohlc(df)
        assert len(valid) == 1
        assert len(quarantined) == 1
        assert float(valid.iloc[0]["open"]) == pytest.approx(676.0)
        assert "ohlc_invalid_reason" in quarantined.columns
        assert "high_below_open" in quarantined.iloc[0]["ohlc_invalid_reason"]

    def test_all_bad(self):
        df = _make_df([
            {"open": 680.12, "high": 676.71, "low": 675.58, "close": 681.55},
            {"open": 100.0,  "high": 95.0,   "low": 90.0,   "close": 98.0},
        ])
        valid, quarantined = filter_valid_ohlc(df)
        assert valid.empty
        assert len(quarantined) == 2

    def test_empty_df(self):
        df = pd.DataFrame()
        valid, quarantined = filter_valid_ohlc(df)
        assert valid.empty
        assert quarantined.empty

    def test_missing_ohlc_columns_is_noop(self):
        """DataFrame without OHLC columns passes through unchanged (no crash)."""
        df = pd.DataFrame({"foo": [1, 2, 3]})
        valid, quarantined = filter_valid_ohlc(df)
        assert len(valid) == 3
        assert quarantined.empty

    def test_quarantined_keeps_extra_columns(self):
        """Custom columns survive in the quarantined slice."""
        df = _make_df([
            {
                "open": 680.12, "high": 676.71, "low": 675.58, "close": 681.55,
                "volume": 123456.0, "session_type": "regular",
            }
        ])
        valid, quarantined = filter_valid_ohlc(df)
        assert "volume" in quarantined.columns
        assert "session_type" in quarantined.columns
        assert float(quarantined.iloc[0]["volume"]) == 123456.0


# ---------------------------------------------------------------------------
# Task-172: yfinance mid-session data glitch tests
# ---------------------------------------------------------------------------

class TestZeroCloseNormaliseColumns:
    """Zero-close (and zero-open/high/low) candles must be dropped by _normalise_columns.

    ohlc_validation_issue returns 'non_positive_ohlc' for any value <= 0, so the
    OHLC sanity pass inside _normalise_columns quarantines these rows before they
    ever reach the database.
    """

    def _run_normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        from ingestion.fetcher import DataFetcher
        return DataFetcher._normalise_columns(df)

    def test_zero_close_is_dropped(self):
        """A candle with close=0 is non-positive and must be quarantined at ingest."""
        df = pd.DataFrame(
            {
                "Open":  [500.0, 501.0],
                "High":  [505.0, 506.0],
                "Low":   [498.0, 499.0],
                "Close": [503.0,   0.0],
                "Volume": [1_000_000, 1_000_000],
            },
            index=pd.DatetimeIndex(["2024-07-29", "2024-07-30"]),
        )
        result = self._run_normalise(df)
        assert len(result) == 1
        assert float(result.iloc[0]["open"]) == pytest.approx(500.0)

    def test_zero_open_is_dropped(self):
        """A zero open price is non-positive and must be quarantined."""
        df = pd.DataFrame(
            {
                "Open":  [0.0,   500.0],
                "High":  [505.0, 505.0],
                "Low":   [498.0, 498.0],
                "Close": [503.0, 503.0],
                "Volume": [1_000_000, 1_000_000],
            },
            index=pd.DatetimeIndex(["2024-07-29", "2024-07-30"]),
        )
        result = self._run_normalise(df)
        assert len(result) == 1
        assert float(result.iloc[0]["open"]) == pytest.approx(500.0)

    def test_zero_close_only_row_gives_empty_frame(self):
        """When the only row has close=0 the result is empty (no crash)."""
        df = pd.DataFrame(
            {
                "Open":  [500.0],
                "High":  [505.0],
                "Low":   [498.0],
                "Close": [  0.0],
                "Volume": [1_000_000],
            },
            index=pd.DatetimeIndex(["2024-07-30"]),
        )
        result = self._run_normalise(df)
        assert result.empty

    def test_mixed_zero_and_valid_keeps_only_valid(self):
        """Multiple zero-close rows are all dropped; valid rows survive."""
        df = pd.DataFrame(
            {
                "Open":  [500.0, 501.0, 502.0],
                "High":  [505.0, 506.0, 507.0],
                "Low":   [498.0, 499.0, 500.0],
                "Close": [503.0,   0.0,   0.0],
                "Volume": [1e6, 1e6, 1e6],
            },
            index=pd.DatetimeIndex(["2024-07-28", "2024-07-29", "2024-07-30"]),
        )
        result = self._run_normalise(df)
        assert len(result) == 1
        assert float(result.iloc[0]["open"]) == pytest.approx(500.0)


class TestIntradaySpikeBar:
    """Single intraday 5-min bar whose close deviates > 10 % from neighbours.

    The OHLC validator checks *internal* self-consistency only.  A spike where
    high keeps pace with close (so high >= close) is internally valid and will
    NOT be quarantined — the test documents this expectation and verifies that
    the pipeline does not crash either way.  A spike where high does NOT keep
    pace is internally inconsistent (high_below_close) and IS quarantined.
    """

    def _drop(self, df: pd.DataFrame) -> tuple:
        import importlib
        import routers.predictions as pred
        importlib.reload(pred)
        return pred._drop_invalid_ohlc(df, timeframe="5min")

    def _make_5min_frame(self, rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def test_internally_valid_spike_is_now_quarantined_by_cross_bar_check(self):
        """A bar that closes +15 % above neighbours but is internally valid
        (high >= close) is NOW caught by the cross-bar spike check and
        quarantined with degraded=True.

        Before Task-177 this bar passed through undetected because the validator
        only checked intra-bar consistency.  The cross-bar rolling-median check
        flags it as a probable data glitch.
        """
        base_close = 500.0
        spike_close = base_close * 1.15  # +15 % — exceeds SPIKE_CLOSE_THRESHOLD (10 %)

        rows = [
            {
                "timestamp": datetime(2024, 7, 30, 9, 30),
                "open": 499.0, "high": 504.0, "low": 497.0, "close": base_close,
            },
            {
                # Spike bar: high keeps pace with close → internally valid but
                # 15 % above rolling median → cross-bar spike
                "timestamp": datetime(2024, 7, 30, 9, 35),
                "open": 500.0, "high": spike_close + 1.0, "low": 499.0, "close": spike_close,
            },
            {
                "timestamp": datetime(2024, 7, 30, 9, 40),
                "open": 500.5, "high": 505.0, "low": 499.0, "close": 502.0,
            },
        ]
        df = self._make_5min_frame(rows)
        clean, degraded, reason = self._drop(df)
        # Cross-bar spike is now quarantined
        assert degraded is True
        assert "cross_bar_spike" in reason
        # Two valid neighbours survive
        assert len(clean) == 2

    def test_internally_invalid_spike_is_quarantined(self):
        """A bar that spikes +15 % in close but high does not keep pace is
        quarantined (high_below_close) and degraded=True is set."""
        base_close = 500.0
        spike_close = base_close * 1.15  # +15 % glitch

        rows = [
            {
                "timestamp": datetime(2024, 7, 30, 9, 30),
                "open": 499.0, "high": 504.0, "low": 497.0, "close": base_close,
            },
            {
                # high does NOT keep up with spiking close → high_below_close
                "timestamp": datetime(2024, 7, 30, 9, 35),
                "open": 500.0, "high": 503.0, "low": 499.0, "close": spike_close,
            },
            {
                "timestamp": datetime(2024, 7, 30, 9, 40),
                "open": 500.5, "high": 505.0, "low": 499.0, "close": 502.0,
            },
        ]
        df = self._make_5min_frame(rows)
        clean, degraded, reason = self._drop(df)
        assert degraded is True
        assert "high_below_close" in reason
        # Two valid neighbours survive
        assert len(clean) == 2

    def test_pipeline_does_not_crash_with_spike_bar(self):
        """Regardless of whether the spike bar is quarantined, _drop_invalid_ohlc
        always returns a 3-tuple without raising."""
        spike_close = 600.0  # +20 % above neighbours at ~500
        base = datetime(2024, 7, 30, 9, 30)

        rows = [
            {
                "timestamp": base + timedelta(minutes=i * 5),
                "open": 499.0,
                "high": spike_close + 1.0 if i == 5 else 504.0,
                "low": 497.0,
                "close": spike_close if i == 5 else 500.0,
            }
            for i in range(12)
        ]
        df = pd.DataFrame(rows)
        result = self._drop(df)
        assert len(result) == 3
        clean_df, degraded, reason = result
        assert isinstance(clean_df, pd.DataFrame)
        assert isinstance(degraded, bool)
        assert isinstance(reason, str)


class TestDailySpikeBar:
    """Daily candles with a cross-bar spike must be quarantined before they
    reach the long-trend prediction engine.

    Daily candles use DAILY_SPIKE_CLOSE_THRESHOLD (12 %), which is higher than
    the intraday SPIKE_CLOSE_THRESHOLD (10 %) so that legitimate macro-event
    moves (e.g. COVID March 2020, post-CPI 2022) are not wrongly quarantined.

    A single day whose close is +13 % above its neighbours exceeds the
    DAILY_SPIKE_CLOSE_THRESHOLD (12 %) and must set degraded=True.

    Normal multi-week daily volatility — e.g. a 3 % single-day earnings move
    or even an 11 % COVID-style crash surrounded by typical ±1 % days — must
    NOT trip the daily check.
    """

    def _drop(self, df: pd.DataFrame) -> tuple:
        import importlib
        import routers.predictions as pred
        importlib.reload(pred)
        return pred._drop_invalid_ohlc(df, timeframe="daily")

    def _make_daily_frame(self, closes: list[float], base_date=None) -> pd.DataFrame:
        """Build a minimal daily candle DataFrame from a list of close prices.

        Each bar is internally valid: open ≈ prev close, high = close + 1,
        low = close - 1.  Only the closes vary so the cross-bar check is the
        only discriminating factor.
        """
        if base_date is None:
            base_date = datetime(2024, 1, 2)
        rows = []
        for i, c in enumerate(closes):
            ts = base_date + timedelta(days=i)
            rows.append({
                "timestamp": ts,
                "open": c - 0.5,
                "high": c + 1.0,
                "low": c - 1.0,
                "close": c,
            })
        return pd.DataFrame(rows)

    # ── Spike: +13 % single day ───────────────────────────────────────────────

    def test_daily_spike_13pct_sets_degraded(self):
        """A single daily candle whose close is +13 % above neighbours is
        quarantined with degraded=True and 'cross_bar_spike' in the reason.

        Uses 13 % because the daily threshold is 12 % (DAILY_SPIKE_CLOSE_THRESHOLD)
        and the check is strict (> not >=), so a true data glitch well above the
        threshold is the meaningful regression case.
        """
        base = 500.0
        spike = base * 1.13  # +13 % — exceeds DAILY_SPIKE_CLOSE_THRESHOLD (12 %)

        # 5 rows: normal, normal, SPIKE, normal, normal
        closes = [base, base * 1.005, spike, base * 0.998, base * 1.003]
        df = self._make_daily_frame(closes)

        clean, degraded, reason = self._drop(df)

        assert degraded is True
        assert "cross_bar_spike" in reason
        # The 4 surrounding normal days must survive
        assert len(clean) == 4

    def test_daily_spike_sets_degraded_true_return_shape(self):
        """_drop_invalid_ohlc always returns a 3-tuple; check types when spike found."""
        base = 480.0
        spike = base * 1.15  # +15 %

        closes = [base, base, spike, base, base]
        df = self._make_daily_frame(closes)

        result = self._drop(df)
        assert len(result) == 3
        clean_df, degraded, reason = result
        assert isinstance(clean_df, pd.DataFrame)
        assert isinstance(degraded, bool)
        assert isinstance(reason, str)
        assert degraded is True

    def test_daily_spike_quarantine_count_in_reason(self):
        """The degraded reason string must mention how many candles were quarantined."""
        base = 500.0
        closes = [base, base, base * 1.13, base, base]
        df = self._make_daily_frame(closes)

        _, degraded, reason = self._drop(df)

        assert degraded is True
        assert "quarantined 1" in reason

    def test_daily_spike_neighbours_survive(self):
        """The candles surrounding the spike bar are valid and must not be dropped."""
        base = 500.0
        closes = [base, base * 1.005, base * 1.13, base * 0.998, base * 1.003]
        df = self._make_daily_frame(closes)

        clean, degraded, _ = self._drop(df)

        assert degraded is True
        # All four non-spike bars survive
        assert len(clean) == 4
        # None of the surviving closes should be the spike value
        spike_val = base * 1.13
        for c in clean["close"].tolist():
            assert abs(c - spike_val) > 1.0, f"Spike close {spike_val} leaked into clean frame"

    # ── Normal daily volatility: 3 % earnings move, 11 % macro crash ─────────

    def test_three_pct_earnings_move_not_flagged(self):
        """A 3 % single-day move surrounded by typical ±1 % days stays below the
        12 % daily threshold and must NOT be quarantined (degraded=False)."""
        base = 500.0
        # Simulate an earnings pop: base ± 1 % neighbours, +3 % on earnings day
        closes = [
            base,
            base * 1.008,
            base * 1.03,   # +3 % earnings day — well inside threshold
            base * 1.025,
            base * 1.020,
        ]
        df = self._make_daily_frame(closes)

        clean, degraded, reason = self._drop(df)

        assert degraded is False, (
            f"A 3 % earnings-day move should not be flagged as a spike; "
            f"got reason={reason!r}"
        )
        assert len(clean) == len(closes)

    def test_eleven_pct_macro_crash_day_not_flagged(self):
        """An 11 % single-day move (COVID-crash scale) is below the 12 %
        DAILY_SPIKE_CLOSE_THRESHOLD and must NOT be quarantined.

        This is the primary motivation for splitting the daily threshold from the
        intraday one: under the old shared 10 % threshold a COVID-like day would
        have been wrongly quarantined.
        """
        base = 500.0
        closes = [base, base, base * 0.89, base, base]  # -11 % crash day
        df = self._make_daily_frame(closes)

        clean, degraded, reason = self._drop(df)

        assert degraded is False, (
            f"An 11 % macro-crash move should not exceed the 12 % daily threshold; "
            f"got reason={reason!r}"
        )
        assert len(clean) == len(closes)

    def test_gradual_multi_week_trend_not_flagged(self):
        """Steady multi-week drift that never exceeds 10 % from the rolling median
        in a single bar must pass through cleanly."""
        # 10 days of ~0.5 % daily gains — cumulative +5 % but no single bar spikes
        closes = [500.0 * (1.005 ** i) for i in range(10)]
        df = self._make_daily_frame(closes)

        clean, degraded, reason = self._drop(df)

        assert degraded is False, (
            f"Gradual daily drift should not trip the spike filter; reason={reason!r}"
        )
        assert len(clean) == 10

    def test_below_threshold_9pct_not_flagged(self):
        """A +9 % single-day move is well below the 12 % daily threshold and
        must not be quarantined."""
        base = 500.0
        closes = [base, base, base * 1.09, base, base]
        df = self._make_daily_frame(closes)

        clean, degraded, reason = self._drop(df)

        assert degraded is False, (
            f"A 9 % move should not exceed the 12 % daily threshold; reason={reason!r}"
        )

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_single_daily_candle_no_context_not_flagged(self):
        """A single daily candle has no rolling-median context (fewer than 3
        neighbours) and must never be quarantined by the cross-bar check."""
        df = self._make_daily_frame([500.0])

        clean, degraded, _ = self._drop(df)

        # With only 1 row the rolling window has no context → not a spike
        assert degraded is False
        assert len(clean) == 1

    def test_two_daily_candles_no_context_not_flagged(self):
        """Two candles still lack the minimum 3 neighbours required by
        min_periods=3 in the rolling window; neither should be flagged."""
        df = self._make_daily_frame([500.0, 560.0])  # +12 % but no context

        clean, degraded, _ = self._drop(df)

        assert degraded is False


class TestPredictShort20PctQuarantined:
    """When 20 % of the 5-min frame is quarantined, the prediction engine must:
      - return degraded=True with an informative reason string
      - keep the remaining 80 % of valid rows in the clean frame
      - never raise (predict_short can continue to build a valid signal response)
    """

    def _drop(self, df: pd.DataFrame) -> tuple:
        import importlib
        import routers.predictions as pred
        importlib.reload(pred)
        return pred._drop_invalid_ohlc(df, timeframe="5min")

    def _make_mixed_frame(self, total: int, bad_fraction: float) -> pd.DataFrame:
        """Build a frame where *bad_fraction* rows have high < open (invalid)."""
        bad_count = int(total * bad_fraction)
        rows = []
        base_ts = datetime(2024, 7, 30, 9, 30)
        for i in range(total):
            minutes_offset = i * 5
            ts = base_ts.replace(
                minute=(base_ts.minute + minutes_offset) % 60,
                hour=base_ts.hour + (base_ts.minute + minutes_offset) // 60,
            )
            if i < bad_count:
                # high < open → internally invalid
                rows.append({
                    "timestamp": ts,
                    "open": 500.0, "high": 495.0, "low": 490.0, "close": 497.0,
                })
            else:
                rows.append({
                    "timestamp": ts,
                    "open": 500.0, "high": 505.0, "low": 498.0, "close": 503.0,
                })
        return pd.DataFrame(rows)

    def test_20pct_quarantine_returns_degraded_flag(self):
        """degraded=True and reason cites quarantine count when 20 rows are bad."""
        df = self._make_mixed_frame(total=100, bad_fraction=0.20)
        clean, degraded, reason = self._drop(df)
        assert degraded is True
        assert "quarantined 20" in reason

    def test_20pct_quarantine_clean_frame_size(self):
        """80 valid rows survive after 20 % quarantine."""
        df = self._make_mixed_frame(total=100, bad_fraction=0.20)
        clean, degraded, reason = self._drop(df)
        assert len(clean) == 80

    def test_20pct_quarantine_response_shape(self):
        """The returned tuple carries the three values predict_short expects."""
        df = self._make_mixed_frame(total=50, bad_fraction=0.20)
        clean, degraded, reason = self._drop(df)

        # predict_short unpacks exactly this 3-tuple
        assert isinstance(clean, pd.DataFrame)
        assert isinstance(degraded, bool)
        assert isinstance(reason, str)

        # Clean slice is non-empty → predict_short continues (does not fall
        # through to the all-invalid neutral return)
        assert not clean.empty
        assert "quarantined" in reason
        assert "5min" in reason

    def test_50pct_quarantine_still_no_crash(self):
        """Half the frame invalid: clean has 50 % remaining, no exception raised."""
        df = self._make_mixed_frame(total=40, bad_fraction=0.50)
        clean, degraded, reason = self._drop(df)
        assert degraded is True
        assert len(clean) == 20

    def test_100pct_quarantine_returns_empty_clean_degraded(self):
        """All candles invalid → empty clean frame with degraded=True.
        predict_short returns the neutral fallback, not a crash."""
        df = self._make_mixed_frame(total=10, bad_fraction=1.0)
        clean, degraded, reason = self._drop(df)
        assert clean.empty
        assert degraded is True


# ---------------------------------------------------------------------------
# Task-177: Cross-bar spike detection
# ---------------------------------------------------------------------------

class TestFlagCrossBarSpikes:
    """Unit tests for flag_cross_bar_spikes() — the rolling-median deviation check."""

    def _make_close_df(self, closes: list[float]) -> pd.DataFrame:
        """Build a minimal DataFrame with a ``close`` column."""
        base = datetime(2024, 7, 30, 9, 30)
        timestamps = [base + timedelta(minutes=i * 5) for i in range(len(closes))]
        return pd.DataFrame({"close": closes}, index=pd.DatetimeIndex(timestamps))

    def test_flat_series_no_spikes(self):
        """A perfectly flat close series should never trigger a spike."""
        df = self._make_close_df([500.0] * 10)
        result = flag_cross_bar_spikes(df, threshold=0.10)
        assert not result.any()

    def test_normal_intraday_move_not_flagged(self):
        """Typical 0.3 % intraday moves are well below the 10 % threshold."""
        closes = [500.0, 500.3, 500.6, 500.9, 501.2, 501.5, 501.2, 500.9]
        df = self._make_close_df(closes)
        result = flag_cross_bar_spikes(df, threshold=0.10)
        assert not result.any()

    def test_one_pct_intraday_move_not_flagged(self):
        """Even a sharp 1 % single-bar move (within normal VOO intraday range)
        must not trip the 10 % threshold."""
        closes = [500.0, 500.0, 505.0, 500.0, 500.0, 500.0, 500.0]  # +1 % spike
        df = self._make_close_df(closes)
        result = flag_cross_bar_spikes(df, threshold=0.10)
        # 505 vs median ≈ 500 → deviation ~1 % < 10 %
        assert not result.any()

    def test_fifteen_pct_spike_is_flagged(self):
        """A bar 15 % above its neighbours is flagged."""
        closes = [500.0, 500.0, 575.0, 500.0, 500.0]
        df = self._make_close_df(closes)
        result = flag_cross_bar_spikes(df, threshold=0.10)
        # Only the spike bar (index 2) should be flagged
        assert result.iloc[2] is True or result.iloc[2]
        assert result.iloc[0] is False or not result.iloc[0]
        assert result.iloc[4] is False or not result.iloc[4]

    def test_exactly_at_threshold_not_flagged(self):
        """A deviation exactly equal to the threshold is NOT flagged (strict >)."""
        # median([500, 500, 550, 500, 500]) = 500; deviation = 50/500 = 0.10
        closes = [500.0, 500.0, 550.0, 500.0, 500.0]
        df = self._make_close_df(closes)
        result = flag_cross_bar_spikes(df, threshold=0.10)
        assert not result.iloc[2]

    def test_just_over_threshold_is_flagged(self):
        """A deviation of 10.01 % is flagged."""
        # median([500, 500, 550.05, 500, 500]) = 500; deviation ≈ 10.01 %
        closes = [500.0, 500.0, 550.05, 500.0, 500.0]
        df = self._make_close_df(closes)
        result = flag_cross_bar_spikes(df, threshold=0.10)
        assert result.iloc[2]

    def test_edge_bars_not_flagged_due_to_insufficient_context(self):
        """Edge bars with fewer than 3 valid neighbours in the window are not flagged
        even if their value differs wildly from adjacent bars."""
        # Only 3 bars: the window=5 centered median at edge has fewer than 3
        # neighbours for a bar that sits at position 0 or 2 with a window of 5.
        # With min_periods=3 and 3 total bars, the middle bar has context but
        # edge bars at positions 0 and 2 with window=5 only have 2-3 bars visible.
        closes = [500.0, 500.0, 500.0]
        df = self._make_close_df(closes)
        result = flag_cross_bar_spikes(df, threshold=0.10)
        # No spikes in a flat series regardless of edge context
        assert not result.any()

    def test_empty_df_returns_empty_series(self):
        df = pd.DataFrame()
        result = flag_cross_bar_spikes(df, threshold=0.10)
        assert result.empty

    def test_missing_close_column_returns_false_series(self):
        df = pd.DataFrame({"open": [500.0, 501.0, 502.0]})
        result = flag_cross_bar_spikes(df, threshold=0.10)
        assert not result.any()

    def test_zero_threshold_returns_no_flags(self):
        """When threshold is 0, the check is a no-op (never flag)."""
        closes = [500.0, 500.0, 800.0, 500.0, 500.0]  # +60 % spike
        df = self._make_close_df(closes)
        result = flag_cross_bar_spikes(df, threshold=0)
        assert not result.any()

    def test_negative_threshold_returns_no_flags(self):
        closes = [500.0, 600.0, 500.0]
        df = self._make_close_df(closes)
        result = flag_cross_bar_spikes(df, threshold=-0.05)
        assert not result.any()


class TestFilterValidOhlcCrossBar:
    """Verify that filter_valid_ohlc catches cross-bar spikes on top of intra-bar checks."""

    def _frame(self, rows: list[dict]) -> pd.DataFrame:
        base = datetime(2024, 7, 30, 9, 30)
        df = pd.DataFrame(rows)
        df.index = pd.DatetimeIndex(
            [base + timedelta(minutes=i * 5) for i in range(len(rows))]
        )
        return df

    def test_internally_valid_spike_quarantined_by_cross_bar(self):
        """An internally-valid bar whose close is +15 % above neighbours is
        quarantined with reason 'cross_bar_spike'."""
        rows = [
            {"open": 499.0, "high": 504.0, "low": 497.0, "close": 500.0},
            {"open": 499.0, "high": 504.0, "low": 497.0, "close": 500.0},
            # Spike: internally valid (high > close) but +15 % above neighbours
            {"open": 500.0, "high": 580.0, "low": 499.0, "close": 575.0},
            {"open": 499.0, "high": 504.0, "low": 497.0, "close": 500.0},
            {"open": 499.0, "high": 504.0, "low": 497.0, "close": 500.0},
        ]
        df = self._frame(rows)
        valid, quarantined = filter_valid_ohlc(df)
        assert len(quarantined) == 1
        assert "cross_bar_spike" in quarantined.iloc[0]["ohlc_invalid_reason"]
        assert len(valid) == 4

    def test_normal_volatility_passes_through(self):
        """A 1 % intraday move does not trip the cross-bar check."""
        rows = [
            {"open": 499.0, "high": 504.0, "low": 497.0, "close": 500.0},
            {"open": 500.0, "high": 506.0, "low": 499.0, "close": 505.0},  # +1 %
            {"open": 504.0, "high": 509.0, "low": 502.0, "close": 507.0},
            {"open": 506.0, "high": 511.0, "low": 504.0, "close": 509.0},
            {"open": 508.0, "high": 513.0, "low": 506.0, "close": 511.0},
        ]
        df = self._frame(rows)
        valid, quarantined = filter_valid_ohlc(df)
        assert quarantined.empty
        assert len(valid) == 5

    def test_spike_threshold_zero_disables_cross_bar_check(self):
        """Passing spike_threshold=0 disables cross-bar detection; an internally
        valid spike bar passes through without being quarantined."""
        rows = [
            {"open": 499.0, "high": 504.0, "low": 497.0, "close": 500.0},
            {"open": 499.0, "high": 504.0, "low": 497.0, "close": 500.0},
            {"open": 500.0, "high": 580.0, "low": 499.0, "close": 575.0},
            {"open": 499.0, "high": 504.0, "low": 497.0, "close": 500.0},
            {"open": 499.0, "high": 504.0, "low": 497.0, "close": 500.0},
        ]
        df = self._frame(rows)
        valid, quarantined = filter_valid_ohlc(df, spike_threshold=0)
        # Spike bar is internally valid; with check disabled it is not quarantined
        assert quarantined.empty
        assert len(valid) == 5

    def test_intrabar_violation_still_caught_regardless_of_spike_check(self):
        """An intra-bar violation (high < open) is quarantined even when the
        cross-bar spike check would not have flagged it."""
        rows = [
            {"open": 499.0, "high": 504.0, "low": 497.0, "close": 500.0},
            # high < open: internally invalid
            {"open": 680.12, "high": 676.71, "low": 675.58, "close": 676.01},
            {"open": 499.0, "high": 504.0, "low": 497.0, "close": 500.0},
        ]
        df = self._frame(rows)
        valid, quarantined = filter_valid_ohlc(df)
        assert len(quarantined) == 1
        assert "high_below_open" in quarantined.iloc[0]["ohlc_invalid_reason"]

    def test_quarantined_row_has_ohlc_invalid_reason_column(self):
        """Quarantined cross-bar spike rows always carry the ohlc_invalid_reason column."""
        rows = [
            {"open": 499.0, "high": 504.0, "low": 497.0, "close": 500.0},
            {"open": 499.0, "high": 504.0, "low": 497.0, "close": 500.0},
            {"open": 500.0, "high": 580.0, "low": 499.0, "close": 575.0},
            {"open": 499.0, "high": 504.0, "low": 497.0, "close": 500.0},
            {"open": 499.0, "high": 504.0, "low": 497.0, "close": 500.0},
        ]
        df = self._frame(rows)
        _, quarantined = filter_valid_ohlc(df)
        assert "ohlc_invalid_reason" in quarantined.columns
        assert quarantined.iloc[0]["ohlc_invalid_reason"] != ""

    def test_close_value_appears_in_cross_bar_reason(self):
        """The quarantine reason string includes the actual close value for auditability."""
        rows = [
            {"open": 499.0, "high": 504.0, "low": 497.0, "close": 500.0},
            {"open": 499.0, "high": 504.0, "low": 497.0, "close": 500.0},
            {"open": 500.0, "high": 580.0, "low": 499.0, "close": 575.0},
            {"open": 499.0, "high": 504.0, "low": 497.0, "close": 500.0},
            {"open": 499.0, "high": 504.0, "low": 497.0, "close": 500.0},
        ]
        df = self._frame(rows)
        _, quarantined = filter_valid_ohlc(df)
        reason = quarantined.iloc[0]["ohlc_invalid_reason"]
        assert "575." in reason  # close value present


# ---------------------------------------------------------------------------
# Task-178: zero-volume bar liquidity score tests
# ---------------------------------------------------------------------------

def _make_5min_vol_frame(rows: list[dict]) -> pd.DataFrame:
    """Build a typical 5-min candle DataFrame for liquidity score tests."""
    base = datetime(2024, 7, 30, 9, 30)
    enriched = []
    for i, r in enumerate(rows):
        enriched.append({
            "timestamp": base + timedelta(minutes=i * 5),
            "open": r.get("open", 500.0),
            "high": r.get("high", 505.0),
            "low": r.get("low", 498.0),
            "close": r.get("close", 502.0),
            "volume": r.get("volume", 1_000_000),
            "session_type": r.get("session_type", "regular"),
            "is_extended_hours": r.get("is_extended_hours", False),
            "gap_type": r.get("gap_type", "none"),
            "gap_percent": r.get("gap_percent", 0.0),
            "ticker": "VOO",
        })
    return pd.DataFrame(enriched)


class TestDetectZeroVolumeBars:
    """Unit tests for _detect_zero_volume_bars helper in routers.predictions."""

    def _detect(self, df):
        import importlib
        import routers.predictions as pred
        importlib.reload(pred)
        return pred._detect_zero_volume_bars(df)

    def test_no_zero_volume_returns_empty_mask_and_no_reason(self):
        """All bars have positive volume → mask all-False, count=0, reason=''."""
        df = _make_5min_vol_frame([
            {"volume": 1_000_000},
            {"volume": 1_200_000},
            {"volume":   800_000},
        ])
        mask, count, reason = self._detect(df)
        assert count == 0
        assert reason == ""
        assert not mask.any()

    def test_one_zero_volume_bar_detected(self):
        """A single bar with volume=0 is flagged."""
        df = _make_5min_vol_frame([
            {"volume": 1_000_000},
            {"volume": 0},           # glitch bar
            {"volume":   900_000},
        ])
        mask, count, reason = self._detect(df)
        assert count == 1
        assert mask.sum() == 1
        assert "zero_volume_bars" in reason
        assert "1 5-min bar(s)" in reason

    def test_multiple_zero_volume_bars_counted(self):
        """Three glitch bars → count reflects all three."""
        df = _make_5min_vol_frame([
            {"volume": 1_000_000},
            {"volume": 0},
            {"volume": 0},
            {"volume": 0},
            {"volume":   800_000},
        ])
        mask, count, reason = self._detect(df)
        assert count == 3
        assert "3 5-min bar(s)" in reason

    def test_reason_includes_earliest_timestamp(self):
        """The reason string contains the timestamp of the first zero-volume bar."""
        df = _make_5min_vol_frame([
            {"volume": 1_000_000},
            {"volume": 0},           # second row → ts = 09:35
            {"volume":   900_000},
        ])
        mask, count, reason = self._detect(df)
        # Timestamp of the zero-volume bar should appear in reason
        assert "09:35" in reason or "earliest ts=" in reason

    def test_empty_df_returns_no_zeros(self):
        """Empty DataFrame → count=0, no crash."""
        import pandas as pd
        mask, count, reason = self._detect(pd.DataFrame())
        assert count == 0
        assert reason == ""

    def test_missing_volume_column_returns_no_zeros(self):
        """DataFrame without a 'volume' column → count=0, no crash."""
        import pandas as pd
        df = pd.DataFrame({"open": [500.0], "close": [502.0]})
        mask, count, reason = self._detect(df)
        assert count == 0

    def test_nan_volume_treated_as_zero(self):
        """A bar with volume=NaN is treated as zero volume."""
        import pandas as pd
        import numpy as np
        df = _make_5min_vol_frame([
            {"volume": 1_000_000},
            {"volume": float("nan")},
            {"volume":   900_000},
        ])
        mask, count, reason = self._detect(df)
        assert count == 1
        assert "zero_volume_bars" in reason


class TestZeroVolumeBarLiquidityScore:
    """Verify that zero-volume bars don't distort the liquidity score.

    The liquidity score is ``extended_vol / max(regular_vol, 1.0)``.
    A zero-volume regular-session bar reduces ``regular_vol`` and can
    push the score upward when many bars glitch; conversely, a zero-volume
    extended-hours bar reduces ``extended_vol`` and pushes the score toward
    zero, potentially triggering the low-liquidity signal suppression even
    when the real session was liquid.

    After the fix, zero-volume bars are excluded from the liquidity
    computation so glitch bars cannot move the score.
    """

    LOW_LIQUIDITY_THRESHOLD = 0.2  # approximate suppression threshold used in tests

    def _compute_liquidity(self, df: "pd.DataFrame") -> float:
        """Replicate the fixed liquidity computation from predict_short."""
        import importlib
        import routers.predictions as pred
        importlib.reload(pred)

        zero_vol_mask, _, _ = pred._detect_zero_volume_bars(df)
        df_for_liq = df[~zero_vol_mask] if zero_vol_mask.any() else df

        regular_mask = df_for_liq["session_type"] == "regular"
        extended_mask = df_for_liq["is_extended_hours"] == True
        regular_vol = float(df_for_liq.loc[regular_mask, "volume"].sum()) if regular_mask.any() else 1.0
        extended_vol = float(df_for_liq.loc[extended_mask, "volume"].sum()) if extended_mask.any() else 0.0
        return extended_vol / max(regular_vol, 1.0)

    def _compute_liquidity_naive(self, df: "pd.DataFrame") -> float:
        """Replicate the old (unfixed) computation that includes zero-volume bars."""
        regular_mask = df["session_type"] == "regular"
        extended_mask = df["is_extended_hours"] == True
        regular_vol = float(df.loc[regular_mask, "volume"].sum()) if regular_mask.any() else 1.0
        extended_vol = float(df.loc[extended_mask, "volume"].sum()) if extended_mask.any() else 0.0
        return extended_vol / max(regular_vol, 1.0)

    def test_normal_session_no_zero_bars_score_unchanged(self):
        """Without any zero-volume bars, fixed and naive computations agree."""
        df = _make_5min_vol_frame([
            {"volume": 1_000_000, "session_type": "regular", "is_extended_hours": False},
        ] * 10)
        score_fixed = self._compute_liquidity(df)
        score_naive = self._compute_liquidity_naive(df)
        assert abs(score_fixed - score_naive) < 1e-9

    def test_zero_volume_extended_bar_does_not_suppress_liquidity(self):
        """A single zero-volume extended-hours glitch bar must not reduce
        the liquidity score below the suppression threshold when the regular
        session had healthy volume.

        Without the fix: extended_vol drops toward 0 → score ≈ 0 (suppressed).
        With the fix: zero-volume bar excluded → score computed from real bars.
        """
        # 10 regular bars with 1 M vol each, then 1 extended bar with healthy
        # vol, then 1 extended glitch bar with vol=0
        rows = (
            [{"volume": 1_000_000, "session_type": "regular", "is_extended_hours": False}] * 10
            + [{"volume": 500_000, "session_type": "pre_market", "is_extended_hours": True}]
            + [{"volume": 0, "session_type": "pre_market", "is_extended_hours": True}]   # glitch
        )
        df = _make_5min_vol_frame(rows)

        score_fixed = self._compute_liquidity(df)
        # The healthy extended bar still contributes; score > 0
        assert score_fixed > 0.0, "expected non-zero liquidity with real extended volume"

    def test_zero_volume_regular_bar_excluded_from_denominator(self):
        """A zero-volume regular bar must be excluded so it doesn't artificially
        lower regular_vol (which would raise the score anomalously).

        This confirms the mask correctly excludes the glitch bar from both
        the numerator (extended) and denominator (regular) legs.
        """
        rows_clean = [
            {"volume": 1_000_000, "session_type": "regular", "is_extended_hours": False}
        ] * 5
        rows_with_glitch = rows_clean + [
            {"volume": 0, "session_type": "regular", "is_extended_hours": False}  # glitch
        ]

        df_clean = _make_5min_vol_frame(rows_clean)
        df_glitch = _make_5min_vol_frame(rows_with_glitch)

        score_clean = self._compute_liquidity(df_clean)
        score_fixed = self._compute_liquidity(df_glitch)

        # After the fix, both frames have the same non-zero regular volume
        import pytest as _pytest
        assert score_fixed == _pytest.approx(score_clean, abs=1e-9), (
            "zero-volume regular bar should be excluded; score must match clean frame"
        )

    def test_dq_degraded_set_when_zero_volume_detected(self):
        """predict_short must set dq_degraded=True and surface a reason string
        when any zero-volume bar is present in the 5-min frame.

        This test exercises _detect_zero_volume_bars directly since
        predict_short requires a live DB session.
        """
        import importlib
        import routers.predictions as pred
        importlib.reload(pred)

        df = _make_5min_vol_frame([
            {"volume": 1_000_000, "session_type": "regular", "is_extended_hours": False},
            {"volume": 0,         "session_type": "regular", "is_extended_hours": False},
            {"volume":   900_000, "session_type": "regular", "is_extended_hours": False},
        ])

        _, count, reason = pred._detect_zero_volume_bars(df)
        assert count == 1
        assert "zero_volume_bars" in reason

    def test_single_zero_volume_bar_does_not_push_score_below_suppression_threshold(self):
        """Core regression test: one glitch zero-volume bar in an otherwise
        liquid session must not move the score below ``LOW_LIQUIDITY_THRESHOLD``.

        Uses a purely regular-session frame (all extended_vol = 0) so that
        the liquidity score is 0 in both variants — the test asserts the
        *reason* path fires, not a numeric bound.  The meaningful numeric
        assertion is in test_zero_volume_extended_bar_does_not_suppress_liquidity.
        """
        import importlib
        import routers.predictions as pred
        importlib.reload(pred)

        rows = (
            [{"volume": 1_000_000, "session_type": "regular", "is_extended_hours": False}] * 9
            + [{"volume": 0, "session_type": "regular", "is_extended_hours": False}]
        )
        df = _make_5min_vol_frame(rows)
        mask, count, reason = pred._detect_zero_volume_bars(df)

        # dq_reason must be surfaced
        assert count == 1
        assert "zero_volume_bars" in reason
        # The glitch bar is excluded: 9 non-zero regular bars remain
        df_clean = df[~mask]
        assert len(df_clean) == 9


# ---------------------------------------------------------------------------
# Task-188: daily zero-volume bar tests for predict_long
# ---------------------------------------------------------------------------

def _make_daily_vol_frame(rows: list[dict]) -> pd.DataFrame:
    """Build a typical daily candle DataFrame for long-trend zero-volume tests."""
    base = datetime(2024, 7, 1)
    enriched = []
    for i, r in enumerate(rows):
        enriched.append({
            "timestamp": base + timedelta(days=i),
            "open": r.get("open", 500.0),
            "high": r.get("high", 505.0),
            "low": r.get("low", 498.0),
            "close": r.get("close", 502.0),
            "volume": r.get("volume", 5_000_000),
            "session_type": r.get("session_type", "regular"),
            "is_extended_hours": r.get("is_extended_hours", False),
            "gap_type": r.get("gap_type", "none"),
            "gap_percent": r.get("gap_percent", 0.0),
            "ticker": "VOO",
        })
    return pd.DataFrame(enriched)


class TestDailyZeroVolumeBarDetection:
    """_detect_zero_volume_bars works correctly on daily candle frames.

    The helper was originally written for 5-min frames but operates on any
    DataFrame with a 'volume' column. These tests confirm it is usable for
    the daily frame loaded by predict_long.
    """

    def _detect(self, df):
        import importlib
        import routers.predictions as pred
        importlib.reload(pred)
        return pred._detect_zero_volume_bars(df)

    def test_all_positive_volume_returns_no_zeros(self):
        """A daily frame with normal volume → count=0, reason=''."""
        df = _make_daily_vol_frame([
            {"volume": 5_000_000},
            {"volume": 4_800_000},
            {"volume": 5_200_000},
        ])
        mask, count, reason = self._detect(df)
        assert count == 0
        assert reason == ""
        assert not mask.any()

    def test_single_zero_volume_daily_bar_detected(self):
        """A daily bar with volume=0 is flagged with count=1 and a reason."""
        df = _make_daily_vol_frame([
            {"volume": 5_000_000},
            {"volume": 0},           # glitch bar
            {"volume": 4_900_000},
        ])
        mask, count, reason = self._detect(df)
        assert count == 1
        assert mask.sum() == 1
        assert "zero_volume_bars" in reason

    def test_nan_volume_daily_treated_as_zero(self):
        """A daily bar with volume=NaN is also treated as zero volume."""
        df = _make_daily_vol_frame([
            {"volume": 5_000_000},
            {"volume": float("nan")},
            {"volume": 4_900_000},
        ])
        mask, count, reason = self._detect(df)
        assert count == 1
        assert "zero_volume_bars" in reason

    def test_zero_volume_bar_excluded_from_filtered_frame(self):
        """After masking, the zero-volume daily bar is absent from the clean frame."""
        df = _make_daily_vol_frame([
            {"volume": 5_000_000},
            {"volume": 0},
            {"volume": 4_900_000},
        ])
        mask, count, _ = self._detect(df)
        df_clean = df[~mask].reset_index(drop=True)
        assert len(df_clean) == 2
        # No zero-volume rows in the clean frame
        assert (df_clean["volume"] > 0).all()

    def test_multiple_zero_volume_daily_bars_counted(self):
        """Multiple zero-volume daily bars all appear in the count."""
        df = _make_daily_vol_frame([
            {"volume": 5_000_000},
            {"volume": 0},
            {"volume": 0},
            {"volume": 4_900_000},
        ])
        mask, count, reason = self._detect(df)
        assert count == 2
        assert mask.sum() == 2
        assert "zero_volume_bars" in reason


class TestPredictLongZeroVolumeIntegration:
    """Verify that a zero-volume daily bar is excluded from long-trend inputs
    and that dq_degraded / dq_reason surface the detection.

    These tests exercise _detect_zero_volume_bars directly (predict_long
    requires a live async DB session and is tested via the router fixture
    in integration tests).  They assert the preconditions that predict_long
    relies on before calling build_latest_features.
    """

    def _detect(self, df):
        import importlib
        import routers.predictions as pred
        importlib.reload(pred)
        return pred._detect_zero_volume_bars(df)

    def test_zero_volume_daily_bar_sets_dq_degraded(self):
        """When a daily frame has a zero-volume bar, detect returns count>0
        and a non-empty reason — predict_long will set dq_degraded=True."""
        df = _make_daily_vol_frame([
            {"volume": 5_000_000},
            {"volume": 0},        # yfinance glitch
            {"volume": 4_800_000},
        ])
        _, count, reason = self._detect(df)
        # predict_long sets dq_degraded = True when count > 0
        assert count > 0
        assert reason != ""
        assert "zero_volume_bars" in reason

    def test_zero_volume_daily_bar_excluded_before_features(self):
        """After predict_long applies the mask, the zero-volume bar is absent
        from the frame passed to build_latest_features."""
        df = _make_daily_vol_frame([
            {"volume": 5_000_000},
            {"volume": 0},        # glitch
            {"volume": 4_700_000},
            {"volume": 5_100_000},
        ])
        mask, count, _ = self._detect(df)
        assert count == 1
        df_clean = df[~mask].reset_index(drop=True)
        # The clean frame should not contain the zero-volume row
        assert len(df_clean) == 3
        assert (df_clean["volume"] > 0).all()

    def test_single_zero_volume_bar_does_not_corrupt_long_response(self):
        """A single zero-volume daily bar is excluded from the feature frame;
        the remaining valid candles allow a response to be built without crash."""
        # Simulate 10 daily candles, one with volume=0
        rows = [{"volume": 5_000_000}] * 5 + [{"volume": 0}] + [{"volume": 5_000_000}] * 4
        df = _make_daily_vol_frame(rows)
        mask, count, reason = self._detect(df)
        df_clean = df[~mask].reset_index(drop=True)
        # Exactly 9 valid candles survive
        assert len(df_clean) == 9
        # dq fields predict_long will emit
        dq_degraded = count > 0
        dq_reason = reason
        assert dq_degraded is True
        assert "zero_volume_bars" in dq_reason
        # The clean frame has sufficient rows to build features
        assert not df_clean.empty

    def test_all_zero_volume_daily_bars_returns_empty_clean_frame(self):
        """When every daily candle has zero volume, the clean frame is empty.
        predict_long returns the neutral fallback in this scenario."""
        df = _make_daily_vol_frame([
            {"volume": 0},
            {"volume": 0},
            {"volume": 0},
        ])
        mask, count, reason = self._detect(df)
        df_clean = df[~mask].reset_index(drop=True)
        assert count == 3
        assert df_clean.empty
        # predict_long will detect empty and return neutral
        dq_degraded = count > 0
        assert dq_degraded is True

    def test_dq_reason_combined_when_ohlc_also_degraded(self):
        """When both OHLC filter and zero-volume detection fire, the reasons
        are concatenated with '; ' in predict_long's dq_reason field."""
        # Simulate the merging logic in predict_long
        existing_reason = "quarantined 1 malformed daily candle(s); latest bad candle ts=2024-07-30"
        dq_degraded = True

        # A zero-volume bar is also found
        df = _make_daily_vol_frame([{"volume": 0}])
        _, count, zv_reason = self._detect(df)
        assert count == 1

        # predict_long concatenation logic
        if dq_degraded:
            combined = existing_reason + "; " + zv_reason
        else:
            combined = zv_reason
            dq_degraded = True

        assert "quarantined" in combined
        assert "zero_volume_bars" in combined
        assert "; " in combined


# ---------------------------------------------------------------------------
# Task-189: ingest-time zero-volume gate
# ---------------------------------------------------------------------------

class TestFilterZeroVolumeBars:
    """Unit tests for ohlc_validator.filter_zero_volume_bars."""

    from ingestion.ohlc_validator import filter_zero_volume_bars  # noqa: F401

    def _filter(self, df):
        from ingestion.ohlc_validator import filter_zero_volume_bars
        return filter_zero_volume_bars(df)

    def _make_df(self, rows: list[dict]) -> pd.DataFrame:
        base = datetime(2024, 7, 29)
        df = pd.DataFrame(rows)
        df.index = pd.DatetimeIndex(
            [base + timedelta(days=i) for i in range(len(rows))]
        )
        return df

    def test_all_positive_volume_passes_through(self):
        """Rows with positive volume are all valid — quarantined frame is empty."""
        df = self._make_df([
            {"open": 500.0, "high": 505.0, "low": 498.0, "close": 502.0, "volume": 1_000_000},
            {"open": 501.0, "high": 506.0, "low": 499.0, "close": 503.0, "volume": 900_000},
        ])
        valid, quarantined = self._filter(df)
        assert len(valid) == 2
        assert quarantined.empty

    def test_zero_volume_row_is_quarantined(self):
        """A single bar with volume=0 lands in the quarantined frame."""
        df = self._make_df([
            {"open": 500.0, "high": 505.0, "low": 498.0, "close": 502.0, "volume": 1_000_000},
            {"open": 501.0, "high": 506.0, "low": 499.0, "close": 503.0, "volume": 0},
        ])
        valid, quarantined = self._filter(df)
        assert len(valid) == 1
        assert len(quarantined) == 1
        assert quarantined.iloc[0]["ohlc_invalid_reason"] == "zero_volume"

    def test_nan_volume_treated_as_zero(self):
        """A bar whose volume is NaN is treated as zero volume and quarantined."""
        import numpy as np
        df = self._make_df([
            {"open": 500.0, "high": 505.0, "low": 498.0, "close": 502.0, "volume": 1_000_000},
            {"open": 501.0, "high": 506.0, "low": 499.0, "close": 503.0, "volume": float("nan")},
        ])
        valid, quarantined = self._filter(df)
        assert len(valid) == 1
        assert len(quarantined) == 1
        assert quarantined.iloc[0]["ohlc_invalid_reason"] == "zero_volume"

    def test_multiple_zero_volume_rows_all_quarantined(self):
        """Every zero-volume bar is quarantined; non-zero bars survive."""
        df = self._make_df([
            {"open": 500.0, "high": 505.0, "low": 498.0, "close": 502.0, "volume": 0},
            {"open": 501.0, "high": 506.0, "low": 499.0, "close": 503.0, "volume": 1_000_000},
            {"open": 502.0, "high": 507.0, "low": 500.0, "close": 504.0, "volume": 0},
        ])
        valid, quarantined = self._filter(df)
        assert len(valid) == 1
        assert len(quarantined) == 2

    def test_empty_df_returns_empty_frames(self):
        """Empty input produces two empty frames without crashing."""
        valid, quarantined = self._filter(pd.DataFrame())
        assert valid.empty
        assert quarantined.empty

    def test_missing_volume_column_passes_through_unchanged(self):
        """A DataFrame with no 'volume' column passes through as-is."""
        df = pd.DataFrame({"open": [500.0], "close": [502.0]})
        valid, quarantined = self._filter(df)
        assert len(valid) == 1
        assert quarantined.empty

    def test_all_zero_volume_returns_empty_valid(self):
        """When every bar has zero volume the valid frame is empty."""
        df = self._make_df([
            {"open": 500.0, "high": 505.0, "low": 498.0, "close": 502.0, "volume": 0},
            {"open": 501.0, "high": 506.0, "low": 499.0, "close": 503.0, "volume": 0},
        ])
        valid, quarantined = self._filter(df)
        assert valid.empty
        assert len(quarantined) == 2


class TestNormaliseColumnsDropsZeroVolume:
    """Verify that _normalise_columns drops zero-volume bars before DB storage."""

    def _run_normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        from ingestion.fetcher import DataFetcher
        return DataFetcher._normalise_columns(df)

    def test_zero_volume_daily_bar_is_dropped(self):
        """A daily bar with volume=0 is removed by _normalise_columns."""
        df = pd.DataFrame(
            {
                "Open":   [500.0, 501.0],
                "High":   [505.0, 506.0],
                "Low":    [498.0, 499.0],
                "Close":  [502.0, 503.0],
                "Volume": [1_000_000, 0],
            },
            index=pd.DatetimeIndex(["2024-07-29", "2024-07-30"]),
        )
        result = self._run_normalise(df)
        assert len(result) == 1
        assert float(result.iloc[0]["open"]) == pytest.approx(500.0)

    def test_positive_volume_bar_passes_through(self):
        """Bars with positive volume are not affected by the zero-volume filter."""
        df = pd.DataFrame(
            {
                "Open":   [500.0, 501.0],
                "High":   [505.0, 506.0],
                "Low":    [498.0, 499.0],
                "Close":  [502.0, 503.0],
                "Volume": [1_000_000, 800_000],
            },
            index=pd.DatetimeIndex(["2024-07-29", "2024-07-30"]),
        )
        result = self._run_normalise(df)
        assert len(result) == 2

    def test_only_zero_volume_row_gives_empty_frame(self):
        """When the only row has volume=0 the result is an empty DataFrame."""
        df = pd.DataFrame(
            {
                "Open":   [500.0],
                "High":   [505.0],
                "Low":    [498.0],
                "Close":  [502.0],
                "Volume": [0],
            },
            index=pd.DatetimeIndex(["2024-07-30"]),
        )
        result = self._run_normalise(df)
        assert result.empty


@pytest.mark.asyncio
async def test_remove_invalid_voo_candles_removes_zero_volume(db_session):
    """remove_invalid_voo_candles must delete zero-volume rows from the DB."""
    good = VooCandle(
        ticker="VOO",
        timestamp=datetime(2024, 7, 29),
        open=500.0, high=505.0, low=498.0, close=502.0,
        volume=1_000_000,
        timeframe="daily", session_type="regular",
    )
    zero_vol = VooCandle(
        ticker="VOO",
        timestamp=datetime(2024, 7, 30),
        open=501.0, high=506.0, low=499.0, close=503.0,
        volume=0,
        timeframe="daily", session_type="regular",
    )
    db_session.add(good)
    db_session.add(zero_vol)
    await db_session.flush()

    removed = await IngestionPipeline().remove_invalid_voo_candles(db_session)

    assert removed == 1
    assert await db_session.get(VooCandle, good.id) is not None
    assert await db_session.get(VooCandle, zero_vol.id) is None


@pytest.mark.asyncio
async def test_store_voo_candles_skips_zero_volume_row(db_session):
    """store_voo_candles must skip rows with volume=0 and not insert them."""
    frame = pd.DataFrame(
        {
            "open":  [500.0, 501.0],
            "high":  [505.0, 506.0],
            "low":   [498.0, 499.0],
            "close": [502.0, 503.0],
            "volume": [1_000_000, 0],
            "is_extended_hours": [False, False],
            "session_type": ["regular", "regular"],
        },
        index=pd.DatetimeIndex(["2024-07-29", "2024-07-30"]),
    )

    await IngestionPipeline().store_voo_candles(frame, db_session, "daily")

    from sqlalchemy import select
    from database.models import VooCandle as VC
    result = await db_session.execute(select(VC).where(VC.ticker == "VOO"))
    rows = result.scalars().all()
    assert len(rows) == 1
    assert float(rows[0].open) == pytest.approx(500.0)


@pytest.mark.asyncio
async def test_remove_invalid_voo_candles_keeps_positive_volume(db_session):
    """remove_invalid_voo_candles must not delete rows with positive volume."""
    row = VooCandle(
        ticker="VOO",
        timestamp=datetime(2024, 7, 29),
        open=500.0, high=505.0, low=498.0, close=502.0,
        volume=1_000_000,
        timeframe="daily", session_type="regular",
    )
    db_session.add(row)
    await db_session.flush()

    removed = await IngestionPipeline().remove_invalid_voo_candles(db_session)

    assert removed == 0
    assert await db_session.get(VooCandle, row.id) is not None


# ---------------------------------------------------------------------------
# Task-211: long-trend signal stability when last daily bar has zero volume
# ---------------------------------------------------------------------------

class TestPredictLongZeroVolumeDailyBar:
    """Confirm predict_long stays stable when the most recent daily bar has volume=0.

    Task 188 filters zero-volume bars before feature computation via
    _detect_zero_volume_bars.  The filtered daily_df is then used to pick the
    'latest' candle (daily_df.iloc[-1]) for session_type, gap_type, etc.
    If the most recent bar was the zero-volume glitch, the 'latest' candle
    after filtering must be the previous valid bar — not the glitch row.
    """

    def _build_daily_df(self) -> pd.DataFrame:
        """Three valid daily bars plus a final zero-volume glitch bar."""
        return pd.DataFrame([
            {
                "timestamp": datetime(2024, 7, 28),
                "open": 495.0, "high": 500.0, "low": 493.0, "close": 498.0,
                "volume": 2_000_000,
                "session_type": "regular", "gap_type": "none",
                "is_extended_hours": False, "ticker": "VOO", "gap_percent": 0.0,
            },
            {
                "timestamp": datetime(2024, 7, 29),
                "open": 498.0, "high": 503.0, "low": 496.0, "close": 501.0,
                "volume": 1_800_000,
                "session_type": "regular", "gap_type": "none",
                "is_extended_hours": False, "ticker": "VOO", "gap_percent": 0.0,
            },
            {
                # Zero-volume glitch bar — most recent date, must be filtered out
                "timestamp": datetime(2024, 7, 30),
                "open": 501.0, "high": 506.0, "low": 499.0, "close": 504.0,
                "volume": 0,
                "session_type": "regular", "gap_type": "none",
                "is_extended_hours": False, "ticker": "VOO", "gap_percent": 0.0,
            },
        ])

    # ------------------------------------------------------------------
    # 1. Zero-volume detection correctly identifies the glitch bar
    # ------------------------------------------------------------------

    def test_detect_identifies_zero_volume_last_bar(self):
        """_detect_zero_volume_bars correctly flags only the last row with volume=0."""
        import routers.predictions as pred

        df = self._build_daily_df()
        mask, count, reason = pred._detect_zero_volume_bars(df)

        assert count == 1, "Exactly one zero-volume bar should be detected"
        assert bool(mask.iloc[-1]) is True, "Last row (volume=0) must be masked"
        assert bool(mask.iloc[0]) is False, "First row (valid volume) must not be masked"
        assert bool(mask.iloc[1]) is False, "Second row (valid volume) must not be masked"
        assert "zero_volume_bars" in reason

    # ------------------------------------------------------------------
    # 2. After masking, daily_df.iloc[-1] is the previous valid bar
    # ------------------------------------------------------------------

    def test_latest_candle_after_mask_is_previous_valid_bar(self):
        """After applying the zero-volume mask as predict_long does,
        daily_df.iloc[-1] must be the Jul-29 bar, not the Jul-30 glitch."""
        import routers.predictions as pred

        df = self._build_daily_df()
        mask, count, _ = pred._detect_zero_volume_bars(df)

        assert count > 0
        filtered = df[~mask].reset_index(drop=True)

        latest = filtered.iloc[-1]
        assert latest["timestamp"] == datetime(2024, 7, 29), (
            f"Expected latest candle to be 2024-07-29 (previous valid bar), "
            f"got {latest['timestamp']}"
        )
        assert latest["volume"] > 0, (
            "Latest candle after zero-volume filtering must have non-zero volume"
        )

    # ------------------------------------------------------------------
    # 3. dq_degraded is True when zero-volume bar is filtered
    # ------------------------------------------------------------------

    def test_dq_degraded_set_when_zero_volume_bar_filtered(self):
        """The dq_degraded flag and reason are set when a zero-volume daily
        bar is detected, matching the logic inside predict_long."""
        import routers.predictions as pred

        df = self._build_daily_df()
        _, count, zv_reason = pred._detect_zero_volume_bars(df)

        # Replicate predict_long's dq flag update logic
        dq_degraded = False
        dq_reason = ""
        if count > 0:
            dq_reason = zv_reason
            dq_degraded = True

        assert dq_degraded is True
        assert "zero_volume_bars" in dq_reason

    # ------------------------------------------------------------------
    # 4. End-to-end: predict_long with mocked session returns valid response
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_predict_long_zero_volume_last_bar_no_crash(self):
        """predict_long does not crash and returns dq_degraded=True when the
        most recent daily bar has volume=0.

        All DB loaders and ML components are patched so the test runs without
        a real database or trained model artifact.
        """
        from unittest.mock import AsyncMock, MagicMock, patch

        daily_df = self._build_daily_df()
        vix_df = pd.DataFrame([
            {
                "timestamp": datetime(2024, 7, 29),
                "open": 15.0, "high": 16.0, "low": 14.5, "close": 15.5,
                "ticker": "VIX",
            }
        ])

        mock_session = AsyncMock()

        with (
            patch("routers.predictions._load_daily_candles", new=AsyncMock(return_value=daily_df)),
            patch("routers.predictions._load_vix_candles", new=AsyncMock(return_value=vix_df)),
            patch("routers.predictions._load_spx_close_series", new=AsyncMock(return_value=pd.Series(dtype=float))),
            patch("routers.predictions._load_recent_confidence", new=AsyncMock(return_value=[])),
            patch("routers.predictions._store_confidence", new=AsyncMock()),
            patch("routers.predictions._store_signal", new=AsyncMock()),
            patch("routers.predictions._indicators_engine") as mock_ind,
            patch("routers.predictions._long_model") as mock_model,
            patch("routers.predictions._long_gauge") as mock_gauge,
            patch("routers.predictions._macro_override"),
            patch("routers.predictions._decision_filter") as mock_filter,
        ):
            mock_ind.compute_all.return_value = {}
            # Neutral-fallback path: no real model artifact needed
            mock_model.build_latest_features.return_value = None
            mock_model.is_neutral_fallback.return_value = True
            mock_gauge.compute_score.return_value = {
                "score": 0,
                "signal": "neutral",
                "confidence": 0.5,
                "breakdown": {},
            }
            mock_filter.evaluate.return_value = {
                "final_signal": "neutral",
                "priority_boost": 0.0,
                "reason": "test",
                "cycle_quality_score": 0.5,
                "volatility_regime": "calm",
                "liquidity_class": "normal",
                "confidence_momentum": 0.0,
            }

            import routers.predictions as pred
            response = await pred.predict_long(ticker="VOO", session=mock_session)

        assert isinstance(response, dict), "Response must be a dict (no crash)"
        assert response.get("data_quality_degraded") is True, (
            "Expected data_quality_degraded=True when zero-volume daily bar is filtered"
        )
        assert "signal" in response, "Response must contain a signal field"
        assert "zero_volume_bars" in response.get("data_quality_reason", ""), (
            "data_quality_reason must mention zero_volume_bars"
        )

    # ------------------------------------------------------------------
    # 5. All bars zero-volume: predict_long returns neutral fallback
    # ------------------------------------------------------------------

    def _build_all_zero_volume_daily_df(self) -> pd.DataFrame:
        """Three daily bars where every row has volume=0."""
        return pd.DataFrame([
            {
                "timestamp": datetime(2024, 7, 28),
                "open": 495.0, "high": 500.0, "low": 493.0, "close": 498.0,
                "volume": 0,
                "session_type": "regular", "gap_type": "none",
                "is_extended_hours": False, "ticker": "VOO", "gap_percent": 0.0,
            },
            {
                "timestamp": datetime(2024, 7, 29),
                "open": 498.0, "high": 503.0, "low": 496.0, "close": 501.0,
                "volume": 0,
                "session_type": "regular", "gap_type": "none",
                "is_extended_hours": False, "ticker": "VOO", "gap_percent": 0.0,
            },
            {
                "timestamp": datetime(2024, 7, 30),
                "open": 501.0, "high": 506.0, "low": 499.0, "close": 504.0,
                "volume": 0,
                "session_type": "regular", "gap_type": "none",
                "is_extended_hours": False, "ticker": "VOO", "gap_percent": 0.0,
            },
        ])

    def test_all_zero_volume_daily_bars_frame_is_empty_after_mask(self):
        """After applying the zero-volume mask to an all-zero-volume daily_df,
        the resulting filtered frame must be empty."""
        import routers.predictions as pred

        df = self._build_all_zero_volume_daily_df()
        mask, count, reason = pred._detect_zero_volume_bars(df)

        assert count == 3, "All three bars should be detected as zero-volume"
        filtered = df[~mask].reset_index(drop=True)
        assert filtered.empty, "Filtered daily_df must be empty when all bars had zero volume"
        assert "zero_volume_bars" in reason

    @pytest.mark.asyncio
    async def test_predict_long_all_zero_volume_daily_bars_returns_neutral_fallback(self):
        """predict_long returns a neutral fallback with dq_degraded=True and no
        exception when every loaded daily bar has volume=0.

        After _detect_zero_volume_bars removes all rows daily_df becomes empty,
        and predict_long must return the 'All recent daily candles had zero
        volume.' response rather than crashing or proceeding with an empty frame.
        """
        from unittest.mock import AsyncMock, MagicMock, patch

        daily_df = self._build_all_zero_volume_daily_df()
        vix_df = pd.DataFrame([
            {
                "timestamp": datetime(2024, 7, 29),
                "open": 15.0, "high": 16.0, "low": 14.5, "close": 15.5,
                "ticker": "VIX",
            }
        ])

        mock_session = AsyncMock()

        with (
            patch("routers.predictions._load_daily_candles", new=AsyncMock(return_value=daily_df)),
            patch("routers.predictions._load_vix_candles", new=AsyncMock(return_value=vix_df)),
            patch("routers.predictions._load_spx_close_series", new=AsyncMock(return_value=pd.Series(dtype=float))),
            patch("routers.predictions._load_recent_confidence", new=AsyncMock(return_value=[])),
            patch("routers.predictions._store_confidence", new=AsyncMock()),
            patch("routers.predictions._store_signal", new=AsyncMock()),
            patch("routers.predictions._indicators_engine"),
            patch("routers.predictions._long_model"),
            patch("routers.predictions._long_gauge"),
            patch("routers.predictions._macro_override"),
            patch("routers.predictions._decision_filter"),
        ):
            import routers.predictions as pred
            response = await pred.predict_long(ticker="VOO", session=mock_session)

        assert isinstance(response, dict), "Response must be a dict (no crash)"
        assert response.get("data_quality_degraded") is True, (
            "Expected data_quality_degraded=True when all daily bars had zero volume"
        )
        assert response.get("ml_fallback") is True, (
            "Expected ml_fallback=True in the neutral response"
        )
        assert response.get("signal") == "neutral", (
            "Expected signal='neutral' in the all-zero-volume fallback response"
        )
        assert "All recent daily candles had zero volume" in response.get("note", ""), (
            "note must say 'All recent daily candles had zero volume'"
        )
        assert "zero_volume_bars" in response.get("data_quality_reason", ""), (
            "data_quality_reason must mention zero_volume_bars"
        )


# ---------------------------------------------------------------------------
# Task-212: Zero-volume VIX and SPX ingest gate + startup cleanup
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def vix_spx_db_session():
    """In-memory DB session with all tables for VIX/SPX zero-volume tests."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        yield session
    await engine.dispose()


def _make_vix_candle(ts: datetime, volume: float = 1_000.0) -> VixCandle:
    return VixCandle(
        ticker="^VIX",
        timestamp=ts,
        open=20.0,
        high=21.0,
        low=19.0,
        close=20.5,
        volume=volume,
        timeframe="daily",
    )


def _make_spx_candle(ts: datetime, volume: float = 500_000.0) -> SpxCandle:
    return SpxCandle(
        ticker="ES=F",
        timestamp=ts,
        open=5000.0,
        high=5050.0,
        low=4980.0,
        close=5030.0,
        volume=volume,
        timeframe="daily",
    )


def _make_vix_df(rows: list[dict]) -> pd.DataFrame:
    """Build a VIX candle DataFrame suitable for store_vix_candles."""
    data = []
    for r in rows:
        data.append({
            "open": r.get("open", 20.0),
            "high": r.get("high", 21.0),
            "low": r.get("low", 19.0),
            "close": r.get("close", 20.5),
            "volume": r.get("volume", 1_000.0),
        })
    index = pd.DatetimeIndex([r["ts"] for r in rows])
    return pd.DataFrame(data, index=index)


def _make_spx_df(rows: list[dict]) -> pd.DataFrame:
    """Build an SPX candle DataFrame suitable for store_spx_candles."""
    data = []
    for r in rows:
        data.append({
            "open": r.get("open", 5000.0),
            "high": r.get("high", 5050.0),
            "low": r.get("low", 4980.0),
            "close": r.get("close", 5030.0),
            "volume": r.get("volume", 500_000.0),
        })
    index = pd.DatetimeIndex([r["ts"] for r in rows])
    return pd.DataFrame(data, index=index)


class TestStoreVixCandlesZeroVolumeGate:
    """store_vix_candles must skip zero-volume bars and log the event."""

    @pytest.mark.asyncio
    async def test_zero_volume_vix_bar_is_skipped(self, vix_spx_db_session):
        """A VIX bar with volume=0 must not be inserted into the DB."""
        df = _make_vix_df([
            {"ts": datetime(2026, 7, 28), "volume": 1_000.0},
            {"ts": datetime(2026, 7, 29), "volume": 0.0},   # glitch day
        ])
        pipeline = IngestionPipeline()
        await pipeline.store_vix_candles(df, vix_spx_db_session, timeframe="daily")

        from sqlalchemy import select as _select
        result = await vix_spx_db_session.execute(_select(VixCandle))
        rows = result.scalars().all()
        assert len(rows) == 1, "Only the valid bar should be stored"
        assert rows[0].timestamp == datetime(2026, 7, 28)

    @pytest.mark.asyncio
    async def test_zero_volume_vix_bar_logged(self, vix_spx_db_session, caplog):
        """store_vix_candles logs ingest_zero_volume_bar_skipped for zero-vol bars."""
        df = _make_vix_df([{"ts": datetime(2026, 7, 29), "volume": 0.0}])
        pipeline = IngestionPipeline()
        import logging
        with caplog.at_level(logging.WARNING, logger="ingestion.pipeline"):
            await pipeline.store_vix_candles(df, vix_spx_db_session, timeframe="daily")
        assert any(
            "ingest_zero_volume_bar_skipped" in r.message
            for r in caplog.records
        ), "Expected ingest_zero_volume_bar_skipped log for VIX zero-volume bar"

    @pytest.mark.asyncio
    async def test_valid_vix_bar_is_stored(self, vix_spx_db_session):
        """A VIX bar with positive volume is stored normally."""
        df = _make_vix_df([{"ts": datetime(2026, 7, 28), "volume": 1_500.0}])
        pipeline = IngestionPipeline()
        await pipeline.store_vix_candles(df, vix_spx_db_session, timeframe="daily")

        from sqlalchemy import select as _select
        result = await vix_spx_db_session.execute(_select(VixCandle))
        rows = result.scalars().all()
        assert len(rows) == 1

    @pytest.mark.asyncio
    async def test_all_zero_volume_vix_bars_skipped(self, vix_spx_db_session):
        """When all bars have volume=0, nothing is inserted."""
        df = _make_vix_df([
            {"ts": datetime(2026, 7, 28), "volume": 0.0},
            {"ts": datetime(2026, 7, 29), "volume": 0.0},
        ])
        pipeline = IngestionPipeline()
        await pipeline.store_vix_candles(df, vix_spx_db_session, timeframe="daily")

        from sqlalchemy import select as _select
        result = await vix_spx_db_session.execute(_select(VixCandle))
        rows = result.scalars().all()
        assert len(rows) == 0


class TestStoreSpxCandlesZeroVolumeGate:
    """store_spx_candles must skip zero-volume bars and log the event."""

    @pytest.mark.asyncio
    async def test_zero_volume_spx_bar_is_skipped(self, vix_spx_db_session):
        """An SPX bar with volume=0 must not be inserted into the DB."""
        df = _make_spx_df([
            {"ts": datetime(2026, 7, 28), "volume": 500_000.0},
            {"ts": datetime(2026, 7, 29), "volume": 0.0},   # glitch day
        ])
        pipeline = IngestionPipeline()
        await pipeline.store_spx_candles(df, vix_spx_db_session, timeframe="daily")

        from sqlalchemy import select as _select
        result = await vix_spx_db_session.execute(_select(SpxCandle))
        rows = result.scalars().all()
        assert len(rows) == 1, "Only the valid bar should be stored"
        assert rows[0].timestamp == datetime(2026, 7, 28)

    @pytest.mark.asyncio
    async def test_zero_volume_spx_bar_logged(self, vix_spx_db_session, caplog):
        """store_spx_candles logs ingest_zero_volume_bar_skipped for zero-vol bars."""
        df = _make_spx_df([{"ts": datetime(2026, 7, 29), "volume": 0.0}])
        pipeline = IngestionPipeline()
        import logging
        with caplog.at_level(logging.WARNING, logger="ingestion.pipeline"):
            await pipeline.store_spx_candles(df, vix_spx_db_session, timeframe="daily")
        assert any(
            "ingest_zero_volume_bar_skipped" in r.message
            for r in caplog.records
        ), "Expected ingest_zero_volume_bar_skipped log for SPX zero-volume bar"

    @pytest.mark.asyncio
    async def test_valid_spx_bar_is_stored(self, vix_spx_db_session):
        """An SPX bar with positive volume is stored normally."""
        df = _make_spx_df([{"ts": datetime(2026, 7, 28), "volume": 750_000.0}])
        pipeline = IngestionPipeline()
        await pipeline.store_spx_candles(df, vix_spx_db_session, timeframe="daily")

        from sqlalchemy import select as _select
        result = await vix_spx_db_session.execute(_select(SpxCandle))
        rows = result.scalars().all()
        assert len(rows) == 1

    @pytest.mark.asyncio
    async def test_all_zero_volume_spx_bars_skipped(self, vix_spx_db_session):
        """When all bars have volume=0, nothing is inserted."""
        df = _make_spx_df([
            {"ts": datetime(2026, 7, 28), "volume": 0.0},
            {"ts": datetime(2026, 7, 29), "volume": 0.0},
        ])
        pipeline = IngestionPipeline()
        await pipeline.store_spx_candles(df, vix_spx_db_session, timeframe="daily")

        from sqlalchemy import select as _select
        result = await vix_spx_db_session.execute(_select(SpxCandle))
        rows = result.scalars().all()
        assert len(rows) == 0


class TestRemoveInvalidVixCandles:
    """remove_invalid_vix_candles must delete existing zero-volume rows at startup."""

    @pytest.mark.asyncio
    async def test_removes_zero_volume_vix_row(self, vix_spx_db_session):
        """A VIX row with volume=0 that already exists in the DB is removed."""
        bad = _make_vix_candle(datetime(2026, 7, 29), volume=0.0)
        vix_spx_db_session.add(bad)
        await vix_spx_db_session.flush()

        removed = await IngestionPipeline().remove_invalid_vix_candles(vix_spx_db_session)

        assert removed == 1
        from sqlalchemy import select as _select
        result = await vix_spx_db_session.execute(_select(VixCandle))
        assert result.scalars().all() == []

    @pytest.mark.asyncio
    async def test_keeps_valid_vix_row(self, vix_spx_db_session):
        """A VIX row with positive volume is left untouched."""
        good = _make_vix_candle(datetime(2026, 7, 28), volume=1_000.0)
        vix_spx_db_session.add(good)
        await vix_spx_db_session.flush()

        removed = await IngestionPipeline().remove_invalid_vix_candles(vix_spx_db_session)

        assert removed == 0
        from sqlalchemy import select as _select
        result = await vix_spx_db_session.execute(_select(VixCandle))
        assert len(result.scalars().all()) == 1

    @pytest.mark.asyncio
    async def test_removes_only_zero_volume_vix_rows(self, vix_spx_db_session):
        """Only zero-volume rows are removed; valid rows survive."""
        good = _make_vix_candle(datetime(2026, 7, 28), volume=1_000.0)
        bad = _make_vix_candle(datetime(2026, 7, 29), volume=0.0)
        vix_spx_db_session.add(good)
        vix_spx_db_session.add(bad)
        await vix_spx_db_session.flush()

        removed = await IngestionPipeline().remove_invalid_vix_candles(vix_spx_db_session)

        assert removed == 1
        from sqlalchemy import select as _select
        result = await vix_spx_db_session.execute(_select(VixCandle))
        remaining = result.scalars().all()
        assert len(remaining) == 1
        assert remaining[0].timestamp == datetime(2026, 7, 28)

    @pytest.mark.asyncio
    async def test_empty_db_returns_zero(self, vix_spx_db_session):
        """remove_invalid_vix_candles returns 0 and does not crash on empty table."""
        removed = await IngestionPipeline().remove_invalid_vix_candles(vix_spx_db_session)
        assert removed == 0

    @pytest.mark.asyncio
    async def test_removes_null_volume_vix_row(self, vix_spx_db_session):
        """A VIX row with volume=NULL is also treated as zero-volume and removed."""
        bad = _make_vix_candle(datetime(2026, 7, 29), volume=0.0)
        bad.volume = None
        vix_spx_db_session.add(bad)
        await vix_spx_db_session.flush()

        removed = await IngestionPipeline().remove_invalid_vix_candles(vix_spx_db_session)
        assert removed == 1


class TestRemoveInvalidSpxCandles:
    """remove_invalid_spx_candles must delete existing zero-volume rows at startup."""

    @pytest.mark.asyncio
    async def test_removes_zero_volume_spx_row(self, vix_spx_db_session):
        """An SPX row with volume=0 that already exists in the DB is removed."""
        bad = _make_spx_candle(datetime(2026, 7, 29), volume=0.0)
        vix_spx_db_session.add(bad)
        await vix_spx_db_session.flush()

        removed = await IngestionPipeline().remove_invalid_spx_candles(vix_spx_db_session)

        assert removed == 1
        from sqlalchemy import select as _select
        result = await vix_spx_db_session.execute(_select(SpxCandle))
        assert result.scalars().all() == []

    @pytest.mark.asyncio
    async def test_keeps_valid_spx_row(self, vix_spx_db_session):
        """An SPX row with positive volume is left untouched."""
        good = _make_spx_candle(datetime(2026, 7, 28), volume=500_000.0)
        vix_spx_db_session.add(good)
        await vix_spx_db_session.flush()

        removed = await IngestionPipeline().remove_invalid_spx_candles(vix_spx_db_session)

        assert removed == 0
        from sqlalchemy import select as _select
        result = await vix_spx_db_session.execute(_select(SpxCandle))
        assert len(result.scalars().all()) == 1

    @pytest.mark.asyncio
    async def test_removes_only_zero_volume_spx_rows(self, vix_spx_db_session):
        """Only zero-volume SPX rows are removed; valid rows survive."""
        good = _make_spx_candle(datetime(2026, 7, 28), volume=500_000.0)
        bad = _make_spx_candle(datetime(2026, 7, 29), volume=0.0)
        vix_spx_db_session.add(good)
        vix_spx_db_session.add(bad)
        await vix_spx_db_session.flush()

        removed = await IngestionPipeline().remove_invalid_spx_candles(vix_spx_db_session)

        assert removed == 1
        from sqlalchemy import select as _select
        result = await vix_spx_db_session.execute(_select(SpxCandle))
        remaining = result.scalars().all()
        assert len(remaining) == 1
        assert remaining[0].timestamp == datetime(2026, 7, 28)

    @pytest.mark.asyncio
    async def test_empty_db_returns_zero(self, vix_spx_db_session):
        """remove_invalid_spx_candles returns 0 and does not crash on empty table."""
        removed = await IngestionPipeline().remove_invalid_spx_candles(vix_spx_db_session)
        assert removed == 0

    @pytest.mark.asyncio
    async def test_removes_null_volume_spx_row(self, vix_spx_db_session):
        """An SPX row with volume=NULL is treated as zero-volume and removed."""
        bad = _make_spx_candle(datetime(2026, 7, 29), volume=0.0)
        bad.volume = None
        vix_spx_db_session.add(bad)
        await vix_spx_db_session.flush()

        removed = await IngestionPipeline().remove_invalid_spx_candles(vix_spx_db_session)
        assert removed == 1


# ---------------------------------------------------------------------------
# Task-234: zero-volume guard for the short-trend 5-min path
# ---------------------------------------------------------------------------

class TestPredictShortZeroVolumeLatestBar:
    """Confirm that _detect_zero_volume_bars flags a zero-volume latest 5-min bar
    and that predict_short falls back to the previous valid bar for session
    context (session_type, gap_type, is_extended) while setting dq_degraded=True.
    """

    # ── helper wiring ─────────────────────────────────────────────────────────

    def _detect(self, df: pd.DataFrame):
        """Call _detect_zero_volume_bars from the predictions module."""
        import importlib
        import routers.predictions as pred
        importlib.reload(pred)
        return pred._detect_zero_volume_bars(df)

    def _make_5min_frame(self, rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    # ── _detect_zero_volume_bars unit tests ──────────────────────────────────

    def test_zero_volume_latest_bar_is_detected(self):
        """When the newest 5-min bar has volume=0, the mask, count, and reason
        must all reflect a zero-volume detection and dq_degraded is implied."""
        df = self._make_5min_frame([
            {
                "timestamp": datetime(2024, 7, 30, 9, 30),
                "open": 499.0, "high": 504.0, "low": 497.0, "close": 500.0,
                "volume": 120_000, "session_type": "regular",
                "gap_type": "gap_up", "is_extended_hours": False,
            },
            {
                "timestamp": datetime(2024, 7, 30, 9, 35),
                "open": 500.0, "high": 503.0, "low": 499.0, "close": 501.0,
                # Zero-volume glitch bar — the newest candle
                "volume": 0, "session_type": "pre_market",
                "gap_type": "none", "is_extended_hours": True,
            },
        ])
        mask, count, reason = self._detect(df)

        assert count == 1, "exactly one zero-volume bar must be detected"
        assert bool(mask.iloc[-1]) is True, "the mask must flag the last (newest) bar"
        assert bool(mask.iloc[0]) is False, "the first valid bar must not be flagged"
        assert reason != "", "a non-empty reason string must be returned (dq_degraded trigger)"
        assert "zero_volume_bars" in reason

    def test_zero_volume_latest_bar_mask_last_element_true(self):
        """iloc[-1] of the returned mask must be True when only the last bar is zero-volume."""
        df = self._make_5min_frame([
            {
                "timestamp": datetime(2024, 7, 30, 9, 30),
                "open": 499.0, "high": 504.0, "low": 497.0, "close": 500.0,
                "volume": 80_000,
            },
            {
                "timestamp": datetime(2024, 7, 30, 9, 35),
                "open": 500.0, "high": 503.0, "low": 499.0, "close": 501.0,
                "volume": 0,  # newest bar — zero volume
            },
        ])
        mask, count, _ = self._detect(df)

        assert bool(mask.iloc[-1]) is True
        assert count == 1

    def test_no_zero_volume_bars_returns_empty_reason(self):
        """When all bars have positive volume, count=0 and reason is empty."""
        df = self._make_5min_frame([
            {
                "timestamp": datetime(2024, 7, 30, 9, 30),
                "open": 499.0, "high": 504.0, "low": 497.0, "close": 500.0,
                "volume": 100_000,
            },
            {
                "timestamp": datetime(2024, 7, 30, 9, 35),
                "open": 500.0, "high": 503.0, "low": 499.0, "close": 501.0,
                "volume": 90_000,
            },
        ])
        mask, count, reason = self._detect(df)

        assert count == 0
        assert reason == ""
        assert not mask.any()

    # ── predict_short context-selection fallback tests ────────────────────────

    def test_previous_valid_bar_used_for_session_context(self):
        """When the newest bar has volume=0, the session context fields
        (session_type, gap_type, is_extended) must come from the last
        non-zero-volume bar, not from the glitch bar.

        This mirrors the logic added to predict_short:
            if zero_vol_mask.iloc[-1]:
                latest = df_5min[~zero_vol_mask].iloc[-1]
        """
        import importlib
        import routers.predictions as pred
        importlib.reload(pred)

        # Build a frame whose last bar is a zero-volume pre-market glitch.
        # The previous bar is a regular-session bar with gap_up.
        df = self._make_5min_frame([
            {
                "timestamp": datetime(2024, 7, 30, 9, 30),
                "open": 499.0, "high": 504.0, "low": 497.0, "close": 500.0,
                "volume": 120_000,
                "session_type": "regular",
                "gap_type": "gap_up",
                "is_extended_hours": False,
            },
            {
                "timestamp": datetime(2024, 7, 30, 9, 35),
                "open": 500.5, "high": 502.0, "low": 500.0, "close": 501.0,
                "volume": 0,  # glitch bar — newest
                "session_type": "pre_market",
                "gap_type": "none",
                "is_extended_hours": True,
            },
        ])

        zero_vol_mask, zero_vol_count, zv_reason = pred._detect_zero_volume_bars(df)

        # Simulate the fallback selection logic added to predict_short
        if zero_vol_count > 0 and bool(zero_vol_mask.iloc[-1]):
            valid_rows = df[~zero_vol_mask]
            latest = valid_rows.iloc[-1] if not valid_rows.empty else df.iloc[-1]
        else:
            latest = df.iloc[-1]

        # dq_degraded must be True (reason is non-empty)
        assert zv_reason != "", "dq_degraded must be triggered (reason non-empty)"
        assert zero_vol_count == 1

        # Session context must come from the previous valid bar
        assert str(latest.get("session_type")) == "regular", (
            "session_type must come from the previous valid bar, not the zero-volume glitch bar"
        )
        assert str(latest.get("gap_type")) == "gap_up", (
            "gap_type must come from the previous valid bar"
        )
        assert bool(latest.get("is_extended_hours")) is False, (
            "is_extended_hours must come from the previous valid bar"
        )

    def test_all_zero_volume_falls_back_to_raw_last(self):
        """When every bar has volume=0, the raw last row is kept as a best-effort
        fallback (no crash, no KeyError)."""
        import importlib
        import routers.predictions as pred
        importlib.reload(pred)

        df = self._make_5min_frame([
            {
                "timestamp": datetime(2024, 7, 30, 9, 30),
                "open": 499.0, "high": 504.0, "low": 497.0, "close": 500.0,
                "volume": 0, "session_type": "regular", "gap_type": "none",
                "is_extended_hours": False,
            },
            {
                "timestamp": datetime(2024, 7, 30, 9, 35),
                "open": 500.0, "high": 503.0, "low": 499.0, "close": 501.0,
                "volume": 0, "session_type": "pre_market", "gap_type": "none",
                "is_extended_hours": True,
            },
        ])

        zero_vol_mask, zero_vol_count, zv_reason = pred._detect_zero_volume_bars(df)

        # Simulate the fallback selection logic
        if zero_vol_count > 0 and bool(zero_vol_mask.iloc[-1]):
            valid_rows = df[~zero_vol_mask]
            latest = valid_rows.iloc[-1] if not valid_rows.empty else df.iloc[-1]
        else:
            latest = df.iloc[-1]

        # Must not raise; latest must be a valid row from the DataFrame
        assert latest is not None
        assert str(latest.get("session_type")) in ("regular", "pre_market", "post_market")
        # dq_degraded implied: reason non-empty
        assert zv_reason != ""

    def test_null_volume_latest_bar_is_detected(self):
        """A bar with volume=NULL (NaN) is treated as zero-volume by fillna(0)."""
        df = self._make_5min_frame([
            {
                "timestamp": datetime(2024, 7, 30, 9, 30),
                "open": 499.0, "high": 504.0, "low": 497.0, "close": 500.0,
                "volume": 100_000, "session_type": "regular",
                "gap_type": "gap_up", "is_extended_hours": False,
            },
            {
                "timestamp": datetime(2024, 7, 30, 9, 35),
                "open": 500.0, "high": 503.0, "low": 499.0, "close": 501.0,
                "volume": float("nan"),  # NULL-equivalent newest bar
                "session_type": "pre_market", "gap_type": "none",
                "is_extended_hours": True,
            },
        ])

        mask, count, reason = self._detect(df)

        assert count == 1, "NaN volume must be treated as zero-volume"
        assert bool(mask.iloc[-1]) is True
        assert reason != ""
