"""Regression coverage for malformed vendor OHLC rows and repairs."""

from datetime import datetime

import pandas as pd
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from database.models import Base, VooCandle
from ingestion.fetcher import DataFetcher, ohlc_validation_issue
from ingestion.pipeline import IngestionPipeline
from ingestion.ohlc_validator import filter_valid_ohlc, validate_ohlc_row


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
