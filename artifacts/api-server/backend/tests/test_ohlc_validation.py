"""Regression coverage for malformed vendor OHLC rows and repairs."""

from datetime import datetime, timedelta

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

    def test_internally_valid_spike_is_not_quarantined(self):
        """A bar that closes +15 % above neighbours but is internally valid
        (high >= close) passes through without degraded=True."""
        base_close = 500.0
        spike_close = base_close * 1.15  # +15 % — plausible data glitch

        rows = [
            {
                "timestamp": datetime(2024, 7, 30, 9, 30),
                "open": 499.0, "high": 504.0, "low": 497.0, "close": base_close,
            },
            {
                # Spike bar: high keeps pace with close → internally valid
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
        # Internally valid spike: not quarantined
        assert not degraded
        assert len(clean) == 3

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
