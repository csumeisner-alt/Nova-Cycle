"""Tests for the /api/model_performance endpoint and performance_engine."""

from datetime import datetime, timedelta

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from database.db import get_session
from database.models import Base, ModelMetadata, SignalHistory, VooCandle
from main import app
from performance_engine import (
    bucket_cycles_by_period,
    compute_calibration_curve,
    compute_confidence_buckets,
    compute_cumulative_pnl,
    compute_streaks,
    filter_cycles_by_confidence,
    find_missed_rallies_in_candles,
    get_model_performance,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pure-helper tests
# ─────────────────────────────────────────────────────────────────────────────

def _cycle(buy_ts, sell_ts, ret, conf):
    return {
        "buy_timestamp": buy_ts,
        "sell_timestamp": sell_ts,
        "return_percent": ret,
        "confidence_at_buy": conf,
        "session_type_at_buy": "regular",
    }


BASE = datetime(2026, 7, 1, 10, 0)


def _cycles_fixture():
    return [
        _cycle(BASE, BASE + timedelta(hours=1), 1.0, 0.85),
        _cycle(BASE + timedelta(days=1), BASE + timedelta(days=1, hours=1), -0.5, 0.55),
        _cycle(BASE + timedelta(days=8), BASE + timedelta(days=8, hours=1), 2.0, 0.92),
        _cycle(BASE + timedelta(days=40), BASE + timedelta(days=40, hours=1), 0.7, 0.30),
    ]


class TestConfidenceFiltering:
    def test_no_filter_returns_all(self):
        cycles = _cycles_fixture()
        assert filter_cycles_by_confidence(cycles, None, None) == cycles

    def test_high_band(self):
        out = filter_cycles_by_confidence(_cycles_fixture(), 0.7, 1.0)
        assert len(out) == 2
        assert all(c["confidence_at_buy"] >= 0.7 for c in out)

    def test_band_with_no_matches(self):
        out = filter_cycles_by_confidence(_cycles_fixture(), 0.0, 0.1)
        assert out == []

    def test_band_is_half_open_except_top(self):
        cycles = [
            _cycle(BASE, BASE, 1.0, 0.4),
            _cycle(BASE, BASE, 1.0, 0.7),
            _cycle(BASE, BASE, 1.0, 1.0),
        ]
        medium = filter_cycles_by_confidence(cycles, 0.4, 0.7)
        assert [c["confidence_at_buy"] for c in medium] == [0.4]
        high = filter_cycles_by_confidence(cycles, 0.7, 1.0)
        assert [c["confidence_at_buy"] for c in high] == [0.7, 1.0]

    def test_cycle_without_confidence_excluded_when_filtering(self):
        cycles = [_cycle(BASE, BASE, 1.0, None)]
        assert filter_cycles_by_confidence(cycles, 0.0, 1.0) == []


class TestConfidenceBuckets:
    def test_buckets_partition_cycles(self):
        buckets = compute_confidence_buckets(_cycles_fixture())
        assert set(buckets) == {"low", "medium", "high"}
        assert buckets["low"]["trade_count"] == 1
        assert buckets["medium"]["trade_count"] == 1
        assert buckets["high"]["trade_count"] == 2
        assert buckets["high"]["win_rate"] == 1.0

    def test_empty(self):
        buckets = compute_confidence_buckets([])
        for b in buckets.values():
            assert b == {"trade_count": 0, "win_rate": 0.0, "avg_return_percent": 0.0}


class TestCalibrationCurve:
    def test_ten_points_always(self):
        curve = compute_calibration_curve(_cycles_fixture())
        assert len(curve) == 10
        mids = [p["confidence_mid"] for p in curve]
        assert mids == [round(0.05 + 0.1 * i, 2) for i in range(10)]

    def test_empty_buckets_have_null_win_rate(self):
        curve = compute_calibration_curve([])
        assert all(p["actual_win_rate"] is None and p["trade_count"] == 0 for p in curve)

    def test_win_rate_per_bucket(self):
        cycles = [
            _cycle(BASE, BASE, 1.0, 0.85),
            _cycle(BASE, BASE, -1.0, 0.86),
        ]
        curve = compute_calibration_curve(cycles)
        bucket = next(p for p in curve if p["confidence_mid"] == 0.85)
        assert bucket["trade_count"] == 2
        assert bucket["actual_win_rate"] == 0.5

    def test_confidence_of_exactly_one_lands_in_top_bucket(self):
        curve = compute_calibration_curve([_cycle(BASE, BASE, 1.0, 1.0)])
        assert curve[9]["trade_count"] == 1


class TestCumulativePnlAndStreaks:
    def test_compounding(self):
        cycles = [
            _cycle(BASE, BASE + timedelta(hours=1), 10.0, 0.8),
            _cycle(BASE, BASE + timedelta(hours=2), 10.0, 0.8),
        ]
        pnl = compute_cumulative_pnl(cycles)
        assert len(pnl) == 2
        assert pnl[-1]["cumulative_return_percent"] == pytest.approx(21.0)

    def test_empty(self):
        assert compute_cumulative_pnl([]) == []
        assert compute_streaks([]) == {
            "current_win": 0, "current_loss": 0,
            "longest_win": 0, "longest_loss": 0,
        }

    def test_streaks_all_wins(self):
        cycles = [_cycle(BASE, BASE + timedelta(hours=i), 1.0, 0.8) for i in range(3)]
        s = compute_streaks(cycles)
        assert s == {"current_win": 3, "current_loss": 0,
                     "longest_win": 3, "longest_loss": 0}

    def test_streaks_all_losses(self):
        cycles = [_cycle(BASE, BASE + timedelta(hours=i), -1.0, 0.8) for i in range(2)]
        s = compute_streaks(cycles)
        assert s == {"current_win": 0, "current_loss": 2,
                     "longest_win": 0, "longest_loss": 2}

    def test_streaks_mixed(self):
        rets = [1.0, 1.0, -1.0, 1.0]
        cycles = [
            _cycle(BASE, BASE + timedelta(hours=i), r, 0.8)
            for i, r in enumerate(rets)
        ]
        s = compute_streaks(cycles)
        assert s["longest_win"] == 2
        assert s["current_win"] == 1
        assert s["longest_loss"] == 1


class TestPeriodBucketing:
    def test_day_buckets(self):
        periods = bucket_cycles_by_period(_cycles_fixture(), "day")
        assert [p["label"] for p in periods] == [
            "2026-07-01", "2026-07-02", "2026-07-09", "2026-08-10",
        ]
        assert periods[0]["buy_count"] == 1
        assert periods[0]["wins"] == 1

    def test_week_buckets_iso_boundaries(self):
        # 2026-07-01 is a Wednesday (ISO week 27); 2026-07-06 is Monday (week 28)
        cycles = [
            _cycle(datetime(2026, 7, 5, 10), datetime(2026, 7, 5, 11), 1.0, 0.8),
            _cycle(datetime(2026, 7, 6, 10), datetime(2026, 7, 6, 11), 1.0, 0.8),
        ]
        periods = bucket_cycles_by_period(cycles, "week")
        assert [p["label"] for p in periods] == ["2026-W27", "2026-W28"]

    def test_month_buckets(self):
        periods = bucket_cycles_by_period(_cycles_fixture(), "month")
        assert [p["label"] for p in periods] == ["2026-07", "2026-08"]
        assert periods[0]["buy_count"] == 3

    def test_missed_rallies_assigned_to_bucket(self):
        periods = bucket_cycles_by_period(
            _cycles_fixture(), "day",
            missed_rally_timestamps=[BASE + timedelta(minutes=30)],
        )
        assert periods[0]["missed_rallies"] == 1
        assert periods[1]["missed_rallies"] == 0

    def test_oos_accuracy_nearest_at_or_before(self):
        history = [
            {"trained_at": (BASE - timedelta(days=5)).isoformat(), "accuracy": 0.6},
            {"trained_at": (BASE + timedelta(days=5)).isoformat(), "accuracy": 0.7},
        ]
        periods = bucket_cycles_by_period(_cycles_fixture(), "day",
                                          accuracy_history=history)
        assert periods[0]["oos_accuracy"] == 0.6   # 2026-07-01
        assert periods[2]["oos_accuracy"] == 0.7   # 2026-07-09

    def test_empty(self):
        assert bucket_cycles_by_period([], "day") == []


class TestMissedRallyScan:
    def test_rally_found(self):
        ts0 = BASE
        candles = [(ts0 + timedelta(minutes=5 * i), 100.0) for i in range(3)]
        candles.append((ts0 + timedelta(minutes=20), 100.5))  # +0.5% within 12 bars
        assert find_missed_rallies_in_candles(candles) == ts0

    def test_flat_gap_skipped(self):
        candles = [(BASE + timedelta(minutes=5 * i), 100.0 + 0.01 * i) for i in range(20)]
        # max rise within any 12-bar span is ~0.12% < 0.3%
        assert find_missed_rallies_in_candles(candles) is None

    def test_rise_beyond_12_bars_not_counted(self):
        candles = [(BASE + timedelta(minutes=5 * i), 100.0) for i in range(13)]
        candles.append((BASE + timedelta(minutes=5 * 13), 101.0))
        # first base candle sees the rise only at bar 13 (> 12 bars ahead)…
        # but candle at index 1 sees it within 12 bars, so it still fires there.
        hit = find_missed_rallies_in_candles(candles)
        assert hit == candles[1][0]

    def test_empty(self):
        assert find_missed_rallies_in_candles([]) is None


# ─────────────────────────────────────────────────────────────────────────────
# DB-backed integration tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def db():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    yield factory
    await engine.dispose()


def _signal(ts, sig_type, conf=0.85):
    return SignalHistory(
        ticker="VOO", timestamp=ts, signal_type=sig_type, gauge_type="short",
        confidence=conf, session_type="regular", is_extended_hours=False,
        gap_type="none", liquidity_score=1.0, macro_override_applied=False,
    )


def _candle5(ts, close):
    return VooCandle(
        ticker="VOO", timestamp=ts, open=close, high=close + 0.1,
        low=close - 0.1, close=close, volume=1e6, timeframe="5min",
        is_extended_hours=False, session_type="regular",
        gap_percent=0.0, gap_type="none",
    )


def _recent(days_ago, hour=10, minute=0):
    now = datetime.utcnow()
    return datetime(now.year, now.month, now.day, hour, minute) - timedelta(days=days_ago)


class TestGetModelPerformance:
    async def test_empty_database_returns_safe_shapes(self, db):
        async with db() as session:
            result = await get_model_performance(session)
        assert result["summary"]["total_trades"] == 0
        assert result["summary"]["buy_precision"] == 0.0
        assert result["periods"] == []
        assert result["cumulative_pnl"] == []
        assert result["best_trade"] is None
        assert result["worst_trade"] is None
        assert len(result["calibration_curve"]) == 10
        assert result["missed_rallies"]["count"] == 0
        assert result["missed_rallies"]["rate"] == 0.0
        assert result["summary"]["missed_rally_rate"] == 0.0
        assert result["accuracy_history"] == []

    async def test_happy_path_multiple_cycles(self, db):
        async with db() as session:
            # Two BUY→SELL cycles + candles so returns are real
            signals = [
                _signal(_recent(10), "buy", 0.9),
                _signal(_recent(9), "sell", 0.8),
                _signal(_recent(5), "buy", 0.6),
                _signal(_recent(4), "sell", 0.7),
            ]
            candles = [
                _candle5(_recent(10, 9, 55), 100.0),
                _candle5(_recent(9, 9, 55), 102.0),   # +2% win
                _candle5(_recent(5, 9, 55), 100.0),
                _candle5(_recent(4, 9, 55), 99.0),    # -1% loss
            ]
            session.add_all(signals + candles)
            session.add(ModelMetadata(
                model_name="short_trend", ticker="VOO",
                trained_at=_recent(20), accuracy=0.82,
            ))
            await session.commit()
            result = await get_model_performance(session, window="30d")

        s = result["summary"]
        assert s["total_trades"] == 2
        assert s["wins"] == 1 and s["losses"] == 1
        assert s["buy_precision"] == 0.5
        assert len(result["periods"]) == 2
        assert result["best_trade"]["return_percent"] == pytest.approx(2.0)
        assert result["worst_trade"]["return_percent"] == pytest.approx(-1.0)
        assert result["accuracy_history"][0]["accuracy"] == 0.82
        assert isinstance(result["best_trade"]["buy_timestamp"], str)
        assert result["streak"]["longest_win"] == 1

    async def test_single_cycle(self, db):
        async with db() as session:
            session.add_all([
                _signal(_recent(3), "buy", 0.75),
                _signal(_recent(2), "sell", 0.7),
                _candle5(_recent(3, 9, 55), 100.0),
                _candle5(_recent(2, 9, 55), 101.0),
            ])
            await session.commit()
            result = await get_model_performance(session, window="30d")
        assert result["summary"]["total_trades"] == 1
        assert result["summary"]["wins"] == 1
        assert result["confidence_buckets"]["high"]["trade_count"] == 1

    async def test_confidence_filter_no_matches(self, db):
        async with db() as session:
            session.add_all([
                _signal(_recent(3), "buy", 0.9),
                _signal(_recent(2), "sell", 0.9),
                _candle5(_recent(3, 9, 55), 100.0),
                _candle5(_recent(2, 9, 55), 101.0),
            ])
            await session.commit()
            result = await get_model_performance(
                session, window="30d", confidence_min=0.0, confidence_max=0.1,
            )
        assert result["summary"]["total_trades"] == 0
        assert result["periods"] == []
        assert result["best_trade"] is None

    async def test_missed_rally_detected_in_hold_gap(self, db):
        async with db() as session:
            # SELL then a later BUY, with a >0.3% rise inside the gap
            sell_ts = _recent(6)
            buy_ts = _recent(4)
            session.add_all([
                _signal(_recent(7), "buy", 0.8),
                _signal(sell_ts, "sell", 0.8),
                _signal(buy_ts, "buy", 0.8),
                _signal(_recent(3), "sell", 0.8),
            ])
            # candles inside the SELL→BUY gap rising 1%
            gap_start = sell_ts + timedelta(hours=1)
            session.add_all([
                _candle5(gap_start + timedelta(minutes=5 * i), 100.0 + 0.2 * i)
                for i in range(6)
            ])
            await session.commit()
            result = await get_model_performance(session, window="30d")
        assert result["missed_rallies"]["count"] >= 1
        assert result["missed_rallies"]["timestamps"]

    async def test_flat_hold_gap_not_counted(self, db):
        async with db() as session:
            sell_ts = _recent(6)
            session.add_all([
                _signal(_recent(7), "buy", 0.8),
                _signal(sell_ts, "sell", 0.8),
            ])
            gap_start = sell_ts + timedelta(hours=1)
            session.add_all([
                _candle5(gap_start + timedelta(minutes=5 * i), 100.0)
                for i in range(6)
            ])
            await session.commit()
            result = await get_model_performance(session, window="30d")
        assert result["missed_rallies"]["count"] == 0


class TestEndpoint:
    @pytest_asyncio.fixture
    async def client(self, db):
        async def override():
            async with db() as session:
                yield session
        app.dependency_overrides[get_session] = override
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c
        app.dependency_overrides.pop(get_session, None)

    async def test_endpoint_empty_db_200(self, client):
        resp = await client.get("/api/model_performance")
        assert resp.status_code == 200
        body = resp.json()
        assert body["summary"]["total_trades"] == 0
        assert len(body["calibration_curve"]) == 10

    async def test_endpoint_rejects_bad_period(self, client):
        resp = await client.get("/api/model_performance?period=year")
        assert resp.status_code == 400

    async def test_endpoint_rejects_bad_ticker(self, client):
        resp = await client.get("/api/model_performance?ticker=SPY")
        assert resp.status_code == 400

    async def test_endpoint_rejects_out_of_range_confidence(self, client):
        resp = await client.get("/api/model_performance?confidence_min=1.5")
        assert resp.status_code == 422
