"""Retry-interval tests: after a failed/rolled-back retrain the trainer must
retry after a short interval instead of the full weekly cadence, and a
successful retrain restores the weekly cadence."""

import json
from datetime import datetime, timedelta

import pytest

from ml import trainer as trainer_mod
from ml import training_status as ts
from ml.trainer import ModelTrainer, _RETRAIN_INTERVAL_DAYS, _FAILED_RETRAIN_INTERVAL_DAYS
from ml.training_status import (
    any_model_failed_last_attempt,
    record_training_result,
)


@pytest.fixture
def isolated_status(tmp_path, monkeypatch):
    monkeypatch.setattr(ts, "STATUS_PATH", tmp_path / "training_status.json")


def test_no_status_file_means_no_failure(isolated_status):
    assert any_model_failed_last_attempt() is False


def test_failed_attempt_detected(isolated_status):
    record_training_result("long_trend", success=False, error="regression rolled back")
    assert any_model_failed_last_attempt() is True


def test_success_clears_failure(isolated_status):
    record_training_result("long_trend", success=False, error="boom")
    record_training_result("long_trend", success=True, accuracy=0.7)
    assert any_model_failed_last_attempt() is False


def test_corrupt_status_file_is_not_failure(isolated_status):
    ts.STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts.STATUS_PATH.write_text("{not json")
    assert any_model_failed_last_attempt() is False


class _FakeSession:
    pass


@pytest.fixture
def trainer(monkeypatch):
    t = ModelTrainer.__new__(ModelTrainer)  # skip heavy __init__
    monkeypatch.setattr(ModelTrainer, "_missing_model_files", staticmethod(lambda: []))

    calls = {"trained": 0}

    async def fake_train(self, db):
        calls["trained"] += 1

    monkeypatch.setattr(ModelTrainer, "run_initial_training", fake_train)
    return t, calls


def _set_last_trained(monkeypatch, days_ago):
    async def fake_last(db):
        return datetime.utcnow() - timedelta(days=days_ago)

    monkeypatch.setattr(
        ModelTrainer, "_get_last_trained", staticmethod(fake_last)
    )


@pytest.mark.asyncio
async def test_failed_attempt_uses_short_interval(isolated_status, trainer, monkeypatch):
    t, calls = trainer
    record_training_result("short_trend", success=False, error="rolled back")
    _set_last_trained(monkeypatch, days_ago=_FAILED_RETRAIN_INTERVAL_DAYS + 1)

    assert await t.retrain_if_needed(_FakeSession()) is True
    assert calls["trained"] == 1


@pytest.mark.asyncio
async def test_success_keeps_weekly_cadence(isolated_status, trainer, monkeypatch):
    t, calls = trainer
    record_training_result("long_trend", success=True, accuracy=0.7)
    record_training_result("short_trend", success=True, accuracy=0.7)
    _set_last_trained(monkeypatch, days_ago=2)

    assert await t.retrain_if_needed(_FakeSession()) is False
    assert calls["trained"] == 0

    _set_last_trained(monkeypatch, days_ago=_RETRAIN_INTERVAL_DAYS)
    assert await t.retrain_if_needed(_FakeSession()) is True
    assert calls["trained"] == 1


@pytest.mark.asyncio
async def test_recent_failure_does_not_retrain_same_day(isolated_status, trainer, monkeypatch):
    t, calls = trainer
    record_training_result("long_trend", success=False, error="rolled back")
    _set_last_trained(monkeypatch, days_ago=0)

    assert await t.retrain_if_needed(_FakeSession()) is False
    assert calls["trained"] == 0
