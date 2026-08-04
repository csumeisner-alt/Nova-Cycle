---
name: Calibration report staleness & test isolation
description: Stale-flag contract for calibration reports and the test-fixture pattern that prevents suite runs from overwriting real ml/models files
---

## Stale-flag contract

- Every `save_calibration_report` write sets `"stale": false` and removes `stale_note` / `marked_stale_at`. A missing key in older reports is treated as false (backward compat).
- `mark_calibration_report_stale(model_name, note, dataset_meta=None)` sets `"stale": true` with a UTC timestamp and an optional human-readable note + dataset block. It is idempotent (re-call when counts haven't changed → no file write).
- `is_calibration_report_stale(model_name)` returns the bool; missing file → False.
- Reports also self-describe via a `"dataset"` key: `total_candles`, `labeled_rows`, `date_start`, `date_end`. `save_calibration_report` writes it when `dataset_meta` is supplied; loading code uses `.get("dataset")`.

## Production call site

`audit_calibration_report_staleness(db_session, model_name="long_trend")` in `ml/trainer.py`:
- Called from `retrain_if_needed` whenever retraining is **skipped** (models up-to-date).
- Queries current VOO daily candle counts via `_fetch_daily_candle_meta`.
- Only marks stale when: report has `evaluated=False`, DB has ≥2× the report's labeled-row count, and DB has ≥500 labeled rows.
- Idempotent: skips re-write when the report is already stale with the exact current counts.
- Never raises; any error is logged and swallowed.

## Test-fixture rule

Any fixture that redirects model `.pkl` paths **must also** patch:
```python
monkeypatch.setattr(cal, "MODEL_DIR", tmp_path)
monkeypatch.setattr(cal, "CALIBRATOR_PATH", tmp_path / "long_trend_calibrator.pkl")
monkeypatch.setattr(cal, "REPORT_PATH", tmp_path / "long_trend_calibration.json")
monkeypatch.setattr(lt, "MODEL_DIR", tmp_path)
monkeypatch.setattr(st, "MODEL_DIR", tmp_path)
```
Without this, `train()` and `_sidecar_files` write/backup calibration files straight into the real `ml/models/` directory, silently corrupting the on-disk report after every suite run.

Use `calibration_report_path(model_name)` and `calibrator_path(model_name)` (functions) rather than `CALIBRATOR_PATH` / `REPORT_PATH` (constants) in any production path — functions always respect the current `MODEL_DIR` after a monkeypatch.
