---
name: NovaCycle backend fixes
description: Runtime issues discovered when booting the FastAPI backend on Replit and how they were fixed
---

## DATABASE_URL env var conflict
Replit injects a PostgreSQL `DATABASE_URL` env var that overrides pydantic-settings defaults.
**Fix:** Hardcode the SQLite URL directly in `database/db.py` — do NOT read it from `settings.DATABASE_URL`.
```python
_SQLITE_URL = "sqlite+aiosqlite:///./novacycle.db"
async_engine = create_async_engine(_SQLITE_URL, ...)
```
**Why:** pydantic-settings env vars take precedence over field defaults; Replit's DATABASE_URL points at PostgreSQL and psycopg2 is not installed.

## Missing db.py aliases
`main.py` imports `create_tables` and `get_session_factory`; `routers/predictions.py` imports `get_session`.
`db.py` only defines `init_db`, `AsyncSessionLocal`, and `get_db`.
**Fix:** Add explicit aliases/wrappers in `db.py`:
- `create_tables = init_db`
- `def get_session_factory(): return AsyncSessionLocal`
- `async def get_session()` — a generator wrapping `AsyncSessionLocal()`

## TensorFlow → scikit-learn MLPClassifier
TensorFlow is too large for Replit's pip install timeout (workflow times out at 270s before port opens).
**Fix:** Replace `ml/short_trend.py` with scikit-learn `MLPClassifier(hidden_layer_sizes=(128,64,32))`.
Model persisted as `.pkl` (model + StandardScaler dict). Functionally equivalent; much faster to install.
**Why:** `tensorflow==2.16.1` pip install alone exceeds the 270s workflow startup timeout.

## pip not in PATH on fresh Replit NixOS
`pip: command not found` because Python wasn't installed yet.
**Fix:** Use `installProgrammingLanguage({ language: "python-3.11" })` then `installLanguagePackages(...)` via the package-management skill callbacks. After that, the workflow's `pip install -r requirements.txt` works.

## Android screen files not persisted (parallel WriteFile race)
When writing many files in parallel via WriteFile and immediately running `find` in a parallel ShellExec, some files may not appear in the listing (write not yet flushed). Always re-check with a separate find after writes complete.
