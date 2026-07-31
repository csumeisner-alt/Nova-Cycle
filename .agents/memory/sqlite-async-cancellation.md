---
name: SQLite async cancellation in tests
description: Cancelling an aiosqlite query mid-flight invalidates the pooled connection; :memory: DBs vanish with it
---

Cancelling a coroutine mid-query (e.g. `asyncio.wait_for` timeout tests against real DB work) invalidates the pooled aiosqlite connection. With a `sqlite+aiosqlite:///:memory:` engine the entire database lives on that one connection, so subsequent sessions see "no such table".

**Why:** hit while load-testing the cleanup timeout guard — the post-timeout row-count check failed with a missing table.

**How to apply:** any test that cancels real DB work and then re-queries must use a file-based SQLite DB (`tmp_path`), not `:memory:`.

Baseline perf note: the batched Python-loop OHLC cleanup scans ~50k rows/s on Replit SQLite, so the 300 s timeout covers ~15M rows.
