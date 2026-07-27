---
name: FastAPI middleware ordering vs CORS
description: Decorator-registered @app.middleware("http") wraps OUTSIDE earlier add_middleware(CORSMiddleware); responses it short-circuits lack CORS headers.
---

`app.add_middleware` prepends to the stack, so a middleware registered later (including via the `@app.middleware("http")` decorator) runs *outside* CORSMiddleware. Any response such an outer middleware returns directly (e.g. a 504 timeout guard) ships **without** CORS headers, so browsers cannot read it cross-origin.

**Why:** the request-timeout guard in the NovaCycle backend returned a structured 504 that browser clients couldn't read until CORS headers were added manually.

**How to apply:** when an outer middleware short-circuits with its own response, mirror the app's CORS policy on that response by hand (e.g. set `Access-Control-Allow-Origin` when the request has an `Origin` header), and keep a regression test asserting CORS headers on that path.
