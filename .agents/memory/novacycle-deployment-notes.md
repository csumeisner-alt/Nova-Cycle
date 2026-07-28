---
name: NovaCycle deployment notes
description: How the NovaCycle backend must be published so the Android app can reach it
---

## Backend must be a Reserved VM, not autoscale

The NovaCycle backend uses two things that break on the default `autoscale` deployment target:

1. **SQLite on the local filesystem** — autoscale has an ephemeral filesystem, so the database disappears between instances.
2. **APScheduler running inside the app** — autoscale does not keep background cron-like jobs alive; it is request-driven.

**Why:** Replit autoscale is designed for stateless request/response services. The NovaCycle backend is a long-running stateful service with periodic ingestion.

**How to apply:** When publishing the backend artifact, choose **Reserved VM** (`vm`) as the deployment type. Only the user can select this in the Publishing pane; the agent cannot set it programmatically.

## Publishing checklist

- Confirm the backend is committed and the dev workflow is healthy.
- Ensure the artifact `production.health.startup` probe path exists (`/api/healthz` is implemented and mounted under the `/api` prefix).
- Start the background ingestion task quickly during lifespan so the startup probe passes before ingestion finishes.
- The seeded SQLite database (`artifacts/api-server/backend/novacycle.db`) is tracked in git so the production VM has historical data on first boot.
- After publishing, read `getDeploymentInfo()` for the production URL — never guess from env vars or dev domains.
- Update `android/app/build.gradle.kts` `buildConfigField("String", "API_BASE_URL", ...)` with the production URL and rebuild the APK.

## Artifact proxy readiness probe

The artifact deployment proxy may probe `/api` during startup even when the configured health path is `/api/healthz`. Keep `/api` as a cheap, dependency-free 200 response so startup probing cannot fail before the database and ingestion tasks finish.

**Why:** A deployment can start the FastAPI process successfully but still report repeated health-check 500/404 failures when `/api` has no route; this makes the published service appear down during promotion.

**How to apply:** Preserve the lightweight `/api` route alongside the detailed `/api/healthz` route. Do not make `/api` perform database queries or model loading.
