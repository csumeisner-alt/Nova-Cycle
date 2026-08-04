# [Project name]

_Replace the heading above with the project's name, and this line with one sentence describing what this app does for users._

## Run & Operate

- `pnpm --filter @workspace/api-server run dev` — run the API server (port 5000)
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- Required env: `DATABASE_URL` — Postgres connection string

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- API: Express 5
- DB: PostgreSQL + Drizzle ORM
- Validation: Zod (`zod/v4`), `drizzle-zod`
- API codegen: Orval (from OpenAPI spec)
- Build: esbuild (CJS bundle)

## Where things live

_Populate as you build — short repo map plus pointers to the source-of-truth file for DB schema, API contracts, theme files, etc._

## Architecture decisions

_Populate as you build — non-obvious choices a reader couldn't infer from the code (3-5 bullets)._

## Product

_Describe the high-level user-facing capabilities of this app once they exist._

## User preferences

- When the user asks for the newest Android APK, build and provide the installable **signed release APK**, never a debug APK.
- Deliver APKs as a clear, usable clickable download link. Do not present only a file attachment/card or ask the user to find the artifact path.
- If a signed release APK cannot be built or retrieved, explain the exact blocker before offering any alternative; do not silently substitute a debug APK.
- Treat the established CI-published release artifact as the correct APK delivery path. A locally debug-signed APK is not an acceptable substitute because it may not update an existing release-signed installation.

## Gotchas

- Android APK delivery must use the CI/release signing path; local debug builds are for internal verification only and must not be delivered as the requested APK.

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
