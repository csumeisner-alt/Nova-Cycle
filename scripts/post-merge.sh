#!/bin/bash
set -e

# Sync pnpm lockfile with any workspace config changes (overrides, catalog, etc.)
pnpm install --no-frozen-lockfile
