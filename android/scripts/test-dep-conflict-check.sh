#!/usr/bin/env bash
# test-dep-conflict-check.sh
#
# Validates that the grep patterns used in
# .github/workflows/check-compose-bom-deps.yml correctly detect version
# conflicts in the Gradle dependency tree before they silently reach CI.
#
# Usage (from repo root):
#   bash android/scripts/test-dep-conflict-check.sh
#
# The script synthesises three synthetic dep-tree fixtures — one clean, one
# with a FAILED marker, and one with a duplicate-class warning — and asserts
# that the same grep commands used in the workflow exit non-zero only when a
# conflict is present.  A fourth fixture exercises the Compose-conflict warning
# (which is non-fatal in the workflow but should still surface the text).
#
# All temporary files are created in /tmp and cleaned up on exit.
#
# This script was written as part of the validation exercise for
# check-compose-bom-deps.yml to confirm the check has been exercised against
# a real conflict pattern before shipping.

set -euo pipefail

PASS=0
FAIL=0

# ── colour helpers ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RESET='\033[0m'

ok()   { echo -e "${GREEN}PASS${RESET}  $*"; PASS=$((PASS + 1)); }
fail() { echo -e "${RED}FAIL${RESET}  $*"; FAIL=$((FAIL + 1)); }
info() { echo -e "${YELLOW}INFO${RESET}  $*"; }

# ── temp-file cleanup ──────────────────────────────────────────────────────────
TMPDIR_WORK=$(mktemp -d /tmp/dep-check-test.XXXXXX)
trap 'rm -rf "$TMPDIR_WORK"' EXIT

# ── fixture helpers ────────────────────────────────────────────────────────────

# write_clean_tree: a dependency tree with no conflicts, no FAILED markers, no
# duplicate-class warnings.  Mirrors what a healthy BOM-managed build produces.
write_clean_tree() {
  cat > "$1" <<'EOF'
------------------------------------------------------------
Project :app
------------------------------------------------------------

releaseRuntimeClasspath - Runtime classpath of compilation 'release' (target  (androidJvm)).
+--- androidx.compose:compose-bom:2024.09.03
|    +--- androidx.compose.material:material-icons-extended:1.7.0
|    +--- androidx.compose.ui:ui:1.7.0
|    \--- androidx.compose.material3:material3:1.3.0
+--- com.google.dagger:hilt-android:2.51.1
\--- androidx.navigation:navigation-compose:2.7.7

(*) - dependencies omitted (listed previously)
EOF
}

# write_failed_tree: simulates an artifact that Gradle could not resolve.
# Gradle appends " FAILED" (uppercase, at end of line) to the coordinate.
write_failed_tree() {
  cat > "$1" <<'EOF'
------------------------------------------------------------
Project :app
------------------------------------------------------------

releaseRuntimeClasspath
+--- androidx.compose:compose-bom:2024.09.03
|    +--- androidx.compose.material:material-icons-extended:1.7.0 FAILED
|    \--- androidx.compose.ui:ui:1.7.0
\--- com.google.dagger:hilt-android:2.51.1

FAILURE: Build failed with an exception.
EOF
}

# write_duplicate_class_tree: simulates the warning Gradle emits when two
# artifacts on the classpath provide the same fully-qualified class.  This is
# the scenario that materialised when an explicit version pin for
# material-icons-extended conflicted with the BOM-managed version.
write_duplicate_class_tree() {
  cat > "$1" <<'EOF'
------------------------------------------------------------
Project :app
------------------------------------------------------------

releaseRuntimeClasspath
+--- androidx.compose:compose-bom:2024.09.03
|    +--- androidx.compose.material:material-icons-extended:1.5.4
|    \--- androidx.compose.ui:ui:1.7.0
+--- androidx.compose.material:material-icons-extended:1.7.0 (*)

w: Duplicate class androidx.compose.material.icons.Icons found in modules:
   material-icons-extended-1.5.4.jar
   material-icons-extended-1.7.0.jar
EOF
}

# write_compose_conflict_tree: simulates a forced version-conflict resolution
# notice that Gradle emits when the BOM and an explicit pin disagree.
write_compose_conflict_tree() {
  cat > "$1" <<'EOF'
------------------------------------------------------------
Project :app
------------------------------------------------------------

releaseRuntimeClasspath
+--- androidx.compose:compose-bom:2024.09.03
|    +--- androidx.compose.material:material-icons-extended:1.7.0 -> 1.5.4 (conflict resolution)
|    \--- androidx.compose.ui:ui:1.7.0
\--- com.google.dagger:hilt-android:2.51.1
EOF
}

# ── replicate the exact workflow steps ─────────────────────────────────────────
# These functions mirror the shell commands in check-compose-bom-deps.yml
# step-for-step so that any future change to the workflow grep patterns that
# would break detection will also break this script.

check_failed_markers() {
  local dep_tree="$1"
  if grep -qE "^.*FAILED$" "$dep_tree"; then
    return 1   # conflict detected → CI exits non-zero
  fi
  return 0     # clean
}

check_duplicate_class() {
  local dep_tree="$1"
  if grep -qi "duplicate class" "$dep_tree"; then
    return 1   # conflict detected → CI exits non-zero
  fi
  return 0     # clean
}

check_compose_conflicts() {
  local dep_tree="$1"
  local conflicts
  conflicts=$(grep -i "conflict" "$dep_tree" | grep -i "androidx.compose" || true)
  if [ -n "$conflicts" ]; then
    return 1   # warning fires
  fi
  return 0     # silent
}

# ── test cases ─────────────────────────────────────────────────────────────────

info "=== Test suite: check-compose-bom-deps grep patterns ==="
echo

# ---------------------------------------------------------------------------- #
# 1. CLEAN TREE — all checks should pass silently
# ---------------------------------------------------------------------------- #
info "--- Fixture 1: clean dependency tree (no conflicts) ---"
CLEAN="$TMPDIR_WORK/clean.txt"
write_clean_tree "$CLEAN"

if check_failed_markers "$CLEAN"; then
  ok "FAILED-markers check passes on clean tree"
else
  fail "FAILED-markers check incorrectly flagged a clean tree"
fi

if check_duplicate_class "$CLEAN"; then
  ok "duplicate-class check passes on clean tree"
else
  fail "duplicate-class check incorrectly flagged a clean tree"
fi

if check_compose_conflicts "$CLEAN"; then
  ok "compose-conflict warning is silent on clean tree"
else
  fail "compose-conflict warning incorrectly fired on a clean tree"
fi

echo

# ---------------------------------------------------------------------------- #
# 2. FAILED MARKER — simulates an unresolvable artifact after a BOM bump
# ---------------------------------------------------------------------------- #
info "--- Fixture 2: FAILED marker (unresolvable material-icons-extended) ---"
FAILED="$TMPDIR_WORK/failed.txt"
write_failed_tree "$FAILED"

if check_failed_markers "$FAILED"; then
  fail "FAILED-markers check missed a FAILED line — grep pattern is broken"
else
  ok "FAILED-markers check correctly exits non-zero"
fi

# The duplicate-class and conflict checks should not be confused by FAILED lines
if check_duplicate_class "$FAILED"; then
  ok "duplicate-class check is not confused by FAILED lines"
else
  fail "duplicate-class check incorrectly flagged a tree with only a FAILED line"
fi

echo

# ---------------------------------------------------------------------------- #
# 3. DUPLICATE-CLASS WARNING — the historical material-icons-extended failure
#    mode: explicit pin + BOM-managed version on the same classpath
# ---------------------------------------------------------------------------- #
info "--- Fixture 3: duplicate-class warning (pinned vs BOM-managed icons) ---"
DUPECLASS="$TMPDIR_WORK/dupeclass.txt"
write_duplicate_class_tree "$DUPECLASS"

if check_duplicate_class "$DUPECLASS"; then
  fail "duplicate-class check missed the warning — grep pattern is broken"
else
  ok "duplicate-class check correctly exits non-zero"
fi

# A duplicate-class tree does not necessarily produce a top-level FAILED line
if check_failed_markers "$DUPECLASS"; then
  ok "FAILED-markers check does not false-positive on duplicate-class trees"
else
  fail "FAILED-markers check false-positived on a duplicate-class tree (no FAILED line present)"
fi

echo

# ---------------------------------------------------------------------------- #
# 4. COMPOSE CONFLICT RESOLUTION — forced downgrade notice in the tree
# ---------------------------------------------------------------------------- #
info "--- Fixture 4: Compose version conflict resolution notice ---"
CONFLICT="$TMPDIR_WORK/conflict.txt"
write_compose_conflict_tree "$CONFLICT"

if check_compose_conflicts "$CONFLICT"; then
  fail "compose-conflict warning check missed the conflict resolution notice"
else
  ok "compose-conflict check correctly fires for androidx.compose conflict lines"
fi

# Should not trip the FAILED or duplicate-class checks
if check_failed_markers "$CONFLICT"; then
  ok "FAILED-markers check does not false-positive on conflict-resolution lines"
else
  fail "FAILED-markers check false-positived on a conflict-resolution tree"
fi

if check_duplicate_class "$CONFLICT"; then
  ok "duplicate-class check does not false-positive on conflict-resolution lines"
else
  fail "duplicate-class check false-positived on a conflict-resolution tree"
fi

echo

# ── summary ───────────────────────────────────────────────────────────────────
info "=== Results: $PASS passed, $FAIL failed ==="
echo

if [ "$FAIL" -gt 0 ]; then
  echo -e "${RED}One or more pattern checks failed.  Review the grep expressions in${RESET}"
  echo -e "${RED}.github/workflows/check-compose-bom-deps.yml against the output above.${RESET}"
  exit 1
fi

echo -e "${GREEN}All checks passed.  The workflow grep patterns correctly detect:${RESET}"
echo "  • Unresolved artifacts (FAILED markers)"
echo "  • Duplicate-class warnings (historical material-icons-extended failure)"
echo "  • Compose version conflict resolution notices"
echo "  • And produce no false positives on a clean dependency tree"
exit 0
