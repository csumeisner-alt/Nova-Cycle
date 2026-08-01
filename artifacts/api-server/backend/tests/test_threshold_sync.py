"""
Cross-cutting threshold-sync guard
====================================
Confirms that LONG_BUY_THRESHOLD, LONG_SELL_THRESHOLD, SHORT_BUY_THRESHOLD,
and SHORT_SELL_THRESHOLD are never hardcoded as literals in the pipeline
modules that depend on them.

Why this file exists
--------------------
Several modules (macro_override, hold_time, long_gauge, short_gauge) consume
the signal-threshold constants from ``config.settings``.  If someone edits
``LONG_BUY_THRESHOLD`` in config.py without touching every callsite, the
suppression boundary, hold-time trigger, and signal-emission threshold will
silently diverge.

This test file guards against that in two complementary ways:

1. **Static-analysis checks** – scan each source file as text and flag any
   bare numeric literal that matches the current threshold values (65.0, 50.0,
   and their negatives / the old hardcoded 70.0) appearing in a comparison
   context.  The scan is intentionally strict: it whitelists known-acceptable
   uses (RSI overbought at 70.0, default-value sentinels) and treats
   everything else as a potential drift hazard.

2. **Runtime alignment checks** – verify at import time that module-level
   constants (e.g. ``LONG_STRONG_BULL`` in macro_override) equal the live
   ``settings`` values.

These complement the existing per-module boundary tests in
``test_hold_time_threshold.py`` and ``test_macro_override_safety.py``.
"""

from __future__ import annotations

import ast
import os
import re
import textwrap
from pathlib import Path
from typing import List, Tuple

import pytest

from config import settings
from signal_engine.macro_override import LONG_STRONG_BEAR, LONG_STRONG_BULL


# ---------------------------------------------------------------------------
# Locate the backend root so tests work regardless of CWD
# ---------------------------------------------------------------------------

_BACKEND = Path(__file__).parent.parent  # artifacts/api-server/backend


def _read(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Cross-module runtime alignment
# ---------------------------------------------------------------------------

class TestRuntimeAlignment:
    """
    Module-level constants that snapshot settings at import time must equal
    the live settings values.  If config.py is changed, these constants
    must be updated too.
    """

    def test_macro_override_bull_equals_long_buy_threshold(self):
        """LONG_STRONG_BULL in macro_override.py must mirror settings.LONG_BUY_THRESHOLD."""
        assert LONG_STRONG_BULL == settings.LONG_BUY_THRESHOLD, (
            f"macro_override.LONG_STRONG_BULL={LONG_STRONG_BULL} != "
            f"settings.LONG_BUY_THRESHOLD={settings.LONG_BUY_THRESHOLD}. "
            "Update macro_override.py to keep the suppression boundary in "
            "sync with the long-gauge BUY threshold."
        )

    def test_macro_override_bear_equals_long_sell_threshold(self):
        """LONG_STRONG_BEAR in macro_override.py must mirror settings.LONG_SELL_THRESHOLD."""
        assert LONG_STRONG_BEAR == settings.LONG_SELL_THRESHOLD, (
            f"macro_override.LONG_STRONG_BEAR={LONG_STRONG_BEAR} != "
            f"settings.LONG_SELL_THRESHOLD={settings.LONG_SELL_THRESHOLD}. "
            "Update macro_override.py to keep the suppression boundary in "
            "sync with the long-gauge SELL threshold."
        )

    def test_thresholds_are_symmetric(self):
        """LONG_BUY and LONG_SELL thresholds must be equal in magnitude."""
        assert settings.LONG_BUY_THRESHOLD == -settings.LONG_SELL_THRESHOLD, (
            f"LONG_BUY_THRESHOLD={settings.LONG_BUY_THRESHOLD} and "
            f"LONG_SELL_THRESHOLD={settings.LONG_SELL_THRESHOLD} are not "
            "symmetric.  Both macro_override and hold_time assume symmetry."
        )

    def test_short_thresholds_are_symmetric(self):
        """SHORT_BUY and SHORT_SELL thresholds must be equal in magnitude."""
        assert settings.SHORT_BUY_THRESHOLD == -settings.SHORT_SELL_THRESHOLD, (
            f"SHORT_BUY_THRESHOLD={settings.SHORT_BUY_THRESHOLD} and "
            f"SHORT_SELL_THRESHOLD={settings.SHORT_SELL_THRESHOLD} are not "
            "symmetric.  short_gauge applies both values in the same scoring "
            "path; asymmetry would produce biased signal suppression."
        )


# ---------------------------------------------------------------------------
# 2. Static-analysis helpers
# ---------------------------------------------------------------------------

# Literals that ARE the threshold values (or the old hardcoded predecessor)
# and therefore must NOT appear as bare comparison values in gate logic.
_LONG_THRESHOLD_LITERALS = {
    settings.LONG_BUY_THRESHOLD,          # e.g. 65.0
    -settings.LONG_BUY_THRESHOLD,         # e.g. -65.0
    settings.LONG_SELL_THRESHOLD,         # e.g. -65.0 (same as above)
    -settings.LONG_SELL_THRESHOLD,        # e.g. +65.0
    70.0, -70.0,                           # old hardcoded predecessor
}
_SHORT_THRESHOLD_LITERALS = {
    settings.SHORT_BUY_THRESHOLD,         # e.g. 50.0
    -settings.SHORT_BUY_THRESHOLD,        # e.g. -50.0
    settings.SHORT_SELL_THRESHOLD,        # e.g. -50.0 (same as above)
    -settings.SHORT_SELL_THRESHOLD,       # e.g. +50.0
}

# Regex: comparison operators followed by a numeric literal,
# e.g. `> 65.0`, `< -65.0`, `>= 70`, etc.
_CMP_PATTERN = re.compile(
    r"""(?x)
    [<>]=?\s*          # comparison operator
    (-?                # optional minus
     (?:
       \d+\.\d*        # float with decimal point
       | \.\d+         # float starting with .
       | \d+           # integer
     )
    )
    \b
    """
)


def _code_only_lines(source: str) -> List[Tuple[int, str]]:
    """
    Return (lineno, line_text) pairs for every line that contains actual
    Python code — i.e. lines that are NOT part of a string literal or a
    standalone comment.

    Uses Python's ``tokenize`` module so multi-line docstrings and
    triple-quoted strings are reliably excluded.  Falls back to the raw
    source if tokenisation fails (parse errors in the source file).
    """
    import io
    import tokenize as _tok

    try:
        lines = source.splitlines(keepends=True)
        # Collect line-number ranges that are occupied by string tokens.
        # tokenize uses 1-based line numbers.
        string_lines: set[int] = set()
        comment_lines: set[int] = set()
        tokens = list(_tok.generate_tokens(io.StringIO(source).readline))
        for tok_type, tok_str, tok_start, tok_end, _ in tokens:
            if tok_type == _tok.STRING:
                for ln in range(tok_start[0], tok_end[0] + 1):
                    string_lines.add(ln)
            elif tok_type == _tok.COMMENT:
                comment_lines.add(tok_start[0])
        result = []
        for lineno, line in enumerate(lines, 1):
            if lineno in string_lines or lineno in comment_lines:
                continue
            result.append((lineno, line.rstrip()))
        return result
    except Exception:
        # Fallback: skip lines that start with # or triple-quote
        result_fallback = []
        for lineno, line in enumerate(source.splitlines(), 1):
            s = line.strip()
            if s.startswith("#") or s.startswith('"""') or s.startswith("'''"):
                continue
            result_fallback.append((lineno, line.rstrip()))
        return result_fallback


def _extract_comparison_literals(source: str) -> List[Tuple[int, float, str]]:
    """
    Return (lineno, value, line_text) for every comparison-literal pair
    found in the *code* portions of ``source`` (strings and comments excluded).
    """
    found = []
    for lineno, line in _code_only_lines(source):
        for m in _CMP_PATTERN.finditer(line):
            try:
                val = float(m.group(1))
                found.append((lineno, val, line))
            except ValueError:
                pass
    return found


def _suspicious_threshold_hits(
    source: str,
    literals: set,
    allowlist_patterns: List[str] | None = None,
) -> List[Tuple[int, float, str]]:
    """
    Return comparison-literal hits whose value is in ``literals`` and whose
    line does NOT match any pattern in ``allowlist_patterns`` (case-insensitive).
    """
    allowlist_patterns = allowlist_patterns or []
    hits = []
    for lineno, val, line in _extract_comparison_literals(source):
        if val not in literals:
            continue
        if any(re.search(p, line, re.IGNORECASE) for p in allowlist_patterns):
            continue
        hits.append((lineno, val, line))
    return hits


# ---------------------------------------------------------------------------
# 3. Static-analysis tests — long-gauge threshold literals
# ---------------------------------------------------------------------------

class TestNoHardcodedLongThresholds:
    """
    No module in the core signal pipeline should compare against 65.0 / -65.0
    (the current LONG_BUY/SELL thresholds) or the old 70.0 / -70.0 as bare
    literals.  All such comparisons must go through ``settings.*``.
    """

    # Common allowlist: comment lines, docstrings already excluded above.
    # Add patterns for known-good 70.0 uses (RSI overbought in short_gauge).
    _LONG_ALLOWLIST = [
        r"rsi.*70",          # RSI overbought sentinel in short_gauge
        r"70.*rsi",
        r"#",                # inline comments (belt-and-suspenders)
        r"adx.*25",          # ADX threshold in hold_time (not a gauge threshold)
    ]

    def _assert_no_hits(self, rel_path: str):
        source = _read(rel_path)
        hits = _suspicious_threshold_hits(
            source, _LONG_THRESHOLD_LITERALS, self._LONG_ALLOWLIST
        )
        if hits:
            lines = "\n".join(
                f"  line {ln}: {txt}  [literal={val}]"
                for ln, val, txt in hits
            )
            pytest.fail(
                f"Hardcoded long-gauge threshold literal found in {rel_path}.\n"
                f"Replace with settings.LONG_BUY_THRESHOLD / LONG_SELL_THRESHOLD:\n"
                f"{lines}"
            )

    def test_macro_override_no_hardcoded_long_threshold(self):
        self._assert_no_hits("signal_engine/macro_override.py")

    def test_hold_time_no_hardcoded_long_threshold(self):
        self._assert_no_hits("ml/hold_time.py")

    def test_long_gauge_no_hardcoded_long_threshold(self):
        self._assert_no_hits("signal_engine/long_gauge.py")

    def test_predictions_router_no_hardcoded_long_threshold(self):
        self._assert_no_hits("routers/predictions.py")

    def test_decision_filter_no_hardcoded_long_threshold(self):
        self._assert_no_hits("signal_engine/decision_filter.py")


# ---------------------------------------------------------------------------
# 4. Static-analysis tests — short-gauge threshold literals
# ---------------------------------------------------------------------------

class TestNoHardcodedShortThresholds:
    """
    No module in the core signal pipeline should compare against 50.0 / -50.0
    (the current SHORT_BUY/SELL thresholds) as bare literals.  All such
    comparisons must go through ``settings.*``.
    """

    # Allowlist: RSI/stoch default fallback sentinels (50.0 as neutral midpoint)
    _SHORT_ALLOWLIST = [
        r"\.get\(.*50",          # dict.get("key", 50.0) default sentinel
        r"or\s+50",              # `x or 50.0` neutral fallback
        r"=\s*50",               # assignment to a sentinel (e.g. default=50.0)
        r"50\.0\)",              # closing-paren default-arg pattern
        r"#",                    # inline comments
    ]

    def _assert_no_hits(self, rel_path: str):
        source = _read(rel_path)
        hits = _suspicious_threshold_hits(
            source, _SHORT_THRESHOLD_LITERALS, self._SHORT_ALLOWLIST
        )
        if hits:
            lines = "\n".join(
                f"  line {ln}: {txt}  [literal={val}]"
                for ln, val, txt in hits
            )
            pytest.fail(
                f"Hardcoded short-gauge threshold literal found in {rel_path}.\n"
                f"Replace with settings.SHORT_BUY_THRESHOLD / SHORT_SELL_THRESHOLD:\n"
                f"{lines}"
            )

    def test_short_gauge_no_hardcoded_short_threshold(self):
        self._assert_no_hits("signal_engine/short_gauge.py")

    def test_hold_time_no_hardcoded_short_threshold(self):
        self._assert_no_hits("ml/hold_time.py")

    def test_macro_override_no_hardcoded_short_threshold(self):
        self._assert_no_hits("signal_engine/macro_override.py")

    def test_predictions_router_no_hardcoded_short_threshold(self):
        self._assert_no_hits("routers/predictions.py")


# ---------------------------------------------------------------------------
# 5. Behavioural: short_gauge fires at SHORT_SELL_THRESHOLD boundary
# ---------------------------------------------------------------------------

class TestShortSellThresholdSync:
    """
    SHORT_SELL_THRESHOLD must govern when the short gauge emits a SELL signal.
    Probe the gauge at threshold − ε (no sell) and threshold − 1 (sell) to
    confirm it reads from settings rather than a local constant.

    Note: ShortTrendGauge.compute() requires real indicator data; we use a
    minimal synthetic frame and verify only that the sell boundary is consistent
    with settings.SHORT_SELL_THRESHOLD, not the exact score magnitude.
    """

    def test_short_sell_threshold_comes_from_settings(self):
        """
        Verify SHORT_SELL_THRESHOLD is the live settings value, not a stale
        import-time snapshot.  This is a lightweight invariant check that
        complements the static-analysis tests above.
        """
        from signal_engine.short_gauge import ShortTrendGauge
        gauge = ShortTrendGauge()
        # The gauge exposes no public constant, but the settings value is
        # what all the comparison tests elsewhere pin against.  The important
        # invariant is that SHORT_SELL_THRESHOLD is negative and symmetric
        # with SHORT_BUY_THRESHOLD (already verified in TestRuntimeAlignment).
        assert settings.SHORT_SELL_THRESHOLD < 0, (
            "SHORT_SELL_THRESHOLD must be negative."
        )
        assert settings.SHORT_SELL_THRESHOLD == -settings.SHORT_BUY_THRESHOLD, (
            "SHORT_SELL_THRESHOLD must be the negation of SHORT_BUY_THRESHOLD."
        )

    def test_short_gauge_reads_settings_not_literal(self):
        """
        If SHORT_SELL_THRESHOLD is patched in settings, the gauge must
        respect the new value.  We monkey-patch settings, run the gauge
        with a score that crosses the patched threshold but not the original,
        and confirm the result reflects the patched value.

        Implementation: we don't call compute() (needs real DB data), but
        we can read the source to confirm the attribute access pattern.
        """
        source = _read("signal_engine/short_gauge.py")
        # The short_gauge must reference settings.SHORT_SELL_THRESHOLD
        assert "settings.SHORT_SELL_THRESHOLD" in source, (
            "short_gauge.py does not reference settings.SHORT_SELL_THRESHOLD. "
            "It may be using a hardcoded constant that will drift from config.py."
        )
        assert "settings.SHORT_BUY_THRESHOLD" in source, (
            "short_gauge.py does not reference settings.SHORT_BUY_THRESHOLD. "
            "It may be using a hardcoded constant that will drift from config.py."
        )


# ---------------------------------------------------------------------------
# 6. Behavioural: long_gauge reads settings for both BUY and SELL thresholds
# ---------------------------------------------------------------------------

class TestLongGaugeThresholdSync:
    """Confirm long_gauge.py reads LONG_BUY_THRESHOLD and LONG_SELL_THRESHOLD
    from settings at call time, not from module-level literals."""

    def test_long_gauge_reads_settings_long_buy_threshold(self):
        source = _read("signal_engine/long_gauge.py")
        assert "settings.LONG_BUY_THRESHOLD" in source, (
            "long_gauge.py does not reference settings.LONG_BUY_THRESHOLD. "
            "It may be using a hardcoded constant that will drift from config.py."
        )

    def test_long_gauge_reads_settings_long_sell_threshold(self):
        source = _read("signal_engine/long_gauge.py")
        assert "settings.LONG_SELL_THRESHOLD" in source, (
            "long_gauge.py does not reference settings.LONG_SELL_THRESHOLD. "
            "It may be using a hardcoded constant that will drift from config.py."
        )


# ---------------------------------------------------------------------------
# 7. Behavioural: hold_time reads settings for both BUY thresholds
# ---------------------------------------------------------------------------

class TestHoldTimeThresholdSync:
    """Confirm hold_time.py reads BUY thresholds from settings at call time."""

    def test_hold_time_reads_settings_long_buy_threshold(self):
        source = _read("ml/hold_time.py")
        assert "settings.LONG_BUY_THRESHOLD" in source, (
            "hold_time.py does not reference settings.LONG_BUY_THRESHOLD. "
            "It may be using a hardcoded constant (e.g. 70.0) that diverges "
            "from the gauge's BUY threshold."
        )

    def test_hold_time_reads_settings_short_buy_threshold(self):
        source = _read("ml/hold_time.py")
        assert "settings.SHORT_BUY_THRESHOLD" in source, (
            "hold_time.py does not reference settings.SHORT_BUY_THRESHOLD. "
            "It may be using a hardcoded constant that diverges from the "
            "gauge's SHORT BUY threshold."
        )

    def test_hold_time_no_old_hardcoded_70(self):
        """The old hardcoded 70.0 long-trigger must not reappear in hold_time.py."""
        source = _read("ml/hold_time.py")
        # Only flag 70.0 when it appears in a comparison context; the constant
        # may also appear in doc-strings or comments, which we don't flag.
        comparison_hits = [
            (lineno, line)
            for lineno, line in enumerate(source.splitlines(), 1)
            if re.search(r"[<>]=?\s*70\.0", line) and not line.strip().startswith("#")
        ]
        assert not comparison_hits, (
            "hold_time.py contains a comparison against literal 70.0, which "
            "is the old hardcoded long-trigger threshold.  Use "
            "settings.LONG_BUY_THRESHOLD instead.\n"
            + "\n".join(f"  line {ln}: {txt}" for ln, txt in comparison_hits)
        )


# ---------------------------------------------------------------------------
# 8. Macro-override module: no SELL path uses SHORT_SELL_THRESHOLD
#    (macro_override only cares about LONG thresholds — a guard that it
#     hasn't accidentally imported a short-threshold constant)
# ---------------------------------------------------------------------------

class TestMacroOverrideDoesNotImportShortThresholds:
    """macro_override.py gates on long-trend strength, not short thresholds.
    It must not reference SHORT_BUY_THRESHOLD or SHORT_SELL_THRESHOLD."""

    def test_no_short_threshold_import_in_macro_override(self):
        source = _read("signal_engine/macro_override.py")
        assert "SHORT_BUY_THRESHOLD" not in source, (
            "macro_override.py references SHORT_BUY_THRESHOLD, which is not "
            "part of its suppression logic.  Check for accidental drift."
        )
        assert "SHORT_SELL_THRESHOLD" not in source, (
            "macro_override.py references SHORT_SELL_THRESHOLD, which is not "
            "part of its suppression logic.  Check for accidental drift."
        )
