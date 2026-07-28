"""
Tests that /confidence_history downsamples instead of truncating: long windows
with >1000 rows must span the whole window (oldest rows retained), capped at
1000 points.
"""

import pytest

from routers.predictions import _downsample_uniform


class TestDownsampleUniform:

    def test_under_cap_returned_unchanged(self):
        rows = list(range(500))
        assert _downsample_uniform(rows, max_points=1000) == rows

    def test_exactly_cap_returned_unchanged(self):
        rows = list(range(1000))
        assert _downsample_uniform(rows, max_points=1000) == rows

    def test_over_cap_is_downsampled_to_cap(self):
        rows = list(range(5000))
        out = _downsample_uniform(rows, max_points=1000)
        assert len(out) == 1000

    def test_endpoints_preserved(self):
        rows = list(range(5000))
        out = _downsample_uniform(rows, max_points=1000)
        # Both the newest (first) and oldest (last) rows must survive so the
        # chart spans the full requested window.
        assert out[0] == rows[0]
        assert out[-1] == rows[-1]

    def test_sampling_is_uniform_and_ordered(self):
        rows = list(range(10000))
        out = _downsample_uniform(rows, max_points=1000)
        assert out == sorted(out)
        gaps = [b - a for a, b in zip(out, out[1:])]
        # Uniform stride: gaps differ by at most 1 index.
        assert max(gaps) - min(gaps) <= 1

    def test_empty_and_tiny_inputs(self):
        assert _downsample_uniform([], max_points=1000) == []
        assert _downsample_uniform([42], max_points=1000) == [42]
