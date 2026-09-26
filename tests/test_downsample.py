"""Unit tests for bounded extrema downsampling used by the timeline.

Replaces the old fixed-stride sampling, which could return more points than
``max_points`` allowed and dropped isolated spikes/dips, flattening anomalies
on the page.
"""

import math

from app.services import downsample


def series(values, start=0):
    """(ts, value, event_id) tuples with distinct ids in ts order."""
    return [(float(start + i), float(v), start + i + 1) for i, v in enumerate(values)]


def ts_of(picked):
    return [t for t, _, _ in picked]


def test_under_budget_returns_every_sample_untouched():
    values = series([1, 2, 3, 4])
    assert downsample(values, 4) is values
    assert downsample(values, 400) is values
    assert downsample([], 4) == []


def test_returns_at_most_max_points_and_keeps_endpoints():
    values = series(range(50))
    for max_points in (2, 3, 4, 5, 7, 10, 49):
        picked = downsample(values, max_points)
        assert len(picked) <= max_points
        assert picked[0] == values[0]
        assert picked[-1] == values[-1]
        timestamps = ts_of(picked)
        assert timestamps == sorted(timestamps)


def test_max_points_two_keeps_only_endpoints():
    values = series(range(20))
    assert downsample(values, 2) == [values[0], values[-1]]


def test_tiny_budgets_are_bounded():
    values = series(range(20))
    assert downsample(values, 1) == [values[0]]
    assert downsample(values, 0) == []
    assert downsample(values, -3) == []


def test_isolated_spike_is_preserved_within_budget():
    # 20 samples, max_points=4: the old stride returned 5 points and missed
    # the spike at ts=7, flattening the anomaly.
    values = series([99 if t == 7 else 10 for t in range(20)])
    picked = downsample(values, 4)
    assert len(picked) == 4
    assert 7.0 in ts_of(picked)
    assert picked[0][0] == 0.0 and picked[-1][0] == 19.0


def test_isolated_dip_is_preserved_within_budget():
    values = series([-50 if t == 13 else 10 for t in range(20)])
    picked = downsample(values, 4)
    assert len(picked) <= 4
    assert 13.0 in ts_of(picked)


def test_single_slot_bucket_prefers_larger_deviation():
    # max_points=3 -> budget 1 -> one bucket covering the whole middle.
    # Baseline between endpoints (both 0) is the zero line; the dip deviates
    # more than the mild rise, so the dip must win the single slot.
    values = series([0, 1, 1, 1, 0, -40, 0, 0])
    picked = downsample(values, 3)
    assert len(picked) == 3
    assert ts_of(picked) == [0.0, 5.0, 7.0]


def test_single_slot_deviation_uses_bucket_endpoint_baseline():
    # Endpoints slope upward (10 -> 20); the sample deviating from that sloped
    # line the most (the dip at ts=4) is kept even though both extreme
    # candidates exist.
    raw = [10, 11, 12, 13, -30, 15, 16, 20]
    values = series(raw)
    picked = downsample(values, 3)
    assert len(picked) == 3
    assert ts_of(picked)[1] == 4.0


def test_single_slot_tie_keeps_earlier_extremum():
    # Equal absolute deviations from the flat endpoint baseline: the earlier
    # spike at ts=3 wins over the later dip at ts=12, deterministically.
    values = series([40 if t == 3 else -40 if t == 12 else 0 for t in range(20)])
    picked = downsample(values, 3)
    assert [(t, v) for t, v, _ in picked] == [(0.0, 0.0), (3.0, 40.0), (19.0, 0.0)]


def test_constant_channel_does_not_exceed_budget():
    # Min and max coincide in every bucket; duplicates collapse to one point.
    values = series([5] * 30)
    for max_points in (2, 3, 4, 5, 8):
        picked = downsample(values, max_points)
        assert len(picked) <= max_points
        assert picked[0] == values[0] and picked[-1] == values[-1]


def test_output_sorted_by_timestamp_and_event_id():
    # Extrema inside buckets must come back in original chronological order,
    # never ordered min-before-max regardless of time.
    values = series([0, 9, 1, 8, 2, 7, 3, 6, 4, 5] * 3)
    picked = downsample(values, 8)
    keys = [(t, i) for t, _, i in picked]
    assert keys == sorted(keys)
    assert len(picked) <= 8


def test_two_slot_bucket_with_constant_tail_still_bounded():
    # Odd budget: one two-slot bucket plus a constant single-slot bucket.
    values = series([0, 10, 0, 10] + [1] * 15 + [20])
    picked = downsample(values, 5)
    assert len(picked) <= 5
    assert picked[0] == values[0] and picked[-1] == values[-1]


def test_many_random_shapes_never_violate_contract():
    import random

    rng = random.Random(20260926)
    for _ in range(2000):
        n = rng.randint(1, 80)
        max_points = rng.randint(2, 40)
        values = [
            (float(i), rng.choice([0.0, 1.0, -1.0, rng.uniform(-100, 100)]), i + 1)
            for i in range(n)
        ]
        picked = downsample(values, max_points)
        assert len(picked) <= max_points
        if n <= max_points:
            assert picked == values
            continue
        assert picked[0] == values[0]
        assert picked[-1] == values[-1]
        keys = [(t, i) for t, _, i in picked]
        assert keys == sorted(keys)
        # ids are preserved verbatim (no fabricated samples)
        assert {i for _, _, i in picked} <= {i for _, _, i in values}
        # finite values
        assert all(math.isfinite(v) for _, v, _ in picked)
