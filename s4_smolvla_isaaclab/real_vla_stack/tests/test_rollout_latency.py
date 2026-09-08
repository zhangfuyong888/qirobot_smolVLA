from __future__ import annotations

import pytest

from real_vla_stack.robot.rollout.main import RollingLatencyEstimator


def test_rtc_latency_uses_rolling_percentile_not_lifetime_max() -> None:
    tracker = RollingLatencyEstimator(window_size=4, percentile=95)
    tracker.add(500.0)
    for _ in range(4):
        tracker.add(100.0)
    assert tracker.estimate_ms == pytest.approx(100.0)
    assert tracker.delay_steps(20, maximum=10) == 2


def test_rtc_latency_delay_is_ceil_and_clamped() -> None:
    tracker = RollingLatencyEstimator(window_size=3, percentile=95)
    tracker.add(201.0)
    assert tracker.delay_steps(20, maximum=10) == 5
    tracker.add(900.0)
    assert tracker.delay_steps(20, maximum=10) == 10
