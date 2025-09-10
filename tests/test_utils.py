from __future__ import annotations

import pytest

from common.utils import BatchSizeMonitor, clamp01


@pytest.mark.parametrize(
    "value, expected",
    [(-1, 0.0), (0.5, 0.5), (2, 1.0), ("bad", 0.0)],
)
def test_clamp01(value: float | str, expected: float) -> None:
    assert clamp01(value) == expected


def test_batch_size_monitor_adjustment() -> None:
    monitor = BatchSizeMonitor(
        100,
        target_time=1.0,
        patience=2,
        min_batch_size=10,
        max_batch_size=200,
    )
    monitor.update(2.0)
    assert monitor.batch_size == 100
    monitor.update(2.0)
    assert monitor.batch_size == 50
    monitor.update(0.1)
    monitor.update(0.1)
    assert monitor.batch_size == 100
