from __future__ import annotations

import pytest

from common.utils import clamp01


@pytest.mark.parametrize(
    "value, expected",
    [(-1, 0.0), (0.5, 0.5), (2, 1.0), ("bad", 0.0)],
)
def test_clamp01(value: float | str, expected: float) -> None:
    assert clamp01(value) == expected
