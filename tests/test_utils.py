from __future__ import annotations

import pytest

import pandas as pd

from common.utils import BatchSizeMonitor, clamp01, _merge_ohlcv_variants


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


def test_merge_ohlcv_variants_prefers_existing_uppercase() -> None:
    df = pd.DataFrame(
        {
            "Open": [10.0, None],
            "open": [11.0, 12.0],
            "Volume": [100.0, None],
            "volume": [150.0, 200.0],
            "Close": [20.0, 21.0],
        }
    )

    merged = _merge_ohlcv_variants(df)

    assert list(merged.columns) == ["Open", "Volume", "Close"]
    assert merged["Open"].tolist() == [11.0, 12.0]
    assert merged["Volume"].tolist() == [150.0, 200.0]


def test_merge_ohlcv_variants_handles_uppercase_names() -> None:
    df = pd.DataFrame(
        [
            [10.0, 99.0, 11.0, 20.0],
            [None, 100.0, 12.0, 21.0],
        ],
        columns=["OPEN", "misc", "open", "Close"],
    )

    merged = _merge_ohlcv_variants(df)

    assert list(merged.columns) == ["Open", "misc", "Close"]
    assert merged["Open"].tolist() == [11.0, 12.0]


def test_merge_ohlcv_variants_unifies_adjclose_variants() -> None:
    df = pd.DataFrame(
        [
            [None, 100.0, 95.0],
            [101.0, 102.0, None],
        ],
        columns=["Adj Close", "adjusted_close", "adjclose"],
    )

    merged = _merge_ohlcv_variants(df)

    assert list(merged.columns) == ["AdjClose"]
    assert merged["AdjClose"].tolist() == [100.0, 102.0]
