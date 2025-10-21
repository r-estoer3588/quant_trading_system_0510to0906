# tests/test_system3_precompute_columns.py
"""Unit tests for System3 precomputed indicator column normalization."""
from __future__ import annotations

import pandas as pd

from core.system3 import prepare_data_vectorized_system3


def test_prepare_data_normalizes_indicator_column_names():
    # Prepare a small sample DataFrame with variant column casing
    dates = pd.date_range("2025-01-01", periods=5, freq="D")
    df = pd.DataFrame(index=dates)
    df["Close"] = [10, 12, 11, 9, 10]
    # Use non-canonical column names intentionally
    df["Drop3D"] = [-0.2, -0.15, -0.1, -0.05, -0.13]
    df["ATR_Ratio"] = [0.06, 0.07, 0.05, 0.08, 0.06]
    df["dollarvolume20"] = [30_000_000, 26_000_000, 40_000_000, 20_000_000, 30_000_000]

    prepared = prepare_data_vectorized_system3({"TST": df}, reuse_indicators=True)

    assert isinstance(prepared, dict)
    assert "TST" in prepared
    out_df = prepared["TST"]
    # canonical lowercase columns should be present after normalization
    assert "drop3d" in out_df.columns
    assert "atr_ratio" in out_df.columns
    assert "dollarvolume20" in out_df.columns
    # setup column should have been computed
    assert "setup" in out_df.columns
