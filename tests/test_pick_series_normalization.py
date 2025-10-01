#!/usr/bin/env python3
"""Tests for _pick_series normalization (case/underscore insensitive).

Focus: ensure that lower-case and underscore variants resolve without
explicitly enumerating every variant in call sites.
"""
from __future__ import annotations

import pandas as pd

from common.today_filters import _pick_series


def make_df():
    return pd.DataFrame(
        {
            "dollarvolume50": [1, 2, 3],
            "avg_volume50": [10, 11, 12],
            "HV50": [5, 6, 7],  # mixed case retained
            "atr_pct": [0.01, 0.02, 0.03],
        }
    )


def test_pick_series_direct_lower():
    df = make_df()
    s = _pick_series(df, ["DollarVolume50"])  # camel case variant
    assert s is not None
    assert list(s.values) == [1, 2, 3]


def test_pick_series_snake_to_camel():
    df = make_df()
    s = _pick_series(df, ["AvgVolume50"])  # stored as avg_volume50
    assert s is not None
    assert list(s.values) == [10, 11, 12]


def test_pick_series_hv_mixed():
    df = make_df()
    s = _pick_series(df, ["hv50"])  # stored as HV50
    assert s is not None
    assert list(s.values) == [5, 6, 7]


def test_pick_series_atr_pct_variant():
    df = make_df()
    s = _pick_series(df, ["ATR_Pct"])  # stored as atr_pct
    assert s is not None
    assert list(s.values) == [0.01, 0.02, 0.03]
