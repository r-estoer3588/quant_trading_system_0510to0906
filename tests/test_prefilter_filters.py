import pandas as pd

from core.system5 import DEFAULT_ATR_PCT_THRESHOLD
from scripts.run_all_systems_today import (
    filter_system1,
    filter_system2,
    filter_system3,
    filter_system4,
    filter_system5,
    filter_system6,
)


def test_filter_system1_prefers_indicator_columns():
    data = {
        "AAA": pd.DataFrame(
            {
                "Close": [6.0, 6.5],
                "Volume": [10_000_000, 11_000_000],
                "DollarVolume20": [45_000_000, 60_000_000],
            }
        ),
        "BBB": pd.DataFrame(
            {
                "Close": [6.2],
                "Volume": [10_000_000],
                "DollarVolume20": [40_000_000],
            }
        ),
    }
    result = filter_system1(["AAA", "BBB"], data)
    assert result == ["AAA"]


def test_filter_system1_fallback_uses_price_volume():
    data = {
        "CCC": pd.DataFrame(
            {
                "Close": [6.0, 6.1],
                "Volume": [11_000_000, 12_000_000],
            }
        )
    }
    result = filter_system1(["CCC"], data)
    assert result == ["CCC"]


def test_filter_system2_uses_atr_ratio_indicator():
    data = {
        "AAA": pd.DataFrame(
            {
                "Close": [10.0],
                "Low": [9.5],
                "DollarVolume20": [30_000_000],
                "ATR_Ratio": [0.04],
            }
        ),
        "BBB": pd.DataFrame(
            {
                "Close": [10.5],
                "Low": [10.0],
                "DollarVolume20": [30_000_000],
                "ATR_Ratio": [0.02],
            }
        ),
    }
    result = filter_system2(["AAA", "BBB"], data)
    assert result == ["AAA"]


def test_filter_system2_fallback_uses_atr10_when_ratio_missing():
    data = {
        "CCC": pd.DataFrame(
            {
                "Close": [10.0],
                "Low": [9.4],
                "High": [10.8],
                "DollarVolume20": [35_000_000],
                "ATR10": [0.45],
            }
        )
    }
    result = filter_system2(["CCC"], data)
    assert result == ["CCC"]


def test_filter_system3_requires_avg_volume_and_ratio():
    data = {
        "AAA": pd.DataFrame(
            {
                "Low": [1.5],
                "AvgVolume50": [1_200_000],
                "ATR_Ratio": [0.06],
            }
        ),
        "BBB": pd.DataFrame(
            {
                "Low": [1.5],
                "AvgVolume50": [900_000],
                "ATR_Ratio": [0.06],
            }
        ),
    }
    result = filter_system3(["AAA", "BBB"], data)
    assert result == ["AAA"]


def test_filter_system4_respects_hv_range():
    data = {
        "AAA": pd.DataFrame(
            {
                "DollarVolume50": [150_000_000],
                "HV50": [25.0],
            }
        ),
        "BBB": pd.DataFrame(
            {
                "DollarVolume50": [150_000_000],
                "HV50": [55.0],
            }
        ),
    }
    result = filter_system4(["AAA", "BBB"], data)
    assert result == ["AAA"]


def test_filter_system5_uses_atr_pct_threshold():
    above = DEFAULT_ATR_PCT_THRESHOLD + 0.02
    below = DEFAULT_ATR_PCT_THRESHOLD - 0.01
    data = {
        "AAA": pd.DataFrame(
            {
                "AvgVolume50": [700_000],
                "DollarVolume50": [3_000_000],
                "ATR_Pct": [above],
            }
        ),
        "BBB": pd.DataFrame(
            {
                "AvgVolume50": [700_000],
                "DollarVolume50": [3_000_000],
                "ATR_Pct": [below],
            }
        ),
    }
    result = filter_system5(["AAA", "BBB"], data)
    assert result == ["AAA"]


def test_filter_system6_requires_liquidity_threshold():
    data = {
        "AAA": pd.DataFrame(
            {
                "Low": [6.0],
                "DollarVolume50": [15_000_000],
            }
        ),
        "BBB": pd.DataFrame(
            {
                "Low": [6.0],
                "DollarVolume50": [5_000_000],
            }
        ),
    }
    result = filter_system6(["AAA", "BBB"], data)
    assert result == ["AAA"]
