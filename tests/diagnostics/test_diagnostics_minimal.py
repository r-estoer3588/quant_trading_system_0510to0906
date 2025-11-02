# tests/diagnostics/test_diagnostics_minimal.py
from __future__ import annotations

import pandas as pd

from core.system2 import generate_candidates_system2
from core.system3 import generate_candidates_system3
from core.system4 import generate_candidates_system4
from core.system5 import generate_candidates_system5
from core.system6 import generate_candidates_system6
from core.system7 import generate_candidates_system7


def _make_df_system2(pass_setup: bool = True) -> pd.DataFrame:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])  # two rows, last is latest
    data = {
        "Close": [10.0, 10.5],
        "dollarvolume20": [30_000_000, 35_000_000],
        "atr_ratio": [0.05, 0.06],
        "rsi3": [50.0, 95.0],
        "twodayup": [False, True],
        "adx7": [10.0, 20.0],
    }
    df = pd.DataFrame(data, index=dates)
    if pass_setup:
        df["filter"] = (
            (df["Close"] >= 5.0)
            & (df["dollarvolume20"] > 25_000_000)
            & (df["atr_ratio"] > 0.03)
        )
        df["setup"] = df["filter"] & (df["rsi3"] > 90.0) & df["twodayup"]
    return df


def _make_df_system7(pass_setup: bool = True) -> pd.DataFrame:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])  # two rows, last is latest
    low = [100.0, 95.0]
    min50 = [96.0, 96.0]  # latest low (95) <= min50 (96) -> setup True
    close = [101.0, 96.5]
    atr50 = [2.5, 2.6]
    df = pd.DataFrame(
        {"Low": low, "min_50": min50, "Close": close, "ATR50": atr50}, index=dates
    )
    if pass_setup:
        df["setup"] = df["Low"] <= df["min_50"]
    return df


def _make_df_system3(pass_setup: bool = True) -> pd.DataFrame:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])  # two rows, last is latest
    data = {
        "Close": [10.0, 11.0],
        "dollarvolume20": [30_000_000, 35_000_000],
        "atr_ratio": [0.06, 0.07],
        "drop3d": [0.13, 0.20],
    }
    df = pd.DataFrame(data, index=dates)
    if pass_setup:
        df["filter"] = (
            (df["Close"] >= 5.0)
            & (df["dollarvolume20"] > 25_000_000)
            & (df["atr_ratio"] >= 0.05)
        )
        df["setup"] = df["filter"] & (df["drop3d"] >= 0.125)
    return df


def _make_df_system4(pass_setup: bool = True) -> pd.DataFrame:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])  # two rows, last is latest
    data = {
        "Close": [100.0, 105.0],
        "sma200": [95.0, 100.0],
        "dollarvolume50": [120_000_000, 150_000_000],
        "hv50": [20.0, 20.0],
        "rsi4": [40.0, 25.0],  # ranking uses low rsi4 (<30)
        "atr_ratio": [0.04, 0.05],
    }
    df = pd.DataFrame(data, index=dates)
    if pass_setup:
        df["filter"] = (df["dollarvolume50"] > 100_000_000) & df["hv50"].between(10, 40)
        df["setup"] = df["filter"] & (df["Close"] > df["sma200"])
    return df


def _make_df_system5(pass_setup: bool = True) -> pd.DataFrame:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])  # two rows, last is latest
    data = {
        "Close": [10.0, 11.0],
        "adx7": [36.0, 60.0],  # ranking uses high adx7 (>35)
        "atr_pct": [0.03, 0.04],  # > 2.5%
    }
    df = pd.DataFrame(data, index=dates)
    if pass_setup:
        df["filter"] = (
            (df["Close"] >= 5.0) & (df["adx7"] > 35.0) & (df["atr_pct"] > 0.025)
        )
        df["setup"] = df["filter"]
    return df


def _make_df_system6(pass_setup: bool = True) -> pd.DataFrame:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])  # two rows, last is latest
    data = {
        "Open": [10.0, 10.5],
        "High": [10.5, 11.0],
        "Low": [9.8, 10.2],
        "Close": [10.2, 11.5],
        "Volume": [2_000_000, 2_200_000],
        "dollarvolume50": [12_000_000, 13_000_000],
        "hv50": [20.0, 22.0],
        "return_6d": [0.15, 0.25],
        "UpTwoDays": [False, True],
        "atr10": [0.5, 0.55],
    }
    df = pd.DataFrame(data, index=dates)
    if pass_setup:
        df["filter"] = (
            (df["Low"] >= 5.0)
            & (df["dollarvolume50"] > 10_000_000)
            & df["hv50"].between(10.0, 40.0)
        )
        df["setup"] = df["filter"] & (df["return_6d"] > 0.20) & df["UpTwoDays"]
    return df


def test_system2_diagnostics_latest_only_shape():
    prepared = {"AAA": _make_df_system2(pass_setup=True)}
    result = generate_candidates_system2(
        prepared, latest_only=True, include_diagnostics=True, top_n=5
    )
    if isinstance(result, tuple) and len(result) == 3:
        by_date, merged, diag = result
    else:
        by_date, merged = result  # type: ignore[misc]
        diag = {}
    # shape checks
    assert isinstance(by_date, dict)
    assert merged is not None
    assert isinstance(diag, dict)
    # required keys
    for k in [
        "ranking_source",
        "setup_predicate_count",
        "ranked_top_n_count",
        "predicate_only_pass_count",
        "mismatch_flag",
    ]:
        assert k in diag
    assert diag["ranking_source"] == "latest_only"
    # with setup True, expect at least one candidate and count >= 1
    assert int(diag["ranked_top_n_count"]) >= 1


def test_system7_diagnostics_latest_only_shape():
    prepared = {"SPY": _make_df_system7(pass_setup=True)}
    result = generate_candidates_system7(
        prepared, latest_only=True, include_diagnostics=True, top_n=1
    )
    if isinstance(result, tuple) and len(result) == 3:
        by_date, merged, diag = result
    else:
        by_date, merged = result  # type: ignore[misc]
        diag = {}
    assert isinstance(by_date, dict)
    assert merged is not None
    assert isinstance(diag, dict)
    for k in [
        "ranking_source",
        "setup_predicate_count",
        "ranked_top_n_count",
        "predicate_only_pass_count",
        "mismatch_flag",
    ]:
        assert k in diag
    assert diag["ranking_source"] == "latest_only"
    assert int(diag["ranked_top_n_count"]) >= 1


def test_system3_diagnostics_latest_only_shape():
    prepared = {"AAA": _make_df_system3(pass_setup=True)}
    result = generate_candidates_system3(
        prepared, latest_only=True, include_diagnostics=True, top_n=5
    )
    if isinstance(result, tuple) and len(result) == 3:
        by_date, merged, diag = result
    else:
        by_date, merged = result  # type: ignore[misc]
        diag = {}
    assert isinstance(by_date, dict)
    assert merged is not None
    assert isinstance(diag, dict)
    for k in [
        "ranking_source",
        "setup_predicate_count",
        "ranked_top_n_count",
        "predicate_only_pass_count",
        "mismatch_flag",
    ]:
        assert k in diag
    assert diag["ranking_source"] == "latest_only"
    assert int(diag["ranked_top_n_count"]) >= 1


def test_system4_diagnostics_latest_only_shape():
    prepared = {"BBB": _make_df_system4(pass_setup=True)}
    result = generate_candidates_system4(
        prepared, latest_only=True, include_diagnostics=True, top_n=5
    )
    if isinstance(result, tuple) and len(result) == 3:
        by_date, merged, diag = result
    else:
        by_date, merged = result  # type: ignore[misc]
        diag = {}
    assert isinstance(by_date, dict)
    assert merged is not None
    assert isinstance(diag, dict)
    for k in [
        "ranking_source",
        "setup_predicate_count",
        "ranked_top_n_count",
        "predicate_only_pass_count",
        "mismatch_flag",
    ]:
        assert k in diag
    assert diag["ranking_source"] == "latest_only"
    assert int(diag["ranked_top_n_count"]) >= 1


def test_system6_diagnostics_latest_only_shape():
    prepared = {"CCC": _make_df_system6(pass_setup=True)}
    result = generate_candidates_system6(
        prepared, latest_only=True, include_diagnostics=True, top_n=5
    )
    if isinstance(result, tuple) and len(result) == 3:
        by_date, merged, diag = result
    else:
        by_date, merged = result  # type: ignore[misc]
        diag = {}
    assert isinstance(by_date, dict)
    # System6 は latest_only でも DataFrame を返す設計
    assert merged is not None
    assert isinstance(diag, dict)
    for k in [
        "ranking_source",
        "setup_predicate_count",
        "ranked_top_n_count",
        "predicate_only_pass_count",
        "mismatch_flag",
    ]:
        assert k in diag
    assert diag["ranking_source"] == "latest_only"
    assert int(diag["ranked_top_n_count"]) >= 1


def test_system5_diagnostics_latest_only_shape():
    prepared = {"CCC": _make_df_system5(pass_setup=True)}
    result = generate_candidates_system5(
        prepared, latest_only=True, include_diagnostics=True, top_n=5
    )
    if isinstance(result, tuple) and len(result) == 3:
        by_date, merged, diag = result
    else:
        by_date, merged = result  # type: ignore[misc]
        diag = {}
    assert isinstance(by_date, dict)
    assert merged is not None
    assert isinstance(diag, dict)
    for k in [
        "ranking_source",
        "setup_predicate_count",
        "ranked_top_n_count",
        "predicate_only_pass_count",
        "mismatch_flag",
    ]:
        assert k in diag
    assert diag["ranking_source"] == "latest_only"
    assert int(diag["ranked_top_n_count"]) >= 1
