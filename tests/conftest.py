from pathlib import Path
import sys

import pandas as pd
import pytest

# プロジェクトルートを import パスに追加(pytest 実行場所に依存しないため)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ========== Minimal DataFrame Fixtures for System 1-7 ==========


@pytest.fixture
def minimal_system1_df():
    """System1 の setup を満たす最小 DataFrame を返すファクトリー関数。

    Returns:
        callable: pass_setup 引数を受け取り、DataFrame を返す関数。
    """

    def _make(pass_setup: bool = True) -> pd.DataFrame:
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])  # two rows, last is latest
        data = {
            "Open": [10.0, 10.0],
            "High": [10.5, 10.6],
            "Low": [9.8, 9.9],
            "Close": [10.0, 10.5],
            "Volume": [5_000_000, 5_500_000],
            "dollarvolume20": [30_000_000, 35_000_000],
            "sma200": [9.0, 9.5],
            "roc200": [0.1, 0.2],
            "sma25": [9.7, 10.0],
            "sma50": [9.6, 9.9],
            "atr20": [0.2, 0.2],
        }
        df = pd.DataFrame(data, index=dates)
        if pass_setup:
            df["filter"] = (df["Close"] >= 5.0) & (df["dollarvolume20"] > 25_000_000)
            df["setup"] = df["filter"] & (df["Close"] > df["sma200"]) & (df["roc200"] > 0)
        return df

    return _make


@pytest.fixture
def minimal_system2_df():
    """System2 の setup を満たす最小 DataFrame を返すファクトリー関数。"""

    def _make(pass_setup: bool = True) -> pd.DataFrame:
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
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

    return _make


@pytest.fixture
def minimal_system3_df():
    """System3 の setup を満たす最小 DataFrame を返すファクトリー関数。"""

    def _make(pass_setup: bool = True) -> pd.DataFrame:
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
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

    return _make


@pytest.fixture
def minimal_system4_df():
    """System4 の setup を満たす最小 DataFrame を返すファクトリー関数。"""

    def _make(pass_setup: bool = True) -> pd.DataFrame:
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        data = {
            "Close": [100.0, 105.0],
            "sma200": [95.0, 100.0],
            "dollarvolume50": [120_000_000, 150_000_000],
            "hv50": [20.0, 20.0],
            "rsi4": [40.0, 25.0],
            "atr_ratio": [0.04, 0.05],
        }
        df = pd.DataFrame(data, index=dates)
        if pass_setup:
            df["filter"] = (df["dollarvolume50"] > 100_000_000) & df["hv50"].between(10, 40)
            df["setup"] = df["filter"] & (df["Close"] > df["sma200"])
        return df

    return _make


@pytest.fixture
def minimal_system5_df():
    """System5 の setup を満たす最小 DataFrame を返すファクトリー関数。"""

    def _make(pass_setup: bool = True) -> pd.DataFrame:
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        data = {
            "Close": [10.0, 11.0],
            "adx7": [36.0, 60.0],
            "atr_pct": [0.03, 0.04],
        }
        df = pd.DataFrame(data, index=dates)
        if pass_setup:
            df["filter"] = (df["Close"] >= 5.0) & (df["adx7"] > 35.0) & (df["atr_pct"] > 0.025)
            df["setup"] = df["filter"]
        return df

    return _make


@pytest.fixture
def minimal_system6_df():
    """System6 の setup を満たす最小 DataFrame を返すファクトリー関数。"""

    def _make(pass_setup: bool = True) -> pd.DataFrame:
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        data = {
            "Open": [100.0, 100.5],
            "High": [101.0, 102.0],
            "Low": [99.0, 99.5],
            "Close": [100.5, 102.0],
            "Volume": [8_000_000, 9_000_000],
            "ATR10": [1.2, 1.3],  # uppercase form (prepare path)
            "atr10": [1.2, 1.3],  # lowercase form (latest_only path)
            "DollarVolume50": [20_000_000, 25_000_000],
            "Return_6D": [0.21, 0.25],  # uppercase form (prepare path)
            "return_6d": [0.21, 0.25],  # lowercase form (latest_only path)
            "UpTwoDays": [False, True],
            "HV50": [20.0, 20.0],
        }
        df = pd.DataFrame(data, index=dates)
        if pass_setup:
            # system6 prepare path will build filter/setup using provided columns
            # We can pre-set for safety
            hv_ok = pd.Series([True, True], index=dates)
            df["filter"] = (df["Low"] >= 5.0) & (df["DollarVolume50"] > 10_000_000) & hv_ok
            df["setup"] = df["filter"] & (df["Return_6D"] > 0.20) & df["UpTwoDays"]
        return df

    return _make


@pytest.fixture
def minimal_system7_df():
    """System7 の setup を満たす最小 DataFrame を返すファクトリー関数。"""

    def _make(pass_setup: bool = True) -> pd.DataFrame:
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
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

    return _make
