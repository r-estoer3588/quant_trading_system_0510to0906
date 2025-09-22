from types import SimpleNamespace

from indicators_common import add_indicators
import numpy as np
import pandas as pd
import pytest

from common.cache_manager import CacheManager


class DummyRolling(SimpleNamespace):
    base_lookback_days = 300
    buffer_days = 30
    prune_chunk_days = 30
    meta_file = "_meta.json"


def _build_cm(tmp_path, file_format: str = "csv"):
    cache = SimpleNamespace(
        full_dir=tmp_path,
        rolling_dir=tmp_path,
        rolling=DummyRolling(),
        file_format=file_format,
    )
    settings = SimpleNamespace(cache=cache)
    return CacheManager(settings)


def test_read_handles_uppercase_date(tmp_path):
    cm = _build_cm(tmp_path)
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=2),
            "Open": [1, 2],
            "High": [1, 2],
            "Low": [1, 2],
            "Close": [1, 2],
            "Volume": [100, 200],
        }
    )
    df.to_csv(tmp_path / "AAA.csv", index=False)
    out = cm.read("AAA", "full")
    assert out is not None
    assert list(out.columns)[:6] == ["date", "open", "high", "low", "close", "volume"]
    assert pd.api.types.is_datetime64_any_dtype(out["date"])


def test_read_supports_feather(tmp_path):
    pytest.importorskip("pyarrow")
    cm = _build_cm(tmp_path, file_format="feather")
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=2),
            "open": [1.0, 2.0],
            "high": [1.5, 2.5],
            "low": [0.5, 1.5],
            "close": [1.2, 2.2],
            "volume": [100, 200],
        }
    )
    df.to_feather(tmp_path / "AAA.feather")
    out = cm.read("AAA", "rolling")
    assert out is not None
    assert "close" in out.columns
    assert len(out) == 2


def test_write_atomic_feather(tmp_path):
    pytest.importorskip("pyarrow")
    cm = _build_cm(tmp_path, file_format="feather")
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-02-01", periods=3),
            "open": [1.0, 2.0, 3.0],
            "high": [1.5, 2.5, 3.5],
            "low": [0.5, 1.5, 2.5],
            "close": [1.2, 2.2, 3.2],
            "volume": [100, 200, 300],
        }
    )
    cm.write_atomic(df, "BBB", "rolling")
    saved = pd.read_feather(tmp_path / "BBB.feather")
    assert list(saved.columns)[:5] == ["date", "open", "high", "low", "close"]
    assert len(saved) == len(df)


def test_nan_rate_ignores_leading_window(tmp_path, caplog):
    cm = _build_cm(tmp_path)
    periods = 260
    dates = pd.date_range("2023-01-02", periods=periods, freq="B")
    sma200 = [np.nan] * 200 + list(np.linspace(150, 160, periods - 200))
    df = pd.DataFrame(
        {
            "date": dates,
            "open": np.linspace(100, 120, periods),
            "high": np.linspace(101, 121, periods),
            "low": np.linspace(99, 119, periods),
            "close": np.linspace(100.5, 120.5, periods),
            "volume": np.linspace(1_000_000, 1_200_000, periods),
            "sma200": sma200,
        }
    )
    df.to_csv(tmp_path / "AAA.csv", index=False)

    with caplog.at_level("WARNING", logger="common.cache_manager"):
        cm.read("AAA", "full")

    assert not any("NaN率高" in message for message in caplog.messages)


def test_nan_rate_warns_when_all_nan(tmp_path, caplog):
    cm = _build_cm(tmp_path)
    dates = pd.date_range("2023-01-02", periods=20, freq="B")
    df = pd.DataFrame(
        {
            "date": dates,
            "open": 100,
            "high": 101,
            "low": 99,
            "close": 100.5,
            "volume": 1_000_000,
            "sma25": [np.nan] * len(dates),
        }
    )
    df.to_csv(tmp_path / "BBB.csv", index=False)

    with caplog.at_level("WARNING", logger="common.cache_manager"):
        cm.read("BBB", "full")

    assert any("NaN率高" in message for message in caplog.messages)


def _prepare_enriched_prices(periods: int) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "Date": pd.date_range("2023-01-02", periods=periods, freq="B"),
            "Open": np.linspace(10, 20, periods),
            "High": np.linspace(10.5, 20.5, periods),
            "Low": np.linspace(9.5, 19.5, periods),
            "Close": np.linspace(10.1, 20.1, periods),
            "Volume": np.linspace(1_000_000, 1_200_000, periods),
        }
    )
    base["date"] = base["Date"]
    enriched = add_indicators(base)
    enriched.columns = [str(c).lower() for c in enriched.columns]
    cols = pd.Index(enriched.columns)
    enriched = enriched.loc[:, ~cols.duplicated(keep="first")]
    ordered = ["date"] + [col for col in enriched.columns if col != "date"]
    return enriched.loc[:, ordered]


def test_upsert_recomputes_indicators(tmp_path):
    cm = _build_cm(tmp_path)
    initial = _prepare_enriched_prices(360)
    cm.write_atomic(initial, "AMRZ", "full")
    cm.write_atomic(initial.tail(cm._rolling_target_len), "AMRZ", "rolling")

    new_dates = pd.date_range(
        initial["date"].iloc[-1] + pd.offsets.BDay(),
        periods=5,
        freq="B",
    )
    new_rows = pd.DataFrame(
        {
            "date": new_dates,
            "open": np.linspace(21, 21.4, len(new_dates)),
            "high": np.linspace(21.5, 21.9, len(new_dates)),
            "low": np.linspace(20.5, 20.9, len(new_dates)),
            "close": np.linspace(21.1, 21.5, len(new_dates)),
            "volume": np.linspace(1_210_000, 1_250_000, len(new_dates)),
        }
    )

    cm.upsert_both("AMRZ", new_rows)

    updated_full = cm.read("AMRZ", "full")
    appended = updated_full[updated_full["date"] >= new_dates.min()]
    for col in ("sma25", "atr10", "rsi3", "dollarvolume20"):
        assert not appended[col].isna().any()

    updated_roll = cm.read("AMRZ", "rolling")
    tail = updated_roll.tail(len(new_dates))
    for col in ("sma25", "atr10", "rsi3", "dollarvolume20"):
        assert not tail[col].isna().any()
