from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from common.cache_manager import CacheManager, save_base_cache
from core.today_pipeline import BasicDataLoadResult, MissingDetail, load_basic_data_phase


def _make_settings(tmp_path) -> SimpleNamespace:
    rolling_dir = tmp_path / "rolling"
    full_dir = tmp_path / "full"
    base_dir = tmp_path / "base"
    rolling_dir.mkdir(parents=True, exist_ok=True)
    full_dir.mkdir(parents=True, exist_ok=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    rolling_cfg = SimpleNamespace(
        meta_file="_meta.json",
        base_lookback_days=5,
        buffer_days=0,
        max_staleness_days=2,
        prune_chunk_days=5,
        load_max_workers=None,
    )
    cache = SimpleNamespace(
        rolling_dir=str(rolling_dir),
        full_dir=str(full_dir),
        rolling=rolling_cfg,
    )
    return SimpleNamespace(
        cache=cache,
        DATA_CACHE_DIR=str(tmp_path),
    )


def _make_prefetched_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "open": [10, 11, 12, 13, 14, 15],
            "high": [11, 12, 13, 14, 15, 16],
            "low": [9, 10, 11, 12, 13, 14],
            "close": [10.5, 11.5, 12.5, 13.5, 14.5, 15.5],
            "volume": [1000, 1100, 1200, 1300, 1400, 1500],
            "sma25": [10] * 6,
            "sma50": [10] * 6,
            "sma100": [10] * 6,
            "sma150": [10] * 6,
            "sma200": [10] * 6,
            "atr14": [1] * 6,
            "atr40": [1] * 6,
            "roc200": [0.1] * 6,
        }
    )


def _write_full_cache(cm: CacheManager, symbol: str) -> None:
    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    base = pd.DataFrame(
        {
            "Date": dates,
            "Open": pd.Series(range(20), dtype="float"),
            "High": pd.Series(range(1, 21), dtype="float"),
            "Low": pd.Series(range(0, 20), dtype="float"),
            "Close": pd.Series(range(1, 21), dtype="float"),
            "Volume": [50_000 + i for i in range(20)],
            "SMA25": [10.0] * 20,
            "SMA50": [10.0] * 20,
            "SMA100": [10.0] * 20,
            "SMA150": [10.0] * 20,
            "SMA200": [10.0] * 20,
            "EMA20": [9.5] * 20,
            "EMA50": [9.5] * 20,
            "ATR10": [1.0] * 20,
            "ATR14": [1.0] * 20,
            "ATR40": [1.0] * 20,
            "ATR50": [1.0] * 20,
            "RSI3": [55.0] * 20,
            "RSI14": [60.0] * 20,
            "ROC200": [0.5] * 20,
            "HV50": [12.0] * 20,
            "DollarVolume20": [100_000.0] * 20,
            "DollarVolume50": [100_000.0] * 20,
        }
    )
    cm.write_atomic(
        base[["Date", "Open", "High", "Low", "Close", "Volume"]],
        symbol,
        "full",
    )
    save_base_cache(symbol, base)


def test_prefetched_data_reused(tmp_path) -> None:
    settings = _make_settings(tmp_path)
    cm = CacheManager(settings)
    frame = _make_prefetched_frame()

    result = load_basic_data_phase(
        ["AAA"],
        cache_manager=cm,
        settings=settings,
        symbol_data={"AAA": frame},
        base_cache={},
    )

    assert isinstance(result, BasicDataLoadResult)
    assert "AAA" in result.data
    assert result.stats.get("prefetched", 0) == 1
    assert not result.missing_details


def test_manual_intervention_required_when_rolling_missing(tmp_path) -> None:
    settings = _make_settings(tmp_path)
    cm = CacheManager(settings)
    _write_full_cache(cm, "BBB")

    logs: list[str] = []

    def capture(msg: str) -> None:
        logs.append(msg)

    result = load_basic_data_phase(
        ["BBB"],
        cache_manager=cm,
        settings=settings,
        base_cache={},
        log=capture,
    )

    assert "BBB" not in result.data
    assert result.stats.get("manual_rebuild_required", 0) == 1
    assert result.stats.get("rolling", 0) == 0
    assert result.missing_details
    detail = result.missing_details[0]
    assert isinstance(detail, MissingDetail)
    assert detail.symbol == "BBB"
    assert detail.action == "manual_rebuild_required"
    assert detail.resolved is False
    assert "manual_rebuild_required" in detail.note
    assert any("手動で rolling キャッシュを更新" in msg for msg in logs)


def test_missing_when_base_not_available(tmp_path) -> None:
    settings = _make_settings(tmp_path)
    cm = CacheManager(settings)

    result = load_basic_data_phase(
        ["CCC"],
        cache_manager=cm,
        settings=settings,
        base_cache={},
    )

    assert "CCC" not in result.data
    assert result.stats.get("manual_rebuild_required", 0) == 1
    assert len(result.missing_details) == 1
    detail = result.missing_details[0]
    assert detail.symbol == "CCC"
    assert detail.action == "manual_rebuild_required"
    assert detail.resolved is False
