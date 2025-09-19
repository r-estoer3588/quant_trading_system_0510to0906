from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

import common.today_signals as today_signals
import common.utils_spy as utils_spy
from strategies.system4_strategy import System4Strategy


def _make_prepared_frame() -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    dates = pd.date_range("2024-06-03", periods=5, freq="B")
    df = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "High": [101.0, 102.0, 103.0, 104.0, 105.0],
            "Low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "Close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "ATR40": [1.0] * 5,
            "RSI4": [10.0] * 5,
            "DollarVolume50": [150_000_000.0] * 5,
            "HV50": [20.0] * 5,
            "SMA200": [95.0] * 5,
            "filter": [True] * 5,
            "setup": [True] * 5,
        },
        index=dates,
    )
    return dates, df


@pytest.fixture(autouse=True)
def _stub_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = SimpleNamespace(
        risk=SimpleNamespace(max_positions=10),
        cache=SimpleNamespace(
            rolling=SimpleNamespace(max_staleness_days=2, max_stale_days=2)
        ),
        backtest=SimpleNamespace(top_n_rank=10),
    )
    monkeypatch.setattr(
        today_signals,
        "get_settings",
        lambda create_dirs=False: settings,
    )


def _patch_calendar(monkeypatch: pytest.MonkeyPatch, base_day: pd.Timestamp) -> None:
    def stub_latest(day=None):
        if day is None:
            return base_day
        return pd.Timestamp(day).normalize()

    def stub_next(day=None):
        if day is None:
            return base_day
        return pd.Timestamp(day).normalize()

    monkeypatch.setattr(utils_spy, "get_latest_nyse_trading_day", stub_latest)
    monkeypatch.setattr(utils_spy, "get_next_nyse_trading_day", stub_next)


def test_system4_no_spy_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    strategy = System4Strategy()
    dates, prepared_df = _make_prepared_frame()

    def fake_prepare(self, raw_data, **kwargs):
        return {"AAA": prepared_df}

    monkeypatch.setattr(System4Strategy, "prepare_data", fake_prepare)
    monkeypatch.setattr(
        today_signals,
        "get_spy_with_indicators",
        lambda *_, **__: None,
    )

    base_day = dates[-1]
    _patch_calendar(monkeypatch, base_day)

    called = False

    def fake_generate(self, prepared, market_df=None, **kwargs):
        nonlocal called
        called = True
        raise AssertionError(
            "generate_candidates should not be called when SPY missing"
        )

    monkeypatch.setattr(System4Strategy, "generate_candidates", fake_generate)

    result = strategy.get_today_signals(
        {"AAA": prepared_df},
        market_df=None,
        today=base_day,
    )

    assert result.empty
    assert list(result.columns) == today_signals.TODAY_SIGNAL_COLUMNS
    assert not called


def test_system4_fast_path_blocks_when_spy_below_sma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = System4Strategy()
    dates, prepared_df = _make_prepared_frame()
    base_day = dates[-1]

    def fake_prepare(self, raw_data, **kwargs):
        return {"AAA": prepared_df}

    monkeypatch.setattr(System4Strategy, "prepare_data", fake_prepare)

    fallback_spy = pd.DataFrame(
        {
            "Close": [350.0, 349.0, 348.0],
            "SMA200": [360.0, 361.0, 362.0],
        },
        index=dates[:3],
    )
    monkeypatch.setattr(
        today_signals,
        "get_spy_with_indicators",
        lambda *_, **__: fallback_spy,
    )

    called = False

    def fake_generate(self, prepared, market_df=None, **kwargs):
        nonlocal called
        called = True
        raise AssertionError(
            "generate_candidates should not run when SPY gate blocks"
        )

    monkeypatch.setattr(System4Strategy, "generate_candidates", fake_generate)
    _patch_calendar(monkeypatch, base_day)

    result = strategy.get_today_signals(
        {"AAA": prepared_df},
        market_df=None,
        today=base_day,
    )

    assert result.empty
    assert list(result.columns) == today_signals.TODAY_SIGNAL_COLUMNS
    assert not called


def test_system4_fast_path_produces_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = System4Strategy()
    dates, prepared_df = _make_prepared_frame()
    base_day = dates[-2]

    def fake_prepare(self, raw_data, **kwargs):
        return {"AAA": prepared_df}

    monkeypatch.setattr(System4Strategy, "prepare_data", fake_prepare)

    fallback_spy = pd.DataFrame(
        {
            "Close": [400.0, 401.0, 402.0],
            "SMA200": [395.0, 396.0, 397.0],
        },
        index=dates[:3],
    )
    monkeypatch.setattr(
        today_signals,
        "get_spy_with_indicators",
        lambda *_, **__: fallback_spy,
    )

    def fake_generate(prepared, market_df=None, **kwargs):
        raise AssertionError("fast path should handle candidate collection")

    monkeypatch.setattr(System4Strategy, "generate_candidates", fake_generate)
    _patch_calendar(monkeypatch, base_day)

    result = strategy.get_today_signals(
        {"AAA": prepared_df},
        market_df=None,
        today=base_day,
    )

    assert not result.empty
    assert set(result.columns) == set(today_signals.TODAY_SIGNAL_COLUMNS)
    assert result.iloc[0]["symbol"] == "AAA"
