from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

import common.today_signals as today_signals
import common.utils_spy as utils_spy
from common.exit_planner import decide_exit_schedule
from strategies.system2_strategy import System2Strategy
from strategies.system4_strategy import System4Strategy
from strategies.system2_strategy import System2Strategy


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
        outputs=SimpleNamespace(results_csv_dir="results_csv_test"),
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


def _make_atr_frame(atr_col: str, atr_value: float = 2.0) -> pd.DataFrame:
    idx = pd.date_range("2024-06-03", periods=2, freq="B")
    data = {
        "Open": [100.0, 101.0],
        "High": [101.0, 102.0],
        "Low": [99.0, 100.0],
        "Close": [100.0, 101.0],
        atr_col: [atr_value, atr_value],
    }
    return pd.DataFrame(data, index=idx)


def test_filter_by_data_freshness_flags_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    base_day = pd.Timestamp("2024-06-14")

    def stub_latest(day=None):
        if day is None:
            return base_day
        candidate = pd.Timestamp(day).normalize()
        if candidate >= base_day:
            return base_day
        return candidate

    monkeypatch.setattr(today_signals, "get_latest_nyse_trading_day", stub_latest)

    skip_stats = today_signals.SkipStats()
    logs: list[str] = []

    def log(msg: str) -> None:
        logs.append(msg)

    prepared = {
        "OLD": pd.DataFrame(
            {"Close": [10.0, 11.0]},
            index=pd.to_datetime(["2024-04-10", "2024-04-12"]),
        ),
        "GAP": pd.DataFrame(
            {"Close": [12.0, 13.0]},
            index=pd.to_datetime(["2024-06-03", "2024-06-05"]),
        ),
        "FRESH": pd.DataFrame(
            {"Close": [14.0]},
            index=pd.to_datetime(["2024-06-14"]),
        ),
    }

    filtered, alerts, suppressed = today_signals._filter_by_data_freshness(
        prepared, pd.Timestamp("2024-06-15"), skip_stats, log
    )

    assert set(filtered.keys()) == {"GAP", "FRESH"}
    assert suppressed == [("OLD", pd.Timestamp("2024-04-12"))]
    assert alerts == [("GAP", pd.Timestamp("2024-06-05"))]
    assert skip_stats.counts.get("stale_over_month") == 1
    assert any("🔕" in msg for msg in logs)
    assert any("⚠️" in msg for msg in logs)


@pytest.mark.parametrize(
    ("system", "side", "atr_col", "expected_mult"),
    [
        ("system1", "long", "ATR20", 5.0),
        ("system2", "short", "ATR10", 3.0),
        ("system3", "long", "ATR10", 2.5),
        ("system4", "long", "ATR40", 1.5),
    ],
)
def test_compute_entry_stop_uses_system_specific_multiplier(
    system: str, side: str, atr_col: str, expected_mult: float
) -> None:
    df = _make_atr_frame(atr_col)
    entry_date = df.index[-1]
    candidate = {"entry_date": entry_date}
    strategy = SimpleNamespace(SYSTEM_NAME=system, config={})

    result = today_signals._compute_entry_stop(strategy, df, candidate, side)

    assert result is not None
    entry_price, stop_price = result
    expected_entry = float(df.iloc[-1]["Open"])
    assert entry_price == pytest.approx(expected_entry)

    atr_value = float(df.iloc[-2][atr_col])
    if side == "long":
        expected_stop = entry_price - expected_mult * atr_value
    else:
        expected_stop = entry_price + expected_mult * atr_value
    assert stop_price == pytest.approx(expected_stop)


def test_compute_entry_stop_prefers_config_multiplier() -> None:
    df = _make_atr_frame("ATR10", atr_value=1.0)
    entry_date = df.index[-1]
    candidate = {"entry_date": entry_date}
    strategy = SimpleNamespace(
        SYSTEM_NAME="system3", config={"stop_atr_multiple": "4"}
    )

    result = today_signals._compute_entry_stop(strategy, df, candidate, "long")

    assert result is not None
    entry_price, stop_price = result
    expected_entry = float(df.iloc[-1]["Open"])
    assert entry_price == pytest.approx(expected_entry)
    expected_stop = entry_price - 4.0 * float(df.iloc[-2]["ATR10"])
    assert stop_price == pytest.approx(expected_stop)


def test_system4_spy_gate_blocks_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    strategy = System4Strategy()
    dates, prepared_df = _make_prepared_frame()
    base_day = dates[-1]

    def fake_prepare(self, raw_data, **kwargs):
        return {"AAA": prepared_df}

    monkeypatch.setattr(System4Strategy, "prepare_data", fake_prepare)

    spy_df = pd.DataFrame(
        {
            "Close": [350.0, 349.0, 348.0],
            "SMA200": [360.0, 361.0, 362.0],
        },
        index=dates[:3],
    )

    monkeypatch.setattr(
        today_signals,
        "get_spy_with_indicators",
        lambda *_, **__: spy_df,
    )

    def fake_generate(self, prepared, market_df=None, **kwargs):
        return ({base_day: [{"symbol": "AAA", "entry_date": base_day}]}, None)

    monkeypatch.setattr(System4Strategy, "generate_candidates", fake_generate)
    _patch_calendar(monkeypatch, base_day)

    result = strategy.get_today_signals(
        {"AAA": prepared_df},
        market_df=spy_df,
        today=base_day,
    )

    assert result.empty
    assert list(result.columns) == today_signals.TODAY_SIGNAL_COLUMNS
    assert (
        result.attrs.get("zero_reason") == "setup_fail: SPY close <= SMA200"
    )


def test_system2_shortability_filter_excludes_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = System2Strategy()
    dates, prepared_df = _make_prepared_frame()
    base_day = dates[-1]

    def fake_prepare(self, raw_data, **kwargs):
        return {"BBB": prepared_df}

    monkeypatch.setattr(System2Strategy, "prepare_data", fake_prepare)

    def fake_generate(self, prepared, **kwargs):
        return ({base_day: [{"symbol": "BBB", "entry_date": base_day}]}, None)

    monkeypatch.setattr(System2Strategy, "generate_candidates", fake_generate)

    monkeypatch.setattr(
        "common.broker_alpaca.get_client", lambda paper=True: object()
    )
    monkeypatch.setattr(
        "common.broker_alpaca.get_shortable_map",
        lambda client, symbols: {sym: False for sym in symbols},
    )

    _patch_calendar(monkeypatch, base_day)

    result = strategy.get_today_signals(
        {"BBB": prepared_df},
        market_df=None,
        today=base_day,
    )

    assert result.empty
    assert list(result.columns) == today_signals.TODAY_SIGNAL_COLUMNS
    assert result.attrs.get("zero_reason") == "setup_pass_zero"


def test_decide_exit_schedule_marks_due_for_past_date() -> None:
    today = pd.Timestamp("2024-06-11")
    due, when = decide_exit_schedule("system2", pd.Timestamp("2024-06-05"), today)
    assert due is True
    assert when == "today_close"


def test_decide_exit_schedule_future_date_planned() -> None:
    today = pd.Timestamp("2024-06-11")
    due, when = decide_exit_schedule("system2", pd.Timestamp("2024-06-13"), today)
    assert due is False
    assert when == "tomorrow_close"


def test_decide_exit_schedule_system5_due_uses_tomorrow_open() -> None:
    today = pd.Timestamp("2024-06-11")
    due, when = decide_exit_schedule("system5", pd.Timestamp("2024-06-05"), today)
    assert due is True
    assert when == "tomorrow_open"


def test_decide_exit_schedule_system5_future_keeps_tomorrow_open() -> None:
    today = pd.Timestamp("2024-06-11")
    due, when = decide_exit_schedule("system5", pd.Timestamp("2024-06-13"), today)
    assert due is False
    assert when == "tomorrow_open"


def test_decide_exit_schedule_system3_future_uses_tomorrow_close() -> None:
    today = pd.Timestamp("2024-06-11")
    due, when = decide_exit_schedule("system3", pd.Timestamp("2024-06-12"), today)
    assert due is False
    assert when == "tomorrow_close"


def test_decide_exit_schedule_system6_due_uses_today_close() -> None:
    today = pd.Timestamp("2024-06-11")
    due, when = decide_exit_schedule("system6", pd.Timestamp("2024-06-10"), today)
    assert due is True
    assert when == "today_close"


def test_decide_exit_schedule_system6_future_uses_tomorrow_close() -> None:
    today = pd.Timestamp("2024-06-11")
    due, when = decide_exit_schedule("system6", pd.Timestamp("2024-06-12"), today)
    assert due is False
    assert when == "tomorrow_close"


def test_normalize_to_naive_day_converts_tokyo_evening_to_previous_us_day() -> None:
    tokyo_time = pd.Timestamp("2024-06-11 09:00", tz="Asia/Tokyo")
    normalized = utils_spy._normalize_to_naive_day(tokyo_time)
    assert normalized == pd.Timestamp("2024-06-10")


def test_normalize_to_naive_day_after_us_midnight_keeps_same_day() -> None:
    tokyo_time = pd.Timestamp("2024-06-11 23:00", tz="Asia/Tokyo")
    normalized = utils_spy._normalize_to_naive_day(tokyo_time)
    assert normalized == pd.Timestamp("2024-06-11")


def test_get_latest_nyse_trading_day_uses_us_calendar_for_tokyo_time() -> None:
    tokyo_time = pd.Timestamp("2024-06-11 09:00", tz="Asia/Tokyo")
    latest = utils_spy.get_latest_nyse_trading_day(tokyo_time)
    assert latest == pd.Timestamp("2024-06-10")
