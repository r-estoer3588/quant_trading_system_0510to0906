from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from common import soft_trailing as soft
from schedulers import runner


class FakeClient:
    def __init__(self, positions):
        self.positions = positions
        self.closed: list[str] = []

    def get_clock(self):
        return SimpleNamespace(is_open=True)

    def get_all_positions(self):
        return self.positions

    def close_position(self, symbol):
        self.closed.append(symbol)
        return SimpleNamespace(id=f"close-{symbol}", status="accepted")


def pos(symbol: str, qty: float, entry: float, current: float):
    return SimpleNamespace(
        symbol=symbol,
        qty=str(qty),
        side="long",
        avg_entry_price=str(entry),
        current_price=str(current),
        market_value=str(qty * current),
    )


def test_canonical_trailing_thresholds_are_hwm_based():
    assert soft.trailing_threshold(
        side="long", hwm=120.0, trail_pct=0.25
    ) == pytest.approx(90.0)
    assert soft.trailing_threshold(
        side="long", hwm=120.0, trail_pct=0.20
    ) == pytest.approx(96.0)


def test_hwm_ratchet_never_moves_down():
    assert (
        soft.update_hwm(side="long", previous=120.0, entry=100.0, current=110.0)
        == 120.0
    )
    assert (
        soft.update_hwm(side="long", previous=120.0, entry=100.0, current=130.0)
        == 130.0
    )


def test_fractional_s1_breach_closes_entire_position_without_integer_cast(
    monkeypatch, tmp_path
):
    client = FakeClient([pos("FRAC", 3.75, 100.0, 74.0)])
    monkeypatch.setattr(soft, "assert_paper_env", lambda: None)
    monkeypatch.setattr(
        soft,
        "load_tracker",
        lambda: {
            "FRAC": {
                "system": "system1",
                "entry_date": "2026-09-01",
                "entry_price": 100.0,
            }
        },
    )
    monkeypatch.setattr(soft, "load_symbol_system_map", lambda: {})
    monkeypatch.setattr(soft, "_order_system_index", lambda _client: {})
    monkeypatch.setattr(
        soft, "_bootstrap_high_from_cache", lambda *args, **kwargs: 100.0
    )
    monkeypatch.setattr(
        soft, "_write_status", lambda rows, stamp: tmp_path / "status.json"
    )
    monkeypatch.setattr(soft, "_append_jsonl", lambda *args, **kwargs: None)

    result = soft.run_soft_trailing(
        client=client,
        state_path=tmp_path / "trailing.json",
        now=datetime(2026, 9, 10, 15, 0),
    )
    assert client.closed == ["FRAC"]
    assert result["closed"] == 1
    assert result["rows"][0]["qty"] == pytest.approx(3.75)
    assert result["rows"][0]["stop_price"] == pytest.approx(75.0)
    assert result["rows"][0]["action"] == "close_market_submitted"


def test_whole_share_s1_is_ignored_to_avoid_racing_native_trailing(
    monkeypatch, tmp_path
):
    client = FakeClient([pos("WHOLE", 4.0, 100.0, 50.0)])
    monkeypatch.setattr(soft, "assert_paper_env", lambda: None)
    monkeypatch.setattr(soft, "load_tracker", lambda: {"WHOLE": {"system": "system1"}})
    monkeypatch.setattr(soft, "load_symbol_system_map", lambda: {})
    monkeypatch.setattr(soft, "_order_system_index", lambda _client: {})
    monkeypatch.setattr(
        soft, "_write_status", lambda rows, stamp: tmp_path / "status.json"
    )

    result = soft.run_soft_trailing(
        client=client, state_path=tmp_path / "trailing.json"
    )
    assert client.closed == []
    assert result["protected"] == 0


def test_s4_uses_20_percent_soft_trail(monkeypatch, tmp_path):
    client = FakeClient([pos("S4F", 2.5, 100.0, 79.0)])
    monkeypatch.setattr(soft, "assert_paper_env", lambda: None)
    monkeypatch.setattr(
        soft,
        "load_tracker",
        lambda: {
            "S4F": {
                "system": "system4",
                "entry_date": "2026-09-01",
                "entry_price": 100.0,
            }
        },
    )
    monkeypatch.setattr(soft, "load_symbol_system_map", lambda: {})
    monkeypatch.setattr(soft, "_order_system_index", lambda _client: {})
    monkeypatch.setattr(
        soft, "_bootstrap_high_from_cache", lambda *args, **kwargs: 100.0
    )
    monkeypatch.setattr(
        soft, "_write_status", lambda rows, stamp: tmp_path / "status.json"
    )
    monkeypatch.setattr(soft, "_append_jsonl", lambda *args, **kwargs: None)

    result = soft.run_soft_trailing(
        client=client, state_path=tmp_path / "trailing.json"
    )
    assert client.closed == ["S4F"]
    assert result["rows"][0]["stop_price"] == pytest.approx(80.0)


def test_market_closed_is_read_only(monkeypatch, tmp_path):
    client = FakeClient([pos("FRAC", 3.75, 100.0, 50.0)])
    client.get_clock = lambda: SimpleNamespace(is_open=False)
    monkeypatch.setattr(soft, "assert_paper_env", lambda: None)
    result = soft.run_soft_trailing(
        client=client, state_path=tmp_path / "trailing.json"
    )
    assert client.closed == []
    assert result["market_open"] is False
    assert not (tmp_path / "trailing.json").exists()


def test_existing_monitor_cron_now_supports_step_and_overnight_session():
    pred = runner.parse_cron("*/30 22-6 * * 1-5")
    assert pred(datetime(2026, 9, 11, 22, 0))  # Friday JST evening
    assert pred(datetime(2026, 9, 12, 0, 30))  # Saturday JST, belongs to Friday session
    assert pred(datetime(2026, 9, 12, 6, 0))
    assert not pred(
        datetime(2026, 9, 12, 22, 0)
    )  # Saturday session is not a weekday start
    assert not pred(datetime(2026, 9, 11, 21, 30))


def test_legacy_blanket_cancel_recreate_task_is_retired():
    assert (
        runner.TASKS["update_trailing_stops"]
        is runner.task_update_trailing_stops_retired
    )
    assert runner.TASKS["monitor_portfolio"] is runner.task_monitor_portfolio


def test_paper_submitter_and_open_run_keep_whole_share_entry_policy():
    root = Path(__file__).resolve().parents[1]
    submitter = (root / "scripts" / "paper_trading_submit.py").read_text(
        encoding="utf-8"
    )
    open_run = (root / "scripts" / "open_auto_run.py").read_text(encoding="utf-8")
    assert "PAPER_WHOLE_SHARE_ONLY = True" in submitter
    assert "prefer_fractional=False" in submitter
    assert "paper_trading_submit.py" in open_run
