from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from common.alpaca_trading import ExitReasonCode, PreparedExit
from scripts.paper_exit_check import _post_entry_protection_only

ROOT = Path(__file__).resolve().parents[1]


def _po(symbol: str, reason: str, *, cancel=False) -> PreparedExit:
    return PreparedExit(
        symbol=symbol,
        system="system1",
        qty=10,
        side="sell",
        order_type="stop" if reason != ExitReasonCode.PROTECT_OCO else "oco",
        reason=reason,
        cancel_client_order_ids=["old"] if cancel else None,
    )


def test_scope_keeps_only_selected_non_destructive_protection():
    keep_stop = _po("NEW", ExitReasonCode.PROTECT_STOP)
    keep_oco = _po("NEW", ExitReasonCode.PROTECT_OCO)
    upgrade = _po("NEW", ExitReasonCode.PROTECT_OCO, cancel=True)
    time_exit = PreparedExit(
        symbol="NEW",
        system="system1",
        qty=10,
        side="sell",
        order_type="market",
        reason=ExitReasonCode.TIME,
    )
    other = _po("OTHER", ExitReasonCode.PROTECT_STOP)

    scoped = _post_entry_protection_only(
        [keep_stop, keep_oco, upgrade, time_exit, other], symbols={"NEW"}
    )
    assert scoped == [keep_stop, keep_oco]


def test_scope_can_filter_same_session_entry_date():
    today = _po("TODAY", ExitReasonCode.PROTECT_STOP)
    today.entry_date = "2026-09-11"
    old = _po("OLD", ExitReasonCode.PROTECT_STOP)
    old.entry_date = "2026-09-10"
    destructive = _po("TODAY2", ExitReasonCode.PROTECT_STOP, cancel=True)
    destructive.entry_date = "2026-09-11"
    assert _post_entry_protection_only(
        [today, old, destructive], entry_date="2026-09-11"
    ) == [today]


def _load_open_auto_run():
    path = ROOT / "scripts" / "open_auto_run.py"
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("open_auto_post_protect_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


oar = _load_open_auto_run()


def _args(**overrides):
    base = dict(
        date="2026-09-11",
        min_signals=10,
        poll_timeout=0.0,
        dry_run=False,
        skip_signals=True,
        allow_closed=True,
        force=True,
        flatten_all=False,
        no_publish=True,
        primary_root=".",
        thin_aborts_run=False,
        trigger="manual",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_reconcile_returns_filled_symbols(tmp_path, monkeypatch):
    monkeypatch.setattr(oar, "ROOT", tmp_path)
    runner = oar.Runner(_args())
    runner.paper_json.parent.mkdir(parents=True, exist_ok=True)
    runner.paper_json.write_text(
        json.dumps({"orders": [{"order_id": "o1", "symbol": "abc", "error": None}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "_client", lambda: object())
    import common.broker_alpaca as ba

    monkeypatch.setattr(
        ba, "get_orders_status_map", lambda _client, _ids: {"o1": "filled"}
    )

    assert runner.reconcile_entry_fills() == {"ABC"}
    assert runner.record["entry_filled_symbols"] == ["ABC"]


def test_post_entry_stage_is_scoped_to_filled_symbols(tmp_path, monkeypatch):
    monkeypatch.setattr(oar, "ROOT", tmp_path)
    runner = oar.Runner(_args())
    calls = []

    def fake_run_step(name, argv):
        calls.append((name, argv))
        return 0, "", ""

    monkeypatch.setattr(runner, "run_step", fake_run_step)
    rc = runner.post_entry_protection_stage({"ZZZ", "AAA"})

    assert rc == 0
    assert calls[0][0] == "post_entry_protection"
    argv = calls[0][1]
    assert "--protection-only-symbols" in argv
    assert argv[argv.index("--protection-only-symbols") + 1] == "AAA,ZZZ"
    assert "--today-entry-protection-only" in argv
    assert "--confirm" in argv and "--yes" in argv
    assert runner.record["post_entry_protection_status"] == "ok"


def test_recurring_sweep_is_scoped_and_handles_jst_midnight():
    sweep = (ROOT / "scripts" / "post_entry_protection_sweep.ps1").read_text(
        encoding="utf-8"
    )
    register = (ROOT / "scripts" / "register_post_entry_protection_task.ps1").read_text(
        encoding="utf-8"
    )
    assert (
        "--today-entry-protection-only" in sweep
        and "$now.Hour -lt 12" in sweep
        and "AddDays(-1)" in sweep
    )
    assert "--confirm" in sweep and "--yes" in sweep
    assert 'Interval = "PT15M"' in register and 'Duration = "PT8H"' in register
