from __future__ import annotations

from types import SimpleNamespace

from common import broker_alpaca as ba
from common.alpaca_trading import ExitReasonCode, PreparedExit
from scripts.paper_exit_check import (
    SYSTEM5_TARGET_NEXT_OPEN,
    _is_already_protected_non_exit,
    _is_mandatory_full_close,
)

HELD = '{"code":40310000,"existing_qty":"10","held_for_orders":"10"}'


def _po(reason: str, order_type: str = "market") -> PreparedExit:
    return PreparedExit(
        symbol="TEST",
        system="system1",
        qty=10,
        side="sell",
        order_type=order_type,
        reason=reason,
    )


def test_mandatory_market_closes_never_become_already_protected():
    for reason in (
        ExitReasonCode.TIME,
        ExitReasonCode.BREAKOUT,
        SYSTEM5_TARGET_NEXT_OPEN,
    ):
        po = _po(reason)
        assert _is_mandatory_full_close(po) is True
        assert _is_already_protected_non_exit(po, HELD) is False


def test_protection_submit_can_still_be_classified_already_protected():
    po = _po(ExitReasonCode.PROTECT_STOP, order_type="stop")
    assert _is_mandatory_full_close(po) is False
    assert _is_already_protected_non_exit(po, HELD) is True


class _SettlingClient:
    def __init__(self, remaining_reads: int):
        self.remaining_reads = remaining_reads

    def get_orders(self, _request):
        if self.remaining_reads > 0:
            self.remaining_reads -= 1
            return [SimpleNamespace(symbol="TEST", id="o1")]
        return []


def test_wait_for_no_open_orders_polls_until_qty_is_released():
    result = ba.wait_for_no_open_orders_for_symbols(
        _SettlingClient(remaining_reads=2),
        {"TEST"},
        timeout_seconds=0.2,
        poll_seconds=0.01,
    )
    assert result == {"settled": True, "pending_symbols": [], "error": None}


def test_wait_for_no_open_orders_reports_timeout_truthfully():
    result = ba.wait_for_no_open_orders_for_symbols(
        _SettlingClient(remaining_reads=999),
        {"TEST"},
        timeout_seconds=0.0,
        poll_seconds=0.01,
    )
    assert result["settled"] is False
    assert result["pending_symbols"] == ["TEST"]
    assert result["error"] is None
