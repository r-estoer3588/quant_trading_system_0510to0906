from __future__ import annotations

from types import SimpleNamespace

from common.alpaca_trading import ExitReasonCode, PreparedExit
from common.system5_migration_execution import execute_system5_protection_migration

OLD = "protect-system5-TEST-20260901-protect-oco"
CANON = "protect-s5c-TEST-20260901-20260911"


class _Client:
    def __init__(self, coids: list[str]):
        self.orders = [self._order(c) for c in coids]

    @staticmethod
    def _order(coid: str):
        return SimpleNamespace(client_order_id=coid, legs=[])

    def get_orders(self, filter=None):  # noqa: A002 - Alpaca-compatible fake
        return list(self.orders)

    def add(self, coid: str) -> None:
        self.orders.append(self._order(coid))

    def remove(self, coid: str) -> None:
        self.orders = [o for o in self.orders if o.client_order_id != coid]


def _po() -> PreparedExit:
    return PreparedExit(
        symbol="TEST",
        system="system5",
        qty=10,
        side="sell",
        order_type="stop",
        reason=ExitReasonCode.PROTECT_STOP,
        entry_date="2026-09-01",
        stop_price=85.0,
        client_order_id=CANON,
        cancel_client_order_ids=[OLD],
        rollback_stop_price=82.5,
        time_in_force="gtc",
    )


def _cancel_remove(client: _Client, coids):
    touched = []
    for coid in coids:
        if any(o.client_order_id == coid for o in client.orders):
            client.remove(coid)
            touched.append(coid)
    return {"canceled": len(touched), "coids": touched}


def test_sequential_migration_cancels_exact_old_then_arms_canonical():
    client = _Client([OLD])
    po = _po()

    def submitter(candidate: PreparedExit) -> PreparedExit:
        assert not any(o.client_order_id == OLD for o in client.orders)
        client.add(candidate.client_order_id or "")
        candidate.order_id = "canonical-order-id"
        candidate.status = "accepted"
        return candidate

    result = execute_system5_protection_migration(
        po,
        client=client,
        canceler=_cancel_remove,
        submitter=submitter,
        cancel_timeout_seconds=0.1,
        poll_seconds=0.01,
    )
    assert result.success is True
    assert result.safe is True
    assert result.rollback is None
    assert po.order_id == "canonical-order-id"
    assert {o.client_order_id for o in client.orders} == {CANON}


def test_stale_old_protection_fails_closed_without_cancel_or_submit():
    client = _Client([])
    calls = {"cancel": 0, "submit": 0}

    def canceler(client, coids):
        calls["cancel"] += 1
        return _cancel_remove(client, coids)

    def submitter(candidate):
        calls["submit"] += 1
        return candidate

    result = execute_system5_protection_migration(
        _po(),
        client=client,
        canceler=canceler,
        submitter=submitter,
        cancel_timeout_seconds=0.0,
        poll_seconds=0.01,
    )
    assert result.success is False
    assert result.safe is True
    assert result.error == "stale_or_unproven_old_protection"
    assert calls == {"cancel": 0, "submit": 0}


def test_canonical_failure_rearms_observed_old_stop():
    client = _Client([OLD])
    po = _po()
    submitted: list[str] = []

    def submitter(candidate: PreparedExit) -> PreparedExit:
        submitted.append(candidate.client_order_id or "")
        if candidate.client_order_id == CANON:
            raise RuntimeError("canonical rejected")
        client.add(candidate.client_order_id or "")
        candidate.order_id = "rollback-order-id"
        candidate.status = "accepted"
        return candidate

    result = execute_system5_protection_migration(
        po,
        client=client,
        canceler=_cancel_remove,
        submitter=submitter,
        cancel_timeout_seconds=0.1,
        poll_seconds=0.01,
    )
    assert result.success is False
    assert result.safe is True
    assert result.rollback is not None
    assert result.rollback.stop_price == 82.5
    assert result.rollback.order_id == "rollback-order-id"
    assert submitted == [CANON, "protect-s5rb-TEST-20260901-20260911"]


def test_ambiguous_canonical_exception_is_reconciled_before_rollback():
    client = _Client([OLD])
    po = _po()
    submitted: list[str] = []

    def submitter(candidate: PreparedExit) -> PreparedExit:
        submitted.append(candidate.client_order_id or "")
        client.add(candidate.client_order_id or "")
        if candidate.client_order_id == CANON:
            raise RuntimeError("transport timeout after broker accept")
        raise AssertionError("rollback must not be submitted")

    result = execute_system5_protection_migration(
        po,
        client=client,
        canceler=_cancel_remove,
        submitter=submitter,
        cancel_timeout_seconds=0.1,
        poll_seconds=0.01,
    )
    assert result.success is True
    assert result.safe is True
    assert result.rollback is None
    assert po.status == "open_reconciled"
    assert submitted == [CANON]


def test_cancel_ack_without_settlement_is_not_declared_safe():
    client = _Client([OLD])
    po = _po()

    def canceler(client, coids):
        return {"canceled": 1, "coids": list(coids)}

    result = execute_system5_protection_migration(
        po,
        client=client,
        canceler=canceler,
        submitter=lambda candidate: candidate,
        cancel_timeout_seconds=0.0,
        poll_seconds=0.01,
    )
    assert result.success is False
    assert result.safe is False
    assert result.error == "exact_cancel_not_settled_old_still_open"
    assert any(o.client_order_id == OLD for o in client.orders)
