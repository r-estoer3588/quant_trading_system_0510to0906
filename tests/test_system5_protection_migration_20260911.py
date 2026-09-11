from __future__ import annotations

from types import SimpleNamespace

from common.alpaca_trading import (
    ExitReasonCode,
    PreparedExit,
    build_stop_rearm_after_failed_oco,
)
from common.system5_protection_migration import (
    defer_extra_system5_migrations,
    defer_system5_migrations_for_recovery,
    observe_protection_fallbacks,
    system5_migrations_only,
)


class _Client:
    def __init__(self, orders):
        self.orders = orders

    def get_orders(self, filter=None):  # noqa: A002 - mirrors Alpaca SDK
        return self.orders


def _order(**kwargs):
    defaults = {
        "client_order_id": "",
        "symbol": "TEST",
        "qty": "10",
        "side": "sell",
        "stop_price": None,
        "legs": [],
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_nested_oco_stop_is_captured_before_cancel():
    coid = "protect-system5-TEST-20260901-protect-oco"
    parent = _order(
        client_order_id=coid,
        legs=[_order(client_order_id="child", stop_price="84.25")],
    )
    got = observe_protection_fallbacks(_Client([parent]), {coid})
    assert got[coid].symbol == "TEST"
    assert got[coid].qty == 10.0
    assert got[coid].side == "sell"
    assert got[coid].stop_price == 84.25


def test_ambiguous_nested_stops_fail_closed():
    coid = "protect-system5-TEST-20260901-protect-oco"
    parent = _order(
        client_order_id=coid,
        legs=[
            _order(client_order_id="child1", stop_price="84.25"),
            _order(client_order_id="child2", stop_price="83.00"),
        ],
    )
    assert observe_protection_fallbacks(_Client([parent]), {coid}) == {}


def test_standalone_stop_is_captured_directly():
    coid = "protect-system5-TEST-20260901-protect-stop"
    parent = _order(client_order_id=coid, stop_price="82.5")
    got = observe_protection_fallbacks(_Client([parent]), {coid})
    assert got[coid].stop_price == 82.5


def test_system5_failed_migration_rearms_observed_old_stop_not_new_stop():
    po = PreparedExit(
        symbol="TEST",
        system="system5",
        qty=10,
        side="sell",
        order_type="stop",
        reason=ExitReasonCode.PROTECT_STOP,
        entry_date="2026-09-01",
        stop_price=85.0,
        client_order_id="protect-s5c-TEST-20260901-20260911",
        cancel_client_order_ids=["protect-system5-TEST-20260901-protect-oco"],
        rollback_stop_price=82.5,
        time_in_force="gtc",
    )
    rollback = build_stop_rearm_after_failed_oco(po)
    assert rollback is not None
    assert rollback.order_type == "stop"
    assert rollback.stop_price == 82.5
    assert rollback.client_order_id == "protect-s5rb-TEST-20260901-20260911"
    assert len(rollback.client_order_id) <= 48


def test_system5_migration_without_rollback_evidence_cannot_build_rearm():
    po = PreparedExit(
        symbol="TEST",
        system="system5",
        qty=10,
        side="sell",
        order_type="stop",
        reason=ExitReasonCode.PROTECT_STOP,
        entry_date="2026-09-01",
        stop_price=85.0,
        client_order_id="protect-s5c-TEST-20260901-20260911",
        cancel_client_order_ids=["protect-system5-TEST-20260901-protect-oco"],
        rollback_stop_price=None,
        time_in_force="gtc",
    )
    assert build_stop_rearm_after_failed_oco(po) is None


def _migration_po(symbol: str) -> PreparedExit:
    return PreparedExit(
        symbol=symbol,
        system="system5",
        qty=10,
        side="sell",
        order_type="stop",
        reason=ExitReasonCode.PROTECT_STOP,
        entry_date="2026-09-01",
        stop_price=85.0,
        client_order_id=f"protect-s5c-{symbol}-20260901-20260911",
        cancel_client_order_ids=[f"protect-system5-{symbol}-20260901-protect-oco"],
        rollback_stop_price=82.5,
        time_in_force="gtc",
    )


def test_only_one_system5_migration_is_actionable_per_run():
    first = _migration_po("AAA")
    second = _migration_po("BBB")
    third = _migration_po("CCC")
    normal = PreparedExit(
        symbol="OTHER",
        system="system2",
        qty=5,
        side="sell",
        order_type="market",
        reason=ExitReasonCode.TIME,
    )
    rows = [normal, first, second, third]
    deferred = defer_extra_system5_migrations(rows)
    assert deferred == 2
    assert first.skip_reason is None
    assert second.skip_reason == "s5_migration_deferred:one_per_run"
    assert third.skip_reason == "s5_migration_deferred:one_per_run"
    assert normal.skip_reason is None


def test_single_system5_migration_is_not_deferred():
    only = _migration_po("AAA")
    assert defer_extra_system5_migrations([only]) == 0
    assert only.skip_reason is None


def test_system5_migration_only_scope_excludes_unrelated_exits():
    migration = _migration_po("AAA")
    deferred = _migration_po("BBB")
    deferred.skip_reason = "s5_migration_deferred:one_per_run"
    unrelated = PreparedExit(
        symbol="OTHER",
        system="system3",
        qty=5,
        side="sell",
        order_type="market",
        reason=ExitReasonCode.TIME,
    )
    scoped = system5_migrations_only([unrelated, migration, deferred])
    assert scoped == [migration, deferred]
    assert unrelated not in scoped


def test_system5_migration_scope_keeps_recovery_stop_after_cancel_settles():
    recovery = PreparedExit(
        symbol="HTFL",
        system="system5",
        qty=43,
        side="sell",
        order_type="stop",
        reason=ExitReasonCode.PROTECT_STOP,
        entry_date="2026-09-08",
        stop_price=41.2,
        client_order_id="protect-system5-HTFL-20260908-protect-stop",
        time_in_force="gtc",
    )
    unrelated = PreparedExit(
        symbol="OTHER",
        system="system2",
        qty=1,
        side="sell",
        order_type="stop",
        reason=ExitReasonCode.PROTECT_STOP,
    )
    scoped = system5_migrations_only([unrelated, recovery])
    assert scoped == [recovery]


def test_recovery_stop_defers_every_actionable_destructive_migration():
    recovery = PreparedExit(
        symbol="REC",
        system="system5",
        qty=10,
        side="sell",
        order_type="stop",
        reason=ExitReasonCode.PROTECT_STOP,
        entry_date="2026-09-01",
        stop_price=80.0,
        client_order_id="protect-system5-REC-20260901-protect-stop",
        time_in_force="gtc",
    )
    first = _migration_po("AAA")
    second = _migration_po("BBB")
    rows = [recovery, first, second]
    assert defer_system5_migrations_for_recovery(rows) == 2
    assert recovery.skip_reason is None
    assert first.skip_reason == "s5_migration_deferred:recovery_first"
    assert second.skip_reason == "s5_migration_deferred:recovery_first"


def test_recovery_priority_preserves_existing_one_per_run_deferral():
    recovery = PreparedExit(
        symbol="REC",
        system="system5",
        qty=10,
        side="sell",
        order_type="stop",
        reason=ExitReasonCode.PROTECT_STOP,
        stop_price=80.0,
        client_order_id="protect-system5-REC-20260901-protect-stop",
        time_in_force="gtc",
    )
    first = _migration_po("AAA")
    second = _migration_po("BBB")
    rows = [recovery, first, second]
    assert defer_extra_system5_migrations(rows) == 1
    assert defer_system5_migrations_for_recovery(rows) == 1
    assert first.skip_reason == "s5_migration_deferred:recovery_first"
    assert second.skip_reason == "s5_migration_deferred:one_per_run"
