"""Sequential execution for one System5 legacy-protection migration.

The planner in ``system5_live_exit`` decides *what* the canonical stop is.
This module only executes one destructive cancel+replace safely: exact scoped
cancel, broker-confirmed settlement, canonical submit, and rollback to the
pre-cancel observed downside stop on failure.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Iterable

from common.alpaca_trading import PreparedExit, build_stop_rearm_after_failed_oco
from common.system5_protection_migration import is_system5_protection_migration


@dataclass(slots=True)
class MigrationExecution:
    success: bool
    safe: bool
    canonical: PreparedExit
    rollback: PreparedExit | None = None
    error: str | None = None
    canceled_client_order_id: str | None = None


def _walk_legs(order: Any):
    for leg in getattr(order, "legs", None) or []:
        yield leg
        yield from _walk_legs(leg)


def _status_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "").strip().lower()


def fetch_open_client_order_states(client: Any) -> dict[str, str] | None:
    """Return broker-open client_order_id -> status; unreadable is None."""
    try:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        orders = client.get_orders(
            filter=GetOrdersRequest(
                status=QueryOrderStatus.OPEN, nested=True, limit=500
            )
        )
    except Exception:
        return None

    out: dict[str, str] = {}
    for parent in orders or []:
        coid = str(getattr(parent, "client_order_id", "") or "")
        if coid:
            out[coid] = _status_value(getattr(parent, "status", None))
        for leg in _walk_legs(parent):
            leg_coid = str(getattr(leg, "client_order_id", "") or "")
            if leg_coid:
                out[leg_coid] = _status_value(getattr(leg, "status", None))
    return out


def fetch_open_client_order_ids(client: Any) -> set[str] | None:
    """Return broker-open client_order_ids, or None if state is unreadable."""
    states = fetch_open_client_order_states(client)
    return None if states is None else set(states)


def wait_for_client_order_absent(
    client: Any,
    client_order_id: str,
    *,
    timeout_seconds: float = 30.0,
    poll_seconds: float = 0.25,
) -> bool:
    """Wait until an exact order is no longer OPEN; unreadable is not PASS."""
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    while True:
        open_coids = fetch_open_client_order_ids(client)
        if open_coids is not None and client_order_id not in open_coids:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(max(0.01, float(poll_seconds)))


def execute_system5_protection_migration(
    po: PreparedExit,
    *,
    client: Any,
    canceler: Callable[[Any, Iterable[str]], dict[str, Any]],
    submitter: Callable[[PreparedExit], PreparedExit],
    cancel_timeout_seconds: float = 30.0,
    poll_seconds: float = 0.25,
) -> MigrationExecution:
    """Execute exactly one S5 cancel+replace, with rollback evidence."""
    if not is_system5_protection_migration(po):
        return MigrationExecution(False, True, po, error="not_system5_migration")

    cancel_coids = list(po.cancel_client_order_ids or [])
    if len(cancel_coids) != 1 or po.rollback_stop_price is None:
        return MigrationExecution(
            False,
            True,
            po,
            error="rollback_evidence_missing_or_ambiguous",
        )

    old_coid = cancel_coids[0]
    before = fetch_open_client_order_states(client)
    if before is None or old_coid not in before:
        return MigrationExecution(
            False,
            True,
            po,
            error="stale_or_unproven_old_protection",
        )

    # A previous run may already have received a cancel acknowledgement and then
    # timed out while Alpaca kept the order in pending_cancel.  Do not issue a
    # second cancel in that state; just wait for the original cancel to settle.
    if before.get(old_coid) != "pending_cancel":
        canceled = canceler(client, {old_coid})
        if old_coid not in set(canceled.get("coids") or []):
            return MigrationExecution(
                False,
                True,
                po,
                error="exact_cancel_not_acknowledged",
            )

    if not wait_for_client_order_absent(
        client,
        old_coid,
        timeout_seconds=cancel_timeout_seconds,
        poll_seconds=poll_seconds,
    ):
        return MigrationExecution(
            False,
            False,
            po,
            error="exact_cancel_not_settled_old_still_open",
            canceled_client_order_id=old_coid,
        )

    canonical_error: str | None = None
    try:
        result = submitter(po)
        if result.error:
            canonical_error = str(result.error)
        elif result.order_id:
            return MigrationExecution(
                True,
                True,
                result,
                canceled_client_order_id=old_coid,
            )
        else:
            canonical_error = "canonical_submit_without_broker_order_id"
    except Exception as exc:  # reconcile ambiguous broker outcome below
        canonical_error = str(exc)

    open_after = fetch_open_client_order_ids(client)
    if open_after is not None and po.client_order_id in open_after:
        po.error = None
        po.status = po.status or "open_reconciled"
        return MigrationExecution(
            True,
            True,
            po,
            canceled_client_order_id=old_coid,
        )

    rollback = build_stop_rearm_after_failed_oco(po)
    if rollback is None:
        return MigrationExecution(
            False,
            False,
            po,
            error=f"canonical_failed_no_rollback:{canonical_error}",
            canceled_client_order_id=old_coid,
        )

    try:
        rb_result = submitter(rollback)
        if rb_result.error or not rb_result.order_id:
            raise RuntimeError(
                rb_result.error or "rollback_submit_without_broker_order_id"
            )
        po.error = f"canonical_failed_rollback_armed:{canonical_error}"
        return MigrationExecution(
            False,
            True,
            po,
            rollback=rb_result,
            error=po.error,
            canceled_client_order_id=old_coid,
        )
    except Exception as exc:  # reconcile ambiguous rollback once
        open_final = fetch_open_client_order_ids(client)
        if open_final is not None and rollback.client_order_id in open_final:
            rollback.error = None
            rollback.status = rollback.status or "open_reconciled"
            po.error = f"canonical_failed_rollback_reconciled:{canonical_error}"
            return MigrationExecution(
                False,
                True,
                po,
                rollback=rollback,
                error=po.error,
                canceled_client_order_id=old_coid,
            )

        po.error = (
            "CRITICAL_unprotected_after_migration:" f"{canonical_error}; rollback={exc}"
        )
        return MigrationExecution(
            False,
            False,
            po,
            rollback=rollback,
            error=po.error,
            canceled_client_order_id=old_coid,
        )


__all__ = [
    "MigrationExecution",
    "execute_system5_protection_migration",
    "fetch_open_client_order_ids",
    "fetch_open_client_order_states",
    "wait_for_client_order_absent",
]
