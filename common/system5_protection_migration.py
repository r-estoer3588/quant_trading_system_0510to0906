"""Safe migration helpers for legacy System5 broker-resident protection.

System5 canonical protection is a standalone GTC stop using ATR10 frozen from the
completed bar before entry.  Older Paper runs may still have an OCO (immediate target)
or a stop built from a drifting ATR.  Replacing those orders necessarily releases the
broker quantity reservation first, so the caller must capture a downside-stop fallback
*before* canceling anything and re-arm it immediately if the canonical replacement
fails.

This module contains the observation/rollback primitives only.  It never cancels or
submits an order by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from common.alpaca_trading import ExitReasonCode, PreparedExit

SYSTEM5 = "system5"


@dataclass(frozen=True, slots=True)
class ProtectionFallback:
    client_order_id: str
    symbol: str
    qty: float
    side: str
    stop_price: float


def _value(value: Any) -> str:
    if value is None:
        return ""
    return str(getattr(value, "value", value))


def _positive_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _walk_legs(order: Any):
    for leg in getattr(order, "legs", None) or []:
        yield leg
        yield from _walk_legs(leg)


def observe_protection_fallbacks(
    client: Any, client_order_ids: Iterable[str]
) -> dict[str, ProtectionFallback]:
    """Read OPEN orders and preserve a downside stop for each requested parent coid.

    For an OCO the requested client_order_id belongs to the parent take-profit order;
    its stop price lives on a nested child leg.  For a standalone stop the price is on
    the parent itself.  Missing/ambiguous data is omitted deliberately: a caller must
    not cancel an order unless a rollback stop can be proven first.
    """
    want = {str(c).strip() for c in client_order_ids if str(c).strip()}
    if not want:
        return {}
    try:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        orders = client.get_orders(
            filter=GetOrdersRequest(
                status=QueryOrderStatus.OPEN,
                nested=True,
                limit=500,
            )
        )
    except Exception:
        return {}

    out: dict[str, ProtectionFallback] = {}
    for parent in orders or []:
        coid = str(getattr(parent, "client_order_id", "") or "")
        if coid not in want:
            continue
        symbol = str(getattr(parent, "symbol", "") or "").upper()
        qty = _positive_float(getattr(parent, "qty", None))
        side = _value(getattr(parent, "side", None)).lower()
        if side not in {"buy", "sell"}:
            side = ""

        stop = _positive_float(getattr(parent, "stop_price", None))
        if stop is None:
            stop_candidates = [
                px
                for leg in _walk_legs(parent)
                if (px := _positive_float(getattr(leg, "stop_price", None))) is not None
            ]
            # More than one distinct stop would be ambiguous.  Do not guess.
            unique = sorted({round(float(px), 8) for px in stop_candidates})
            if len(unique) == 1:
                stop = unique[0]

        if not symbol or qty is None or not side or stop is None:
            continue
        out[coid] = ProtectionFallback(
            client_order_id=coid,
            symbol=symbol,
            qty=qty,
            side=side,
            stop_price=float(stop),
        )
    return out


def is_system5_protect_stop(po: PreparedExit) -> bool:
    return (
        str(po.system or "").lower() == SYSTEM5
        and po.order_type == "stop"
        and po.reason == ExitReasonCode.PROTECT_STOP
    )


def is_system5_protection_migration(po: PreparedExit) -> bool:
    return is_system5_protect_stop(po) and bool(po.cancel_client_order_ids)


def system5_migrations_only(exits: list[PreparedExit]) -> list[PreparedExit]:
    """Return only System5 stop-protection work for the migration operator scope.

    Besides destructive cancel+replace proposals, keep a non-destructive canonical
    stop proposal.  This is required when a prior migration run timed out while the
    broker order was ``pending_cancel`` and that cancel settles before the next run;
    otherwise the migration-only scope would accidentally discard the recovery stop.
    Time/target exits and every other system remain excluded.
    """
    return [po for po in exits if is_system5_protect_stop(po)]


def defer_extra_system5_migrations(exits: list[PreparedExit]) -> int:
    """Allow at most one destructive System5 protection migration per run.

    Migration requires canceling the currently resident protection before placing the
    canonical stop.  Even with rollback evidence, canceling five symbols as one batch
    unnecessarily widens the simultaneous protection gap.  Keep the first proposal
    actionable and mark later proposals as visible deferrals.  The next run naturally
    advances to the next symbol because a successfully migrated stop is idempotent and
    a same-day rollback stop is not churned again.

    Returns the number of proposals deferred.  No broker I/O occurs here.
    """
    candidates = [po for po in exits if is_system5_protection_migration(po)]
    deferred = 0
    for po in candidates[1:]:
        po.skip_reason = "s5_migration_deferred:one_per_run"
        deferred += 1
    return deferred


__all__ = [
    "ProtectionFallback",
    "observe_protection_fallbacks",
    "is_system5_protect_stop",
    "is_system5_protection_migration",
    "system5_migrations_only",
    "defer_extra_system5_migrations",
]
