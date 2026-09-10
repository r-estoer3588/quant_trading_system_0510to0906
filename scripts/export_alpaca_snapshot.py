"""Read-only Alpaca snapshot wrapper with authoritative trailing protection data.

The pre-2026-09-10 exporter is preserved verbatim as
``export_alpaca_snapshot_legacy.py``. This layer changes only S1/S4 trailing
protection observation/presentation data. Broker OPEN orders win, then broker
HWM, then (for legacy fractional positions only) the intraday soft-monitor HWM.
It never fabricates a trailing threshold from entry ATR and never uses a $0.01
display floor.

All accounting, P&L, ledger, exposure and win-rate calculations remain delegated
to the preserved exporter unchanged.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from common.trade_management import SYSTEM_TRADE_RULES
from scripts import export_alpaca_snapshot_legacy as _legacy

_LEGACY_BUILD_SNAPSHOT = _legacy.build_snapshot
ROOT = _legacy.ROOT
PAPER_BASE = _legacy.PAPER_BASE
SCHEMA = _legacy.SCHEMA
PROVIDER = _legacy.PROVIDER
_QTY_EPS = 1e-6


def __getattr__(name: str) -> Any:
    """Keep the legacy exporter helper surface compatible for callers/tests."""
    return getattr(_legacy, name)


def _f(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _enum_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "").strip().lower().split(".")[-1]


def _is_fractional(qty: Any) -> bool:
    q = _f(qty)
    return bool(q is not None and abs(abs(q) - round(abs(q))) > _QTY_EPS)


def _target_only(
    *, side: str, avg_entry: float, rules: Any, atr: dict[int, float]
) -> float | None:
    _legacy_stop, target = _legacy._estimate_stop_target(
        side=side, avg_entry=avg_entry, rules=rules, atr=atr
    )
    return target


def _estimate_stop_target(
    *,
    side: str,
    avg_entry: float,
    rules: Any,
    atr: dict[int, float],
    actual_stop: float | None = None,
    hwm: float | None = None,
    trail_pct: float | None = None,
) -> tuple[float | None, float | None]:
    """Resolve stop/target with real-data priority for trailing strategies.

    Trailing rules use: positive broker resting stop -> measured HWM -> ``None``.
    ATR is deliberately *not* a trailing display fallback. Non-trailing systems
    retain the canonical legacy ATR/target calculation.
    """
    if rules is None or not (avg_entry > 0):
        return None, None
    if not getattr(rules, "use_trailing_stop", False):
        return _legacy._estimate_stop_target(
            side=side, avg_entry=avg_entry, rules=rules, atr=atr
        )

    target = _target_only(side=side, avg_entry=avg_entry, rules=rules, atr=atr)
    stop = _f(actual_stop)
    if stop is not None and stop > 0:
        return round(stop, 4), target

    measured_hwm = _f(hwm)
    width = _f(trail_pct)
    if width is None:
        width = _f(getattr(rules, "trailing_stop_pct", None))
    if (
        measured_hwm is None
        or measured_hwm <= 0
        or width is None
        or not (0 < width < 1)
    ):
        return None, target
    if side == "long":
        stop = measured_hwm * (1.0 - width)
    elif side == "short":
        stop = measured_hwm * (1.0 + width)
    else:
        return None, target
    return round(stop, 4), target


def _fetch_open_protection(
    client: Any,
) -> tuple[bool, dict[str, list[dict[str, Any]]], str | None]:
    """Observe OPEN broker protection orders only; never infer failure as absence."""
    out: dict[str, list[dict[str, Any]]] = {}
    try:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        orders = client.get_orders(
            GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500)
        )
    except Exception as exc:
        return False, out, str(exc)

    for order in orders or []:
        symbol = str(getattr(order, "symbol", "") or "").upper()
        order_type = _enum_value(
            getattr(order, "type", None) or getattr(order, "order_type", None)
        )
        if not symbol or order_type not in {"stop", "stop_limit", "trailing_stop"}:
            continue
        trail_percent = _f(getattr(order, "trail_percent", None))
        if trail_percent is not None and trail_percent > 1:
            trail_percent /= 100.0
        out.setdefault(symbol, []).append(
            {
                "order_id": str(getattr(order, "id", "") or "") or None,
                "client_order_id": str(getattr(order, "client_order_id", "") or "")
                or None,
                "order_type": order_type,
                "side": _enum_value(getattr(order, "side", None)),
                "stop_price": _f(getattr(order, "stop_price", None)),
                "hwm": _f(getattr(order, "hwm", None)),
                "trail_pct": trail_percent,
            }
        )
    return True, out, None


def _load_soft_state() -> dict[str, dict[str, Any]]:
    path = ROOT / "data" / "trailing_stops.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except (OSError, ValueError):
        return {}


def _select_broker_threshold(
    rows: list[dict[str, Any]], *, side: str, strategy_trail_pct: float
) -> dict[str, Any] | None:
    """Pick the most protective measured OPEN broker threshold for a position."""
    candidates: list[dict[str, Any]] = []
    exit_side = "sell" if side == "long" else "buy"
    for row in rows:
        order_side = str(row.get("side") or "")
        if order_side and order_side != exit_side:
            continue
        stop = _f(row.get("stop_price"))
        source = "broker_order"
        hwm = _f(row.get("hwm"))
        width = _f(row.get("trail_pct")) or strategy_trail_pct
        if (stop is None or stop <= 0) and row.get("order_type") == "trailing_stop":
            if hwm is not None and hwm > 0 and 0 < width < 1:
                stop = hwm * (1.0 - width) if side == "long" else hwm * (1.0 + width)
                source = "broker_hwm"
        if stop is None or stop <= 0:
            continue
        candidates.append(
            {
                **row,
                "effective_stop": stop,
                "source": source,
                "effective_trail_pct": width,
            }
        )
    if not candidates:
        return None
    return (
        max(candidates, key=lambda row: row["effective_stop"])
        if side == "long"
        else min(candidates, key=lambda row: row["effective_stop"])
    )


def _enrich_trailing_protection(
    snapshot: dict[str, Any], client: Any
) -> dict[str, Any]:
    """Replace trailing display estimates only; accounting blocks are untouched."""
    observed_at = datetime.now(timezone.utc).isoformat()
    measured, broker, observation_error = _fetch_open_protection(client)
    soft = _load_soft_state()
    protection_rows: list[dict[str, Any]] = []

    for pos in snapshot.get("positions", []) or []:
        if not isinstance(pos, dict) or pos.get("exit_type") != "trailing":
            continue
        system = str(pos.get("system") or "").lower()
        rules = SYSTEM_TRADE_RULES.get(system)
        if rules is None or not getattr(rules, "use_trailing_stop", False):
            continue
        symbol = str(pos.get("symbol") or "").upper()
        side = str(pos.get("side") or "").lower()
        avg = _f(pos.get("avg_entry_price")) or 0.0
        current = _f(pos.get("current_price"))
        strategy_width = float(getattr(rules, "trailing_stop_pct", 0.0) or 0.0)
        fractional = _is_fractional(pos.get("qty"))
        selected = (
            _select_broker_threshold(
                broker.get(symbol, []),
                side=side,
                strategy_trail_pct=strategy_width,
            )
            if measured
            else None
        )

        actual_stop = None
        hwm = None
        width = strategy_width
        source: str | None = None
        order_id = None
        client_order_id = None
        order_type = None
        observed = "unmeasured" if not measured else "none"
        state = "unmeasured" if not measured else "missing_after_arm"

        if selected:
            actual_stop = _f(selected.get("effective_stop"))
            hwm = _f(selected.get("hwm"))
            width = _f(selected.get("effective_trail_pct")) or strategy_width
            source = str(selected.get("source") or "broker_order")
            order_id = selected.get("order_id")
            client_order_id = selected.get("client_order_id")
            order_type = selected.get("order_type")
            observed = "trailing" if order_type == "trailing_stop" else "stop"
            state = "verified"
        elif fractional:
            local = soft.get(symbol) or {}
            if (
                isinstance(local, dict)
                and str(local.get("system") or "").lower() == system
            ):
                hwm = _f(local.get("highest_price"))
                if hwm is not None and hwm > 0:
                    source = "soft_hwm"
                    width = _f(local.get("trailing_stop_pct")) or strategy_width
                    state = "soft_monitored"
                else:
                    state = "soft_unmeasured"
            else:
                state = "soft_unmeasured"
        elif measured:
            # Conservative lifecycle boundary: an entry dated today may still be in the
            # entry->protection arm interval. Older whole-share positions must have OPEN
            # native protection; measured absence is an actionable fault.
            entry_date = str(pos.get("entry_date") or "")[:10]
            state = (
                "pending_arm"
                if entry_date and entry_date == str(snapshot.get("date") or "")[:10]
                else "missing_after_arm"
            )

        stop, target = _estimate_stop_target(
            side=side,
            avg_entry=avg,
            rules=rules,
            atr={},
            actual_stop=actual_stop,
            hwm=hwm,
            trail_pct=width,
        )
        pos["stop_price_est"] = stop
        if target is not None:
            pos["target_price_est"] = target
        pos["stop_price_source"] = source
        pos["trailing_hwm"] = round(hwm, 4) if hwm is not None and hwm > 0 else None
        pos["protection_mode"] = (
            "synthetic_intraday" if fractional else "native_trailing_expected"
        )
        pos["protection_observed"] = observed
        pos["protection_state"] = state
        pos["protection_verified"] = state == "verified"
        pos["protection_observed_at"] = observed_at
        pos["protection_observation_error"] = (
            observation_error if not measured else None
        )
        pos["protection_order_id"] = order_id
        pos["resting_client_order_id"] = client_order_id
        pos["protection_order_type"] = order_type
        pos["broker_stop_price"] = (
            round(actual_stop, 4)
            if actual_stop is not None and actual_stop > 0
            else None
        )
        pos["broker_hwm"] = (
            round(hwm, 4) if selected and hwm is not None and hwm > 0 else None
        )
        if current is not None and current > 0 and stop is not None:
            pos["distance_to_stop_pct"] = round((stop - current) / current * 100.0, 3)
        else:
            pos["distance_to_stop_pct"] = None

        protection_rows.append(
            {
                "symbol": symbol,
                "system": system,
                "qty": pos.get("qty"),
                "fractional": fractional,
                "trail_pct": width,
                "stop_price": stop,
                "source": source,
                "hwm": pos.get("trailing_hwm"),
                "observed": observed,
                "state": state,
                "verified": state == "verified",
                "order_id": order_id,
                "client_order_id": client_order_id,
                "order_type": order_type,
            }
        )

    snapshot["trailing_protection"] = {
        "measured": measured,
        "observed_at": observed_at,
        "observation_error": observation_error,
        "priority": ["broker_order", "broker_hwm", "soft_hwm"],
        "rows": protection_rows,
    }
    return snapshot


def build_snapshot(
    client: Any, *, date_str: str, results_dir: Path, period: str
) -> dict[str, Any]:
    snapshot = _LEGACY_BUILD_SNAPSHOT(
        client, date_str=date_str, results_dir=results_dir, period=period
    )
    return _enrich_trailing_protection(snapshot, client)


def main(argv: list[str] | None = None) -> int:
    original = _legacy.build_snapshot
    _legacy.build_snapshot = build_snapshot
    try:
        return _legacy.main(argv)
    finally:
        _legacy.build_snapshot = original


if __name__ == "__main__":
    raise SystemExit(main())
