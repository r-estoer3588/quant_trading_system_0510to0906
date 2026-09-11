"""System5 live-exit adapter that preserves the Bensdorp strategy contract.

System5 is intentionally special here because its documented/backtest profit-taking
semantics are *not* a broker-resident take-profit order:

* entry: previous close - 3% LIMIT (handled elsewhere)
* stop: entry - 3 * ATR10, where ATR10 is frozen from the completed bar before entry
* target trigger: High reaches entry + 1 * that same frozen ATR10
* target execution: market close at the NEXT session open
* timeout: observe six sessions after entry; if neither stop nor target fired, close at
  the NEXT (seventh) session open

The generic Alpaca protection builder can express a resting target/OCO, but that would
sell at the target touch itself and therefore changes the strategy. This module keeps
System5's trigger and execution timing explicit and side-effect free.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from common.alpaca_trading import (
    ExitReasonCode,
    PositionSnapshot,
    PreparedExit,
    compute_holding_days,
    protective_stop_price,
    round_to_alpaca_tick,
)
from common.trade_management import SYSTEM_TRADE_RULES
from common.trading_days import add_trading_days, count_trading_days

SYSTEM5 = "system5"
SYSTEM5_TARGET_NEXT_OPEN = "system5_target_next_open"
_TARGET_SUFFIX = "exit-target-next-open"
_STOP_SUFFIX = "protect-stop"
_STOP_REARM_SUFFIX = "protect-stop-rearm"
_OCO_SUFFIX = "protect-oco"
_TARGET_PROTECT_SUFFIX = "protect-target"
_MIGRATED_STOP_PREFIX = "protect-s5c"
_ROLLBACK_STOP_PREFIX = "protect-s5rb"


@dataclass(frozen=True, slots=True)
class System5ExitState:
    entry_atr10: float | None
    target_price: float | None
    target_hit_date: str | None
    holding_days: int
    timeout_exit_date: str | None
    target_exit_due: bool
    timeout_exit_due: bool


def _history_dates(df: pd.DataFrame) -> pd.Series:
    """Best-effort normalized daily dates without assuming one cache CSV layout."""
    for name in ("Date", "date", "datetime", "Datetime", "timestamp", "Timestamp"):
        if name in df.columns:
            return pd.to_datetime(df[name], errors="coerce").dt.normalize()

    # Rolling CSVs commonly persist an unnamed DatetimeIndex in column 0. Only accept
    # it when most values parse as dates; otherwise fail closed rather than guessing.
    if len(df.columns):
        candidate = pd.to_datetime(df.iloc[:, 0], errors="coerce")
        if len(candidate) and float(candidate.notna().mean()) >= 0.8:
            return candidate.dt.normalize()
    return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")


def _numeric_col(df: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    lower = {str(col).lower(): col for col in df.columns}
    for name in names:
        col = lower.get(name.lower())
        if col is not None:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(float("nan"), index=df.index)


def load_history(rolling_dir: Path, symbol: str) -> pd.DataFrame | None:
    """Load one rolling CSV. Missing/invalid data is represented as None."""
    path = rolling_dir / f"{str(symbol).upper()}.csv"
    try:
        df = pd.read_csv(path)
    except (OSError, ValueError):
        return None
    return df if not df.empty else None


def entry_atr10_from_history(
    df: pd.DataFrame | None, entry_date: str | None
) -> float | None:
    """ATR10 from the last completed bar strictly before the actual entry date.

    This mirrors ``System5Strategy.compute_entry`` / ``compute_exit``: the strategy
    freezes ATR from ``entry_idx - 1`` and does not move the target or stop as later ATR
    changes. Returning None is deliberate when the historical row cannot be proven.
    """
    if df is None or df.empty or not entry_date:
        return None
    try:
        entry = pd.Timestamp(str(entry_date)[:10]).normalize()
    except Exception:
        return None
    dates = _history_dates(df)
    atr10 = _numeric_col(df, "atr10", "ATR10", "atr_10", "ATR_10")
    mask = dates.notna() & (dates < entry) & atr10.notna() & (atr10 > 0)
    if not bool(mask.any()):
        return None
    value = float(atr10.loc[mask].iloc[-1])
    return value if value > 0 else None


def target_hit_date_from_history(
    df: pd.DataFrame | None,
    *,
    entry_date: str | None,
    today: str,
    target_price: float | None,
    max_observation_sessions: int = 6,
) -> str | None:
    """First completed post-entry session whose High reached the frozen target.

    ``today`` is excluded because open-run executes near the session open; using today's
    unfinished High would turn a next-open rule into a same-session exit. Observation
    is capped to the first six post-entry sessions, matching the backtest loop.
    """
    if (
        df is None
        or df.empty
        or not entry_date
        or not target_price
        or target_price <= 0
    ):
        return None
    try:
        entry = pd.Timestamp(str(entry_date)[:10]).normalize()
        now = pd.Timestamp(str(today)[:10]).normalize()
    except Exception:
        return None
    dates = _history_dates(df)
    high = _numeric_col(df, "High", "high")
    rows: list[tuple[pd.Timestamp, float]] = []
    for idx in df.index:
        d = dates.loc[idx]
        h = high.loc[idx]
        if pd.isna(d) or pd.isna(h) or not (entry < d < now):
            continue
        # The strategy checks only days 1..6 after entry before timeout next open.
        elapsed = count_trading_days(entry.date(), d.date())
        if 1 <= elapsed <= int(max_observation_sessions):
            rows.append((d, float(h)))
    rows.sort(key=lambda item: item[0])
    for d, h in rows:
        if h >= float(target_price):
            return d.date().isoformat()
    return None


def evaluate_system5_state(
    snap: PositionSnapshot,
    *,
    today: str,
    history: pd.DataFrame | None,
) -> System5ExitState:
    rules = SYSTEM_TRADE_RULES[SYSTEM5]
    holding = compute_holding_days(snap.entry_date, today) or 0
    atr10 = entry_atr10_from_history(history, snap.entry_date)
    target = (
        float(snap.avg_entry_price) + atr10 * float(rules.profit_target_value)
        if atr10 is not None and snap.avg_entry_price > 0
        else None
    )
    hit = target_hit_date_from_history(
        history,
        entry_date=snap.entry_date,
        today=today,
        target_price=target,
        max_observation_sessions=int(rules.max_holding_days),
    )
    target_due = False
    if hit:
        try:
            target_due = (
                count_trading_days(
                    date.fromisoformat(hit), date.fromisoformat(str(today)[:10])
                )
                >= 1
            )
        except ValueError:
            target_due = False

    timeout_date = None
    if snap.entry_date:
        try:
            d0 = date.fromisoformat(str(snap.entry_date)[:10])
            timeout_date = add_trading_days(
                d0, int(rules.max_holding_days) + 1
            ).isoformat()
        except ValueError:
            timeout_date = None
    # Six sessions are observed in full. Timeout therefore starts at session seven.
    timeout_due = holding >= int(rules.max_holding_days) + 1
    return System5ExitState(
        entry_atr10=atr10,
        target_price=target,
        target_hit_date=hit,
        holding_days=holding,
        timeout_exit_date=timeout_date,
        target_exit_due=target_due,
        timeout_exit_due=timeout_due,
    )


def _exit_coid(snap: PositionSnapshot, today: str, suffix: str) -> str:
    compact = str(today).replace("-", "")[:8]
    return f"exit-{SYSTEM5}-{snap.symbol}-{compact}-{suffix}"


def _protect_base(snap: PositionSnapshot) -> str:
    entry = (snap.entry_date or "").replace("-", "")
    return f"protect-{SYSTEM5}-{snap.symbol}-{entry}"


def _short_protect_tag(snap: PositionSnapshot) -> str:
    return (snap.entry_date or "noentry").replace("-", "")[:8] or "noentry"


def _migrated_stop_coid(snap: PositionSnapshot, today: str) -> str:
    day = str(today).replace("-", "")[:8]
    return f"{_MIGRATED_STOP_PREFIX}-{snap.symbol.upper()}-{_short_protect_tag(snap)}-{day}"


def _rollback_stop_prefix(snap: PositionSnapshot) -> str:
    return f"{_ROLLBACK_STOP_PREFIX}-{snap.symbol.upper()}-{_short_protect_tag(snap)}-"


def _stop_price_matches(observed: float | None, expected: float | None) -> bool | None:
    if observed is None or expected is None:
        return None
    tolerance = 0.011 if float(expected) >= 1.0 else 0.00011
    return abs(float(observed) - float(expected)) <= tolerance


def build_system5_exit_orders(
    snap: PositionSnapshot,
    *,
    today: str,
    history: pd.DataFrame | None,
    existing_protect_coids: set[str] | None = None,
    existing_exit_coids: set[str] | None = None,
    existing_protect_stop_prices: dict[str, float] | None = None,
    coverage_out: list[dict[str, Any]] | None = None,
) -> list[PreparedExit]:
    """Build System5 exits without altering the strategy.

    Priority matches the backtest: an already-observed target wins at the next open;
    otherwise timeout can fire at the seventh open; otherwise only the frozen-ATR stop
    is broker-resident. No native target/OCO is created for System5.

    Existing pre-fix protection is migrated only through a scoped cancel+replace
    proposal.  The runner snapshots the old downside stop before it cancels anything
    and rolls that stop back immediately if replacement fails.  This builder remains
    pure and never touches the broker.
    """
    if str(snap.system or "").lower() != SYSTEM5 or snap.abs_qty <= 0:
        return []
    rules = SYSTEM_TRADE_RULES[SYSTEM5]
    state = evaluate_system5_state(snap, today=today, history=history)
    existing_protect = existing_protect_coids or set()
    existing_exit = existing_exit_coids or set()
    observed_stop_prices = existing_protect_stop_prices or {}
    close_side = "sell" if snap.side == "long" else "buy"

    if state.target_exit_due:
        coid = _exit_coid(snap, today, _TARGET_SUFFIX)
        if coid in existing_exit:
            return []
        return [
            PreparedExit(
                symbol=snap.symbol,
                system=SYSTEM5,
                qty=snap.exit_qty(),
                side=close_side,
                order_type="market",
                reason=SYSTEM5_TARGET_NEXT_OPEN,
                entry_date=snap.entry_date,
                holding_days=state.holding_days,
                max_holding_days=int(rules.max_holding_days),
                client_order_id=coid,
                dry_run=True,
                time_in_force="day",
            )
        ]

    if state.timeout_exit_due:
        coid = _exit_coid(snap, today, "exit-time")
        if coid in existing_exit:
            return []
        return [
            PreparedExit(
                symbol=snap.symbol,
                system=SYSTEM5,
                qty=snap.exit_qty(),
                side=close_side,
                order_type="market",
                reason=ExitReasonCode.TIME,
                entry_date=snap.entry_date,
                holding_days=state.holding_days,
                max_holding_days=int(rules.max_holding_days),
                client_order_id=coid,
                dry_run=True,
                time_in_force="day",
            )
        ]

    base = _protect_base(snap)
    stop_coid = f"{base}-{_STOP_SUFFIX}"
    rearm_coid = f"{base}-{_STOP_REARM_SUFFIX}"
    oco_coid = f"{base}-{_OCO_SUFFIX}"
    target_coid = f"{base}-{_TARGET_PROTECT_SUFFIX}"
    today_compact = str(today).replace("-", "")[:8]
    migrated_prefix = (
        f"{_MIGRATED_STOP_PREFIX}-{snap.symbol.upper()}-{_short_protect_tag(snap)}-"
    )
    rollback_prefix = _rollback_stop_prefix(snap)
    migrated_coids = sorted(
        c for c in existing_protect if c.startswith(migrated_prefix)
    )
    rollback_coids = sorted(
        c for c in existing_protect if c.startswith(rollback_prefix)
    )

    expected_stop: float | None = None
    if state.entry_atr10 is not None:
        raw_stop = protective_stop_price(
            side=snap.side,
            avg_entry_price=snap.avg_entry_price,
            rules=rules,
            atr_value=state.entry_atr10,
            symbol=snap.symbol,
        )
        if raw_stop is not None and raw_stop > 0:
            expected_stop = round_to_alpaca_tick(raw_stop)

    related = []
    for coid in (stop_coid, rearm_coid, oco_coid, target_coid):
        if coid in existing_protect:
            related.append(coid)
    related.extend(migrated_coids)
    related.extend(rollback_coids)
    related = sorted(set(related))

    cancel_for_migration: str | None = None
    migration_detail: str | None = None
    keep_existing = False

    if len(related) > 1:
        # More than one full-quantity protection order is unexpected.  Never guess
        # which one can be removed; leave all broker state untouched and surface it.
        keep_existing = True
        migration_detail = "ambiguous_multiple_protection"
    elif len(related) == 1:
        current = related[0]
        if current in (oco_coid, target_coid):
            cancel_for_migration = current
            migration_detail = "legacy_target_or_oco_migration_pending"
        elif current.startswith(rollback_prefix):
            if current.endswith(f"-{today_compact}"):
                # A failed migration already rolled protection back in this run/day.
                # Do not churn it again until a later session gets a fresh coid.
                keep_existing = True
                migration_detail = "rollback_stop_active_retry_next_session"
            else:
                cancel_for_migration = current
                migration_detail = "rollback_stop_retry_pending"
        else:
            observed = observed_stop_prices.get(current)
            matches = _stop_price_matches(observed, expected_stop)
            if matches is False:
                current_migration_coid = _migrated_stop_coid(snap, today)
                if current == current_migration_coid:
                    keep_existing = True
                    migration_detail = "migrated_stop_mismatch_retry_next_session"
                else:
                    cancel_for_migration = current
                    migration_detail = "legacy_stop_price_migration_pending"
            else:
                # Unknown observed price is kept fail-safe.  A read failure must not
                # trigger a cancellation merely because the price cannot be proved.
                keep_existing = True
                migration_detail = (
                    "stop_already_open"
                    if matches is True
                    else "stop_price_unmeasured_keep_existing"
                )

    if coverage_out is not None:
        coverage_out.append(
            {
                "symbol": snap.symbol,
                "system": SYSTEM5,
                "qty": snap.qty,
                "is_fractional": snap.is_fractional,
                "mode": "system5_canonical",
                "resident_order": bool(related),
                "detail": migration_detail
                or (
                    "stop_pending_arm"
                    if state.entry_atr10 is not None and not snap.is_fractional
                    else (
                        "fractional_daily_stop"
                        if snap.is_fractional
                        else "missing_entry_atr10"
                    )
                ),
                "entry_atr10": state.entry_atr10,
                "target_price": state.target_price,
                "target_hit_date": state.target_hit_date,
                "timeout_exit_date": state.timeout_exit_date,
                "expected_stop_price": expected_stop,
                "existing_protection_coids": related,
            }
        )

    if cancel_for_migration and observed_stop_prices.get(cancel_for_migration) is None:
        # Cancel+replace is forbidden unless the currently broker-accepted downside
        # stop was observed first.  Without that evidence there is no proven rollback.
        if coverage_out is not None:
            coverage_out[-1]["detail"] = "migration_blocked_missing_rollback_stop"
        return []
    if keep_existing:
        return []
    if state.entry_atr10 is None or expected_stop is None:
        # Do not cancel legacy protection when the canonical frozen stop cannot be
        # reconstructed.  Existing protection stays resident until evidence exists.
        return []

    if snap.is_fractional:
        # Legacy fractional positions cannot host native stops. Evaluate the frozen
        # stop at this run; new Paper entries are whole-share-only as of PR #176.
        cur = snap.current_price
        if cur is None or cur <= 0:
            return []
        breached = cur <= expected_stop if snap.side == "long" else cur >= expected_stop
        if not breached:
            return []
        coid = _exit_coid(snap, today, "exit-synstop")
        if coid in existing_exit:
            return []
        return [
            PreparedExit(
                symbol=snap.symbol,
                system=SYSTEM5,
                qty=snap.exit_qty(),
                side=close_side,
                order_type="market",
                reason=ExitReasonCode.PROTECT_STOP,
                entry_date=snap.entry_date,
                stop_price=round(expected_stop, 4),
                client_order_id=coid,
                dry_run=True,
                time_in_force="day",
            )
        ]

    replacement_coid = (
        _migrated_stop_coid(snap, today) if cancel_for_migration else stop_coid
    )
    return [
        PreparedExit(
            symbol=snap.symbol,
            system=SYSTEM5,
            qty=snap.exit_qty(),
            side=close_side,
            order_type="stop",
            reason=ExitReasonCode.PROTECT_STOP,
            entry_date=snap.entry_date,
            stop_price=expected_stop,
            client_order_id=replacement_coid,
            dry_run=True,
            time_in_force="gtc",
            cancel_client_order_ids=(
                [cancel_for_migration] if cancel_for_migration else []
            ),
            rollback_stop_price=(
                observed_stop_prices.get(cancel_for_migration)
                if cancel_for_migration
                else None
            ),
        )
    ]


__all__ = [
    "SYSTEM5",
    "SYSTEM5_TARGET_NEXT_OPEN",
    "System5ExitState",
    "load_history",
    "entry_atr10_from_history",
    "target_hit_date_from_history",
    "evaluate_system5_state",
    "build_system5_exit_orders",
]
