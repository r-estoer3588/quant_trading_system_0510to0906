"""System5 live exit adapter that preserves the documented Bensdorp strategy.

Why this module exists
----------------------
The generic Alpaca protection engine models a profit target as a resting limit and
models ``max_holding_days=N`` as a close when ``holding_days >= N``.  System5's
canonical contract is different:

* entry: previous close - 3% limit buy
* stop: entry - 3 * ATR10, where ATR10 is fixed at entry
* target trigger: entry + 1 * entry ATR10
* when the target is touched, close at the *next trading session open*
* observe six trading sessions after entry; if neither stop nor target resolves the
  trade, close at the following session open (the seventh post-entry session)

``strategies/system5_strategy.py::compute_exit`` implements exactly that bar sequence.
This module translates the same contract into the live/paper exit engine without
changing any strategy threshold or parameter.

The adapter is intentionally System5-only.  Other Bensdorp systems have different
close timing semantics and remain on the shared engine until separately audited.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from common.alpaca_trading import (
    ExitReasonCode,
    PositionSnapshot,
    PreparedExit,
    protective_stop_price,
    round_to_alpaca_tick,
)
from common.trade_management import SYSTEM_TRADE_RULES
from common.trading_days import count_trading_days

SYSTEM5 = "system5"
SYSTEM5_TARGET_NEXT_OPEN = "profit_target_next_open"

# Existing common/alpaca_trading client_order_id suffixes.  They are repeated here
# deliberately so the adapter can recognize/migrate already-resting legacy S5
# protection without importing private module globals.
_PROTECT_STOP_SUFFIX = "protect-stop"
_PROTECT_STOP_REARM_SUFFIX = "protect-stop-rearm"
_PROTECT_TARGET_SUFFIX = "protect-target"
_PROTECT_OCO_SUFFIX = "protect-oco"
_EXIT_TIME_SUFFIX = "exit-time"
_EXIT_SYN_STOP_SUFFIX = "exit-synstop"
_EXIT_TARGET_NEXT_OPEN_SUFFIX = "exit-target-next-open"


@dataclass(slots=True)
class System5ExitContext:
    symbol: str
    entry_date: str | None
    today: str
    holding_days: int | None
    entry_atr10: float | None
    entry_atr_source: str
    target_price: float | None
    target_hit_date: str | None
    target_due: bool
    time_due: bool
    data_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_date(value: Any) -> date | None:
    try:
        return datetime.fromisoformat(str(value)[:10]).date()
    except Exception:
        return None


def _date_series(df: pd.DataFrame) -> pd.Series | None:
    """Return normalized calendar dates for a rolling CSV, or None if unavailable."""
    candidates = ("Date", "date", "Datetime", "datetime", "timestamp", "Timestamp")
    col = next((c for c in candidates if c in df.columns), None)
    if col is None:
        return None
    try:
        parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
        return parsed.dt.date
    except Exception:
        return None


def _numeric_col(df: pd.DataFrame, *names: str) -> pd.Series | None:
    by_lower = {str(c).lower(): c for c in df.columns}
    for name in names:
        actual = by_lower.get(name.lower())
        if actual is None:
            continue
        return pd.to_numeric(df[actual], errors="coerce")
    return None


def _load_history(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    if not path.exists():
        return None, f"rolling_missing:{path.name}"
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - caller surfaces diagnostic
        return None, f"rolling_read_error:{type(exc).__name__}:{exc}"
    if df.empty:
        return None, f"rolling_empty:{path.name}"
    dates = _date_series(df)
    if dates is None or dates.isna().all():
        return None, f"rolling_date_missing:{path.name}"
    out = df.copy()
    out["__s5_date"] = dates
    out = out[out["__s5_date"].notna()].sort_values("__s5_date")
    if out.empty:
        return None, f"rolling_date_unusable:{path.name}"
    return out, None


def resolve_system5_context(
    snap: PositionSnapshot,
    *,
    today: str,
    rolling_dir: Path,
) -> System5ExitContext:
    """Resolve the immutable entry ATR and exact next-open exit state for one S5 leg.

    Target inspection uses only completed sessions strictly before ``today``.  It also
    examines at most the first six post-entry sessions, matching the backtest loop
    ``for offset in range(1, fallback_days + 1)``.  This avoids a later missed runner
    incorrectly reclassifying a move that happened *after* the trade should already
    have time-exited.
    """
    rules = SYSTEM_TRADE_RULES[SYSTEM5]
    entry_d = _as_date(snap.entry_date)
    today_d = _as_date(today)
    holding = (
        count_trading_days(entry_d, today_d)
        if entry_d is not None and today_d is not None
        else None
    )
    max_days = int(rules.max_holding_days)

    base = System5ExitContext(
        symbol=str(snap.symbol).upper(),
        entry_date=str(snap.entry_date)[:10] if snap.entry_date else None,
        today=today,
        holding_days=holding,
        entry_atr10=None,
        entry_atr_source="unresolved",
        target_price=None,
        target_hit_date=None,
        target_due=False,
        # The strategy observes days 1..6 and exits at the next open = day 7.
        time_due=bool(holding is not None and holding > max_days),
    )
    if entry_d is None or today_d is None:
        base.data_error = "invalid_entry_or_today_date"
        return base

    path = rolling_dir / f"{base.symbol}.csv"
    hist, err = _load_history(path)
    if hist is None:
        base.data_error = err
        return base

    atr10 = _numeric_col(hist, "atr10", "atr_10")
    high = _numeric_col(hist, "High", "high")
    if atr10 is None:
        base.data_error = "atr10_missing"
        return base

    # Backtest compute_entry uses the row immediately before entry_idx.
    before_entry = hist[hist["__s5_date"] < entry_d]
    if before_entry.empty:
        base.data_error = "no_pre_entry_row"
        return base
    atr_value = atr10.loc[before_entry.index[-1]]
    try:
        atr = float(atr_value)
    except (TypeError, ValueError):
        atr = 0.0
    if not (atr > 0):
        base.data_error = "entry_atr10_nonpositive"
        return base

    base.entry_atr10 = atr
    base.entry_atr_source = "rolling_pre_entry"
    base.target_price = float(snap.avg_entry_price) + (
        atr * float(rules.profit_target_value)
    )

    if high is None:
        base.data_error = "high_missing"
        return base

    # Only completed bars after entry, and only the first six observation sessions.
    post_entry = hist[hist["__s5_date"] > entry_d].head(max_days)
    completed = post_entry[post_entry["__s5_date"] < today_d]
    if completed.empty:
        return base

    for idx in completed.index:
        try:
            hi = float(high.loc[idx])
        except (TypeError, ValueError):
            continue
        if hi >= float(base.target_price):
            hit_d = hist.loc[idx, "__s5_date"]
            base.target_hit_date = str(hit_d)
            # Any later session is already at/after the specified next-open exit.
            base.target_due = today_d > hit_d
            break
    return base


def _coids(snap: PositionSnapshot) -> dict[str, str]:
    entry_tag = (snap.entry_date or "").replace("-", "") or "noentry"
    base = f"protect-{SYSTEM5}-{snap.symbol}-{entry_tag}"
    return {
        "stop": f"{base}-{_PROTECT_STOP_SUFFIX}",
        "stop_rearm": f"{base}-{_PROTECT_STOP_REARM_SUFFIX}",
        "target": f"{base}-{_PROTECT_TARGET_SUFFIX}",
        "oco": f"{base}-{_PROTECT_OCO_SUFFIX}",
    }


def _market_close(
    snap: PositionSnapshot,
    *,
    today: str,
    reason: str,
    holding_days: int | None,
    max_holding_days: int,
) -> PreparedExit:
    date_compact = today.replace("-", "")
    suffix = (
        _EXIT_TARGET_NEXT_OPEN_SUFFIX
        if reason == SYSTEM5_TARGET_NEXT_OPEN
        else _EXIT_TIME_SUFFIX
    )
    return PreparedExit(
        symbol=snap.symbol,
        system=SYSTEM5,
        qty=snap.exit_qty(),
        side="sell",  # System5 is canonical long-only.
        order_type="market",
        reason=reason,
        entry_date=snap.entry_date,
        holding_days=holding_days,
        max_holding_days=max_holding_days,
        client_order_id=f"exit-{SYSTEM5}-{snap.symbol}-{date_compact}-{suffix}",
        dry_run=True,
        time_in_force="day",
    )


def build_system5_live_exits(
    snapshots: list[PositionSnapshot],
    *,
    today: str,
    rolling_dir: Path,
    existing_protect_coids: set[str] | None = None,
    existing_exit_coids: set[str] | None = None,
    latest_atr_by_symbol: Mapping[str, Mapping[int, float]] | None = None,
    price_by_symbol: Mapping[str, float] | None = None,
    protection_coverage_out: list[dict[str, Any]] | None = None,
    diagnostics_out: list[dict[str, Any]] | None = None,
) -> list[PreparedExit]:
    """Build System5 orders using the canonical strategy timing.

    This intentionally does **not** create a resting profit-target limit/OCO.  The
    target is a trigger whose consequence is a market exit at the next session open.
    A broker-resident stop remains the only resting protection for whole shares.
    Existing legacy S5 target/OCO orders are migrated to stop-only by placing their
    client_order_ids in ``cancel_client_order_ids`` on the replacement stop proposal.
    """
    rules = SYSTEM_TRADE_RULES[SYSTEM5]
    existing_protect = existing_protect_coids or set()
    existing_exit = existing_exit_coids or set()
    latest_atr = latest_atr_by_symbol or {}
    prices = price_by_symbol or {}
    out: list[PreparedExit] = []

    for snap in snapshots:
        if str(snap.system or "").lower() != SYSTEM5 or snap.abs_qty <= 0:
            continue

        ctx = resolve_system5_context(snap, today=today, rolling_dir=rolling_dir)
        if diagnostics_out is not None:
            diagnostics_out.append(ctx.to_dict())

        # Profit trigger has priority over time exit on the same next-open session,
        # matching compute_exit which tests target during days 1..6 before fallback.
        if ctx.target_due:
            po = _market_close(
                snap,
                today=today,
                reason=SYSTEM5_TARGET_NEXT_OPEN,
                holding_days=ctx.holding_days,
                max_holding_days=int(rules.max_holding_days),
            )
            if not po.client_order_id or po.client_order_id not in existing_exit:
                out.append(po)
            if protection_coverage_out is not None:
                protection_coverage_out.append(
                    {
                        "symbol": snap.symbol,
                        "system": SYSTEM5,
                        "qty": snap.qty,
                        "is_fractional": snap.is_fractional,
                        "mode": "closing_target_next_open",
                        "resident_order": False,
                        "detail": f"target_hit={ctx.target_hit_date}",
                    }
                )
            continue

        if ctx.time_due:
            po = _market_close(
                snap,
                today=today,
                reason=ExitReasonCode.TIME,
                holding_days=ctx.holding_days,
                max_holding_days=int(rules.max_holding_days),
            )
            if not po.client_order_id or po.client_order_id not in existing_exit:
                out.append(po)
            if protection_coverage_out is not None:
                protection_coverage_out.append(
                    {
                        "symbol": snap.symbol,
                        "system": SYSTEM5,
                        "qty": snap.qty,
                        "is_fractional": snap.is_fractional,
                        "mode": "closing_time",
                        "resident_order": False,
                        "detail": f"holding_days={ctx.holding_days}",
                    }
                )
            continue

        # Stop distance is frozen at entry ATR, exactly like System5Strategy._last_entry_atr.
        stop_atr = ctx.entry_atr10
        stop_source = ctx.entry_atr_source
        if stop_atr is None:
            try:
                stop_atr = float(latest_atr.get(snap.symbol, {}).get(10) or 0.0)
            except (TypeError, ValueError):
                stop_atr = 0.0
            if stop_atr > 0:
                # Safety fallback only: preserve downside protection if historical data is
                # unavailable, but never use this drifting ATR to manufacture a target hit.
                stop_source = "latest_atr_fallback_for_stop_only"
            else:
                stop_atr = None

        stop_price = protective_stop_price(
            side=snap.side,
            avg_entry_price=snap.avg_entry_price,
            rules=rules,
            atr_value=stop_atr,
            symbol=snap.symbol,
        )

        if snap.is_fractional:
            cur = snap.current_price
            if cur is None:
                try:
                    cur = float(prices.get(snap.symbol) or 0.0) or None
                except (TypeError, ValueError):
                    cur = None
            fired = False
            if stop_price is not None and cur is not None and cur <= stop_price:
                coid = (
                    f"exit-{SYSTEM5}-{snap.symbol}-{today.replace('-', '')}-"
                    f"{_EXIT_SYN_STOP_SUFFIX}"
                )
                if coid not in existing_exit:
                    out.append(
                        PreparedExit(
                            symbol=snap.symbol,
                            system=SYSTEM5,
                            qty=snap.exit_qty(),
                            side="sell",
                            order_type="market",
                            reason=ExitReasonCode.PROTECT_STOP,
                            entry_date=snap.entry_date,
                            stop_price=round(stop_price, 4),
                            client_order_id=coid,
                            dry_run=True,
                            time_in_force="day",
                        )
                    )
                    fired = True
            if protection_coverage_out is not None:
                protection_coverage_out.append(
                    {
                        "symbol": snap.symbol,
                        "system": SYSTEM5,
                        "qty": snap.qty,
                        "is_fractional": True,
                        "mode": "synthetic_daily",
                        "resident_order": False,
                        "detail": "fired" if fired else "evaluated_no_stop_breach",
                        "evaluated_at": today,
                        "current_price": cur,
                        "stop_atr_source": stop_source,
                    }
                )
            continue

        ids = _coids(snap)
        stop_already_open = (
            ids["stop"] in existing_protect
            or ids["stop_rearm"] in existing_protect
        )
        legacy_wrong = [
            coid
            for coid in (ids["target"], ids["oco"])
            if coid in existing_protect
        ]

        if not stop_already_open and stop_price is not None and stop_price > 0:
            out.append(
                PreparedExit(
                    symbol=snap.symbol,
                    system=SYSTEM5,
                    qty=snap.exit_qty(),
                    side="sell",
                    order_type="stop",
                    reason=ExitReasonCode.PROTECT_STOP,
                    entry_date=snap.entry_date,
                    stop_price=round_to_alpaca_tick(stop_price),
                    client_order_id=ids["stop"],
                    dry_run=True,
                    time_in_force="gtc",
                    # A legacy OCO/target encodes the wrong immediate-target semantics.
                    # Cancel only those exact S5 protection orders before stop replacement.
                    cancel_client_order_ids=legacy_wrong,
                )
            )

        if protection_coverage_out is not None:
            proposed = (not stop_already_open) and stop_price is not None and stop_price > 0
            protection_coverage_out.append(
                {
                    "symbol": snap.symbol,
                    "system": SYSTEM5,
                    "qty": snap.qty,
                    "is_fractional": False,
                    "mode": "native_resident",
                    # Preserve existing artifact convention: a valid stop proposal counts as
                    # protection coverage for this pass, while detail records observation.
                    "resident_order": bool(stop_already_open or proposed),
                    "detail": (
                        "existing_stop"
                        if stop_already_open
                        else ("stop_proposed" if proposed else "stop_unavailable")
                    ),
                    "stop_atr_source": stop_source,
                    "legacy_wrong_protection": legacy_wrong,
                }
            )

    return out
