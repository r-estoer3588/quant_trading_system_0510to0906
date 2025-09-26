"""Final allocation stage utilities.

This module extracts the core logic for the *allocation & final list* stage
from ``scripts.run_all_systems_today`` so that it can be unit tested in
isolation.  The helpers here are intentionally side-effect free – all
I/O (Alpaca API calls, CSV writes, etc.) should be handled by the caller.

The key entry point is :func:`finalize_allocation` which combines the
per-system candidate tables into the final trade list using either the
slot-based or capital allocation mode.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_LONG_ALLOCATIONS: dict[str, float] = {
    "system1": 0.25,
    "system3": 0.25,
    "system4": 0.25,
    "system5": 0.25,
}

DEFAULT_SHORT_ALLOCATIONS: dict[str, float] = {
    "system2": 0.40,
    "system6": 0.40,
    "system7": 0.20,
}


def _safe_positive_float(value: Any, *, allow_zero: bool = False) -> float | None:
    """Attempt to convert ``value`` to a positive float.

    Returns ``None`` if conversion fails, value is ``None``/empty string, or the
    numeric result is negative (and zero when ``allow_zero`` is False).

    This helper centralises Optional[float] sanitation so that mypy does not
    see patterns like ``float(x)`` where ``x`` is ``float | None``.
    """
    if value in (None, ""):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f < 0:
        return None
    if not allow_zero and f == 0:
        return None
    return f


@dataclass(slots=True)
class AllocationSummary:
    """Summary of the final allocation step."""

    mode: str
    long_allocations: dict[str, float]
    short_allocations: dict[str, float]
    active_positions: dict[str, int]
    available_slots: dict[str, int]
    final_counts: dict[str, int]
    slot_allocation: dict[str, int] | None = None
    slot_candidates: dict[str, int] | None = None
    budgets: dict[str, float] | None = None
    budget_remaining: dict[str, float] | None = None
    capital_long: float | None = None
    capital_short: float | None = None


def load_symbol_system_map(path: Path | str | None = None) -> dict[str, str]:
    """Load ``data/symbol_system_map.json``.

    The helper normalises keys/values to lower case so that lookups become
    case-insensitive.
    """

    if path is None:
        path = Path("data/symbol_system_map.json")
    path = Path(path)
    if not path.exists():
        return {}
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        raw = json.loads(content)
    except json.JSONDecodeError:
        return {}
    result: dict[str, str] = {}
    for key, value in raw.items():
        key_str = str(key).strip()
        val_str = str(value).strip()
        if not key_str or not val_str:
            continue
        result[key_str.lower()] = val_str.lower()
    return result


def _get_position_attr(obj: object, name: str) -> Any:
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, Mapping):
        return obj.get(name)
    return None


def count_active_positions_by_system(
    positions: Sequence[object] | None,
    symbol_system_map: Mapping[str, str] | None,
) -> dict[str, int]:
    """Return a mapping of ``system -> active position count``.

    Parameters
    ----------
    positions:
        Iterable of Alpaca position objects (or dictionaries).  Only entries
        with a positive ``qty`` are counted.
    symbol_system_map:
        Mapping of ``symbol`` to ``system``.  Keys are compared in a
        case-insensitive fashion.
    """

    if positions is None:
        positions = []
    if symbol_system_map is None:
        symbol_system_map = {}

    norm_map: dict[str, str] = {}
    for key, value in symbol_system_map.items():
        key_str = str(key).strip()
        val_str = str(value).strip()
        if not key_str or not val_str:
            continue
        norm_map[key_str.upper()] = val_str.lower()

    counts: dict[str, int] = {}
    for pos in positions:
        symbol_raw = _get_position_attr(pos, "symbol")
        if symbol_raw is None:
            continue
        sym = str(symbol_raw).strip().upper()
        if not sym:
            continue
        qty_raw = _get_position_attr(pos, "qty")
        try:
            qty_val = abs(float(qty_raw)) if qty_raw is not None else 0.0
        except (TypeError, ValueError):
            qty_val = 0.0
        if qty_val <= 0:
            continue
        side_raw = _get_position_attr(pos, "side")
        side = str(side_raw).strip().lower() if side_raw is not None else ""
        system = norm_map.get(sym) or norm_map.get(sym.lower())
        if not system:
            if sym == "SPY" and side == "short":
                system = "system7"
            else:
                continue
        counts[system] = counts.get(system, 0) + 1
    return counts


def _normalize_allocations(
    weights: Mapping[str, float] | None,
    defaults: Mapping[str, float],
) -> dict[str, float]:
    filtered: dict[str, float] = {}
    if weights:
        for key, value in weights.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if numeric <= 0:
                continue
            filtered[str(key).strip().lower()] = numeric
    if not filtered:
        filtered = {k: float(v) for k, v in defaults.items() if float(v) > 0}
    total = sum(filtered.values())
    if total <= 0:
        # Fallback to equal weights
        n = len(defaults)
        if n == 0:
            return {}
        return {k: 1.0 / n for k in defaults}
    return {k: v / total for k, v in filtered.items()}


def _candidate_count(df: pd.DataFrame | None) -> int:
    if df is None or getattr(df, "empty", True):
        return 0
    return int(len(df))


def _ensure_system_columns(df: pd.DataFrame, system: str, side: str) -> pd.DataFrame:
    out = df.copy()
    if "system" not in out.columns:
        out.insert(0, "system", system)
    if "side" not in out.columns:
        out["side"] = side
    return out


def _distribute_slots(
    weights: Mapping[str, float],
    total_slots: int,
    available_counts: Mapping[str, int],
) -> dict[str, int]:
    if total_slots <= 0:
        return {k: 0 for k in weights}

    base: dict[str, int] = {}
    for key, weight in weights.items():
        slots = int(total_slots * weight)
        cand = int(available_counts.get(key, 0))
        if cand <= 0:
            slots = 0
        elif slots == 0:
            slots = 1
        base[key] = min(slots, cand)

    used = sum(base.values())
    remain = max(0, total_slots - used)
    if remain and weights:
        order = sorted(
            weights.keys(),
            key=lambda k: (available_counts.get(k, 0), weights.get(k, 0.0)),
            reverse=True,
        )
        idx = 0
        while remain > 0 and order:
            name = order[idx % len(order)]
            if available_counts.get(name, 0) > base.get(name, 0):
                base[name] += 1
                remain -= 1
            idx += 1
            if idx > 10000:
                break

    for key in list(base.keys()):
        base[key] = min(base[key], int(available_counts.get(key, 0)))
    return base


@dataclass(slots=True)
class SlotAllocationResult:
    frame: pd.DataFrame
    distribution: dict[str, int]
    candidate_counts: dict[str, int]


def _allocate_by_slots(
    per_system: Mapping[str, pd.DataFrame],
    *,
    long_alloc: Mapping[str, float],
    short_alloc: Mapping[str, float],
    available_slots: Mapping[str, int],
    slots_long: int,
    slots_short: int,
) -> SlotAllocationResult:
    candidate_counts: dict[str, int] = {}

    long_available: dict[str, int] = {}
    long_raw: dict[str, int] = {}
    for name in long_alloc:
        df = per_system.get(name)
        cand_cnt = _candidate_count(df)
        long_raw[name] = cand_cnt
        candidate_counts[name] = cand_cnt
        long_available[name] = min(cand_cnt, int(available_slots.get(name, 0)))

    short_available: dict[str, int] = {}
    short_raw: dict[str, int] = {}
    for name in short_alloc:
        df = per_system.get(name)
        cand_cnt = _candidate_count(df)
        short_raw[name] = cand_cnt
        candidate_counts[name] = cand_cnt
        short_available[name] = min(cand_cnt, int(available_slots.get(name, 0)))

    long_slots = _distribute_slots(long_alloc, slots_long, long_available)
    short_slots = _distribute_slots(short_alloc, slots_short, short_available)

    frames: list[pd.DataFrame] = []
    distribution: dict[str, int] = {}

    for name, slot in {**long_slots, **short_slots}.items():
        if slot <= 0:
            continue
        df = per_system.get(name)
        if df is None or getattr(df, "empty", True):
            continue
        free_slots = int(available_slots.get(name, 0))
        take = min(int(slot), free_slots)
        if take <= 0:
            continue
        side = "long" if name in long_alloc else "short"
        subset = _ensure_system_columns(df.head(take), name, side)
        subset = subset.copy()
        subset["alloc_weight"] = (
            long_alloc.get(name) if name in long_alloc else short_alloc.get(name, 0.0)
        )
        frames.append(subset)
        distribution[name] = take

    if frames:
        frame = pd.concat(frames, ignore_index=True)
    else:
        frame = pd.DataFrame()

    return SlotAllocationResult(
        frame=frame,
        distribution=distribution,
        candidate_counts=candidate_counts,
    )


@dataclass(slots=True)
class _StrategyAllocationMeta:
    calc_fn: Any
    risk_pct: float
    max_pct: float
    max_positions: int


@dataclass(slots=True)
class CapitalAllocationResult:
    frame: pd.DataFrame
    budgets: dict[str, float]
    remaining: dict[str, float]
    counts: dict[str, int]


def _allocate_by_capital(
    per_system: Mapping[str, pd.DataFrame],
    *,
    strategies: Mapping[str, object],
    total_budget: float,
    weights: Mapping[str, float],
    side: str,
    active_positions: Mapping[str, int],
) -> CapitalAllocationResult:
    budgets = {name: float(total_budget) * float(weights.get(name, 0.0)) for name in weights}
    remaining = budgets.copy()

    # stable ordering
    order = [f"system{i}" for i in range(1, 8)]
    ordered_names = [name for name in order if name in weights]

    meta_map: dict[str, _StrategyAllocationMeta] = {}
    candidates: dict[str, list[dict[str, Any]]] = {}
    index_map: dict[str, int] = {}

    for name in ordered_names:
        stg = strategies.get(name)
        config = getattr(stg, "config", {}) if stg is not None else {}
        calc_fn = getattr(stg, "calculate_position_size", None)
        if not callable(calc_fn):
            calc_fn = None
        try:
            risk_pct = float(config.get("risk_pct", 0.02))
        except (TypeError, ValueError):
            risk_pct = 0.02
        try:
            max_pct = float(config.get("max_pct", 0.10))
        except (TypeError, ValueError):
            max_pct = 0.10
        try:
            max_positions = int(config.get("max_positions", 10))
        except (TypeError, ValueError):
            max_positions = 10
        meta_map[name] = _StrategyAllocationMeta(
            calc_fn=calc_fn,
            risk_pct=risk_pct,
            max_pct=max_pct,
            max_positions=max_positions,
        )
        df = per_system.get(name)
        if df is None or getattr(df, "empty", True):
            candidates[name] = []
        else:
            records = df.to_dict("records")
            # 正規化: dict[Hashable, Any] -> dict[str, Any]
            norm_records: list[dict[str, Any]] = []
            for rec in records:
                try:
                    norm_records.append({str(k): v for k, v in rec.items()})
                except Exception:
                    norm_records.append({str(k): rec.get(k) for k in rec})
            candidates[name] = norm_records
        index_map[name] = 0

    counts = {name: 0 for name in ordered_names}
    max_pos_map = {
        name: max(0, meta_map[name].max_positions - int(active_positions.get(name, 0)))
        for name in ordered_names
    }

    chosen: list[dict[str, Any]] = []
    chosen_symbols: set[str] = set()

    def _normalize_shares(value: Any) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0

    still = True
    while still:
        still = False
        for name in ordered_names:
            rows = candidates.get(name, [])
            if (
                not rows
                or remaining.get(name, 0.0) <= 0.0
                or counts.get(name, 0) >= max_pos_map.get(name, 0)
                or index_map.get(name, 0) >= len(rows)
            ):
                continue

            meta = meta_map[name]
            idx = index_map[name]

            while idx < len(rows):
                row = rows[idx]
                idx += 1
                index_map[name] = idx

                sym = str(row.get("symbol", "")).upper()
                if not sym or sym in chosen_symbols:
                    continue

                entry = _safe_positive_float(row.get("entry_price"), allow_zero=False)
                stop = _safe_positive_float(row.get("stop_price"), allow_zero=False)
                if entry is None or stop is None or entry <= 0:
                    continue

                desired_shares = 0
                if meta.calc_fn is not None:
                    try:
                        desired_shares = _normalize_shares(
                            meta.calc_fn(
                                budgets[name],
                                entry,
                                stop,
                                risk_pct=meta.risk_pct,
                                max_pct=meta.max_pct,
                            )
                        )
                    except Exception:
                        desired_shares = 0
                if desired_shares <= 0:
                    continue

                max_by_cash = int(remaining[name] // abs(entry)) if entry else 0
                shares = min(desired_shares, max_by_cash)
                if shares <= 0:
                    continue

                position_value = shares * abs(entry)
                if position_value <= 0:
                    continue

                record = dict(row)
                record["shares"] = int(shares)
                record["position_value"] = float(round(position_value, 2))
                record["system"] = record.get("system", name)
                record["side"] = record.get("side", side)
                record["system_budget"] = float(round(remaining[name], 2))
                remaining_after = remaining[name] - position_value
                record["remaining_after"] = float(round(remaining_after, 2))
                chosen.append(record)
                chosen_symbols.add(sym)
                remaining[name] -= position_value
                counts[name] = counts.get(name, 0) + 1
                still = True
                break

    if chosen:
        frame = pd.DataFrame(chosen)
    else:
        frame = pd.DataFrame()
    return CapitalAllocationResult(
        frame=frame,
        budgets=budgets,
        remaining=remaining,
        counts=counts,
    )


def _sort_final_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "system" not in df.columns:
        return df.copy() if df is not None else df

    tmp = df.copy()
    try:
        tmp["system"] = tmp["system"].astype(str)
    except Exception:
        tmp["system"] = tmp["system"]
    try:
        tmp["_system_no"] = tmp["system"].str.extract(r"(\d+)").fillna(0).astype(int)
    except Exception:
        tmp["_system_no"] = 0

    sort_cols = [c for c in ["side", "_system_no"] if c in tmp.columns]
    if sort_cols:
        tmp = tmp.sort_values(sort_cols, kind="stable")

    try:
        parts: list[pd.DataFrame] = []
        for name, group in tmp.groupby("system", sort=False):
            if "score" not in group.columns:
                parts.append(group)
                continue
            asc = False
            try:
                if isinstance(name, str) and name.lower() == "system4":
                    asc = True
            except Exception:
                asc = False
            parts.append(
                group.sort_values(
                    "score",
                    ascending=asc,
                    kind="stable",
                    na_position="last",
                )
            )
        tmp = pd.concat(parts, ignore_index=True)
    except Exception:
        pass

    tmp = tmp.drop(columns=["_system_no"], errors="ignore")
    try:
        tmp.insert(0, "no", range(1, len(tmp) + 1))
    except Exception:
        tmp["no"] = range(1, len(tmp) + 1)
    return tmp.reset_index(drop=True)


def _resolve_max_positions(
    strategies: Mapping[str, object] | None,
    systems: Sequence[str],
    default_max_positions: int,
) -> dict[str, int]:
    result: dict[str, int] = {name: default_max_positions for name in systems}
    if not strategies:
        return result
    for name, obj in strategies.items():
        key = str(name).strip().lower()
        if not key:
            continue
        config = getattr(obj, "config", {})
        try:
            value = int(config.get("max_positions", default_max_positions))
        except (TypeError, ValueError, AttributeError):
            value = default_max_positions
        result[key] = max(0, value)
    return result


def finalize_allocation(
    per_system: Mapping[str, pd.DataFrame],
    *,
    strategies: Mapping[str, object] | None = None,
    positions: Sequence[object] | None = None,
    symbol_system_map: Mapping[str, str] | None = None,
    long_allocations: Mapping[str, float] | None = None,
    short_allocations: Mapping[str, float] | None = None,
    slots_long: int | None = None,
    slots_short: int | None = None,
    capital_long: float | None = None,
    capital_short: float | None = None,
    default_capital: float = 100000.0,
    default_long_ratio: float = 0.5,
    default_max_positions: int = 10,
) -> tuple[pd.DataFrame, AllocationSummary]:
    """Combine per-system candidates into the final trade list.

    Parameters mirror the behaviour of ``scripts.run_all_systems_today`` but
    are greatly simplified so that tests can provide deterministic inputs.
    """

    per_system_norm: dict[str, pd.DataFrame] = {
        str(name).strip().lower(): df for name, df in per_system.items()
    }

    long_alloc = _normalize_allocations(long_allocations, DEFAULT_LONG_ALLOCATIONS)
    short_alloc = _normalize_allocations(short_allocations, DEFAULT_SHORT_ALLOCATIONS)

    systems = sorted({*per_system_norm.keys(), *long_alloc.keys(), *short_alloc.keys()})
    max_pos_map = _resolve_max_positions(strategies, systems, default_max_positions)

    active_positions = count_active_positions_by_system(positions, symbol_system_map)
    available_slots: dict[str, int] = {}
    for name in systems:
        taken = int(active_positions.get(name, 0))
        limit = int(max_pos_map.get(name, default_max_positions))
        available_slots[name] = max(0, limit - taken)

    candidate_counts = {name: _candidate_count(per_system_norm.get(name)) for name in systems}

    # Determine allocation mode.
    mode = "slot"
    if capital_long is not None or capital_short is not None:
        mode = "capital"

    if mode == "slot":
        if slots_long is None:
            slots_long = sum(available_slots.get(name, 0) for name in long_alloc)
        if slots_short is None:
            slots_short = sum(available_slots.get(name, 0) for name in short_alloc)
        slot_result = _allocate_by_slots(
            per_system_norm,
            long_alloc=long_alloc,
            short_alloc=short_alloc,
            available_slots=available_slots,
            slots_long=int(slots_long or 0),
            slots_short=int(slots_short or 0),
        )
        final_df = slot_result.frame
        summary = AllocationSummary(
            mode="slot",
            long_allocations=dict(long_alloc),
            short_allocations=dict(short_alloc),
            active_positions=dict(active_positions),
            available_slots=dict(available_slots),
            final_counts={},
            slot_allocation=dict(slot_result.distribution),
            slot_candidates=dict(slot_result.candidate_counts),
        )
    else:
        # Capital mode replicates ``run_all_systems_today`` defaults, with
        # stricter Optional handling for mypy friendliness.
        ratio_conv = _safe_positive_float(default_long_ratio, allow_zero=True)
        ratio = ratio_conv if ratio_conv is not None else 0.5
        cap_conv = _safe_positive_float(default_capital, allow_zero=True)
        default_cap_float = cap_conv if cap_conv is not None else 100000.0

        long_cap_opt = _safe_positive_float(capital_long)
        short_cap_opt = _safe_positive_float(capital_short)

        # Derive missing sides if one or both unspecified / non-positive.
        if long_cap_opt is None and short_cap_opt is None:
            total: float = default_cap_float
            long_cap_val: float = total * ratio
            short_cap_val: float = total * (1.0 - ratio)
        elif long_cap_opt is None and short_cap_opt is not None:
            total = short_cap_opt  # short_cap_opt is float (not None) in this branch
            long_cap_val = total * ratio
            short_cap_val = total * (1.0 - ratio)
        elif short_cap_opt is None and long_cap_opt is not None:
            total = long_cap_opt  # long_cap_opt is float (not None) here
            long_cap_val = total * ratio
            short_cap_val = total * (1.0 - ratio)
        else:
            # both provided (and positive) -> enforce non-None with assert for mypy
            assert long_cap_opt is not None and short_cap_opt is not None
            long_cap_val = long_cap_opt
            short_cap_val = short_cap_opt

        long_cap = long_cap_val
        short_cap = short_cap_val

        strategies_norm: dict[str, object] = {
            str(name).strip().lower(): obj for name, obj in (strategies or {}).items()
        }

        long_result = _allocate_by_capital(
            per_system_norm,
            strategies=strategies_norm,
            total_budget=long_cap,
            weights=long_alloc,
            side="long",
            active_positions=active_positions,
        )
        short_result = _allocate_by_capital(
            per_system_norm,
            strategies=strategies_norm,
            total_budget=short_cap,
            weights=short_alloc,
            side="short",
            active_positions=active_positions,
        )
        frames = [df for df in [long_result.frame, short_result.frame] if not df.empty]
        if frames:
            final_df = pd.concat(frames, ignore_index=True)
        else:
            final_df = pd.DataFrame()

        budgets_combined: dict[str, float] = {}
        budgets_combined.update(long_result.budgets)
        budgets_combined.update(short_result.budgets)
        remaining_combined: dict[str, float] = {}
        remaining_combined.update(long_result.remaining)
        remaining_combined.update(short_result.remaining)

        summary = AllocationSummary(
            mode="capital",
            long_allocations=dict(long_alloc),
            short_allocations=dict(short_alloc),
            active_positions=dict(active_positions),
            available_slots=dict(available_slots),
            final_counts={},
            budgets=budgets_combined,
            budget_remaining=remaining_combined,
            capital_long=long_cap,
            capital_short=short_cap,
        )

    if not final_df.empty:
        final_df = _sort_final_frame(final_df)
    else:
        final_df = final_df.copy()

    if "system" in final_df.columns:
        try:
            counts_series = final_df["system"].astype(str).str.strip().str.lower().value_counts()
            summary.final_counts = {str(k): int(v) for k, v in counts_series.items()}
        except Exception:
            summary.final_counts = {}
    else:
        summary.final_counts = {}

    # Add candidate counts for capital mode (slot mode already has them)
    if summary.slot_candidates is None:
        summary.slot_candidates = candidate_counts

    return final_df, summary


__all__ = [
    "AllocationSummary",
    "count_active_positions_by_system",
    "finalize_allocation",
    "load_symbol_system_map",
]
