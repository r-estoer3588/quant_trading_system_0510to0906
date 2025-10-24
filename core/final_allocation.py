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

import json
import logging
import os
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias, TypedDict

import pandas as pd

from common.trade_management import TradeManager
from config.environment import get_env_config

# Type aliases for better readability
PositionDict: TypeAlias = dict[str, Any]
SystemName: TypeAlias = str
Symbol: TypeAlias = str
StrategyMapping: TypeAlias = Mapping[str, object]


class AllocationConfig(TypedDict, total=False):
    """Type definition for allocation configuration."""

    long_allocations: dict[SystemName, float]
    short_allocations: dict[SystemName, float]
    max_positions: int
    risk_pct: float
    max_pct: float


# Configure logger
logger = logging.getLogger(__name__)


class AllocationConstants:
    """Centralized constants for allocation logic."""

    DEFAULT_RISK_PCT = 0.02
    DEFAULT_MAX_PCT = 0.10
    DEFAULT_MAX_POSITIONS = 10
    DEFAULT_CAPITAL = 100_000.0
    DEFAULT_LONG_RATIO = 0.5
    MAX_ITERATIONS = 10_000  # 無限ループ防止

    # エラーメッセージ
    MSG_INVALID_ALLOCATION = "Invalid allocation weights: {}"
    MSG_POSITION_CALC_ERROR = "Error calculating position size for {}: {}"
    MSG_EMPTY_DATA = "No data available for system {}"


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


def _load_allocations_from_settings() -> tuple[dict[str, float], dict[str, float]]:
    """Load allocation settings from configuration.

    Returns:
        Tuple of (long_allocations, short_allocations) dictionaries.
        Falls back to DEFAULT_*_ALLOCATIONS if settings are unavailable.
    """
    try:
        from config.settings import get_settings

        settings = get_settings()

        # UIセクションから配分設定を取得
        long_alloc = getattr(settings.ui, "long_allocations", {}) or {}
        short_alloc = getattr(settings.ui, "short_allocations", {}) or {}

        # 設定がある場合はそれを使用、無い場合はデフォルトを使用
        if long_alloc:
            long_result = {}
            for k, v in long_alloc.items():
                try:
                    fv = float(v)
                    if fv > 0:
                        long_result[str(k)] = fv
                except (TypeError, ValueError):
                    continue
        else:
            long_result = DEFAULT_LONG_ALLOCATIONS.copy()

        if short_alloc:
            short_result = {}
            for k, v in short_alloc.items():
                try:
                    fv = float(v)
                    if fv > 0:
                        short_result[str(k)] = fv
                except (TypeError, ValueError):
                    continue
        else:
            short_result = DEFAULT_SHORT_ALLOCATIONS.copy()
        return long_result, short_result

    except Exception:
        # 設定読み込みに失敗した場合はデフォルトを返す
        return DEFAULT_LONG_ALLOCATIONS.copy(), DEFAULT_SHORT_ALLOCATIONS.copy()


def _safe_positive_float(value: Any, *, allow_zero: bool = False) -> float | None:
    """Attempt to convert value to a positive float with improved error handling.

    Args:
        value: Value to convert
        allow_zero: Whether to allow zero values

    Returns:
        Converted float value or None if invalid

    Raises:
        Never raises - returns None for all errors
    """
    if value in (None, ""):
        logger.debug("Empty value provided for float conversion")
        return None

    try:
        f = float(value)
    except (TypeError, ValueError) as e:
        logger.debug("Failed to convert %r to float: %s", value, e)
        return None

    # Validation checks
    if f < 0:
        logger.debug("Negative value rejected: %s", f)
        return None
    if not allow_zero and f == 0:
        logger.debug("Zero value rejected (allow_zero=False)")
        return None
    if not (0 <= f < float("inf")):
        logger.debug("Invalid numeric value: %s", f)
        return None

    return f


@dataclass(slots=True)
class AllocationSummary:
    """Summary of the final allocation step.

    Attributes:
        mode: Allocation mode ('slot' or 'capital')
        long_allocations: Weight distribution for long systems
        short_allocations: Weight distribution for short systems
        active_positions: Current position count by system
        available_slots: Available slots for new positions by system
        final_counts: Final allocated positions by system
        slot_allocation: Slot distribution by system (slot mode only)
        slot_candidates: Available candidates by system (slot mode only)
        budgets: Budget allocation by system (capital mode only)
        budget_remaining: Remaining budget by system (capital mode only)
        capital_long: Total long-side capital (capital mode only)
        capital_short: Total short-side capital (capital mode only)
    """

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
    system_diagnostics: dict[str, Any] | None = None
    # Optional final metrics (added for diagnostics/reporting)
    final_count: int = 0
    final_symbols: list[str] | None = None
    final_long_count: int = 0
    final_short_count: int = 0

    def __post_init__(self) -> None:
        """Validate allocation summary after initialization."""
        if self.mode not in ("slot", "capital"):
            logger.warning("Unknown allocation mode: %s", self.mode)

        # Validate that slot mode has slot-specific data
        if self.mode == "slot" and self.slot_allocation is None:
            logger.warning("Slot mode summary missing slot_allocation data")

        # Validate that capital mode has capital-specific data
        if self.mode == "capital" and self.budgets is None:
            logger.warning("Capital mode summary missing budgets data")


def load_symbol_system_map(path: Path | str | None = None) -> dict[str, str]:
    """Load symbol-to-system mapping from JSON file.

    The helper normalises keys/values to lower case so that lookups become
    case-insensitive.

    Args:
        path: Path to the JSON file. Defaults to 'data/symbol_system_map.json'

    Returns:
        Dictionary mapping symbols to system names. Empty dict if file
        cannot be loaded or parsed.

    Raises:
        Never raises - returns empty dict on all errors
    """
    if path is None:
        path = Path("data/symbol_system_map.json")

    path = Path(path)
    if not path.exists():
        logger.debug("Symbol system map file not found: %s", path)
        return {}

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("Failed to read symbol system map: %s", e)
        return {}

    try:
        raw = json.loads(content)
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in symbol system map: %s", e)
        return {}

    if not isinstance(raw, dict):
        logger.error("Symbol system map must be a dictionary, got %s", type(raw))
        return {}

    result: dict[str, str] = {}
    for key, value in raw.items():
        key_str = str(key).strip()
        val_str = str(value).strip()
        if not key_str or not val_str:
            logger.debug("Skipping empty key/value pair: %r -> %r", key, value)
            continue
        result[key_str.lower()] = val_str.lower()

    logger.debug("Loaded %d symbol-system mappings", len(result))
    return result


def _get_position_attr(obj: object, name: str) -> Any:
    """Get attribute from position object safely.

    Args:
        obj: Position object (could be object with attributes or dict)
        name: Attribute name to retrieve

    Returns:
        Attribute value or None if not found
    """
    try:
        if hasattr(obj, name):
            return getattr(obj, name)
        if isinstance(obj, Mapping):
            return obj.get(name)
    except Exception as e:
        logger.debug("Error accessing attribute %s: %s", name, e)
    return None


def count_active_positions_by_system(
    positions: Sequence[object] | None,
    symbol_system_map: Mapping[str, str] | None,
) -> dict[str, int]:
    """Return a mapping of system names to active position counts.

    Args:
        positions: Iterable of Alpaca position objects (or dictionaries).
                  Only entries with a positive qty are counted.
        symbol_system_map: Mapping of symbol to system name.
                          Keys are compared case-insensitively.

    Returns:
        Dictionary mapping system names to position counts

    Notes:
        - SPY short positions are automatically assigned to system7
        - Invalid or zero quantity positions are ignored
        - Position objects can be either objects with attributes or dictionaries
    """
    if positions is None:
        positions = []
    if symbol_system_map is None:
        symbol_system_map = {}

    # Normalize the symbol system map for case-insensitive lookups
    norm_map: dict[str, str] = {}
    for key, value in symbol_system_map.items():
        try:
            key_str = str(key).strip()
            val_str = str(value).strip()
            if not key_str or not val_str:
                logger.debug(
                    "Skipping empty key/value in symbol_system_map: %r -> %r",
                    key,
                    value,
                )
                continue
            norm_map[key_str.upper()] = val_str.lower()
        except Exception as e:
            logger.warning(
                "Error processing symbol_system_map entry %r -> %r: %s",
                key,
                value,
                e,
            )
            continue

    counts: dict[str, int] = {}
    for i, pos in enumerate(positions):
        try:
            # Extract symbol
            symbol_raw = _get_position_attr(pos, "symbol")
            if symbol_raw is None:
                logger.debug("Position %d missing symbol", i)
                continue

            sym = str(symbol_raw).strip().upper()
            if not sym:
                logger.debug("Position %d has empty symbol", i)
                continue

            # Extract and validate quantity
            qty_raw = _get_position_attr(pos, "qty")
            try:
                qty_val = abs(float(qty_raw)) if qty_raw is not None else 0.0
            except (TypeError, ValueError) as e:
                logger.debug("Position %d invalid quantity %r: %s", i, qty_raw, e)
                qty_val = 0.0

            if qty_val <= 0:
                logger.debug("Position %d has zero/negative quantity: %s", i, qty_val)
                continue

            # Extract side for special SPY handling
            side_raw = _get_position_attr(pos, "side")
            side = str(side_raw).strip().lower() if side_raw is not None else ""

            # Determine system
            system = norm_map.get(sym) or norm_map.get(sym.lower())
            if not system:
                # Special case: SPY short positions go to system7
                if sym == "SPY" and side == "short":
                    system = "system7"
                else:
                    logger.debug("No system mapping found for symbol: %s", sym)
                    continue

            counts[system] = counts.get(system, 0) + 1

        except Exception as e:
            logger.warning("Error processing position %d: %s", i, e)
            continue

    logger.debug("Counted active positions: %s", dict(counts))
    return counts


def _normalize_allocations(
    weights: Mapping[str, float] | None,
    defaults: Mapping[str, float],
) -> dict[str, float]:
    """Normalize allocation weights to sum to 1.0.

    Args:
        weights: Raw allocation weights by system
        defaults: Default weights to use if weights is empty/invalid

    Returns:
        Normalized weights that sum to 1.0

    Notes:
        - Invalid weights (negative, zero, non-numeric) are filtered out
        - If all weights are invalid, defaults are used
        - If defaults are also invalid, equal weights are assigned
    """
    filtered: dict[str, float] = {}

    if weights:
        for key, value in weights.items():
            try:
                numeric = float(value)
                if numeric > 0:  # Only positive weights allowed
                    filtered[str(key).strip().lower()] = numeric
                else:
                    logger.debug("Skipping non-positive weight: %s=%s", key, value)
            except (TypeError, ValueError) as e:
                logger.debug("Skipping invalid weight %s=%r: %s", key, value, e)
                continue

    # Fall back to defaults if no valid weights provided
    if not filtered:
        logger.debug("No valid weights provided, using defaults")
        try:
            filtered = {k: float(v) for k, v in defaults.items() if float(v) > 0}
        except (TypeError, ValueError) as e:
            logger.error("Invalid default weights: %s", e)
            filtered = {}

    # Calculate total for normalization
    total = sum(filtered.values())
    if total <= 0:
        # Final fallback: equal weights
        logger.warning("All weights are zero/negative, using equal weights")
        n = len(defaults)
        if n == 0:
            logger.error("No systems available for equal weight allocation")
            return {}
        return {k: 1.0 / n for k in defaults}

    # Normalize to sum to 1.0
    normalized = {k: v / total for k, v in filtered.items()}
    logger.debug("Normalized allocations: %s", normalized)
    return normalized


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
        # 重要: 元データにside列が存在しても、配分バケット（long/short）に合わせて明示的に上書きする
        # これにより、system4等のロング専用システムが誤ってショート側へ流れるのを防ぐ
        try:
            subset["side"] = side
        except Exception:
            # 失敗時も最低限の列は確保されているので継続
            pass
        try:
            if name in long_alloc:
                subset["alloc_weight"] = long_alloc.get(name)
            else:
                subset["alloc_weight"] = short_alloc.get(name, 0.0)
        except Exception:
            subset["alloc_weight"] = 0.0
        frames.append(subset)
        distribution[name] = take

    if frames:
        frame = _concat_nonempty(frames)
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
    # map reason -> set of symbols excluded at allocation time
    allocator_excludes: dict[str, set] | None = None
    # map symbol -> system that first selected it (for already-selected diagnostics)
    already_selected_map: dict[str, str] | None = None


def _concat_nonempty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Safely concat DataFrames while avoiding pandas FutureWarning.

    - Skip None, empty, and all-NA frames
    - Suppress dtype-inference FutureWarning during concat
    """
    valid: list[pd.DataFrame] = []
    for _df in frames or []:
        try:
            if _df is None or getattr(_df, "empty", True):
                continue
            if bool(_df.isna().all().all()):
                continue
            # 1 行以上あることを厳格に確認（念のため）
            if getattr(_df, "shape", None) and _df.shape[0] <= 0:
                continue
            valid.append(_df)
        except Exception:
            continue

    if not valid:
        return pd.DataFrame()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        return pd.concat(valid, ignore_index=True)


def _allocate_by_capital(
    per_system: Mapping[str, pd.DataFrame],
    *,
    strategies: Mapping[str, object],
    total_budget: float,
    weights: Mapping[str, float],
    side: str,
    active_positions: Mapping[str, int],
) -> CapitalAllocationResult:
    env = get_env_config()
    debug_mode = env.allocation_debug
    # デバッグ時は簡潔に呼び出し元の欠落コンテキストを警告するが、
    # 具体的なパラメータの存在チェックは外側の呼び出し元で行う。

    budgets: dict[str, float] = {}
    for name in weights:
        try:
            budgets[name] = float(total_budget) * float(weights.get(name, 0.0))
        except Exception:
            budgets[name] = 0.0
    remaining = budgets.copy()

    if debug_mode:
        logger.info(
            "[ALLOC_DEBUG] Starting capital allocation: side=%s, total_budget=$%s",
            side,
            f"{total_budget:,.0f}",
        )
        logger.info("[ALLOC_DEBUG] System weights: %s", dict(weights))
        logger.info("[ALLOC_DEBUG] System budgets: %s", dict(budgets))
        logger.info("[ALLOC_DEBUG] Active positions: %s", dict(active_positions))

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
            if debug_mode:
                logger.warning(
                    "[ALLOC_DEBUG] %s: calculate_position_size not found",
                    name,
                )

        # リスク設定の読み込み
        try:
            risk_pct = float(config.get("risk_pct", 0.02))
        except (TypeError, ValueError):
            risk_pct = 0.02
        try:
            max_pct = float(config.get("max_pct", 0.10))
        except (TypeError, ValueError):
            max_pct = 0.10
        # Guard against non-positive or non-finite settings coming from config
        try:
            import math as _math

            if (not _math.isfinite(risk_pct)) or risk_pct <= 0:
                if debug_mode:
                    logger.info(
                        "[ALLOC_DEBUG] %s: risk_pct=%s invalid -> fallback to default 0.020",
                        name,
                        risk_pct,
                    )
                risk_pct = 0.02
            if (not _math.isfinite(max_pct)) or max_pct <= 0:
                if debug_mode:
                    logger.info(
                        "[ALLOC_DEBUG] %s: max_pct=%s invalid -> fallback to default 0.100",
                        name,
                        max_pct,
                    )
                max_pct = 0.10
        except Exception:
            # On any unexpected error, keep already assigned values
            pass
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

        # 候補データの準備
        df = per_system.get(name)
        if df is None or getattr(df, "empty", True):
            candidates[name] = []
            if debug_mode:
                logger.info(f"[ALLOC_DEBUG] {name}: No candidates")
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
            if debug_mode:
                logger.info(
                    "[ALLOC_DEBUG] %s: %s candidates loaded",
                    name,
                    len(norm_records),
                )
        index_map[name] = 0

    counts = {name: 0 for name in ordered_names}
    max_pos_map = {}
    for name in ordered_names:
        try:
            max_pos = meta_map[name].max_positions
            taken = int(active_positions.get(name, 0))
            max_pos_map[name] = max(0, max_pos - taken)
        except Exception:
            max_pos_map[name] = 0

    if debug_mode:
        logger.info(f"[ALLOC_DEBUG] Max positions available: {dict(max_pos_map)}")

    chosen: list[dict[str, Any]] = []
    # map symbol -> system that selected it first
    chosen_symbols_map: dict[str, str] = {}
    # collect allocator-level excludes: reason -> set(symbols)
    from collections import defaultdict

    allocator_excludes: dict[str, set] = defaultdict(set)
    # record mapping symbol -> first selecting system for already_selected reason
    already_selected_map: dict[str, str] = {}

    def _normalize_shares(value: Any) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0

    # システム別の処理統計
    # Note: per-system processing stats were considered but are currently
    # unused in the final output. Keep logic minimal to avoid lint noise.

    still = True
    round_num = 0
    while still:
        still = False
        round_num += 1

        if debug_mode and round_num <= 3:  # 最初の3ラウンドのみログ
            logger.debug(f"[ALLOC_DEBUG] === Round {round_num} ===")

        for name in ordered_names:
            rows = candidates.get(name, [])
            if (
                not rows
                or remaining.get(name, 0.0) <= 0.0
                or counts.get(name, 0) >= max_pos_map.get(name, 0)
                or index_map.get(name, 0) >= len(rows)
            ):
                # スキップ理由の診断
                if debug_mode and round_num <= 3:
                    skip_reasons = []
                    if not rows:
                        skip_reasons.append("no_rows")
                    if remaining.get(name, 0.0) <= 0.0:
                        rem = remaining.get(name, 0.0)
                        skip_reasons.append(f"no_budget({rem:.0f})")
                    if counts.get(name, 0) >= max_pos_map.get(name, 0):
                        curr = counts.get(name, 0)
                        mp = max_pos_map.get(name, 0)
                        skip_reasons.append(f"max_pos({curr}/{mp})")
                    if index_map.get(name, 0) >= len(rows):
                        exhausted = index_map.get(name, 0)
                        total_rows = len(rows)
                        skip_reasons.append(f"exhausted({exhausted}/{total_rows})")
                    if skip_reasons:
                        logger.debug(
                            "[ALLOC_DEBUG] %s skipped: %s",
                            name,
                            ",".join(skip_reasons),
                        )
                continue

            meta = meta_map[name]
            idx = index_map[name]

            while idx < len(rows):
                row = rows[idx]
                idx += 1
                index_map[name] = idx

                sym = str(row.get("symbol", "")).upper()
                if not sym:
                    if debug_mode and round_num <= 2:
                        logger.debug(
                            "[ALLOC_DEBUG] %s: Skipping row with missing symbol",
                            name,
                        )
                    allocator_excludes["missing_symbol"].add(str(sym))
                    continue
                if sym in chosen_symbols_map:
                    prev = chosen_symbols_map.get(sym)
                    if debug_mode:
                        logger.info(
                            "[ALLOC] %s %s: already selected by %s",
                            name,
                            sym,
                            prev,
                        )
                    allocator_excludes["already_selected"].add(sym)
                    try:
                        if prev:
                            already_selected_map[str(sym)] = str(prev)
                    except Exception:
                        pass
                    continue

                # entry_price/stop_price の取得 - 存在しない場合は Close/ATR から計算
                entry = _safe_positive_float(row.get("entry_price"), allow_zero=False)
                stop = _safe_positive_float(row.get("stop_price"), allow_zero=False)

                # Fallback: entry_price が存在しない場合は Close を使用
                if entry is None or entry <= 0:
                    entry = _safe_positive_float(row.get("Close"), allow_zero=False)
                    if debug_mode and round_num <= 2:
                        logger.debug(
                            "[ALLOC_DEBUG] %s %s: Using Close as entry_price=%s",
                            name,
                            sym,
                            entry,
                        )

                # Fallback: stop_price が存在しない場合は Close ± ATR から計算
                if stop is None or stop <= 0:
                    # システム別優先順: S2/3/5/6=atr10, S1=atr20, S4=atr40, S7=atr50
                    atr = _safe_positive_float(row.get("atr10"), allow_zero=False)
                    if atr is None:
                        atr = _safe_positive_float(row.get("ATR10"), allow_zero=False)
                    if atr is None:
                        atr = _safe_positive_float(row.get("atr20"), allow_zero=False)
                    if atr is None:
                        atr = _safe_positive_float(row.get("atr40"), allow_zero=False)
                    if atr is None:
                        atr = _safe_positive_float(row.get("atr50"), allow_zero=False)

                    if entry and entry > 0 and atr and atr > 0:
                        if side == "long":
                            stop = entry - (atr * 2.0)  # ロング: エントリー - ATR*2
                        else:
                            stop = entry + (atr * 2.0)  # ショート: エントリー + ATR*2

                        if debug_mode and round_num <= 2:
                            logger.debug(
                                "[ALLOC_DEBUG] %s %s: Calculated stop_price=%s from entry=%s atr=%s",
                                name,
                                sym,
                                stop,
                                entry,
                                atr,
                            )

                if entry is None or stop is None or entry <= 0:
                    if debug_mode and round_num <= 2:
                        logger.debug(
                            ("[ALLOC_DEBUG] %s %s: Invalid prices entry=%s stop=%s"),
                            name,
                            sym,
                            entry,
                            stop,
                        )
                    allocator_excludes["invalid_price"].add(sym)
                    continue

                desired_shares = 0
                if meta.calc_fn is not None:
                    try:
                        # 正しい引数で calculate_position_size を呼び出し
                        entry_price = entry
                        stop_price = stop

                        if entry_price and stop_price and entry_price > 0:
                            desired_shares = _normalize_shares(
                                meta.calc_fn(
                                    remaining[name],  # available capital
                                    entry_price,  # entry_price
                                    stop_price,  # stop_price
                                    risk_pct=meta.risk_pct,
                                    max_pct=meta.max_pct,
                                )
                            )
                            # 詳細診断: リスク/資金上限ベースの概算株数も併記
                            if debug_mode and round_num <= 3:
                                try:
                                    rem_cap = float(remaining[name])
                                    e = float(entry_price)
                                    s = float(stop_price)
                                    # compute intermediate values in steps
                                    # to avoid long lines
                                    if e is not None and s is not None:
                                        rps = abs(e - s)
                                    else:
                                        rps = 0.0
                                    allow_risk = rem_cap * float(meta.risk_pct)
                                    if rps > 0:
                                        shares_by_risk = int(allow_risk // rps)
                                    else:
                                        shares_by_risk = 0
                                    cap_limit_dollars = rem_cap * float(meta.max_pct)
                                    if e > 0:
                                        try:
                                            abs_e = abs(e)
                                            tmp_cap = cap_limit_dollars // abs_e
                                            shares_by_capital = int(tmp_cap)
                                        except Exception:
                                            shares_by_capital = 0
                                        try:
                                            abs_e2 = abs(e)
                                            shares_by_cash = int(rem_cap // abs_e2)
                                        except Exception:
                                            shares_by_cash = 0
                                    else:
                                        shares_by_capital = 0
                                        shares_by_cash = 0
                                    # Log shares calculation details
                                    try:
                                        logger.info(
                                            "[ALLOC_DEBUG] %s %s: shares_calc rem=$%s "
                                            "entry=%s stop=%s "
                                            "risk_pct=%.4f max_pct=%.4f "
                                            "rps=%s shr_risk=%s "
                                            "shr_cap=%s shr_cash=%s desired=%s",
                                            name,
                                            sym,
                                            f"{rem_cap:.0f}",
                                            e,
                                            s,
                                            float(meta.risk_pct),
                                            float(meta.max_pct),
                                            (f"{rps:.4f}" if rps else "0"),
                                            shares_by_risk,
                                            shares_by_capital,
                                            shares_by_cash,
                                            desired_shares,
                                        )
                                    except Exception:
                                        # 診断ログは失敗しても処理継続
                                        pass
                                except Exception:
                                    # 診断ログは失敗しても処理継続
                                    pass
                        else:
                            if debug_mode and round_num <= 2:
                                logger.debug(
                                    "[ALLOC_DEBUG] %s %s: Invalid entry/stop entry_price=%s stop_price=%s",
                                    name,
                                    sym,
                                    entry_price,
                                    stop_price,
                                )
                    except Exception as e:
                        desired_shares = 0
                        if debug_mode and round_num <= 2:
                            logger.debug(
                                "[ALLOC_DEBUG] %s %s: calc_fn error %s",
                                name,
                                sym,
                                e,
                            )
                else:
                    if debug_mode and round_num <= 2:
                        logger.debug(
                            "[ALLOC_DEBUG] %s %s: No calc_fn available",
                            name,
                            sym,
                        )

                if desired_shares <= 0:
                    if debug_mode and round_num <= 2:
                        logger.debug(
                            "[ALLOC_DEBUG] %s %s: Invalid shares %s",
                            name,
                            sym,
                            desired_shares,
                        )
                    allocator_excludes["desired_shares_zero"].add(sym)
                    continue

                max_by_cash = int(remaining[name] // abs(entry)) if entry else 0
                shares = min(desired_shares, max_by_cash)
                if debug_mode and round_num <= 3:
                    try:
                        logger.info(
                            "[ALLOC_DEBUG] %s %s: shares_decision desired=%s max_by_cash=%s -> shares=%s",
                            name,
                            sym,
                            desired_shares,
                            max_by_cash,
                            shares,
                        )
                    except Exception:
                        pass
                if shares <= 0:
                    if debug_mode and round_num <= 2:
                        logger.debug(
                            "[ALLOC_DEBUG] %s %s: No cash shares=%s (desired=%s, max_cash=%s, remaining=$%s)",
                            name,
                            sym,
                            shares,
                            desired_shares,
                            max_by_cash,
                            f"{remaining[name]:.0f}",
                        )
                    allocator_excludes["no_cash_shares"].add(sym)
                    continue

                position_value = shares * abs(entry)
                if position_value <= 0:
                    if debug_mode and round_num <= 2:
                        logger.debug(
                            "[ALLOC_DEBUG] %s %s: Invalid position_value %s",
                            name,
                            sym,
                            position_value,
                        )
                    allocator_excludes["position_value_zero"].add(sym)
                    continue

                # 成功 - ポジション追加
                record = dict(row)
                record["shares"] = int(shares)
                record["position_value"] = float(round(position_value, 2))
                record["system"] = record.get("system", name)
                record["side"] = record.get("side", side)
                record["system_budget"] = float(round(remaining[name], 2))

                # フォールバック計算された entry_price と stop_price を設定
                record["entry_price"] = entry
                record["stop_price"] = stop
                remaining_after = remaining[name] - position_value
                record["remaining_after"] = float(round(remaining_after, 2))
                chosen.append(record)
                chosen_symbols_map[sym] = name
                remaining[name] -= position_value
                counts[name] = counts.get(name, 0) + 1
                still = True

                if debug_mode:
                    logger.debug(
                        "[ALLOC_DEBUG] %s %s: ALLOCATED shares=%s value=$%s remaining=$%s",
                        name,
                        sym,
                        shares,
                        f"{position_value:.0f}",
                        f"{remaining_after:.0f}",
                    )
                break

    # convert allocator_excludes defaultdict to normal dict for return
    try:
        alloc_excl_out = {k: set(v) for k, v in allocator_excludes.items()}
    except Exception:
        alloc_excl_out = {
            k: set(v) for k, v in getattr(allocator_excludes, "items", lambda: {})()
        }

    if chosen:
        frame = pd.DataFrame(chosen)
    else:
        frame = pd.DataFrame()

    # デバッグサマリー
    if debug_mode:
        total_allocated = len(chosen)
        total_budget_used = 0.0
        for name in budgets:
            try:
                total_budget_used += budgets[name] - remaining.get(name, 0)
            except Exception:
                continue
        logger.info(f"[ALLOC_DEBUG] === ALLOCATION SUMMARY ({side}) ===")
        logger.info(f"[ALLOC_DEBUG] Total allocated: {total_allocated} positions")
        logger.info(
            "[ALLOC_DEBUG] Budget used: $%s / $%s",
            f"{total_budget_used:,.0f}",
            f"{total_budget:,.0f}",
        )
        logger.info(f"[ALLOC_DEBUG] System counts: {dict(counts)}")
        logger.info(f"[ALLOC_DEBUG] Remaining budgets: {dict(remaining)}")

        if total_allocated == 0:
            # サイドを明示して誤解を避ける（例: ショート側のみ候補なし）
            logger.warning("[ALLOC_DEBUG] ⚠️ NO POSITIONS ALLOCATED! Check (%s):", side)
            for name in ordered_names:
                cand_count = len(candidates.get(name, []))
                calc_fn_available = meta_map[name].calc_fn is not None
                logger.warning(
                    f"[ALLOC_DEBUG]   {name}: candidates={cand_count}, "
                    f"calc_fn={calc_fn_available}, "
                    f"budget=${remaining.get(name, 0):.0f}, "
                    f"max_pos={max_pos_map.get(name, 0)}"
                )

    return CapitalAllocationResult(
        frame=frame,
        budgets=budgets,
        remaining=remaining,
        counts=counts,
        allocator_excludes=alloc_excl_out,
        already_selected_map=(already_selected_map if already_selected_map else None),
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
        tmp = _concat_nonempty(parts)
    except Exception:
        pass

    # Drop _system_no column if it exists
    if "_system_no" in tmp.columns:
        tmp = tmp.drop(columns=["_system_no"])

    # Add sequential numbering
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
    system_diagnostics: Mapping[str, Any] | None = None,
    market_data_dict: Mapping[str, pd.DataFrame] | None = None,
    signal_date: Any | None = None,
    include_trade_management: bool = False,
) -> tuple[pd.DataFrame, AllocationSummary]:
    """Combine per-system candidates into the final trade list.

    Parameters mirror the behaviour of ``scripts.run_all_systems_today`` but
    are greatly simplified so that tests can provide deterministic inputs.
    This function also includes trade management field calculations.
    """
    # 入力候補データのデバッグ情報を出力
    env = get_env_config()
    debug_mode = env.allocation_debug
    if debug_mode:
        logger.info("[ALLOC_DEBUG] === INPUT CANDIDATES DEBUG ===")
        for sys_name, df in per_system.items():
            if not df.empty:
                cols_str = f"{list(df.columns)}"
                msg = f"[ALLOC_DEBUG] {sys_name}: {len(df)} rows, columns: {cols_str}"
                logger.info(msg)
                # サンプル行の詳細を出力
                if len(df) > 0:
                    sample_row = df.iloc[0]
                    sample_dict = dict(sample_row)
                    logger.info(f"[ALLOC_DEBUG] {sys_name} sample row: {sample_dict}")
            else:
                logger.info(f"[ALLOC_DEBUG] {sys_name}: EMPTY DataFrame")

    # 非破壊の簡潔な診断ログ: 呼び出し元が strategies/symbol_system_map を
    # 渡していない場合、資本配分や現有ポジションのカウントに影響するため
    # デバッグ時のみ警告を出す。
    if debug_mode:
        if strategies is None:
            logger.warning(
                "[ALLOC_DEBUG] finalize_allocation called without strategies"
            )
        if symbol_system_map is None:
            logger.warning(
                "[ALLOC_DEBUG] called without symbol_system_map; "
                "active counts may be incomplete"
            )

    per_system_norm: dict[str, pd.DataFrame] = {}
    for name, df in per_system.items():
        try:
            per_system_norm[str(name).strip().lower()] = df
        except Exception:
            continue

    # 配分設定が提供されていない場合は、設定ファイルから読み込む
    if long_allocations is None and short_allocations is None:
        config_long_alloc, config_short_alloc = _load_allocations_from_settings()
        long_alloc = _normalize_allocations(config_long_alloc, DEFAULT_LONG_ALLOCATIONS)
        short_alloc = _normalize_allocations(
            config_short_alloc,
            DEFAULT_SHORT_ALLOCATIONS,
        )
    else:
        long_alloc = _normalize_allocations(long_allocations, DEFAULT_LONG_ALLOCATIONS)
        short_alloc = _normalize_allocations(
            short_allocations,
            DEFAULT_SHORT_ALLOCATIONS,
        )

    systems = sorted({*per_system_norm.keys(), *long_alloc.keys(), *short_alloc.keys()})
    max_pos_map = _resolve_max_positions(strategies, systems, default_max_positions)

    active_positions = count_active_positions_by_system(positions, symbol_system_map)
    available_slots: dict[str, int] = {}
    for name in systems:
        taken = int(active_positions.get(name, 0))
        limit = int(max_pos_map.get(name, default_max_positions))
        available_slots[name] = max(0, limit - taken)

    candidate_counts: dict[str, int] = {}
    for name in systems:
        candidate_counts[name] = _candidate_count(per_system_norm.get(name))

    # Remember whether strategies was explicitly provided by the caller
    original_strategies_provided = strategies is not None

    # Structured diagnostics to assist root-cause analysis when allocations
    # produce zero final entries. Populated and attached to AllocationSummary
    # before return. This includes whether callers provided strategies/symbol map
    # and per-system candidacy/slot/budget info.
    diagnostics: dict[str, Any] = {}
    diagnostics["callers"] = {
        # strategies_provided_initial reflects whether the caller passed
        # strategies explicitly. strategies_provided reflects the final
        # state after any fallback construction.
        "strategies_provided_initial": bool(original_strategies_provided),
        "strategies_provided": bool(strategies),
        "symbol_system_map_provided": bool(symbol_system_map),
        "include_trade_management": bool(include_trade_management),
    }
    diagnostics["candidate_counts"] = dict(candidate_counts)
    diagnostics["active_positions"] = dict(active_positions)
    diagnostics["available_slots"] = dict(available_slots)
    diagnostics["max_pos_map"] = dict(max_pos_map)

    # Fallbacks: if callers didn't provide strategies or symbol_system_map,
    # attempt to build safe defaults. This reduces the chance of getting
    # zero entries when callers simply omitted these optional arguments.
    if not strategies:
        try:
            # Import known strategy classes and build mapping similar to other
            # tools/scripts that construct strategies. This is a best-effort
            # fallback and will be skipped on import errors.
            from strategies.system1_strategy import System1Strategy
            from strategies.system2_strategy import System2Strategy
            from strategies.system3_strategy import System3Strategy
            from strategies.system4_strategy import System4Strategy
            from strategies.system5_strategy import System5Strategy
            from strategies.system6_strategy import System6Strategy
            from strategies.system7_strategy import System7Strategy

            objs = [
                System1Strategy(),
                System2Strategy(),
                System3Strategy(),
                System4Strategy(),
                System5Strategy(),
                System6Strategy(),
                System7Strategy(),
            ]
            strategies = {getattr(s, "SYSTEM_NAME", "").lower(): s for s in objs}
            diagnostics["callers"]["strategies_provided"] = bool(strategies)
            logger.debug("[ALLOC_DEBUG] built fallback strategies mapping")
        except Exception:
            # Leave strategies as-is (None) if build fails
            logger.debug("[ALLOC_DEBUG] could not build fallback strategies mapping")

    if not symbol_system_map:
        try:
            # Use the module helper to load persisted mapping if available
            symbol_system_map = load_symbol_system_map()
            diagnostics["callers"]["symbol_system_map_provided"] = bool(
                symbol_system_map
            )
            logger.debug("[ALLOC_DEBUG] loaded symbol_system_map fallback")
        except Exception:
            symbol_system_map = {}
            logger.debug("[ALLOC_DEBUG] could not load symbol_system_map fallback")

    # Production safety: optionally require strategies to be explicitly provided.
    # Controlled by env var ALLOCATION_REQUIRE_STRATEGIES. Default is permissive
    # (do not raise) to avoid breaking existing tests/tools. When enabled, this
    # forces callers to pass strategies and helps catch omission errors early.
    try:
        require_strat = os.environ.get("ALLOCATION_REQUIRE_STRATEGIES", "0") == "1"
    except Exception:
        require_strat = False

    if require_strat and not original_strategies_provided:
        # In strict mode, raise to make omission explicit. We check the
        # original flag so that internal fallback construction does not
        # bypass the safety guard.
        raise RuntimeError("finalize_allocation: strategies required")

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

        # Add slot-mode diagnostics
        try:
            diagnostics.setdefault("slot", {})
            diagnostics["slot"]["distribution"] = dict(slot_result.distribution)
            diagnostics["slot"]["candidate_counts"] = dict(slot_result.candidate_counts)
        except Exception:
            pass

        # Slot モードでも実株数が必要な場合（UI/TradeManager 連携）、
        # 選抜済み候補に対して資本配分ロジックでサイズ決定を実施する。
        # 条件: include_trade_management=True（既定の呼び出しパス）
        try:
            if include_trade_management and not final_df.empty:
                # 総資本/比率の解決（既定値で安全に）
                ratio_conv = _safe_positive_float(default_long_ratio, allow_zero=True)
                ratio = ratio_conv if ratio_conv is not None else 0.5
                cap_conv = _safe_positive_float(default_capital, allow_zero=True)
                total_cap = cap_conv if cap_conv is not None else 100000.0
                long_cap = total_cap * ratio
                short_cap = total_cap * (1.0 - ratio)

                # 選抜済みスロットから side 別の per_system マップを作る
                selected_per_system_long: dict[str, pd.DataFrame] = {}
                selected_per_system_short: dict[str, pd.DataFrame] = {}
                if "system" in final_df.columns:
                    for pair in final_df.groupby("system", sort=False):
                        try:
                            sys_name = str(pair[0])
                            group = pair[1]
                        except Exception:
                            continue
                        key = str(sys_name).strip().lower()
                        try:
                            # そのまま side 列を尊重（slot 結果には付与済み）
                            is_short = False
                            if "side" in group.columns:
                                first_side = str(group["side"].iloc[0]).strip().lower()
                                is_short = first_side == "short"

                            grp = group.reset_index(drop=True)
                            if is_short:
                                selected_per_system_short[key] = grp
                            else:
                                selected_per_system_long[key] = grp
                        except Exception:
                            # 何かあれば long 側へ
                            selected_per_system_long[key] = group.reset_index(drop=True)

                # strategies 正規化 (slot-mode local variable)
                strategies_norm_slot: dict[str, object] = {}
                try:
                    for name, obj in (strategies or {}).items():
                        try:
                            strategies_norm_slot[str(name).strip().lower()] = obj
                        except Exception:
                            continue
                except Exception:
                    strategies_norm_slot = {}

                # 選抜済み候補に対し、各 side で資本配分により shares を算出
                long_sized = _allocate_by_capital(
                    selected_per_system_long,
                    strategies=strategies_norm_slot,
                    total_budget=long_cap,
                    weights=long_alloc,
                    side="long",
                    active_positions=active_positions,
                )
                short_sized = _allocate_by_capital(
                    selected_per_system_short,
                    strategies=strategies_norm_slot,
                    total_budget=short_cap,
                    weights=short_alloc,
                    side="short",
                    active_positions=active_positions,
                )

                # Merge allocator excludes from sizing results into diagnostics
                try:
                    alloc_excl_slot: dict[str, Any] = {}
                    try:
                        if getattr(long_sized, "allocator_excludes", None):
                            alloc_excl_slot["long"] = {
                                k: sorted(list(v))
                                for k, v in (
                                    long_sized.allocator_excludes or {}
                                ).items()
                            }
                            if getattr(long_sized, "already_selected_map", None):
                                try:
                                    alloc_excl_slot["long"]["already_selected_map"] = (
                                        dict(long_sized.already_selected_map or {})
                                    )
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    try:
                        if getattr(short_sized, "allocator_excludes", None):
                            alloc_excl_slot["short"] = {
                                k: sorted(list(v))
                                for k, v in (
                                    short_sized.allocator_excludes or {}
                                ).items()
                            }
                            if getattr(short_sized, "already_selected_map", None):
                                try:
                                    alloc_excl_slot["short"]["already_selected_map"] = (
                                        dict(short_sized.already_selected_map or {})
                                    )
                                except Exception:
                                    pass
                    except Exception:
                        pass

                    if alloc_excl_slot:
                        diagnostics.setdefault("allocator_excludes", {})
                        diagnostics["allocator_excludes"].update(alloc_excl_slot)
                except Exception:
                    pass

                # 非空のみ連結（FutureWarning 回避のため all-NA は除外）
                def _is_effectively_empty(_df: pd.DataFrame) -> bool:
                    try:
                        if _df is None or getattr(_df, "empty", True):
                            return True
                        return bool(_df.isna().all().all())
                    except Exception:
                        return True

                frames_cap = []
                for df in [long_sized.frame, short_sized.frame]:
                    try:
                        if df is not None and not _is_effectively_empty(df):
                            frames_cap.append(df)
                    except Exception:
                        continue
                if frames_cap:
                    final_df = _concat_nonempty(frames_cap)
                else:
                    # sizing に失敗した場合は従来の slot 結果を返す（後段で shares=0 警告になる）
                    final_df = slot_result.frame
        except Exception:
            # sizing での例外は致命的ではないため、slot 結果をそのまま継続
            final_df = slot_result.frame
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
        # Concatenate non-empty and non-all-NA frames only to avoid
        # pandas FutureWarning about dtype inference with empty/all-NA entries

        def _is_effectively_empty(_df: pd.DataFrame) -> bool:
            try:
                if _df is None or getattr(_df, "empty", True):
                    return True
                # all cells are NA -> treat as empty for concat purposes
                return bool(_df.isna().all().all())
            except Exception:
                return True

        frames = []
        for df in [long_result.frame, short_result.frame]:
            try:
                if df is not None and not _is_effectively_empty(df):
                    frames.append(df)
            except Exception:
                continue
        if frames:
            final_df = _concat_nonempty(frames)
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

        # Add capital-mode diagnostics from allocation results
        try:
            diagnostics.setdefault("capital", {})
            diagnostics["capital"]["long"] = {
                "budgets": dict(long_result.budgets),
                "remaining": dict(long_result.remaining),
                "counts": dict(long_result.counts),
            }
            diagnostics["capital"]["short"] = {
                "budgets": dict(short_result.budgets),
                "remaining": dict(short_result.remaining),
                "counts": dict(short_result.counts),
            }
            # Merge allocator_excludes for diagnostics visibility
            try:
                alloc_excl: dict[str, Any] = {}
                # long side
                try:
                    if getattr(long_result, "allocator_excludes", None):
                        alloc_excl["long"] = {
                            k: sorted(list(v))
                            for k, v in (long_result.allocator_excludes or {}).items()
                        }
                        if getattr(long_result, "already_selected_map", None):
                            try:
                                alloc_excl["long"]["already_selected_map"] = dict(
                                    long_result.already_selected_map or {}
                                )
                            except Exception:
                                pass
                except Exception:
                    pass

                # short side
                try:
                    if getattr(short_result, "allocator_excludes", None):
                        alloc_excl["short"] = {
                            k: sorted(list(v))
                            for k, v in (short_result.allocator_excludes or {}).items()
                        }
                        if getattr(short_result, "already_selected_map", None):
                            try:
                                alloc_excl["short"]["already_selected_map"] = dict(
                                    short_result.already_selected_map or {}
                                )
                            except Exception:
                                pass
                except Exception:
                    pass

                if alloc_excl:
                    diagnostics.setdefault("allocator_excludes", {})
                    diagnostics["allocator_excludes"].update(alloc_excl)
            except Exception:
                # non-fatal: diagnostics best-effort
                pass
        except Exception:
            pass

    if system_diagnostics:
        try:
            # Merge user-provided diagnostics with auto-collected diagnostics
            merged = dict(diagnostics or {})
            try:
                merged.update({str(k): v for k, v in system_diagnostics.items()})
            except Exception:
                # fallback: attach as-is
                merged["user"] = system_diagnostics
            summary.system_diagnostics = merged
        except Exception:
            try:
                summary.system_diagnostics = dict(system_diagnostics)
            except Exception:
                summary.system_diagnostics = None
    else:
        # Attach our collected diagnostics when caller didn't provide any
        try:
            summary.system_diagnostics = dict(diagnostics)
        except Exception:
            summary.system_diagnostics = diagnostics or None

    if not final_df.empty:
        final_df = _sort_final_frame(final_df)
    else:
        final_df = final_df.copy()

    if "system" in final_df.columns:
        try:
            series = final_df["system"].astype(str)
            series = series.str.strip().str.lower()
            counts_series = series.value_counts()
            summary.final_counts = {str(k): int(v) for k, v in counts_series.items()}
        except Exception:
            summary.final_counts = {}
    else:
        summary.final_counts = {}

    # Add candidate counts for capital mode (slot mode already has them)
    if summary.slot_candidates is None:
        summary.slot_candidates = candidate_counts

    if include_trade_management:
        if market_data_dict is None or signal_date is None:
            raise ValueError(
                "market_data_dict and signal_date must be provided "
                "when include_trade_management is True."
            )
        # Convert cached frames so TradeManager receives a datetime index
        # Rolling cache commonly stores an integer index
        prepared_market_data = {}
        for sym, df in market_data_dict.items():
            if df is None or df.empty:
                continue
            # Check if 'date' column exists and index is not already datetime
            if "date" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                df_copy = df.copy()
                df_copy["date"] = pd.to_datetime(df_copy["date"])
                df_copy = df_copy.set_index("date", drop=False)
                prepared_market_data[sym] = df_copy
            else:
                prepared_market_data[sym] = df

        # TradeManager API: instantiate without args and enhance allocation
        tm = TradeManager()
        final_df = tm.enhance_allocation_with_trade_management(
            final_df, prepared_market_data, signal_date
        )

    # Final summary diagnostics
    # If the caller provided system_diagnostics we attempted to merge them
    # earlier. Do not unconditionally overwrite the collected diagnostics
    # here; only use the caller's raw diagnostics as a fallback when no
    # diagnostics have been attached.
    if system_diagnostics and not summary.system_diagnostics:
        summary.system_diagnostics = {str(k): v for k, v in system_diagnostics.items()}

    # final_count
    try:
        summary.final_count = int(len(final_df))
    except Exception:
        summary.final_count = 0

    # final_symbols
    try:
        if "symbol" in final_df.columns:
            summary.final_symbols = [str(s) for s in final_df["symbol"].tolist()]
        else:
            summary.final_symbols = []
    except Exception:
        summary.final_symbols = []

    # Compute long/short counts. Prefer 'side' column if available, otherwise
    # fall back to numeric 'long'/'short' columns when present.
    try:
        if "side" in final_df.columns:
            side_series = final_df["side"].astype(str).str.lower()
            summary.final_long_count = int((side_series == "long").sum())
            summary.final_short_count = int((side_series == "short").sum())
        else:
            # Fallback to numeric columns named 'long'/'short'
            try:
                if "long" in final_df.columns:
                    long_series = pd.to_numeric(
                        final_df["long"], errors="coerce"
                    ).fillna(0)
                    summary.final_long_count = int(long_series.sum())
                else:
                    summary.final_long_count = 0
            except Exception:
                summary.final_long_count = 0

            try:
                if "short" in final_df.columns:
                    short_series = pd.to_numeric(
                        final_df["short"], errors="coerce"
                    ).fillna(0)
                    summary.final_short_count = int(short_series.sum())
                else:
                    summary.final_short_count = 0
            except Exception:
                summary.final_short_count = 0
    except Exception:
        summary.final_long_count = 0
        summary.final_short_count = 0

    return final_df, summary


def to_allocation_summary_dict(summary: AllocationSummary | Any) -> dict[str, Any]:
    """AllocationSummary もしくは類似オブジェクトを dict へ安全変換。

    - 既知フィールドを優先的に収集
    - 失敗しても空 dict
    - 追加フィールドにある程度耐性
    """
    try:
        fields = [
            "mode",
            "long_allocations",
            "short_allocations",
            "active_positions",
            "available_slots",
            "final_counts",
            "slot_allocation",
            "slot_candidates",
            "budgets",
            "budget_remaining",
            "capital_long",
            "capital_short",
            "system_diagnostics",
        ]
        out: dict[str, Any] = {}
        for f in fields:
            if hasattr(summary, f):
                try:
                    out[f] = getattr(summary, f)
                except Exception:
                    pass
        # 追加で *_allocations など緩く拾う（既存キー除外）
        try:
            for name in dir(summary):
                if name.startswith("_") or name in out:
                    continue
                if callable(getattr(summary, name, None)):
                    continue
                if name.endswith("_allocations"):
                    try:
                        out[name] = getattr(summary, name)
                    except Exception:
                        pass
        except Exception:
            pass
        return out
    except Exception:
        return {}


__all__ = [
    "AllocationSummary",
    "count_active_positions_by_system",
    "finalize_allocation",
    "load_symbol_system_map",
    "to_allocation_summary_dict",
]
