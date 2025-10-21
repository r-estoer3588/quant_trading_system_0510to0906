from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from common import broker_alpaca as ba
from common.data_loader import load_price
from common.position_age import (
    fetch_entry_dates_from_alpaca,
    load_entry_dates,
    save_entry_dates,
)
from common.exit_planner import decide_exit_schedule
from common.utils_spy import get_latest_nyse_trading_day
from core.final_allocation import load_symbol_system_map

# strategy classes (import directly to avoid circular UI imports)
from strategies.system1_strategy import System1Strategy
from strategies.system2_strategy import System2Strategy
from strategies.system3_strategy import System3Strategy
from strategies.system4_strategy import System4Strategy
from strategies.system5_strategy import System5Strategy
from strategies.system6_strategy import System6Strategy
from strategies.system7_strategy import System7Strategy


STRATEGY_CLASS_MAP: dict[str, Callable[[], Any]] = {
    "system1": System1Strategy,
    "system2": System2Strategy,
    "system3": System3Strategy,
    "system4": System4Strategy,
    "system5": System5Strategy,
    "system6": System6Strategy,
    "system7": System7Strategy,
}


def _latest_trading_day() -> pd.Timestamp | None:
    try:
        cal = get_latest_nyse_trading_day()
    except Exception:
        cal = None
    # Try to use SPY price last date if available
    try:
        spy_df = load_price("SPY", cache_profile="rolling")
        if spy_df is not None and not spy_df.empty:
            try:
                # index may be date-like or have 'date' column
                if "date" in spy_df.columns:
                    pd_last = pd.to_datetime(spy_df["date"]).max()
                else:
                    pd_last = pd.to_datetime(spy_df.index).max()
                price_day = pd.Timestamp(pd_last).normalize()
            except Exception:
                price_day = None
        else:
            price_day = None
    except Exception:
        price_day = None

    if cal is not None and price_day is not None:
        return max(pd.Timestamp(cal), pd.Timestamp(price_day)).normalize()
    return pd.Timestamp(cal).normalize() if cal is not None else price_day


def _normalize_symbol_system_map(path: Path | None) -> dict[str, str]:
    try:
        data = load_symbol_system_map(path)
        if isinstance(data, dict):
            # Ensure stable key/value types
            return {str(k).upper(): str(v).lower() for k, v in data.items()}
    except Exception:
        pass
    return {}


def _get_col_value_safe(row: pd.Series, keys: Sequence[str]) -> float | None:
    for k in keys:
        try:
            if k in row and row[k] is not None:
                return float(row[k])
        except Exception:
            continue
    return None


def _find_entry_index(index: pd.Index, entry_dt: pd.Timestamp) -> int:
    try:
        if entry_dt in index:
            arr = index.get_indexer([entry_dt])
        else:
            arr = index.get_indexer([entry_dt], method="bfill")
        if len(arr) and arr[0] >= 0:
            return int(arr[0])
    except Exception:
        pass
    return -1


def _entry_and_stop_prices(
    system: str,
    strategy: Any,
    df: pd.DataFrame,
    entry_idx: int,
    prev_close: float,
) -> tuple[float | None, float | None]:
    try:
        # Access both capitalized and lower-case column names
        if system == "system1":
            row_entry = df.iloc[int(entry_idx)]
            entry_price = _get_col_value_safe(row_entry, ["Open", "open"])
            if entry_price is None:
                return None, None
            entry_price_f = float(entry_price)
            row_prev = df.iloc[int(max(0, entry_idx - 1))]
            atr20 = _get_col_value_safe(row_prev, ["ATR20", "atr20"]) or 0.0
            stop_mult = float(strategy.config.get("stop_atr_multiple", 5.0))
            return entry_price_f, float(entry_price_f - stop_mult * float(atr20))
        if system == "system2":
            row_entry = df.iloc[int(entry_idx)]
            entry_price = _get_col_value_safe(row_entry, ["Open", "open"])
            if entry_price is None:
                return None, None
            entry_price_f = float(entry_price)
            row_prev = df.iloc[int(max(0, entry_idx - 1))]
            atr = _get_col_value_safe(row_prev, ["ATR10", "atr10"]) or 0.0
            stop_mult = float(strategy.config.get("stop_atr_multiple", 3.0))
            return entry_price_f, float(entry_price_f + stop_mult * float(atr))
        if system == "system6":
            ratio = float(strategy.config.get("entry_price_ratio_vs_prev_close", 1.05))
            entry_price = round(prev_close * ratio, 2)
            entry_price_f = float(entry_price)
            row_prev = df.iloc[int(max(0, entry_idx - 1))]
            atr = _get_col_value_safe(row_prev, ["ATR10", "atr10"]) or 0.0
            stop_mult = float(strategy.config.get("stop_atr_multiple", 3.0))
            return entry_price_f, float(entry_price_f + stop_mult * float(atr))
        if system == "system3":
            ratio = float(strategy.config.get("entry_price_ratio_vs_prev_close", 0.93))
            entry_price = round(prev_close * ratio, 2)
            entry_price_f = float(entry_price)
            row_prev = df.iloc[int(max(0, entry_idx - 1))]
            atr = _get_col_value_safe(row_prev, ["ATR10", "atr10"]) or 0.0
            stop_mult = float(strategy.config.get("stop_atr_multiple", 2.5))
            return entry_price_f, float(entry_price_f - stop_mult * float(atr))
        if system == "system4":
            row_entry = df.iloc[int(entry_idx)]
            entry_price = _get_col_value_safe(row_entry, ["Open", "open"])
            if entry_price is None:
                return None, None
            entry_price_f = float(entry_price)
            row_prev = df.iloc[int(max(0, entry_idx - 1))]
            atr40 = _get_col_value_safe(row_prev, ["ATR40", "atr40"]) or 0.0
            stop_mult = float(strategy.config.get("stop_atr_multiple", 1.5))
            return entry_price_f, float(entry_price_f - stop_mult * float(atr40))
        if system == "system5":
            ratio = float(strategy.config.get("entry_price_ratio_vs_prev_close", 0.97))
            entry_price = round(prev_close * ratio, 2)
            entry_price_f = float(entry_price)
            row_prev = df.iloc[int(max(0, entry_idx - 1))]
            atr = _get_col_value_safe(row_prev, ["ATR10", "atr10"]) or 0.0
            stop_mult = float(strategy.config.get("stop_atr_multiple", 3.0))
            return entry_price_f, float(entry_price_f - stop_mult * float(atr))
    except Exception:
        return None, None
    return None, None


def _apply_strategy_state(
    system: str,
    strategy: Any,
    df: pd.DataFrame,
    entry_idx: int,
    prev_close: float,
) -> None:
    try:
        if system == "system5":
            row_prev = df.iloc[int(max(0, entry_idx - 1))]
            atr = _get_col_value_safe(row_prev, ["ATR10", "atr10"])
            if atr is not None:
                strategy._last_entry_atr = atr
    except Exception:
        pass
    try:
        if system in {"system3", "system5", "system6"}:
            strategy._last_prev_close = prev_close
    except Exception:
        pass


def _evaluate_position_for_exit(
    pos: Any,
    entry_map: Mapping[str, Any],
    symbol_system_map: Mapping[str, str],
    latest_trading_day: pd.Timestamp | None,
    strategy_classes: Mapping[str, Callable[[], Any]],
) -> tuple[str, str, int, str, dict[str, Any], bool] | None:
    try:
        sym = str(getattr(pos, "symbol", "") or "").upper()
        if not sym:
            return None
        qty = int(abs(float(getattr(pos, "qty", 0)) or 0))
        if qty <= 0:
            return None
        pos_side = str(getattr(pos, "side", "") or "").lower()
        system = symbol_system_map.get(sym, "").lower()
        if not system:
            if sym == "SPY" and pos_side == "short":
                system = "system7"
            else:
                return None
        if system == "system7":
            return None
        entry_date_str = entry_map.get(sym)
        if not entry_date_str:
            return None
        entry_dt = pd.to_datetime(entry_date_str).normalize()
        df_price = load_price(sym, cache_profile="full")
        if df_price is None or df_price.empty:
            return None
        df = df_price.copy(deep=False)
        # Normalize index to dates
        if "Date" in df.columns or "date" in df.columns:
            if "Date" in df.columns:
                df.index = pd.Index(pd.to_datetime(df["Date"]).dt.normalize())
            else:
                df.index = pd.Index(pd.to_datetime(df["date"]).dt.normalize())
        else:
            df.index = pd.Index(pd.to_datetime(df.index).normalize())
        if latest_trading_day is None and len(df.index) > 0:
            latest_trading_day = pd.to_datetime(df.index[-1]).normalize()
        entry_idx = _find_entry_index(df.index, entry_dt)
        if entry_idx < 0:
            return None
        strategy_cls = strategy_classes.get(system)
        if strategy_cls is None:
            return None
        strategy = strategy_cls()
        row_prev = df.iloc[int(max(0, entry_idx - 1))]
        prev_close = _get_col_value_safe(row_prev, ["Close", "close"])
        if prev_close is None:
            return None
        entry_price, stop_price = _entry_and_stop_prices(system, strategy, df, entry_idx, float(prev_close))
        if entry_price is None or stop_price is None:
            return None
        _apply_strategy_state(system, strategy, df, entry_idx, float(prev_close))
        exit_price, exit_date = strategy.compute_exit(df, int(entry_idx), float(entry_price), float(stop_price))
        today_norm = pd.to_datetime(df.index[-1]).normalize()
        if latest_trading_day is not None:
            today_norm = latest_trading_day
        is_today_exit, when = decide_exit_schedule(system, exit_date, today_norm)
        row_base = {
            "symbol": sym,
            "qty": qty,
            "position_side": pos_side,
            "system": system,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "exit_price": exit_price,
        }
        return system, pos_side, qty, when, row_base, is_today_exit
    except Exception:
        return None


def analyze_exit_candidates(
    paper_mode: bool = True, *, client: Any | None = None, skip_external: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int], str | None]:
    """Analyze current positions and classify exits for today vs planned.

    Returns (exits_today_df, planned_df, exit_counts, error_message)
    """
    exits_today_rows: list[dict[str, Any]] = []
    planned_rows: list[dict[str, Any]] = []
    exit_counts: dict[str, int] = {f"system{i}": 0 for i in range(1, 8)}

    if skip_external:
        # External API calls disabled: return empty results
        return pd.DataFrame(), pd.DataFrame(), exit_counts, None

    try:
        client_tmp = client or ba.get_client(paper=paper_mode)
        try:
            positions = list(client_tmp.get_all_positions())
        except Exception:
            positions = []

        raw_entry_map = load_entry_dates()
        entry_map: dict[str, str] = {}
        for k, v in (raw_entry_map or {}).items():
            try:
                entry_map[str(k).upper()] = str(v)
            except Exception:
                continue

        # Fill missing entry dates via Alpaca if any
        missing = [
            str(getattr(p, "symbol", "") or "").upper()
            for p in positions
            if str(getattr(p, "symbol", "") or "").upper()
            and str(getattr(p, "symbol", "") or "").upper() not in entry_map
        ]
        if missing:
            try:
                fetched = fetch_entry_dates_from_alpaca(client_tmp, missing)
            except Exception:
                fetched = None
            if fetched:
                for sym, ts in fetched.items():
                    if sym not in entry_map:
                        try:
                            entry_map[sym] = pd.Timestamp(ts).strftime("%Y-%m-%d")
                        except Exception:
                            continue
                try:
                    save_entry_dates(entry_map)
                except Exception:
                    pass

        symbol_system_map = _normalize_symbol_system_map(Path("data/symbol_system_map.json"))
        latest_trading_day = _latest_trading_day()
        strategy_classes = STRATEGY_CLASS_MAP

        for pos in positions:
            result = _evaluate_position_for_exit(
                pos, entry_map, symbol_system_map, latest_trading_day, strategy_classes
            )
            if result is None:
                continue
            system, _pos_side, _qty, exit_when, row_base, exit_today = result
            when_val = str(exit_when or "").strip()
            when_lower = when_val.lower()
            when_display = when_lower or when_val
            if exit_today:
                exit_counts[system] = exit_counts.get(system, 0) + 1
                if when_lower == "tomorrow_open":
                    planned_rows.append(row_base | {"when": when_display})
                else:
                    exits_today_rows.append(row_base | {"when": when_display})
            else:
                planned_rows.append(row_base | {"when": when_display})
        exits_today_df = pd.DataFrame(exits_today_rows)
        planned_df = pd.DataFrame(planned_rows)
        return (
            exits_today_df,
            planned_df,
            exit_counts,
            None,
        )
    except Exception as exc:
        return pd.DataFrame(), pd.DataFrame(), exit_counts, str(exc)
