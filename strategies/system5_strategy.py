from __future__ import annotations

import pandas as pd

from .base_strategy import StrategyBase
from .constants import STOP_ATR_MULTIPLE_DEFAULT, FALLBACK_EXIT_DAYS_DEFAULT
from common.alpaca_order import AlpacaOrderMixin
from common.backtest_utils import simulate_trades_with_risk
from common.utils import resolve_batch_size
from core.system5 import (
    prepare_data_vectorized_system5,
    generate_candidates_system5,
    get_total_days_system5,
)


class System5Strategy(AlpacaOrderMixin, StrategyBase):
    SYSTEM_NAME = "system5"
    PREFER_PROCESS_POOL = True

    def __init__(self):
        super().__init__()

    def prepare_data(
        self,
        raw_data_or_symbols,
        progress_callback=None,
        log_callback=None,
        skip_callback=None,
        batch_size: int | None = None,
        use_process_pool: bool = False,
        **kwargs,
    ):
        if isinstance(raw_data_or_symbols, dict):
            symbols = list(raw_data_or_symbols.keys())
            raw_dict = None if use_process_pool else raw_data_or_symbols
        else:
            symbols = list(raw_data_or_symbols)
            raw_dict = None

        if batch_size is None and not use_process_pool and raw_dict is not None:
            try:
                from config.settings import get_settings

                batch_size = get_settings(create_dirs=False).data.batch_size
            except Exception:
                batch_size = 100
            batch_size = resolve_batch_size(len(raw_dict), batch_size)
        return prepare_data_vectorized_system5(
            raw_dict,
            progress_callback=progress_callback,
            log_callback=log_callback,
            batch_size=batch_size,
            symbols=symbols,
            use_process_pool=use_process_pool,
            skip_callback=skip_callback,
        )

    def generate_candidates(
        self,
        prepared_dict,
        progress_callback=None,
        log_callback=None,
        batch_size: int | None = None,
    ):
        try:
            from config.settings import get_settings

            top_n = int(get_settings(create_dirs=False).backtest.top_n_rank)
        except Exception:
            top_n = 10
        if batch_size is None:
            try:
                from config.settings import get_settings

                batch_size = get_settings(create_dirs=False).data.batch_size
            except Exception:
                batch_size = 100
            batch_size = resolve_batch_size(len(prepared_dict), batch_size)
        return generate_candidates_system5(
            prepared_dict,
            top_n=top_n,
            progress_callback=progress_callback,
            log_callback=log_callback,
            batch_size=batch_size,
        )

    def run_backtest(
        self, prepared_dict, candidates_by_date, capital, on_progress=None, on_log=None
    ):
        trades_df, _ = simulate_trades_with_risk(
            candidates_by_date,
            prepared_dict,
            capital,
            self,
            on_progress=on_progress,
            on_log=on_log,
        )
        return trades_df

    def compute_entry(self, df: pd.DataFrame, candidate: dict, current_capital: float):
        try:
            entry_idx = df.index.get_loc(candidate["entry_date"])
        except Exception:
            return None
        if entry_idx <= 0 or entry_idx >= len(df):
            return None
        prev_close = float(df.iloc[entry_idx - 1]["Close"])
        ratio = float(
            getattr(self, "config", {}).get(
                "entry_price_ratio_vs_prev_close", 0.97
            )
        )
        entry_price = round(prev_close * ratio, 2)
        try:
            atr = float(df.iloc[entry_idx - 1]["ATR10"])
        except Exception:
            return None
        stop_mult = float(
            getattr(self, "config", {}).get(
                "stop_atr_multiple", STOP_ATR_MULTIPLE_DEFAULT
            )
        )
        stop_price = entry_price - stop_mult * atr
        if entry_price - stop_price <= 0:
            return None
        self._last_entry_atr = atr
        return entry_price, stop_price

    def compute_exit(
        self, df: pd.DataFrame, entry_idx: int, entry_price: float, stop_price: float
    ):
        atr = getattr(self, "_last_entry_atr", None)
        if atr is None:
            try:
                atr = float(df.iloc[entry_idx - 1]["ATR10"])
            except Exception:
                atr = 0.0
        target_mult = float(
            getattr(self, "config", {}).get("target_atr_multiple", 1.0)
        )
        target_price = entry_price + target_mult * atr
        fallback_days = int(
            getattr(self, "config", {}).get(
                "fallback_exit_after_days", FALLBACK_EXIT_DAYS_DEFAULT
            )
        )

        offset = 1
        while offset <= fallback_days and entry_idx + offset < len(df):
            row = df.iloc[entry_idx + offset]
            if float(row["High"]) >= target_price:
                exit_idx = min(entry_idx + offset + 1, len(df) - 1)
                exit_date = df.index[exit_idx]
                exit_price = float(df.iloc[exit_idx]["Open"])
                return exit_price, exit_date
            if float(row["Low"]) <= stop_price:
                if entry_idx + offset < len(df) - 1:
                    prev_close2 = float(df.iloc[entry_idx + offset]["Close"])
                    ratio = float(
                        getattr(self, "config", {}).get(
                            "entry_price_ratio_vs_prev_close", 0.97
                        )
                    )
                    entry_price = round(prev_close2 * ratio, 2)
                    atr2 = float(df.iloc[entry_idx + offset]["ATR10"])
                    stop_mult = float(
                        getattr(self, "config", {}).get(
                            "stop_atr_multiple", STOP_ATR_MULTIPLE_DEFAULT
                        )
                    )
                    stop_price = entry_price - stop_mult * atr2
                    target_price = entry_price + target_mult * atr2
                    entry_idx = entry_idx + offset
                    offset = 0
                else:
                    exit_date = df.index[entry_idx + offset]
                    exit_price = float(stop_price)
                    return exit_price, exit_date
            offset += 1

        idx2 = min(entry_idx + fallback_days, len(df) - 1)
        exit_date = df.index[idx2]
        exit_price = float(df.iloc[idx2]["Open"])
        return exit_price, exit_date

    def compute_pnl(self, entry_price: float, exit_price: float, shares: int) -> float:
        return (exit_price - entry_price) * shares

    def prepare_minimal_for_test(self, raw_data_dict: dict) -> dict:
        out = {}
        for sym, df in raw_data_dict.items():
            x = df.copy()
            x["SMA100"] = x["Close"].rolling(100).mean()
            out[sym] = x
        return out

    def get_total_days(self, data_dict: dict) -> int:
        return get_total_days_system5(data_dict)
