"""System1 strategy wrapper class using shared core functions.

This class integrates with YAML-driven settings for backtest parameters
and relies on StrategyBase to inject risk/system-specific config.  As an
extension example, Alpaca 発注処理も組み込み、バックテストと実売双方に
対応できるようにする。
"""

from __future__ import annotations

import pandas as pd

from common.alpaca_order import AlpacaOrderMixin
from common.backtest_utils import simulate_trades_with_risk
from core.system1 import (
    generate_roc200_ranking_system1,
    get_total_days_system1,
    prepare_data_vectorized_system1,
)

from .base_strategy import StrategyBase
from .constants import STOP_ATR_MULTIPLE_SYSTEM1


class System1Strategy(AlpacaOrderMixin, StrategyBase):
    SYSTEM_NAME = "system1"

    def prepare_data(self, raw_data_or_symbols, **kwargs):
        progress_callback = kwargs.pop("progress_callback", None)
        log_callback = kwargs.pop("log_callback", None)
        skip_callback = kwargs.pop("skip_callback", None)
        use_process_pool = kwargs.pop("use_process_pool", False)

        if isinstance(raw_data_or_symbols, dict):
            symbols = list(raw_data_or_symbols.keys())
            raw_dict = None if use_process_pool else raw_data_or_symbols
        else:
            symbols = list(raw_data_or_symbols)
            raw_dict = None

        return prepare_data_vectorized_system1(
            raw_dict,
            progress_callback=progress_callback,
            log_callback=log_callback,
            skip_callback=skip_callback,
            use_process_pool=use_process_pool,
            symbols=symbols,
            **kwargs,
        )

    def generate_candidates(self, prepared_dict, market_df=None, **kwargs):
        # Pull top-N from YAML backtest config
        try:
            from config.settings import get_settings

            top_n = get_settings(create_dirs=False).backtest.top_n_rank
        except Exception:
            top_n = 10
        if market_df is None:
            market_df = prepared_dict.get("SPY")
            if market_df is None:
                raise ValueError("SPY data not found in prepared_dict.")
        return generate_roc200_ranking_system1(
            prepared_dict,
            market_df,
            top_n=top_n,
            **kwargs,
        )

    def run_backtest(
        self, prepared_dict, candidates_by_date, capital, on_progress=None, on_log=None
    ) -> pd.DataFrame:
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
        """翌日寄り付きで成行仕掛けし、ATR20×5 を損切りに設定"""
        try:
            entry_idx = df.index.get_loc(candidate["entry_date"])
        except Exception:
            return None
        if entry_idx <= 0 or entry_idx >= len(df):
            return None
        entry_price = float(df.iloc[entry_idx]["Open"])
        try:
            atr = float(df.iloc[entry_idx - 1]["ATR20"])
        except Exception:
            return None
        stop_mult = float(self.config.get("stop_atr_multiple", STOP_ATR_MULTIPLE_SYSTEM1))
        stop_price = entry_price - stop_mult * atr
        if entry_price - stop_price <= 0:
            return None
        return entry_price, stop_price

    def get_total_days(self, data_dict: dict) -> int:
        return get_total_days_system1(data_dict)

    def compute_exit(self, df: pd.DataFrame, entry_idx: int, entry_price: float, stop_price: float):
        """Day-based exit for System1 (long):
        - Stop hit: if Low <= stop -> exit same day at stop_price
        - Otherwise, max-hold days then exit on close
        """
        try:
            from .constants import MAX_HOLD_DAYS_DEFAULT
        except Exception:
            MAX_HOLD_DAYS_DEFAULT = 3
        max_hold_days = int(self.config.get("max_hold_days", MAX_HOLD_DAYS_DEFAULT))
        n = len(df)
        for offset in range(max_hold_days):
            idx = entry_idx + offset
            if idx >= n:
                break
            row = df.iloc[idx]
            try:
                if float(row["Low"]) <= float(stop_price):
                    return float(stop_price), df.index[idx]
            except Exception:
                pass
        exit_idx = min(entry_idx + max_hold_days, n - 1)
        return float(df.iloc[exit_idx]["Close"]), df.index[exit_idx]
