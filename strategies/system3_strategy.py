# strategies/system3_strategy.py
from __future__ import annotations

import pandas as pd

from .base_strategy import StrategyBase
from common.alpaca_order import AlpacaOrderMixin
from common.backtest_utils import simulate_trades_with_risk
from core.system3 import (
    prepare_data_vectorized_system3,
    generate_candidates_system3,
    get_total_days_system3,
)


class System3Strategy(AlpacaOrderMixin, StrategyBase):
    SYSTEM_NAME = "system3"

    def __init__(self):
        super().__init__()

    # データ準備（共通コアへ委譲）
    def prepare_data(
        self,
        raw_data_dict,
        progress_callback=None,
        log_callback=None,
        skip_callback=None,
        batch_size=50,
    ):
        return prepare_data_vectorized_system3(
            raw_data_dict,
            progress_callback=progress_callback,
            log_callback=log_callback,
            batch_size=batch_size,
        )

    # 候補生成（共通コアへ委譲）
    def generate_candidates(
        self,
        prepared_dict,
        progress_callback=None,
        log_callback=None,
        batch_size=50,
        **kwargs,
    ):
        try:
            from config.settings import get_settings

            top_n = int(get_settings(create_dirs=False).backtest.top_n_rank)
        except Exception:
            top_n = 10
        return generate_candidates_system3(
            prepared_dict,
            top_n=top_n,
            progress_callback=progress_callback,
            log_callback=log_callback,
            batch_size=batch_size,
        )

    # バックテスト実行（共通シミュレーター）
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

    # 共通シミュレーター用フック（System3）
    def compute_entry(self, df: pd.DataFrame, candidate: dict, current_capital: float):
        try:
            entry_idx = df.index.get_loc(candidate["entry_date"])
        except Exception:
            return None
        if entry_idx <= 0 or entry_idx >= len(df):
            return None
        prev_close = float(df.iloc[entry_idx - 1]["Close"])
        ratio = float(self.config.get("entry_price_ratio_vs_prev_close", 0.93))
        entry_price = round(prev_close * ratio, 2)
        try:
            atr = float(df.iloc[entry_idx - 1]["ATR10"])
        except Exception:
            return None
        stop_mult = float(self.config.get("stop_atr_multiple", 2.5))
        stop_price = entry_price - stop_mult * atr
        if entry_price - stop_price <= 0:
            return None
        return entry_price, stop_price

    def compute_exit(
        self, df: pd.DataFrame, entry_idx: int, entry_price: float, stop_price: float
    ):
        profit_take_pct = float(self.config.get("profit_take_pct", 0.04))
        max_days = int(self.config.get("profit_take_max_days", 3))
        for offset in range(1, max_days + 1):
            idx2 = entry_idx + offset
            if idx2 >= len(df):
                break
            future_close = float(df.iloc[idx2]["Close"])
            gain = (future_close - entry_price) / entry_price
            if gain >= profit_take_pct:
                exit_idx = min(idx2 + 1, len(df) - 1)
                exit_date = df.index[exit_idx]
                exit_price = float(df.iloc[exit_idx]["Close"])
                return exit_price, exit_date
        idx2 = min(entry_idx + max_days, len(df) - 1)
        exit_date = df.index[idx2]
        exit_price = float(df.iloc[idx2]["Close"])
        return exit_price, exit_date

    def compute_pnl(self, entry_price: float, exit_price: float, shares: int) -> float:
        return (exit_price - entry_price) * shares

    def prepare_minimal_for_test(self, raw_data_dict: dict) -> dict:
        out = {}
        for sym, df in raw_data_dict.items():
            x = df.copy()
            x["SMA150"] = x["Close"].rolling(150).mean()
            out[sym] = x
        return out

    def get_total_days(self, data_dict: dict) -> int:
        return get_total_days_system3(data_dict)
