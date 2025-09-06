from __future__ import annotations

import pandas as pd

from .base_strategy import StrategyBase
from common.alpaca_order import AlpacaOrderMixin
from common.backtest_utils import simulate_trades_with_risk
from core.system6 import (
    prepare_data_vectorized_system6,
    generate_candidates_system6,
    get_total_days_system6,
)


class System6Strategy(AlpacaOrderMixin, StrategyBase):
    SYSTEM_NAME = "system6"

    def __init__(self):
        super().__init__()

    def prepare_data(
        self,
        raw_data_dict,
        progress_callback=None,
        log_callback=None,
        skip_callback=None,
        batch_size=50,
    ):
        return prepare_data_vectorized_system6(
            raw_data_dict,
            progress_callback=progress_callback,
            log_callback=log_callback,
            skip_callback=skip_callback,
            batch_size=batch_size,
        )

    def generate_candidates(
        self,
        prepared_dict,
        progress_callback=None,
        log_callback=None,
        skip_callback=None,
        batch_size=50,
    ):
        try:
            from config.settings import get_settings

            top_n = int(get_settings(create_dirs=False).backtest.top_n_rank)
        except Exception:
            top_n = 10
        return generate_candidates_system6(
            prepared_dict,
            top_n=top_n,
            progress_callback=progress_callback,
            log_callback=log_callback,
            skip_callback=skip_callback,
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
            side="short",
        )
        return trades_df

    # シミュレーター用フック（System6: Short）
    def compute_entry(self, df: pd.DataFrame, candidate: dict, current_capital: float):
        try:
            entry_idx = df.index.get_loc(candidate["entry_date"])
        except Exception:
            return None
        if entry_idx <= 0 or entry_idx >= len(df):
            return None
        prev_close = float(df.iloc[entry_idx - 1]["Close"])
        ratio = float(self.config.get("entry_price_ratio_vs_prev_close", 1.05))
        entry_price = round(prev_close * ratio, 2)
        try:
            atr = float(df.iloc[entry_idx - 1]["ATR10"])
        except Exception:
            return None
        stop_mult = float(self.config.get("stop_atr_multiple", 3.0))
        stop_price = entry_price + stop_mult * atr
        if stop_price - entry_price <= 0:
            return None
        return entry_price, stop_price

    def compute_exit(
        self, df: pd.DataFrame, entry_idx: int, entry_price: float, stop_price: float
    ):
        profit_take_pct = float(self.config.get("profit_take_pct", 0.05))
        max_days = int(self.config.get("profit_take_max_days", 3))
        offset = 1
        while offset <= max_days and entry_idx + offset < len(df):
            row = df.iloc[entry_idx + offset]
            gain = (entry_price - float(row["Close"])) / entry_price
            if gain >= profit_take_pct:
                exit_idx = min(entry_idx + offset + 1, len(df) - 1)
                exit_date = df.index[exit_idx]
                exit_price = float(df.iloc[exit_idx]["Close"])
                return exit_price, exit_date
            if float(row["High"]) >= stop_price:
                if entry_idx + offset < len(df) - 1:
                    prev_close2 = float(df.iloc[entry_idx + offset]["Close"])
                    ratio = float(self.config.get("entry_price_ratio_vs_prev_close", 1.05))
                    entry_price = round(prev_close2 * ratio, 2)
                    atr2 = float(df.iloc[entry_idx + offset]["ATR10"])
                    stop_mult = float(self.config.get("stop_atr_multiple", 3.0))
                    stop_price = entry_price + stop_mult * atr2
                    entry_idx = entry_idx + offset + 1
                    offset = 0
                else:
                    exit_date = df.index[entry_idx + offset]
                    exit_price = float(stop_price)
                    return exit_price, exit_date
            offset += 1
        idx2 = min(entry_idx + max_days, len(df) - 1)
        exit_date = df.index[idx2]
        exit_price = float(df.iloc[idx2]["Close"])
        return exit_price, exit_date

    def compute_pnl(self, entry_price: float, exit_price: float, shares: int) -> float:
        return (entry_price - exit_price) * shares

    def prepare_minimal_for_test(self, raw_data_dict: dict) -> dict:
        out = {}
        for sym, df in raw_data_dict.items():
            x = df.copy()
            high, low, close = x["High"], x["Low"], x["Close"]
            tr = pd.concat(
                [
                    (high - low),
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs(),
                ],
                axis=1,
            ).max(axis=1)
            x["ATR10"] = tr.rolling(10).mean()
            out[sym] = x
        return out

    def get_total_days(self, data_dict: dict) -> int:
        return get_total_days_system6(data_dict)

