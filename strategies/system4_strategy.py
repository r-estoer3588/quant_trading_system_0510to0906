# strategies/system4_strategy.py
from __future__ import annotations

import pandas as pd

from .base_strategy import StrategyBase
from common.alpaca_order import AlpacaOrderMixin
from common.backtest_utils import simulate_trades_with_risk
from core.system4 import (
    prepare_data_vectorized_system4,
    generate_candidates_system4,
    get_total_days_system4,
)


class System4Strategy(AlpacaOrderMixin, StrategyBase):
    SYSTEM_NAME = "system4"

    def __init__(self):
        super().__init__()

    # インジケータ計算（コア委譲）
    def prepare_data(
        self,
        raw_data_dict,
        progress_callback=None,
        log_callback=None,
        skip_callback=None,
        batch_size=50,
    ):
        return prepare_data_vectorized_system4(
            raw_data_dict,
            progress_callback=progress_callback,
            log_callback=log_callback,
            batch_size=batch_size,
        )

    # 候補抽出（SPYフィルタ適用。market_df 後方互換あり）
    def generate_candidates(
        self,
        prepared_dict,
        market_df=None,
        progress_callback=None,
        log_callback=None,
        batch_size=50,
    ):
        # market_df 未指定時は prepared_dict から SPY を使用（後方互換）
        if market_df is None:
            market_df = prepared_dict.get("SPY")
        if market_df is None or getattr(market_df, "empty", False):
            raise ValueError("System4 には SPYデータ (market_df) が必要です")
        try:
            from config.settings import get_settings

            top_n = int(get_settings(create_dirs=False).backtest.top_n_rank)
        except Exception:
            top_n = 10
        return generate_candidates_system4(
            prepared_dict,
            market_df,
            top_n=top_n,
            progress_callback=progress_callback,
            log_callback=log_callback,
            batch_size=batch_size,
        )

    # バックテスト実行（コアシミュレーター）
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

    # システムフック群
    def compute_entry(self, df: pd.DataFrame, candidate: dict, current_capital: float):
        try:
            entry_idx = df.index.get_loc(candidate["entry_date"])
        except Exception:
            return None
        if entry_idx <= 0 or entry_idx >= len(df):
            return None
        entry_price = float(df.iloc[entry_idx]["Open"])
        try:
            atr40 = float(df.iloc[entry_idx - 1]["ATR40"])
        except Exception:
            return None
        stop_mult = float(getattr(self, "config", {}).get("stop_atr_multiple", 1.5))
        stop_price = entry_price - stop_mult * atr40
        if entry_price - stop_price <= 0:
            return None
        return entry_price, stop_price

    def compute_exit(
        self, df: pd.DataFrame, entry_idx: int, entry_price: float, stop_price: float
    ):
        trail_pct = float(getattr(self, "config", {}).get("trailing_pct", 0.20))
        highest = entry_price
        for idx2 in range(entry_idx + 1, len(df)):
            close = float(df.iloc[idx2]["Close"])
            if close > highest:
                highest = close
            if close <= highest * (1 - trail_pct):
                return close, df.index[idx2]
            if close <= stop_price:
                return close, df.index[idx2]
        last_close = float(df.iloc[-1]["Close"])
        return last_close, df.index[-1]

    def compute_pnl(self, entry_price: float, exit_price: float, shares: int) -> float:
        return (exit_price - entry_price) * shares

    def prepare_minimal_for_test(self, raw_data_dict: dict) -> dict:
        out = {}
        for sym, df in raw_data_dict.items():
            x = df.copy()
            x["SMA200"] = x["Close"].rolling(200).mean()
            out[sym] = x
        return out

    def get_total_days(self, data_dict: dict) -> int:
        return get_total_days_system4(data_dict)

