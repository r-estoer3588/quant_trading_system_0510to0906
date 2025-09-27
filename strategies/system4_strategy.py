# strategies/system4_strategy.py
from __future__ import annotations

import numpy as np
import pandas as pd

from common.alpaca_order import AlpacaOrderMixin
from common.backtest_utils import simulate_trades_with_risk
from common.utils import resolve_batch_size
from core.system4 import (
    generate_candidates_system4,
    get_total_days_system4,
    prepare_data_vectorized_system4,
)

from .base_strategy import StrategyBase
from .constants import STOP_ATR_MULTIPLE_SYSTEM4


class System4Strategy(AlpacaOrderMixin, StrategyBase):
    SYSTEM_NAME = "system4"

    def __init__(self):
        super().__init__()

    # インジケータ計算（コア委譲）
    def prepare_data(
        self,
        raw_data_or_symbols,
        reuse_indicators: bool | None = None,
        **kwargs,
    ):
        """System4のデータ準備（共通テンプレート使用）"""
        return self._prepare_data_template(
            raw_data_or_symbols,
            prepare_data_vectorized_system4,
            reuse_indicators=reuse_indicators,
            **kwargs,
        )

    # 候補抽出（SPYフィルタ適用。market_df 後方互換あり）
    def generate_candidates(
        self,
        data_dict,
        market_df=None,
        progress_callback=None,
        log_callback=None,
        batch_size: int | None = None,
        **kwargs,
    ):
        prepared_dict = data_dict
        top_n = kwargs.pop("top_n", None)
        # market_df 未指定時は prepared_dict から SPY を使用（後方互換）
        if market_df is None:
            market_df = prepared_dict.get("SPY")
        if market_df is None or getattr(market_df, "empty", False):
            raise ValueError("System4 には SPYデータ (market_df) が必要です")
        if top_n is None:
            try:
                from config.settings import get_settings

                top_n = int(get_settings(create_dirs=False).backtest.top_n_rank)
            except Exception:
                top_n = 10
        else:
            try:
                top_n = max(0, int(top_n))
            except Exception:
                top_n = 10
        if batch_size is None:
            try:
                from config.settings import get_settings

                batch_size = get_settings(create_dirs=False).data.batch_size
            except Exception:
                batch_size = 100
            batch_size = resolve_batch_size(len(prepared_dict), batch_size)
        return generate_candidates_system4(
            prepared_dict,
            top_n=top_n,
            progress_callback=progress_callback,
            log_callback=log_callback,
            batch_size=batch_size,
        )

    # バックテスト実行（コアシミュレーター）
    def run_backtest(self, data_dict, candidates_by_date, capital, **kwargs):
        on_progress = kwargs.get("on_progress", None)
        on_log = kwargs.get("on_log", None)
        trades_df, _ = simulate_trades_with_risk(
            candidates_by_date,
            data_dict,
            capital,
            self,
            on_progress=on_progress,
            on_log=on_log,
        )
        return trades_df

    # システムフック群
    def compute_entry(self, df: pd.DataFrame, candidate: dict, current_capital: float):
        try:
            entry_loc = df.index.get_loc(candidate["entry_date"])
        except Exception:
            return None
        if isinstance(entry_loc, slice) or isinstance(entry_loc, np.ndarray):
            return None
        if not isinstance(entry_loc, int | np.integer):
            return None
        entry_idx = int(entry_loc)
        if entry_idx <= 0 or entry_idx >= len(df):
            return None
        entry_price = float(df.iloc[entry_idx]["Open"])
        try:
            atr40 = float(df.iloc[entry_idx - 1]["ATR40"])
        except Exception:
            return None
        stop_mult = float(
            getattr(self, "config", {}).get("stop_atr_multiple", STOP_ATR_MULTIPLE_SYSTEM4)
        )
        stop_price = entry_price - stop_mult * atr40
        if entry_price - stop_price <= 0:
            return None
        return entry_price, stop_price

    def compute_exit(self, df: pd.DataFrame, entry_idx: int, entry_price: float, stop_price: float):
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
            # テスト用の軽量処理では浅いコピーで十分
            x = df.copy(deep=False)
            x["SMA200"] = x["Close"].rolling(200).mean()
            out[sym] = x
        return out

    def get_total_days(self, data_dict: dict) -> int:
        return get_total_days_system4(data_dict)
