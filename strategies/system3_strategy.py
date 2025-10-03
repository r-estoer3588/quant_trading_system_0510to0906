# strategies/system3_strategy.py
from __future__ import annotations

import numpy as np
import pandas as pd

from common.alpaca_order import AlpacaOrderMixin
from core.system3 import (
    generate_candidates_system3,
    get_total_days_system3,
    prepare_data_vectorized_system3,
)

from .base_strategy import StrategyBase
from .constants import MAX_HOLD_DAYS_DEFAULT, PROFIT_TAKE_PCT_DEFAULT_4, STOP_ATR_MULTIPLE_SYSTEM3


class System3Strategy(AlpacaOrderMixin, StrategyBase):
    SYSTEM_NAME = "system3"

    # データ準備（共通コアへ委譲）
    def prepare_data(
        self,
        raw_data_or_symbols,
        reuse_indicators: bool | None = None,
        **kwargs,
    ):
        """System3のデータ準備（共通テンプレート使用）"""
        return self._prepare_data_template(
            raw_data_or_symbols,
            prepare_data_vectorized_system3,
            reuse_indicators=reuse_indicators,
            **kwargs,
        )

    # 候補生成（共通コアへ委譲）
    def generate_candidates(
        self,
        data_dict,
        market_df=None,
        **kwargs,
    ):
        """候補生成（共通メソッド使用）"""
        top_n = self._get_top_n_setting(kwargs.get("top_n"))
        batch_size = self._get_batch_size_setting(len(data_dict))
        # 重複渡し防止: kwargs に残っている latest_only を取り除いてから明示引数で渡す
        latest_only = bool(kwargs.pop("latest_only", False))
        return generate_candidates_system3(
            data_dict,
            top_n=top_n,
            batch_size=batch_size,
            latest_only=latest_only,
            **kwargs,
        )

    # 共通シミュレーター用フック（System3）
    def compute_entry(self, df: pd.DataFrame, candidate: dict, _current_capital: float):
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
        prev_close = float(df.iloc[entry_idx - 1]["Close"])
        ratio = float(self.config.get("entry_price_ratio_vs_prev_close", 0.93))
        entry_price = round(prev_close * ratio, 2)
        atr = None
        for col in ("atr10", "ATR10"):
            try:
                atr = float(df.iloc[entry_idx - 1][col])
                break
            except Exception:
                continue
        if atr is None:
            return None
        stop_mult = float(self.config.get("stop_atr_multiple", STOP_ATR_MULTIPLE_SYSTEM3))
        stop_price = entry_price - stop_mult * atr
        if entry_price - stop_price <= 0:
            return None
        return entry_price, stop_price

    def compute_exit(self, df: pd.DataFrame, entry_idx: int, entry_price: float, stop_price: float):
        """利確/損切りロジック。
        - 終値ベースで4%以上の利益なら翌日大引けで決済
        - 損切り価格到達時は当日決済
        - 3日経過しても未達なら4日目の大引けで決済
        """
        profit_take_pct = float(self.config.get("profit_take_pct", PROFIT_TAKE_PCT_DEFAULT_4))
        max_hold_days = int(self.config.get("max_hold_days", MAX_HOLD_DAYS_DEFAULT))

        for offset in range(max_hold_days + 1):
            idx = entry_idx + offset
            if idx >= len(df):
                break
            row = df.iloc[idx]
            if float(row["Low"]) <= stop_price:
                return stop_price, df.index[idx]
            gain = (float(row["Close"]) - entry_price) / entry_price
            if gain >= profit_take_pct:
                exit_idx = min(idx + 1, len(df) - 1)
                return float(df.iloc[exit_idx]["Close"]), df.index[exit_idx]

        exit_idx = min(entry_idx + max_hold_days + 1, len(df) - 1)
        return float(df.iloc[exit_idx]["Close"]), df.index[exit_idx]

    def compute_pnl(self, entry_price: float, exit_price: float, shares: int) -> float:
        """ロングのPnL - 基底クラスのメソッドを使用。"""
        return self.compute_pnl_long(entry_price, exit_price, shares)

    def prepare_minimal_for_test(self, raw_data_dict: dict) -> dict:
        out = {}
        for sym, df in raw_data_dict.items():
            x = df.copy()
            x["sma150"] = x["Close"].rolling(150).mean()
            out[sym] = x
        return out

    def get_total_days(self, data_dict: dict) -> int:
        return get_total_days_system3(data_dict)
