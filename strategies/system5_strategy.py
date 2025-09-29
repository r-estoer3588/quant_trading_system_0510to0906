from __future__ import annotations

import numpy as np
import pandas as pd

from common.alpaca_order import AlpacaOrderMixin
from common.utils import resolve_batch_size
from core.system5 import (
    generate_candidates_system5,
    get_total_days_system5,
    prepare_data_vectorized_system5,
)

from .base_strategy import StrategyBase
from .constants import FALLBACK_EXIT_DAYS_DEFAULT, STOP_ATR_MULTIPLE_DEFAULT


class System5Strategy(AlpacaOrderMixin, StrategyBase):
    SYSTEM_NAME = "system5"

    def __init__(self):
        """System5初期化、_last_entry_atr属性を追加。"""
        super().__init__()
        self._last_entry_atr: float = 0.0

    def prepare_data(
        self,
        raw_data_or_symbols,
        reuse_indicators: bool | None = None,
        **kwargs,
    ):
        """System5のデータ準備（共通テンプレート使用）"""
        return self._prepare_data_template(
            raw_data_or_symbols,
            prepare_data_vectorized_system5,
            reuse_indicators=reuse_indicators,
            **kwargs,
        )

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
        return generate_candidates_system5(
            prepared_dict,
            top_n=top_n,
            progress_callback=progress_callback,
            log_callback=log_callback,
            batch_size=batch_size,
        )

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
        ratio = float(getattr(self, "config", {}).get("entry_price_ratio_vs_prev_close", 0.97))
        entry_price = round(prev_close * ratio, 2)
        try:
            atr = float(df.iloc[entry_idx - 1]["ATR10"])
        except Exception:
            return None
        stop_mult = float(
            getattr(self, "config", {}).get("stop_atr_multiple", STOP_ATR_MULTIPLE_DEFAULT)
        )
        stop_price = entry_price - stop_mult * atr
        if entry_price - stop_price <= 0:
            return None
        self._last_entry_atr = atr
        return entry_price, stop_price

    def compute_exit(self, df: pd.DataFrame, entry_idx: int, entry_price: float, stop_price: float):
        """System5 の利確・損切り・時間退出ロジック。

        - 利益目標: 過去10日ATR×設定倍率を上回ったら翌営業日の寄り付きで決済
        - 損切り: 当日の安値がストップ以下になった時点でストップ価格で決済
        - 時間退出: 6営業日経過後も未決済なら7日目の寄り付きで決済
        """

        atr = getattr(self, "_last_entry_atr", None)
        if atr is None:
            try:
                atr = float(df.iloc[entry_idx - 1]["ATR10"])
            except Exception:
                atr = 0.0
        target_mult = float(getattr(self, "config", {}).get("target_atr_multiple", 1.0))
        target_price = entry_price + target_mult * atr
        fallback_days = int(
            getattr(self, "config", {}).get("fallback_exit_after_days", FALLBACK_EXIT_DAYS_DEFAULT)
        )

        last_idx = len(df) - 1

        for offset in range(1, fallback_days + 1):
            idx = entry_idx + offset
            if idx >= len(df):
                break
            row = df.iloc[idx]

            if float(row["Low"]) <= stop_price:
                return float(stop_price), df.index[idx]

            if float(row["High"]) >= target_price:
                exit_idx = idx + 1
                if exit_idx < len(df):
                    exit_price = float(df.iloc[exit_idx]["Open"])
                    exit_date = df.index[exit_idx]
                else:
                    exit_price = float(df.iloc[idx]["Close"])
                    exit_date = df.index[idx]
                return exit_price, exit_date

        fallback_exit_idx = entry_idx + fallback_days + 1
        if fallback_exit_idx < len(df):
            exit_price = float(df.iloc[fallback_exit_idx]["Open"])
            exit_date = df.index[fallback_exit_idx]
        else:
            fallback_idx = min(entry_idx + fallback_days, last_idx)
            exit_price = float(df.iloc[fallback_idx]["Close"])
            exit_date = df.index[fallback_idx]

        return exit_price, exit_date

    def compute_pnl(self, entry_price: float, exit_price: float, shares: int) -> float:
        """ロングのPnL - 基底クラスのメソッドを使用。"""
        return self.compute_pnl_long(entry_price, exit_price, shares)

    def prepare_minimal_for_test(self, raw_data_dict: dict) -> dict:
        out = {}
        for sym, df in raw_data_dict.items():
            x = df.copy()
            x["SMA100"] = x["Close"].rolling(100).mean()
            out[sym] = x
        return out

    def get_total_days(self, data_dict: dict) -> int:
        return get_total_days_system5(data_dict)
