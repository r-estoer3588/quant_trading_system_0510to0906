# strategies/system2_strategy.py
from __future__ import annotations

import numpy as np
import pandas as pd

from common.alpaca_order import AlpacaOrderMixin
from core.system2 import (
    generate_candidates_system2,
    get_total_days_system2,
    prepare_data_vectorized_system2,
)

from .base_strategy import StrategyBase
from .constants import (
    ENTRY_MIN_GAP_PCT_DEFAULT,
    MAX_HOLD_DAYS_DEFAULT,
    PROFIT_TAKE_PCT_DEFAULT_4,
    STOP_ATR_MULTIPLE_DEFAULT,
)


class System2Strategy(AlpacaOrderMixin, StrategyBase):
    SYSTEM_NAME = "system2"

    def get_trading_side(self) -> str:
        """System2 はショート戦略"""
        return "short"

    # -------------------------------
    # データ準備（共通コアへ委譲）
    # -------------------------------
    def prepare_data(
        self,
        raw_data_or_symbols,
        reuse_indicators: bool | None = None,
        **kwargs,
    ):
        """System2のデータ準備（共通テンプレート使用）"""
        return self._prepare_data_template(
            raw_data_or_symbols,
            prepare_data_vectorized_system2,
            reuse_indicators=reuse_indicators,
            **kwargs,
        )

    # -------------------------------
    # 候補生成（共通コアへ委譲）
    # -------------------------------
    def generate_candidates(self, data_dict, market_df=None, **kwargs):
        """候補生成（共通メソッド使用）"""
        top_n = self._get_top_n_setting(kwargs.get("top_n"))
        latest_only = bool(kwargs.get("latest_only", False))
        return generate_candidates_system2(
            data_dict,
            top_n=top_n,
            latest_only=latest_only,
        )

    # -------------------------------
    # 共通シミュレーター用フック（System2ルール）
    # -------------------------------
    def compute_entry(self, df: pd.DataFrame, candidate: dict, _current_capital: float):
        """
        エントリー価格とストップを返す（ショート）。
        - candidate["entry_date"] の行をもとに、ギャップ条件とATRベースのストップを計算。

        Args:
            df: 価格データ
            candidate: エントリー候補情報
            _current_capital: 現在資本（未使用、インターフェース互換性のため）

        Returns:
            (entry_price, stop_price) または None
        """
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
        prior_close = float(df.iloc[entry_idx - 1]["Close"])
        entry_price = float(df.iloc[entry_idx]["Open"])
        min_gap = float(self.config.get("entry_min_gap_pct", ENTRY_MIN_GAP_PCT_DEFAULT))
        # 上窓（前日終値比+4%）未満なら見送り（ショート前提）
        if entry_price < prior_close * (1 + min_gap):
            return None
        atr = None
        for col in ("atr10", "ATR10"):
            try:
                atr = float(df.iloc[entry_idx - 1][col])
                break
            except Exception:
                continue
        if atr is None:
            return None
        stop_mult = float(self.config.get("stop_atr_multiple", STOP_ATR_MULTIPLE_DEFAULT))
        stop_price = entry_price + stop_mult * atr
        return entry_price, stop_price

    def compute_exit(self, df: pd.DataFrame, entry_idx: int, entry_price: float, stop_price: float):
        """利確/損切りロジック。
        - ストップ到達: その日の高値>=stop で当日決済
        - 利確到達: 前日終値で判定し、翌日大引けで決済
        - 未達: 2営業日待っても利確に届かない場合は3日目の大引けで決済
        返り値: (exit_price, exit_date)
        """
        profit_take_pct = float(self.config.get("profit_take_pct", PROFIT_TAKE_PCT_DEFAULT_4))
        max_hold_days = int(self.config.get("max_hold_days", MAX_HOLD_DAYS_DEFAULT))

        for offset in range(max_hold_days):
            idx = entry_idx + offset
            if idx >= len(df):
                break
            row = df.iloc[idx]
            # ストップ到達（ショート）
            if float(row["High"]) >= stop_price:
                return stop_price, df.index[idx]
            # 利確判定（ショートの含み益）
            gain = (entry_price - float(row["Close"])) / entry_price
            if gain >= profit_take_pct:
                exit_idx = min(idx + 1, len(df) - 1)
                return float(df.iloc[exit_idx]["Close"]), df.index[exit_idx]

        exit_idx = min(entry_idx + max_hold_days, len(df) - 1)
        return float(df.iloc[exit_idx]["Close"]), df.index[exit_idx]

    def compute_pnl(self, entry_price: float, exit_price: float, shares: int) -> float:
        """ショートのPnL - 基底クラスのメソッドを使用。"""
        return self.compute_pnl_short(entry_price, exit_price, shares)

    # --- テスト用の最小RSI3計算 ---
    def prepare_minimal_for_test(self, raw_data_dict: dict) -> dict:
        out = {}
        for sym, df in raw_data_dict.items():
            x = df.copy()
            close = x["Close"].astype(float)
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(3).mean()
            loss = -delta.clip(upper=0).rolling(3).mean()
            rs = gain / loss.replace(0, pd.NA)
            x["rsi3"] = 100 - (100 / (1 + rs))
            out[sym] = x
        return out

    def get_total_days(self, data_dict: dict) -> int:
        return get_total_days_system2(data_dict)
