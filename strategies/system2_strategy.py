# strategies/system2_strategy.py
from __future__ import annotations

import pandas as pd

from .base_strategy import StrategyBase
from common.alpaca_order import AlpacaOrderMixin
from common.backtest_utils import simulate_trades_with_risk
from core.system2 import (
    prepare_data_vectorized_system2,
    generate_candidates_system2,
    get_total_days_system2,
)


class System2Strategy(AlpacaOrderMixin, StrategyBase):
    SYSTEM_NAME = "system2"

    def __init__(self):
        super().__init__()

    # ===============================
    # データ準備（共通コアへ委譲）
    # ===============================
    def prepare_data(
        self,
        raw_data_dict,
        progress_callback=None,
        log_callback=None,
        batch_size=50,
        **kwargs,
    ):
        """インジケーター計算をコア関数へ委譲。

        UI 側から渡される追加キーワード（例: `limit_symbols`, `skip_callback` など）が
        混在してもここで受け止めて無視することで、予期しない TypeError を回避する。
        """
        return prepare_data_vectorized_system2(
            raw_data_dict,
            progress_callback=progress_callback,
            log_callback=log_callback,
            batch_size=batch_size,
        )

    # ===============================
    # 候補生成（共通コアへ委譲）
    # ===============================
    def generate_candidates(self, prepared_dict, **kwargs):
        try:
            from config.settings import get_settings

            top_n = int(get_settings(create_dirs=False).backtest.top_n_rank)
        except Exception:
            top_n = 10
        return generate_candidates_system2(prepared_dict, top_n=top_n)

    # ===============================
    # バックテスト実行（共通シミュレーター）
    # ===============================
    def run_backtest(
        self, data_dict, candidates_by_date, capital, on_progress=None, on_log=None
    ):
        trades_df, _ = simulate_trades_with_risk(
            candidates_by_date,
            data_dict,
            capital,
            self,
            on_progress=on_progress,
            on_log=on_log,
            side="short",
        )
        return trades_df

    # ===============================
    # 共通シミュレーター用フック（System2ルール）
    # ===============================
    def compute_entry(self, df: pd.DataFrame, candidate: dict, current_capital: float):
        """エントリー価格とストップを返す（ショート）。
        - candidate["entry_date"] の行をもとに、ギャップ条件とATRベースのストップを計算。
        """
        try:
            entry_idx = df.index.get_loc(candidate["entry_date"])
        except Exception:
            return None
        if entry_idx <= 0 or entry_idx >= len(df):
            return None
        prior_close = float(df.iloc[entry_idx - 1]["Close"])
        entry_price = float(df.iloc[entry_idx]["Open"])
        min_gap = float(self.config.get("entry_min_gap_pct", 0.04))
        # 上窓（前日終値比+4%）未満なら見送り（ショート前提）
        if entry_price < prior_close * (1 + min_gap):
            return None
        try:
            atr = float(df.iloc[entry_idx - 1]["ATR10"])
        except Exception:
            return None
        stop_mult = float(self.config.get("stop_atr_multiple", 3.0))
        stop_price = entry_price + stop_mult * atr
        return entry_price, stop_price

    def compute_exit(
        self, df: pd.DataFrame, entry_idx: int, entry_price: float, stop_price: float
    ):
        """利確/損切りロジック。
        - ストップ到達: その日の高値>=stop で当日決済
        - 利確到達: 翌日寄りで決済（前日終値で利確条件判定）
        - 未達: 指定日数後の翌日寄りで撤退
        返り値: (exit_price, exit_date)
        """
        exit_date, exit_price = None, None
        profit_take_pct = float(self.config.get("profit_take_pct", 0.04))
        max_days = int(self.config.get("profit_take_max_days", 3))

        for offset in range(1, max_days + 1):
            idx2 = entry_idx + offset
            if idx2 >= len(df):
                break
            row = df.iloc[idx2]
            # ストップ到達（ショート）
            if float(row["High"]) >= stop_price:
                exit_date = df.index[idx2]
                exit_price = stop_price
                break
            # 利確判定（ショートの含み益）
            gain = (entry_price - float(row["Close"])) / entry_price
            if gain >= profit_take_pct:
                next_idx = min(idx2 + 1, len(df) - 1)
                exit_date = df.index[next_idx]
                exit_price = float(df.iloc[next_idx]["Open"])
                break

        if exit_price is None:
            fallback_days = int(self.config.get("fallback_exit_after_days", 2))
            idx2 = min(entry_idx + fallback_days, len(df) - 1)
            next_idx = min(idx2 + 1, len(df) - 1)
            exit_date = df.index[next_idx]
            exit_price = float(df.iloc[next_idx]["Open"])
        return exit_price, exit_date

    def compute_pnl(self, entry_price: float, exit_price: float, shares: int) -> float:
        """ショートのPnL。"""
        return (entry_price - exit_price) * shares

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
            x["RSI3"] = 100 - (100 / (1 + rs))
            out[sym] = x
        return out

    def get_total_days(self, data_dict: dict) -> int:
        return get_total_days_system2(data_dict)

