# strategies/system2_strategy.py
from __future__ import annotations

import pandas as pd

from .base_strategy import StrategyBase
from .constants import (
    PROFIT_TAKE_PCT_DEFAULT_4,
    MAX_HOLD_DAYS_DEFAULT,
    STOP_ATR_MULTIPLE_DEFAULT,
    ENTRY_MIN_GAP_PCT_DEFAULT,
)
from common.alpaca_order import AlpacaOrderMixin
from common.backtest_utils import simulate_trades_with_risk
from common.utils import resolve_batch_size
from core.system2 import (
    prepare_data_vectorized_system2,
    generate_candidates_system2,
    get_total_days_system2,
)


class System2Strategy(AlpacaOrderMixin, StrategyBase):
    SYSTEM_NAME = "system2"

    def __init__(self):
        super().__init__()

    # -------------------------------
    # データ準備（共通コアへ委譲）
    # -------------------------------
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
        """インジケーター計算をコア関数へ委譲。"""
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
        return prepare_data_vectorized_system2(
            raw_dict,
            progress_callback=progress_callback,
            log_callback=log_callback,
            batch_size=batch_size,
            symbols=symbols,
            use_process_pool=use_process_pool,
            skip_callback=skip_callback,
        )

    # -------------------------------
    # 候補生成（共通コアへ委譲）
    # -------------------------------
    def generate_candidates(self, prepared_dict, **kwargs):
        try:
            from config.settings import get_settings

            top_n = int(get_settings(create_dirs=False).backtest.top_n_rank)
        except Exception:
            top_n = 10
        return generate_candidates_system2(prepared_dict, top_n=top_n)

    # -------------------------------
    # バックテスト実行（共通シミュレーター）
    # -------------------------------
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

    # -------------------------------
    # 共通シミュレーター用フック（System2ルール）
    # -------------------------------
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
        min_gap = float(
            self.config.get("entry_min_gap_pct", ENTRY_MIN_GAP_PCT_DEFAULT)
        )
        # 上窓（前日終値比+4%）未満なら見送り（ショート前提）
        if entry_price < prior_close * (1 + min_gap):
            return None
        try:
            atr = float(df.iloc[entry_idx - 1]["ATR10"])
        except Exception:
            return None
        stop_mult = float(
            self.config.get("stop_atr_multiple", STOP_ATR_MULTIPLE_DEFAULT)
        )
        stop_price = entry_price + stop_mult * atr
        return entry_price, stop_price

    def compute_exit(
        self, df: pd.DataFrame, entry_idx: int, entry_price: float, stop_price: float
    ):
        """利確/損切りロジック。
        - ストップ到達: その日の高値>=stop で当日決済
        - 利確到達: 前日終値で判定し、翌日大引けで決済
        - 未達: 2営業日待っても利確に届かない場合は3日目の大引けで決済
        返り値: (exit_price, exit_date)
        """
        profit_take_pct = float(
            self.config.get("profit_take_pct", PROFIT_TAKE_PCT_DEFAULT_4)
        )
        max_hold_days = int(
            self.config.get("max_hold_days", MAX_HOLD_DAYS_DEFAULT)
        )

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
