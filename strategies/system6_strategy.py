from __future__ import annotations

import numpy as np
import pandas as pd

from common.alpaca_order import AlpacaOrderMixin
from common.backtest_utils import simulate_trades_with_risk
from common.utils import resolve_batch_size
from core.system6 import (
    generate_candidates_system6,
    get_total_days_system6,
    prepare_data_vectorized_system6,
)
from core.system6_optimized import prepare_data_optimized_system6
from core.system6_ultra_optimized import (
    prepare_data_ultra_optimized_system6,
    generate_candidates_ultra_fast_system6,
)
from core.system6_fixed import (
    prepare_data_fixed_system6,
    generate_candidates_system6_fixed,
)

from .base_strategy import StrategyBase
from .constants import MAX_HOLD_DAYS_DEFAULT, PROFIT_TAKE_PCT_DEFAULT_5, STOP_ATR_MULTIPLE_DEFAULT


class System6Strategy(AlpacaOrderMixin, StrategyBase):
    SYSTEM_NAME = "system6"

    def __init__(self):
        super().__init__()

    def prepare_data(
        self,
        raw_data_or_symbols,
        reuse_indicators: bool | None = None,
        progress_callback=None,
        log_callback=None,
        skip_callback=None,
        batch_size: int | None = None,
        use_process_pool: bool = False,
        enable_optimization: bool = True,  # 最適化版の有効化フラグ
        ultra_mode: bool = False,  # 超最適化モード（デフォルト無効）
        fixed_mode: bool = True,  # 固定版モード（デフォルト有効）
        **kwargs,
    ):
        if isinstance(raw_data_or_symbols, dict):
            symbols = list(raw_data_or_symbols.keys())
            raw_dict = None if use_process_pool else raw_data_or_symbols
        else:
            symbols = list(raw_data_or_symbols)
            raw_dict = None

        # 固定版の使用判定（既存インジケーター活用）
        if fixed_mode and not use_process_pool and raw_dict is not None:
            # 固定版を使用（既存インジケーター活用、再計算なし）
            return prepare_data_fixed_system6(
                raw_dict,
                progress_callback=progress_callback,
                log_callback=log_callback,
                skip_callback=skip_callback,
                reuse_indicators=reuse_indicators if reuse_indicators is not None else True,
                **kwargs,
            )

        # 超最適化版の使用判定（30分達成のため）
        if ultra_mode and enable_optimization and not use_process_pool and raw_dict is not None:
            # 超最適化版を使用
            return prepare_data_ultra_optimized_system6(
                raw_dict,
                progress_callback=progress_callback,
                log_callback=log_callback,
                skip_callback=skip_callback,
                reuse_indicators=reuse_indicators if reuse_indicators is not None else True,
                **kwargs,
            )

        # 最適化版の使用判定（当日実行時に有効化）
        if enable_optimization and not use_process_pool and raw_dict is not None:
            # 30分達成のための最適化版を使用
            return prepare_data_optimized_system6(
                raw_dict,
                progress_callback=progress_callback,
                log_callback=log_callback,
                skip_callback=skip_callback,
                reuse_indicators=reuse_indicators if reuse_indicators is not None else True,
                **kwargs,
            )

        # 従来版（並列処理や無効化時）
        if batch_size is None and not use_process_pool and raw_dict is not None:
            try:
                from config.settings import get_settings

                batch_size = get_settings(create_dirs=False).data.batch_size
            except Exception:
                batch_size = 100
            batch_size = resolve_batch_size(len(raw_dict), batch_size)
        return prepare_data_vectorized_system6(
            raw_dict,
            progress_callback=progress_callback,
            log_callback=log_callback,
            skip_callback=skip_callback,
            batch_size=batch_size,
            symbols=symbols,
            use_process_pool=use_process_pool,
        )

    def generate_candidates(
        self,
        data_dict,
        market_df=None,
        progress_callback=None,
        log_callback=None,
        skip_callback=None,
        batch_size: int | None = None,
        fixed_mode: bool = True,  # 固定版モード（デフォルト有効）
        ultra_mode: bool = False,  # 超最適化モード
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
            
        # 固定版を優先使用
        if fixed_mode:
            return generate_candidates_system6_fixed(
                prepared_dict,
                top_n=top_n,
                progress_callback=progress_callback,
                log_callback=log_callback,
                skip_callback=skip_callback,
                batch_size=batch_size,
            )
        
        # 超最適化版
        if ultra_mode:
            return generate_candidates_ultra_fast_system6(
                prepared_dict,
                top_n=top_n,
                progress_callback=progress_callback,
                log_callback=log_callback,
                skip_callback=skip_callback,
                batch_size=batch_size,
            )
            
        # 従来版
        return generate_candidates_system6(
            prepared_dict,
            top_n=top_n,
            progress_callback=progress_callback,
            log_callback=log_callback,
            skip_callback=skip_callback,
            batch_size=batch_size,
        )

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
            side="short",
        )
        return trades_df

    # シミュレーター用フック（System6: Short）
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
        prev_close = float(df.iloc[entry_idx - 1]["Close"])
        ratio = float(self.config.get("entry_price_ratio_vs_prev_close", 1.05))
        entry_price = round(prev_close * ratio, 2)
        try:
            atr = float(df.iloc[entry_idx - 1]["ATR10"])
        except Exception:
            return None
        stop_mult = float(self.config.get("stop_atr_multiple", STOP_ATR_MULTIPLE_DEFAULT))
        stop_price = entry_price + stop_mult * atr
        if stop_price - entry_price <= 0:
            return None
        return entry_price, stop_price

    def compute_exit(self, df: pd.DataFrame, entry_idx: int, entry_price: float, stop_price: float):
        """System6 の利確・損切り・時間退出ルールを実装。"""

        profit_take_pct = float(self.config.get("profit_take_pct", PROFIT_TAKE_PCT_DEFAULT_5))
        max_days = int(self.config.get("profit_take_max_days", MAX_HOLD_DAYS_DEFAULT))
        last_idx = len(df) - 1

        for offset in range(1, max_days + 1):
            idx = entry_idx + offset
            if idx >= len(df):
                break
            row = df.iloc[idx]

            if float(row["High"]) >= stop_price:
                return float(stop_price), df.index[idx]

            gain = (entry_price - float(row["Close"])) / entry_price
            if gain >= profit_take_pct:
                exit_idx = idx + 1
                if exit_idx < len(df):
                    exit_price = float(df.iloc[exit_idx]["Close"])
                    exit_date = df.index[exit_idx]
                else:
                    exit_price = float(row["Close"])
                    exit_date = df.index[idx]
                return exit_price, exit_date

        fallback_idx = entry_idx + max_days
        if fallback_idx < len(df):
            exit_price = float(df.iloc[fallback_idx]["Close"])
            exit_date = df.index[fallback_idx]
        else:
            exit_price = float(df.iloc[last_idx]["Close"])
            exit_date = df.index[last_idx]

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
