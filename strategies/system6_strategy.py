from __future__ import annotations

import pandas as pd

from .base_strategy import StrategyBase
from .constants import (
    PROFIT_TAKE_PCT_DEFAULT_5,
    MAX_HOLD_DAYS_DEFAULT,
    STOP_ATR_MULTIPLE_DEFAULT,
)
from common.alpaca_order import AlpacaOrderMixin
from common.backtest_utils import simulate_trades_with_risk
from common.utils import resolve_batch_size
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
        raw_data_or_symbols,
        progress_callback=None,
        log_callback=None,
        skip_callback=None,
        batch_size: int | None = None,
        use_process_pool: bool = False,
        **kwargs,
    ):
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
        prepared_dict,
        progress_callback=None,
        log_callback=None,
        skip_callback=None,
        batch_size: int | None = None,
    ):
        try:
            from config.settings import get_settings

            top_n = int(get_settings(create_dirs=False).backtest.top_n_rank)
        except Exception:
            top_n = 10
        if batch_size is None:
            try:
                from config.settings import get_settings

                batch_size = get_settings(create_dirs=False).data.batch_size
            except Exception:
                batch_size = 100
            batch_size = resolve_batch_size(len(prepared_dict), batch_size)
        
        # 基本候補を生成
        candidates_by_date, extra_df = generate_candidates_system6(
            prepared_dict,
            top_n=top_n,
            progress_callback=progress_callback,
            log_callback=log_callback,
            skip_callback=skip_callback,
            batch_size=batch_size,
        )
        
        # ショート可能チェックを実行（設定で有効な場合のみ）
        try:
            enable_shortable_check = getattr(
                get_settings(create_dirs=False).risk, 'enable_shortable_check', False
            )
            
            if enable_shortable_check:
                # 全候補銘柄を抽出
                all_symbols = set()
                for date_candidates in candidates_by_date.values():
                    for candidate in date_candidates:
                        all_symbols.add(candidate.get('symbol', ''))
                
                if all_symbols:
                    from common.broker_alpaca import check_shortable_stocks
                    
                    shortable_status = check_shortable_stocks(
                        list(all_symbols),
                        log_callback=log_callback
                    )
                    
                    # ショート不可の銘柄を除外
                    filtered_candidates = {}
                    for date, candidates in candidates_by_date.items():
                        filtered = []
                        for candidate in candidates:
                            symbol = candidate.get('symbol', '')
                            if shortable_status.get(symbol, False):
                                filtered.append(candidate)
                            elif log_callback:
                                log_callback(f"🚫 System6: {symbol} - ショート不可のため除外")
                        filtered_candidates[date] = filtered
                    
                    candidates_by_date = filtered_candidates
                    
        except Exception as e:
            if log_callback:
                log_callback(f"⚠️ System6: ショート可能チェックでエラー: {e}")
        
        return candidates_by_date, extra_df
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
        stop_mult = float(
            self.config.get("stop_atr_multiple", STOP_ATR_MULTIPLE_DEFAULT)
        )
        stop_price = entry_price + stop_mult * atr
        if stop_price - entry_price <= 0:
            return None
        return entry_price, stop_price

    def compute_exit(
        self, df: pd.DataFrame, entry_idx: int, entry_price: float, stop_price: float
    ):
        profit_take_pct = float(
            self.config.get("profit_take_pct", PROFIT_TAKE_PCT_DEFAULT_5)
        )
        max_days = int(
            self.config.get("profit_take_max_days", MAX_HOLD_DAYS_DEFAULT)
        )
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
                    ratio = float(
                        self.config.get(
                            "entry_price_ratio_vs_prev_close", 1.05
                        )
                    )
                    entry_price = round(prev_close2 * ratio, 2)
                    atr2 = float(df.iloc[entry_idx + offset]["ATR10"])
                    stop_mult = float(
                        self.config.get(
                            "stop_atr_multiple", STOP_ATR_MULTIPLE_DEFAULT
                        )
                    )
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
