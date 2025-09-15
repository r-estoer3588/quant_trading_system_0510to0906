"""System7 strategy with cleaned Japanese comments and safe typing."""

from __future__ import annotations

import time

import pandas as pd

from common.alpaca_order import AlpacaOrderMixin
from common.utils import get_cached_data
from core.system7 import (
    generate_candidates_system7,
    get_total_days_system7,
    prepare_data_vectorized_system7,
)

from .base_strategy import StrategyBase
from .constants import STOP_ATR_MULTIPLE_DEFAULT


class System7Strategy(AlpacaOrderMixin, StrategyBase):
    """
    SPY専用のショート・カタストロフィー・ヘッジ
    - セットアップ: SPYの安値が直近50日最安値(min_50)を更新
    - エントリー: セットアップ日の翌営業日寄りでショート
    - ストップ: エントリー + 3×ATR50（ATR50は損切り幅の計算専用）
    - 利確: SPYが直近70日高値(max_70)を更新した翌営業日寄り
    """

    SYSTEM_NAME = "system7"

    def __init__(self):
        super().__init__()

    def prepare_data(
        self,
        raw_data_dict,
        progress_callback=None,
        log_callback=None,
        skip_callback=None,
        **kwargs,
    ):
        """インジケーター計算をコア関数へ委譲"""
        kwargs.pop("single_mode", None)
        if isinstance(raw_data_dict, dict):
            tmp: dict[str, pd.DataFrame] = {}
            for k, v in raw_data_dict.items():
                if isinstance(k, str) and isinstance(v, pd.DataFrame):
                    tmp[k] = v
            prepared_input = tmp
        else:
            tmp2: dict[str, pd.DataFrame] = {}
            try:
                for s in list(raw_data_dict):  # type: ignore[arg-type]
                    df = get_cached_data(s)
                    if df is not None and isinstance(df, pd.DataFrame):
                        tmp2[str(s)] = df
            except Exception:
                tmp2 = {}
            prepared_input = tmp2
        return prepare_data_vectorized_system7(
            prepared_input,
            progress_callback=progress_callback,
            log_callback=log_callback,
            skip_callback=skip_callback,
        )

    def generate_candidates(self, data_dict, market_df=None, **kwargs):
        kwargs.pop("single_mode", None)
        return generate_candidates_system7(data_dict, **kwargs)

    def run_backtest(
        self,
        data_dict,
        candidates_by_date,
        capital,
        **kwargs,
    ) -> pd.DataFrame:
        results: list[dict] = []
        if "SPY" not in data_dict:
            return pd.DataFrame()

        df: pd.DataFrame = data_dict["SPY"]
        total_days = len(candidates_by_date)
        start_time = time.time()

        capital_current = float(capital)
        position_open = False
        current_exit_date = None

        on_progress = kwargs.get("on_progress")
        on_log = kwargs.get("on_log")
        single_mode = bool(kwargs.get("single_mode", False))

        risk_pct = float(self.config.get("risk_pct", 0.02))
        max_pct = float(self.config.get("max_pct", 0.20))
        if "single_mode" in self.config:
            single_mode = (
                bool(self.config.get("single_mode", False)) if not single_mode else single_mode
            )

        stop_mult = float(self.config.get("stop_atr_multiple", STOP_ATR_MULTIPLE_DEFAULT))

        for i, (entry_date, candidates) in enumerate(
            sorted(candidates_by_date.items()),
            1,
        ):
            if position_open and entry_date >= current_exit_date:
                position_open = False
                current_exit_date = None

            if position_open:
                if on_progress:
                    on_progress(i, total_days, start_time)
                continue

            for c in candidates:
                idxers = df.index.get_indexer([pd.Timestamp(entry_date)])
                if len(idxers) == 0 or int(idxers[0]) == -1:
                    continue
                entry_idx = int(idxers[0])
                entry_price = float(df.iloc[entry_idx]["Open"])
                atr_val = None
                try:
                    atr_val = c.get("ATR50") if isinstance(c, dict) else c["ATR50"]
                except Exception:
                    atr_val = None
                if atr_val is None or pd.isna(atr_val):
                    continue
                atr = float(atr_val)
                stop_price = entry_price + stop_mult * atr
                diff = stop_price - entry_price
                if diff <= 0:
                    continue

                risk_per_trade = risk_pct * capital_current
                max_position_value = capital_current if single_mode else capital_current * max_pct

                shares_by_risk = risk_per_trade / (stop_price - entry_price)
                shares_by_cap = max_position_value // entry_price
                shares = int(min(shares_by_risk, shares_by_cap))
                if shares <= 0:
                    continue

                exit_date, exit_price = None, None
                for idx2 in range(entry_idx + 1, len(df)):
                    if float(df.iloc[idx2]["High"]) >= stop_price:
                        exit_date = df.index[idx2]
                        exit_price = stop_price
                        break
                    if float(df.iloc[idx2]["High"]) >= float(df.iloc[idx2]["max_70"]):
                        exit_idx = min(idx2 + 1, len(df) - 1)
                        exit_date = df.index[exit_idx]
                        exit_price = float(df.iloc[exit_idx]["Open"])
                        break
                if exit_date is None:
                    exit_date = df.index[-1]
                    exit_price = float(df.iloc[-1]["Close"])

                exit_price_safe = (
                    float(exit_price) if exit_price is not None else float(df.iloc[-1]["Close"])
                )
                pnl = (entry_price - exit_price_safe) * shares
                return_pct = pnl / capital_current * 100 if capital_current else 0.0

                results.append(
                    {
                        "symbol": "SPY",
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "entry_price": entry_price,
                        "exit_price": round(exit_price_safe, 2),
                        "shares": shares,
                        "pnl": round(pnl, 2),
                        "return_%": round(return_pct, 2),
                    }
                )

                capital_current += pnl
                position_open = True
                current_exit_date = exit_date

            if on_progress:
                on_progress(i, total_days, start_time)
            if on_log and (i % 10 == 0 or i == total_days):
                try:
                    on_log(i, total_days, start_time)
                except TypeError:
                    on_log(f"💹 バックテスト進捗 {int(i)}/{int(total_days)} 日")

        return pd.DataFrame(results)

    def compute_entry(self, df: pd.DataFrame, candidate: dict, current_capital: float):
        key = candidate.get("entry_date")
        if key is None:
            return None
        try:
            key_ts = pd.Timestamp(key)
        except Exception:
            return None
        idxers = df.index.get_indexer([key_ts])
        if len(idxers) == 0 or int(idxers[0]) == -1:
            return None
        entry_idx = int(idxers[0])
        if entry_idx <= 0 or entry_idx >= len(df):
            return None
        entry_price = float(df.iloc[entry_idx]["Open"])
        atr_val = None
        try:
            atr_val = candidate.get("ATR50") if isinstance(candidate, dict) else None
        except Exception:
            atr_val = None
        if atr_val is None:
            try:
                atr_val = df.iloc[entry_idx - 1]["ATR50"]
            except Exception:
                return None
        if atr_val is None or pd.isna(atr_val):
            return None
        atr = float(atr_val)
        stop_mult = float(self.config.get("stop_atr_multiple", STOP_ATR_MULTIPLE_DEFAULT))
        stop_price = entry_price + stop_mult * atr
        if stop_price - entry_price <= 0:
            return None
        return entry_price, stop_price

    def prepare_minimal_for_test(self, raw_data_dict: dict) -> dict:
        out: dict = {}
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
            x["ATR50"] = tr.rolling(50).mean()
            out[sym] = x
        return out

    def get_total_days(self, data_dict: dict) -> int:
        return get_total_days_system7(data_dict)
