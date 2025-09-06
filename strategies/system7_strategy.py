# strategies/system7_strategy.py
from __future__ import annotations

import pandas as pd
import time

from .base_strategy import StrategyBase
from common.alpaca_order import AlpacaOrderMixin
from core.system7 import (
    prepare_data_vectorized_system7,
    generate_candidates_system7,
    get_total_days_system7,
)


class System7Strategy(AlpacaOrderMixin, StrategyBase):
    """
    SPY専用のショート・カタストロフィー・ヘッジ。
    - エントリー: SPYが直近50日安値を更新の翌日寄りでショート
    - ストップ: エントリー + 3×ATR50
    - 利確: SPYが直近70日高値を更新した翌日寄り
    """

    SYSTEM_NAME = "system7"

    def __init__(self):
        super().__init__()

    def prepare_data(self, raw_data_dict, **kwargs):
        return prepare_data_vectorized_system7(raw_data_dict, **kwargs)

    def generate_candidates(self, prepared_dict, **kwargs):
        return generate_candidates_system7(prepared_dict, **kwargs)

    def run_backtest(
        self,
        prepared_dict,
        candidates_by_date,
        capital,
        on_progress=None,
        on_log=None,
        single_mode=False,
    ):
        results = []
        if "SPY" not in prepared_dict:
            return pd.DataFrame()

        df = prepared_dict["SPY"]
        total_days = len(candidates_by_date)
        start_time = time.time()

        capital_current = capital
        position_open = False
        current_exit_date = None

        risk_pct = float(self.config.get("risk_pct", 0.02))
        max_pct = float(self.config.get("max_pct", 0.20))
        if "single_mode" in self.config:
            single_mode = bool(self.config.get("single_mode", False))

        stop_mult = float(self.config.get("stop_atr_multiple", 3.0))

        for i, (entry_date, candidates) in enumerate(
            sorted(candidates_by_date.items()), 1
        ):
            if position_open and entry_date >= current_exit_date:
                position_open = False
                current_exit_date = None

            if position_open:
                continue

            for c in candidates:
                entry_price = float(df.loc[entry_date, "Open"])
                atr = float(c["ATR50"])
                stop_price = entry_price + stop_mult * atr

                risk_per_trade = risk_pct * capital_current
                max_position_value = (
                    capital_current if single_mode else capital_current * max_pct
                )

                shares_by_risk = risk_per_trade / (stop_price - entry_price)
                shares_by_cap = max_position_value // entry_price
                shares = int(min(shares_by_risk, shares_by_cap))
                if shares <= 0:
                    continue

                exit_date, exit_price = None, None
                entry_idx = df.index.get_loc(entry_date)
                for idx2 in range(entry_idx + 1, len(df)):
                    if float(df.iloc[idx2]["High"]) >= stop_price:
                        exit_date = df.index[idx2]
                        exit_price = stop_price
                        break
                    if float(df.iloc[idx2]["High"]) >= float(df.iloc[idx2]["max_70"]):
                        exit_date = df.index[min(idx2 + 1, len(df) - 1)]
                        exit_price = float(df.loc[exit_date, "Open"])
                        break
                if exit_date is None:
                    exit_date = df.index[-1]
                    exit_price = float(df.iloc[-1]["Close"])

                pnl = (entry_price - exit_price) * shares
                return_pct = pnl / capital_current * 100 if capital_current else 0.0

                results.append(
                    {
                        "symbol": "SPY",
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "entry_price": entry_price,
                        "exit_price": round(exit_price, 2),
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
                    # on_log が1引数（msg）の実装に対応（日本語・他システムと整合）
                    on_log(f"💹 バックテスト: {int(i)}/{int(total_days)} 日")

        return pd.DataFrame(results)

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
            x["ATR50"] = tr.rolling(50).mean()
            out[sym] = x
        return out

    def get_total_days(self, data_dict: dict) -> int:
        return get_total_days_system7(data_dict)
