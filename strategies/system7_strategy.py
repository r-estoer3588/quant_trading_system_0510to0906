"""System7 strategy with cleaned Japanese comments and safe typing."""

from __future__ import annotations

import math
import time

import pandas as pd

from common.alpaca_order import AlpacaOrderMixin
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

    def prepare_data(
        self,
        raw_data_or_symbols,
        reuse_indicators: bool | None = None,
        **kwargs,
    ):
        """System7のデータ準備（共通テンプレート使用）"""
        kwargs.pop("single_mode", None)  # System7固有の引数を除去
        return self._prepare_data_template(
            raw_data_or_symbols,
            prepare_data_vectorized_system7,
            reuse_indicators=reuse_indicators,
            **kwargs,
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

    @staticmethod
    def _safe_positive(value) -> float | None:
        """値を安全に正の浮動小数点数に変換"""
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(out) or out <= 0:
            return None
        return out

    @staticmethod
    def _latest_positive(series: pd.Series | None) -> float | None:
        if series is None:
            return None
        try:
            numeric = pd.to_numeric(series, errors="coerce").dropna()
        except Exception:
            return None
        numeric = numeric[numeric > 0]
        if numeric.empty:
            return None
        val = float(numeric.iloc[-1])
        if not math.isfinite(val) or val <= 0:
            return None
        return val

    @staticmethod
    def _infer_atr_window(name: str | None, default: int = 50) -> int:
        if not name:
            return default
        digits = "".join(ch for ch in str(name) if ch.isdigit())
        if not digits:
            return default
        try:
            val = int(digits)
        except ValueError:
            return default
        return max(1, val)

    @staticmethod
    def _detect_atr_columns(df: pd.DataFrame) -> list[str]:
        return [
            col
            for col in getattr(df, "columns", [])
            if isinstance(col, str) and col.upper().startswith("ATR")
        ]

    @staticmethod
    def _fallback_atr(df: pd.DataFrame, window: int) -> float | None:
        required = {"High", "Low", "Close"}
        if df is None or df.empty or any(col not in df.columns for col in required):
            return None
        try:
            high = pd.to_numeric(df["High"], errors="coerce")
            low = pd.to_numeric(df["Low"], errors="coerce")
            close = pd.to_numeric(df["Close"], errors="coerce")
        except Exception:
            return None
        tr = pd.concat(
            [
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        if tr.empty:
            return None
        window = max(1, int(window or 50))
        min_periods = min(window, max(2, min(5, len(tr))))
        atr_series = tr.rolling(window, min_periods=min_periods).mean()
        return System7Strategy._latest_positive(atr_series)

    def compute_entry(self, df: pd.DataFrame, candidate: dict, _current_capital: float):
        key = candidate.get("entry_date")
        if key is None:
            return None
        try:
            key_ts = pd.Timestamp(key)
        except Exception:
            return None
        if df is None or df.empty:
            return None

        entry_price: float | None = None
        atr_val: float | None = None
        entry_idx = -1
        try:
            idxers = df.index.get_indexer([key_ts])
            if len(idxers) > 0:
                entry_idx = int(idxers[0])
        except Exception:
            entry_idx = -1

        atr_columns = self._detect_atr_columns(df)
        atr_column = atr_columns[0] if atr_columns else "ATR50"
        atr_window = self._infer_atr_window(atr_column, 50)

        if 0 <= entry_idx < len(df):
            row = df.iloc[entry_idx]
            try:
                entry_price = self._safe_positive(row.get("Open"))
            except Exception:
                entry_price = None
            if entry_idx > 0:
                prev_row = df.iloc[max(entry_idx - 1, 0)]
                for col in atr_columns:
                    try:
                        candidate_val = self._safe_positive(prev_row.get(col))
                    except Exception:
                        candidate_val = None
                    if candidate_val is not None:
                        atr_val = candidate_val
                        atr_column = col
                        atr_window = self._infer_atr_window(col, atr_window)
                        break

        if isinstance(candidate, dict):
            if entry_price is None:
                entry_candidate = self._safe_positive(candidate.get("entry_price"))
                if entry_candidate is not None:
                    entry_price = entry_candidate
            if entry_price is None:
                for key in ("open", "close", "price", "last_price"):
                    if key in candidate:
                        entry_candidate = self._safe_positive(candidate.get(key))
                        if entry_candidate is not None:
                            entry_price = entry_candidate
                            break
            if atr_val is None:
                atr_candidate = self._safe_positive(candidate.get("ATR50"))
                if atr_candidate is not None:
                    atr_val = atr_candidate
                    atr_window = self._infer_atr_window("ATR50", atr_window)
            if atr_val is None:
                for key, value in candidate.items():
                    if not isinstance(key, str):
                        continue
                    if "atr" not in key.lower():
                        continue
                    atr_candidate = self._safe_positive(value)
                    if atr_candidate is not None:
                        atr_val = atr_candidate
                        atr_window = self._infer_atr_window(key, atr_window)
                        break

        if entry_price is None:
            entry_price = self._latest_positive(df.get("Close"))
        if entry_price is None:
            entry_price = self._latest_positive(df.get("Open"))

        if atr_val is None and atr_column:
            atr_val = self._latest_positive(df.get(atr_column))
        if atr_val is None:
            atr_val = self._fallback_atr(df, atr_window)

        if entry_price is None or atr_val is None:
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
