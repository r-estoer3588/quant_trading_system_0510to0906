# common/cache_validation.py
"""
キャッシュデータの健全性チェック・バリデーション処理
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from common.utils import describe_dtype

logger = logging.getLogger(__name__)


# 健全性チェックで参照する主要指標列（読み込み後は小文字化される）
MAIN_INDICATOR_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "sma25",
    "sma50",
    "sma100",
    "sma150",
    "sma200",
    "ema20",
    "ema50",
    "atr10",
    "atr14",
    "atr20",
    "atr40",
    "atr50",
    "adx7",
    "rsi3",
    "rsi4",
    "rsi14",
    "roc200",
    "hv50",
    "dollarvolume20",
    "dollarvolume50",
    "avgvolume50",
    "return_3d",
    "return_6d",
    "return_pct",
    "drop3d",
    "atr_ratio",
    "atr_pct",
)

# 各指標列が有効値を持つために最低限必要とする観測日数の目安
_INDICATOR_MIN_OBSERVATIONS: dict[str, int] = {
    "sma25": 20,
    "sma50": 50,
    "sma100": 100,
    "sma150": 150,
    "sma200": 200,
    "ema20": 1,
    "ema50": 1,
    "atr10": 11,
    "atr14": 15,
    "atr20": 21,
    "atr40": 41,
    "atr50": 51,
    "adx7": 14,
    "rsi3": 3,
    "rsi4": 4,
    "rsi14": 14,
    "roc200": 201,
    "hv50": 51,
    "dollarvolume20": 20,
    "dollarvolume50": 50,
    "avgvolume50": 50,
    "return_3d": 4,
    "return_6d": 7,
    "return_pct": 2,
    "drop3d": 4,
    "atr_ratio": 11,
    "atr_pct": 11,
}


class CacheValidator:
    """キャッシュデータの健全性チェック機能を提供するクラス。"""

    _GLOBAL_WARNED: ClassVar[set[tuple[str, str, str]]] = set()

    def __init__(self):
        self._warned = self._GLOBAL_WARNED

    def _warn_once(self, ticker: str, profile: str, category: str, message: str) -> None:
        """同じ警告を一度だけ出力する。"""
        key = (ticker, profile, category)
        if key in self._warned:
            return
        self._warned.add(key)
        logger.warning(message)

    def check_nan_rates(self, df: pd.DataFrame, ticker: str, profile: str) -> None:
        """主要指標のNaN率をチェックし、高い場合は警告を出力する。"""
        if df is None or df.empty:
            return

        present_indicators = [col for col in MAIN_INDICATOR_COLUMNS if col in df.columns]
        if not present_indicators:
            return

        total_rows = len(df)
        problematic_cols = []

        for col in present_indicators:
            try:
                series = df[col]
                nan_count = series.isna().sum()
                nan_rate = nan_count / total_rows if total_rows > 0 else 0.0

                min_obs = _INDICATOR_MIN_OBSERVATIONS.get(col, 50)
                valid_count = total_rows - nan_count

                # 警告条件: NaN率60%超過 または 有効観測数不足
                if nan_rate > 0.6 or valid_count < min_obs:
                    problematic_cols.append(f"{col}({nan_rate:.1%})")
            except Exception:
                problematic_cols.append(f"{col}(err)")

        if problematic_cols:
            msg = (
                f"[{profile}] {ticker}: 指標NaN率が高い列: {', '.join(problematic_cols[:5])}"
                + (f" (+{len(problematic_cols)-5})" if len(problematic_cols) > 5 else "")
                + f" (rows={total_rows})"
            )
            self._warn_once(ticker, profile, "high_nan", msg)

    def check_column_dtypes(self, df: pd.DataFrame, ticker: str, profile: str) -> None:
        """主要列のデータ型をチェックする。"""
        if df is None or df.empty:
            return
        dtype_info = describe_dtype(df, max_columns=8)
        logger.debug(f"[{profile}] {ticker}: dtypes={dtype_info}")

    def check_non_positive_prices(self, df: pd.DataFrame, ticker: str, profile: str) -> None:
        """価格列で非正値の割合をチェックする。"""
        price_cols = ["open", "high", "low", "close"]
        issues = []

        for col in price_cols:
            if col in df.columns:
                try:
                    non_pos_count = (df[col] <= 0).sum()
                    if non_pos_count > 0:
                        rate = non_pos_count / len(df) if len(df) > 0 else 0.0
                        issues.append(f"{col}({non_pos_count}, {rate:.1%})")
                except Exception:
                    pass

        if issues:
            msg = f"[{profile}] {ticker}: 非正価格検出: {', '.join(issues)}"
            self._warn_once(ticker, profile, "non_positive", msg)

    def perform_health_check(self, df: pd.DataFrame, ticker: str, profile: str) -> None:
        """総合的な健全性チェックを実行する。"""
        if df is None or df.empty:
            return

        try:
            self.check_nan_rates(df, ticker, profile)
            self.check_column_dtypes(df, ticker, profile)
            self.check_non_positive_prices(df, ticker, profile)
        except Exception as e:
            logger.warning(f"[{profile}] {ticker}: 健全性チェック中にエラー: {e}")


# グローバルバリデーターインスタンス
_global_validator = CacheValidator()


def get_cache_validator() -> CacheValidator:
    """グローバルバリデーターインスタンスを取得。"""
    return _global_validator


def perform_cache_health_check(df: pd.DataFrame, ticker: str, profile: str) -> None:
    """キャッシュデータの健全性チェックを実行する（便利関数）。"""
    _global_validator.perform_health_check(df, ticker, profile)
