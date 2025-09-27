"""チE�Eタの健全性チェチE��機�E"""

from __future__ import annotations

import logging
from typing import ClassVar

import pandas as pd

from common.utils import describe_dtype

logger = logging.getLogger(__name__)


class CacheHealthChecker:
    """キャチE��ュチE�Eタの健全性をチェチE��するクラス"""

    _GLOBAL_WARNINGS: ClassVar[set[tuple[str, str, str]]] = set()

    # 吁E��標�Eが有効値を持つために最低限忁E��とする観測日数
    INDICATOR_MIN_OBSERVATIONS = {
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

    def __init__(self, ui_prefix: str = "[CacheHealthChecker]"):
        self.ui_prefix = ui_prefix
        self.warnings = self._GLOBAL_WARNINGS

    def check_dataframe_health(
        self, df: pd.DataFrame, ticker: str, profile: str
    ) -> dict[str, bool]:
        """DataFrameの健全性を総合皁E��チェチE��"""
        if df is None or df.empty:
            return {"overall_health": True}

        results = {
            "overall_health": True,
            "nan_rates_ok": True,
            "dtypes_ok": True,
            "prices_positive": True,
        }

        try:
            # NaN玁E��ェチE��
            if not self._check_nan_rates(df, ticker, profile):
                results["nan_rates_ok"] = False
                results["overall_health"] = False

            # チE�Eタ型チェチE��
            if not self._check_column_dtypes(df, ticker, profile):
                results["dtypes_ok"] = False
                results["overall_health"] = False

            # 価格値チェチE��
            if not self._check_non_positive_prices(df, ticker, profile):
                results["prices_positive"] = False
                results["overall_health"] = False

        except Exception as e:
            error_msg = (
                f"{self.ui_prefix} ⚠�E�E{ticker} {profile} cache: " f"健全性チェチE��失敁E({e})"
            )
            self._warn_once(ticker, profile, f"healthcheck_error:{type(e).__name__}", error_msg)
            results["overall_health"] = False

        return results

    def _warn_once(self, ticker: str, profile: str, category: str, message: str) -> None:
        """同じ警告を重褁E��て出力しなぁE��ぁE��する"""
        key = (ticker, profile, category)
        if key in self.warnings:
            return
        self.warnings.add(key)
        logger.warning(message)

    def _check_nan_rates(self, df: pd.DataFrame, ticker: str, profile: str) -> bool:
        """持E���EのNaN玁E��チェチE��"""
        try:
            if df is None or df.empty:
                return True

            # DataFrame冁E��存在する持E���Eを特宁E
            indicator_columns = [
                col for col in df.columns if col.lower() in self.INDICATOR_MIN_OBSERVATIONS
            ]

            if not indicator_columns:
                return True

            # 最近�EチE�EタウィンドウでNaN玁E��評価
            recent_window = min(len(df), 120)
            problematic_columns = []

            for col in indicator_columns:
                try:
                    series = pd.to_numeric(df[col], errors="coerce").reset_index(drop=True)
                except Exception:
                    continue

                lookback_days = self.INDICATOR_MIN_OBSERVATIONS.get(col.lower(), 0)

                # 全体がNaNの場吁E
                if series.isna().all():
                    problematic_columns.append((col, 1.0))
                    continue

                # チE�Eタが少なすぎる場合�EスキチE�E
                if lookback_days and len(series) <= lookback_days:
                    continue

                # 最近�EチE�EタでNaN玁E��計箁E
                recent_data = series.tail(recent_window)
                try:
                    nan_ratio = float(recent_data.isna().mean())
                except Exception:
                    nan_ratio = 1.0

                if nan_ratio >= 0.99:  # 99%以上がNaN
                    problematic_columns.append((col, nan_ratio))

            if problematic_columns:
                issues = ", ".join(f"{col}:{ratio:.2%}" for col, ratio in problematic_columns)
                warning_msg = (
                    f"{self.ui_prefix} ⚠�E�E{ticker} {profile} cache: " f"NaN玁E��E({issues})"
                )
                self._warn_once(ticker, profile, f"nan_rate:{issues}", warning_msg)
                return False

            return True

        except Exception as e:
            logger.error(f"{self.ui_prefix} NaN玁E��ェチE��失敁E {e}")
            return False

    def _check_column_dtypes(self, df: pd.DataFrame, ticker: str, profile: str) -> bool:
        """OHLCV列�EチE�Eタ型をチェチE��"""
        health_ok = True
        ohlcv_columns = ["open", "high", "low", "close", "volume"]

        for col in ohlcv_columns:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                dtype_description = describe_dtype(df[col])
                warning_msg = (
                    f"{self.ui_prefix} ⚠�E�E{ticker} {profile} cache: "
                    f"{col}型不一致 ({dtype_description})"
                )
                self._warn_once(ticker, profile, f"dtype:{col}:{dtype_description}", warning_msg)
                health_ok = False

        return health_ok

    def _check_non_positive_prices(self, df: pd.DataFrame, ticker: str, profile: str) -> bool:
        """価格列が非正値でなぁE��チェチE��"""
        health_ok = True
        price_columns = ["close", "high", "low"]

        for col in price_columns:
            if col in df.columns:
                try:
                    values = pd.to_numeric(df[col], errors="coerce")
                    if not values.empty and (values <= 0).all():
                        warning_msg = (
                            f"{self.ui_prefix} ⚠�E�E{ticker} {profile} cache: " f"{col}全て非正値"
                        )
                        self._warn_once(ticker, profile, f"non_positive:{col}", warning_msg)
                        health_ok = False
                except Exception:
                    # 数値変換に失敗した場合もチE�Eタ型�E問題として扱ぁE
                    health_ok = False

        return health_ok
