# common/cache_validation.py
"""
キャッシュデータの健全性チェック・バリデーション処理
"""

from __future__ import annotations

from datetime import datetime
import logging
import os
from pathlib import Path
from typing import ClassVar

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
        # 出力方針（環境変数で制御）
        # - COMPACT_TODAY_LOGS=1 のときは既定でファイル集約（CLI抑制）
        # - CACHE_VALIDATION_TO_FILE=1 でもファイル集約（CLI抑制）
        # - CACHE_VALIDATION_LOG_PATH で出力先を指定可能（未指定は logs/cache_validation_warnings.log）
        # - CACHE_VALIDATION_SILENT_CLI=1 でCLI出力を強制的に抑制
        try:
            compact = (os.getenv("COMPACT_TODAY_LOGS") or "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        except Exception:
            compact = False
        try:
            to_file = (os.getenv("CACHE_VALIDATION_TO_FILE") or "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        except Exception:
            to_file = False

        self._aggregate_to_file = bool(compact or to_file)
        try:
            self._silent_cli = self._aggregate_to_file or (
                (os.getenv("CACHE_VALIDATION_SILENT_CLI") or "").strip().lower() in {"1", "true", "yes", "on"}
            )
        except Exception:
            self._silent_cli = self._aggregate_to_file

        # ログ保存先
        try:
            custom_path = (os.getenv("CACHE_VALIDATION_LOG_PATH") or "").strip()
        except Exception:
            custom_path = ""
        if custom_path:
            self._aggregate_path = Path(custom_path)
        else:
            self._aggregate_path = Path("logs") / "cache_validation_warnings.log"

    def _append_file(self, message: str) -> None:
        """検証メッセージをログファイルへ追記する（失敗は握りつぶす）。"""
        try:
            path = self._aggregate_path
            path.parent.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with path.open("a", encoding="utf-8") as fh:
                fh.write(f"{ts} | {message}\n")
        except Exception:
            # ファイル書き込みに失敗しても処理は継続
            pass

    def _log_once(
        self,
        level: int,
        ticker: str,
        profile: str,
        category: str,
        message: str,
    ) -> None:
        """同じメッセージカテゴリは一度だけ出力する。"""
        key = (ticker, profile, category)
        if key in self._warned:
            return
        self._warned.add(key)
        # ファイル集約（必要なら）
        if self._aggregate_to_file:
            self._append_file(message)
        # CLI 出力（必要なときだけ）
        if not self._silent_cli:
            logger.log(level, message)

    def _warn_once(
        self,
        ticker: str,
        profile: str,
        category: str,
        message: str,
    ) -> None:
        """後方互換のための WARN ラッパー。"""
        self._log_once(logging.WARNING, ticker, profile, category, message)

    def check_nan_rates(self, df: pd.DataFrame, ticker: str, profile: str) -> None:
        """主要指標のNaN率をチェックし、高い場合は警告を出力する。

        ポリシー:
        - 行数不足で生じるNaNは想定内として情報レベル(またはデバッグ)へ格下げ。
        - 行数は十分だがNaN率が極端に高い(>=90%)場合のみ警告。
        - COMPACT_TODAY_LOGS=1 のときは警告を INFO に格下げしてログノイズを抑制。
        """
        if df is None or df.empty:
            return

        present_indicators = [col for col in MAIN_INDICATOR_COLUMNS if col in df.columns]
        if not present_indicators:
            return

        total_rows = len(df)
        severe_cols: list[str] = []
        expected_cols: list[str] = []
        compact = str(os.getenv("COMPACT_TODAY_LOGS", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        for col in present_indicators:
            try:
                series = df[col]
                nan_count = series.isna().sum()
                nan_rate = nan_count / total_rows if total_rows > 0 else 0.0

                min_obs = _INDICATOR_MIN_OBSERVATIONS.get(col, 50)

                # 行数がそもそも足りないケースは想定内として情報扱い
                if total_rows < min_obs:
                    expected_cols.append(f"{col}({nan_rate:.1%})")
                    continue

                # 行数は十分だがNaNが極端に多いときのみ「重度」として扱う
                # 例: 計算漏れ・列壊れ等
                if nan_rate >= 0.9:
                    severe_cols.append(f"{col}({nan_rate:.1%})")
            except Exception:
                severe_cols.append(f"{col}(err)")

        # 想定内(行数不足)は詳細ログへ
        if expected_cols:
            msg_exp = (
                f"[{profile}] {ticker}: 指標NaN(データ不足で想定内): "
                f"{', '.join(expected_cols[:5])}"
                + (f" (+{len(expected_cols) - 5})" if len(expected_cols) > 5 else "")
                + f" (rows={total_rows})"
            )
            # 既定はDEBUG、コンパクト時はINFOに一段上げる（見たいときだけ見えるように）
            level = logging.INFO if compact else logging.DEBUG
            self._log_once(level, ticker, profile, "nan_expected", msg_exp)

        # 重度ケースのみ警告（COMPACT_TODAY_LOGS が有効なら INFO に格下げ）
        if severe_cols:
            msg = (
                f"[{profile}] {ticker}: 指標NaN率が高い列: {', '.join(severe_cols[:5])}"
                + (f" (+{len(severe_cols) - 5})" if len(severe_cols) > 5 else "")
                + f" (rows={total_rows})"
            )
            if compact:
                self._log_once(logging.INFO, ticker, profile, "high_nan_info", msg)
            else:
                self._warn_once(ticker, profile, "high_nan", msg)

    def check_column_dtypes(self, df: pd.DataFrame, ticker: str, profile: str) -> None:
        """主要列のデータ型をチェックする。"""
        if df is None or df.empty:
            return
        dtype_info = describe_dtype(df, max_columns=8)
        logger.debug(f"[{profile}] {ticker}: dtypes={dtype_info}")

    def check_non_positive_prices(
        self,
        df: pd.DataFrame,
        ticker: str,
        profile: str,
    ) -> None:
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
