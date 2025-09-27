"""キャッシュファイルのI/O操作を担当するクラス"""

from __future__ import annotations

import logging
from pathlib import Path
import shutil
from typing import ClassVar

import pandas as pd

from config.settings import Settings

logger = logging.getLogger(__name__)


class CacheFileIOError(Exception):
    """キャッシュファイルI/O操作で発生するエラー"""

    pass


class CacheFileIO:
    """キャッシュファイルの読み書きを専門に扱うクラス"""

    SUPPORTED_FORMATS: ClassVar[set[str]] = {".csv", ".parquet", ".feather"}

    def __init__(self, settings: Settings):
        self.settings = settings
        self.file_format = getattr(settings.cache, "file_format", "auto")

    def detect_file_path(self, base_dir: Path, ticker: str) -> Path:
        """ファイル形式を自動検出し、適切なパスを返す"""
        for ext in [".csv", ".parquet", ".feather"]:
            if (candidate := base_dir / f"{ticker}{ext}").exists():
                return candidate

        # 既存ファイルが見つからない場合、設定に基づいて決定
        fmt = (self.file_format or "auto").lower()
        if fmt == "parquet":
            return base_dir / f"{ticker}.parquet"
        elif fmt == "feather":
            return base_dir / f"{ticker}.feather"
        else:
            return base_dir / f"{ticker}.csv"

    def read_dataframe(self, path: Path) -> pd.DataFrame | None:
        """ファイル形式に応じてDataFrameを読み込み"""
        if not path.exists():
            return None

        try:
            if path.suffix == ".feather":
                return pd.read_feather(path)
            elif path.suffix == ".parquet":
                return pd.read_parquet(path)
            elif path.suffix == ".csv":
                return self._read_csv_with_fallback(path)
            else:
                raise CacheFileIOError(f"サポートされていないファイル形式: {path.suffix}")
        except Exception as e:
            logger.error(f"ファイル読み込み失敗: {path.name} ({e})")
            # CSVフォールバック
            if path.suffix != ".csv":
                csv_fallback = path.with_suffix(".csv")
                if csv_fallback.exists():
                    return self._read_csv_with_fallback(csv_fallback)
            return None

    def _read_csv_with_fallback(self, path: Path) -> pd.DataFrame:
        """CSVファイル読み込み（日付解析のフォールバック処理付き）"""
        try:
            return pd.read_csv(path, parse_dates=["date"])
        except ValueError as e:
            if "Missing column provided to 'parse_dates': 'date'" in str(e):
                df = pd.read_csv(path)
                # Date列をdate列にリネーム
                if "Date" in df.columns:
                    df = df.rename(columns={"Date": "date"})
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                return df
            raise

    def write_dataframe_atomic(self, df: pd.DataFrame, path: Path) -> None:
        """アトミック書き込み（一時ファイル経由）"""
        if df is None or df.empty:
            logger.warning(f"空のDataFrameの書き込みをスキップ: {path.name}")
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")

        try:
            self._write_by_format(df, tmp_path, path.suffix)
            # アトミックな移動
            shutil.move(str(tmp_path), str(path))
        except Exception as e:
            logger.error(f"ファイル書き込み失敗: {path.name} ({e})")
            raise CacheFileIOError(f"書き込み失敗: {path.name}") from e
        finally:
            # 一時ファイルのクリーンアップ
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError as e:
                    logger.error(f"一時ファイル削除失敗: {tmp_path} ({e})")

    def _write_by_format(self, df: pd.DataFrame, path: Path, format_ext: str) -> None:
        """ファイル形式に応じた書き込み処理"""
        if format_ext == ".parquet":
            df.to_parquet(path, index=False)
        elif format_ext == ".feather":
            df.reset_index(drop=True).to_feather(path)
        elif format_ext == ".csv":
            self._write_csv_with_formatting(df, path)
        else:
            raise CacheFileIOError(f"サポートされていない書き込み形式: {format_ext}")

    def _write_csv_with_formatting(self, df: pd.DataFrame, path: Path) -> None:
        """CSV書き込み（フォーマット設定適用）"""
        try:
            # 設定からCSVフォーマットオプションを取得
            cache_cfg = getattr(self.settings, "cache", None)
            csv_cfg = getattr(cache_cfg, "csv", None)

            if csv_cfg:
                decimal_point = getattr(csv_cfg, "decimal_point", ".")
                thousands_sep = getattr(csv_cfg, "thousands_sep", None)
                field_sep = getattr(csv_cfg, "field_sep", ",")

                formatters = self._make_csv_formatters(df, decimal_point, thousands_sep)

                if formatters:
                    formatted_df = df.copy()
                    for col, formatter in formatters.items():
                        if col in formatted_df.columns:
                            try:
                                formatted_df[col] = formatted_df[col].apply(formatter)
                            except Exception:
                                formatted_df[col] = formatted_df[col].astype(str)
                    formatted_df.to_csv(path, index=False, sep=field_sep)
                else:
                    df.to_csv(path, index=False, decimal=decimal_point, sep=field_sep)
            else:
                # デフォルトCSV書き込み
                df.to_csv(path, index=False)
        except Exception as e:
            logger.error(f"CSVフォーマット書き込み失敗: {path.name} ({e})")
            # フォールバック: プレーンCSV書き込み
            df.to_csv(path, index=False)

    def _make_csv_formatters(
        self,
        df: pd.DataFrame,
        decimal_point: str = ".",
        thousands_sep: str | None = None,
    ) -> dict:
        """CSV書き込み用のフォーマッタ辞書を生成"""
        if df is None or df.empty:
            return {}

        lowercase_map = {c.lower(): c for c in df.columns}
        formatters = {}

        def _add_thousands_separator(int_str: str, sep: str) -> str:
            if not sep:
                return int_str
            negative = int_str.startswith("-")
            prefix = "-" if negative else ""
            number_part = int_str[1:] if negative else int_str
            parts = []
            while number_part:
                parts.append(number_part[-3:])
                number_part = number_part[:-3]
            return prefix + sep.join(reversed(parts))

        def _create_number_formatter(decimal_places: int):
            def formatter(x):
                if pd.isna(x):
                    return ""
                try:
                    formatted = f"{float(x):.{decimal_places}f}"
                    if thousands_sep:
                        integer_part, _, fractional_part = formatted.partition(".")
                        integer_part = _add_thousands_separator(integer_part, thousands_sep)
                        formatted = (
                            f"{integer_part}.{fractional_part}" if fractional_part else integer_part
                        )
                    return (
                        formatted.replace(".", decimal_point) if decimal_point != "." else formatted
                    )
                except (ValueError, TypeError):
                    return str(x)

            return formatter

        def _create_integer_formatter():
            def formatter(x):
                if pd.isna(x):
                    return ""
                try:
                    int_str = f"{int(round(float(x))):d}"
                    return (
                        _add_thousands_separator(int_str, thousands_sep)
                        if thousands_sep
                        else int_str
                    )
                except (ValueError, TypeError):
                    return str(x)

            return formatter

        # フォーマットルール定義
        formatting_rules = {
            _create_number_formatter(2): [
                "open",
                "close",
                "high",
                "low",
                "atr10",
                "atr14",
                "atr20",
                "atr40",
                "atr50",
                "rsi3",
                "rsi4",
                "rsi14",
                "adx7",
            ],
            _create_number_formatter(4): [
                "roc200",
                "return_3d",
                "return_6d",
                "return_6d",
                "atr_ratio",
                "atr_pct",
                "hv50",
            ],
            _create_integer_formatter(): [
                "volume",
                "dollarvolume20",
                "dollarvolume50",
                "avgvolume50",
            ],
        }

        for formatter_func, column_names in formatting_rules.items():
            for name in column_names:
                if name in lowercase_map:
                    formatters[lowercase_map[name]] = formatter_func

        return formatters
