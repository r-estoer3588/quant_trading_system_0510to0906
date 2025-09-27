# common/cache_io.py
"""
キャッシュファイルの入出力操作
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from common.cache_format import write_dataframe_to_csv

logger = logging.getLogger(__name__)


class CacheFileManager:
    """キャッシュファイルの読み込み・書き込みを管理するクラス。"""

    def __init__(self, settings: Any):
        self.settings = settings
        self.file_format = getattr(settings.cache, "file_format", "auto")

    def detect_path(self, base_dir: Path, ticker: str) -> Path:
        """ファイル形式を自動検出して適切なパスを返す。"""
        if self.file_format == "feather":
            return base_dir / f"{ticker}.feather"
        elif self.file_format == "csv":
            return base_dir / f"{ticker}.csv"
        else:  # auto
            # Feather優先、存在しなければCSV
            feather_path = base_dir / f"{ticker}.feather"
            csv_path = base_dir / f"{ticker}.csv"
            if feather_path.exists():
                return feather_path
            else:
                return csv_path

    def read_with_fallback(self, path: Path, ticker: str, profile: str) -> pd.DataFrame | None:
        """指定パスからデータを読み込み、失敗時はフォールバック。"""
        if not path.exists():
            return None

        try:
            if path.suffix == ".feather":
                df = pd.read_feather(path)
            else:
                df = pd.read_csv(path)

            if df is not None and not df.empty:
                # 列名を小文字に統一
                df.columns = [str(col).lower() for col in df.columns]

            return df
        except Exception as e:
            logger.warning(f"[{profile}] {ticker}: 読み込み失敗 ({path}): {e}")

            # フォールバック試行
            if path.suffix == ".feather":
                csv_path = path.with_suffix(".csv")
                if csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path)
                        if df is not None and not df.empty:
                            df.columns = [str(col).lower() for col in df.columns]
                        logger.info(f"[{profile}] {ticker}: CSV フォールバック成功")
                        return df
                    except Exception:
                        pass

            return None

    def write_atomic(self, df: pd.DataFrame, path: Path, ticker: str, profile: str) -> None:
        """データフレームをアトミック書き込みで保存する。"""
        if df is None or df.empty:
            logger.warning(f"[{profile}] {ticker}: 空のDataFrameをスキップ")
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(f"{path.suffix}.tmp")

        try:
            if path.suffix == ".feather":
                # Feather形式で保存
                df.to_feather(temp_path)
            else:
                # CSV形式で保存（フォーマット処理込み）
                write_dataframe_to_csv(df, temp_path, self.settings)

            # アトミック移動
            temp_path.replace(path)

        except Exception as e:
            # 一時ファイルのクリーンアップ
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            raise RuntimeError(f"[{profile}] {ticker}: 書き込み失敗 ({path})") from e

    def optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """DataFrameのメモリ使用量を最適化する。"""
        if df is None or df.empty:
            return df

        optimized = df.copy()

        for col in optimized.columns:
            if pd.api.types.is_numeric_dtype(optimized[col]):
                try:
                    # 浮動小数点数の最適化
                    if pd.api.types.is_float_dtype(optimized[col]):
                        if optimized[col].notna().any():
                            min_val = optimized[col].min()
                            max_val = optimized[col].max()

                            # float32で表現可能な範囲内ならダウンキャスト
                            if -3.4e38 <= min_val <= 3.4e38 and -3.4e38 <= max_val <= 3.4e38:
                                optimized[col] = optimized[col].astype("float32")

                    # 整数の最適化
                    elif pd.api.types.is_integer_dtype(optimized[col]):
                        optimized[col] = pd.to_numeric(optimized[col], downcast="integer")

                except Exception:
                    # 最適化に失敗しても元のデータを保持
                    continue

        return optimized

    def remove_unnecessary_columns(
        self, df: pd.DataFrame, keep_columns: list[str] | None = None
    ) -> pd.DataFrame:
        """不要な列を除去する。"""
        if df is None or df.empty:
            return df

        if keep_columns is None:
            return df

        # keep_columns に含まれる列のみ保持
        keep_columns_lower = [col.lower() for col in keep_columns]
        available_cols = [col for col in df.columns if col.lower() in keep_columns_lower]

        if not available_cols:
            return df

        return df[available_cols].copy()
