"""DataFrame処理のユーティリティ関数群"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DataFrameNormalizationError(Exception):
    """DataFrame正規化処理で発生するエラー"""

    pass


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """列名を小文字に正規化し、重複を除去"""
    if df is None or df.empty:
        return df

    normalized = df.copy()
    normalized.columns = [str(c).lower() for c in normalized.columns]

    # 重複列の除去
    if normalized.columns.has_duplicates:
        normalized = normalized.loc[:, ~normalized.columns.duplicated(keep="first")]

    return normalized


def normalize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """日付列を正規化し、ソート・重複除去を実行"""
    if df is None or df.empty or "date" not in df.columns:
        return df

    normalized = df.copy()

    # 日付列をDatetime型に変換
    try:
        normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    except Exception as e:
        logger.warning(f"日付列の変換に失敗: {e}")
        return df

    # NaN日付の除去、ソート、重複除去
    normalized = (
        normalized.dropna(subset=["date"])
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )

    return normalized


def ensure_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """指定した列を数値型に変換（エラー時はNaNに変換）"""
    if df is None or df.empty:
        return df

    result = df.copy()
    for col in columns:
        if col in result.columns:
            try:
                result[col] = pd.to_numeric(result[col], errors="coerce")
            except Exception as e:
                logger.warning(f"列 '{col}' の数値変換に失敗: {e}")

    return result


def standardize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV列名を統一フォーマットに変換"""
    if df is None or df.empty:
        return df

    # Date列の処理
    rename_map = {}
    if "date" in df.columns and "Date" not in df.columns:
        rename_map["date"] = "Date"

    # OHLCV列の標準化
    ohlcv_mapping = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "adjusted_close": "Close",
        "adj_close": "Close",
        "adjclose": "Close",
        "close": "Close",
        "volume": "Volume",
        "vol": "Volume",
    }

    # 実際に存在する列のみマッピングに追加
    for old_name, new_name in ohlcv_mapping.items():
        if old_name in df.columns:
            rename_map[old_name] = new_name

    if rename_map:
        result = df.rename(columns=rename_map)
    else:
        result = df.copy()

    return result


def validate_required_columns(
    df: pd.DataFrame, required_cols: set[str]
) -> tuple[bool, set[str]]:
    """必須列の存在チェック"""
    if df is None or df.empty:
        return False, required_cols

    existing_cols = set(df.columns)
    missing_cols = required_cols - existing_cols

    return len(missing_cols) == 0, missing_cols


def round_dataframe(df: pd.DataFrame, decimals: int | None) -> pd.DataFrame:
    """DataFrameを指定桁数で丸める（列種別に応じた丸め処理）"""
    if df is None or decimals is None:
        return df

    try:
        decimals_int = int(decimals)
    except (ValueError, TypeError):
        return df

    # カテゴリ別の丸め設定
    price_atr_cols = {
        "open",
        "close",
        "high",
        "low",
        "atr10",
        "atr14",
        "atr20",
        "atr40",
        "atr50",
        "adjusted_close",
        "adjclose",
        "adj_close",
    }
    volume_cols = {"volume", "dollarvolume20", "dollarvolume50", "avgvolume50"}
    oscillator_cols = {"rsi3", "rsi4", "rsi14", "adx7"}
    pct_cols = {
        "roc200",
        "return_3d",
        "return_6d",
        "return6d",
        "atr_ratio",
        "atr_pct",
        "hv50",
        "return_pct",
    }

    result = df.copy()
    lowercase_map = {c.lower(): c for c in result.columns}

    def _safe_round(series: pd.Series, ndigits: int) -> pd.Series:
        try:
            return pd.to_numeric(series, errors="coerce").round(ndigits)
        except Exception:
            return series

    # グループ別丸め処理
    rounding_groups = {
        2: price_atr_cols | oscillator_cols,
        4: pct_cols,
    }

    rounded_columns = set()

    for ndigits, column_names in rounding_groups.items():
        cols_to_round = [
            lowercase_map[name] for name in column_names if name in lowercase_map
        ]
        for col in cols_to_round:
            result[col] = _safe_round(result[col], ndigits)
        rounded_columns.update(cols_to_round)

    # Volume系列の特別処理
    vol_cols_to_round = [
        lowercase_map[name] for name in volume_cols if name in lowercase_map
    ]
    for col in vol_cols_to_round:
        try:
            series = pd.to_numeric(result[col], errors="coerce").round(0)
            result[col] = series.astype("Int64")
        except Exception:
            result[col] = _safe_round(result[col], 0)
    rounded_columns.update(vol_cols_to_round)

    # 残りの数値列
    for col in result.columns:
        if col not in rounded_columns and pd.api.types.is_numeric_dtype(result[col]):
            result[col] = _safe_round(result[col], decimals_int)

    return result


def prepare_dataframe_for_cache(df: pd.DataFrame) -> pd.DataFrame:
    """キャッシュ用にDataFrameを準備（正規化、ソート、重複除去）"""
    if df is None or df.empty:
        return df

    # 段階的な正規化処理
    result = normalize_column_names(df)
    result = normalize_date_column(result)

    # OHLCV列の数値化
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    result = ensure_numeric_columns(result, ohlcv_cols)

    return result
