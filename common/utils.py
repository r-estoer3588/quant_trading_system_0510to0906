# common/utils.py
import os
import pandas as pd
from datetime import datetime

# Windows予約語（safe_filename用）
RESERVED_WORDS = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
}


def safe_filename(symbol: str) -> str:
    """
    Windows予約語を避けたファイル名を返す
    """
    if symbol.upper() in RESERVED_WORDS:
        return symbol + "_RESV"
    return symbol


def clean_date_column(df: pd.DataFrame, col_name: str = "Date") -> pd.DataFrame:
    """
    指定されたDate列を正規化（datetime化・昇順ソート）して返す
    """
    if col_name not in df.columns:
        raise ValueError(f"{col_name} 列が存在しません")
    df[col_name] = pd.to_datetime(df[col_name])
    df = df.sort_values(col_name).reset_index(drop=True)
    return df


def get_cached_data(symbol: str, folder: str = "data_cache") -> pd.DataFrame:
    """
    キャッシュ済みCSVから銘柄データを読み込む
    """
    safe_symbol = safe_filename(symbol)
    path = os.path.join(folder, f"{safe_symbol}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, parse_dates=["Date"])
        df.set_index("Date", inplace=True)
        df = df.sort_index()
        return df
    except Exception as e:
        print(f"{symbol}: 読み込み失敗 - {e}")
        return None


def get_manual_data(symbol: str, folder: str = "data_cache") -> pd.DataFrame:
    """
    ユーザー指定シンボルを手動読み込み用に取得
    get_cached_dataのラッパー
    """
    return get_cached_data(symbol, folder=folder)


def clamp01(value: float) -> float:
    """Clamp numeric value into the 0..1 range."""
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0
