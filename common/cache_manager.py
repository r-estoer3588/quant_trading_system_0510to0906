from __future__ import annotations

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

from config.settings import get_settings
from common.utils import get_cached_data, safe_filename


BASE_SUBDIR = "base"


def _base_dir() -> Path:
    settings = get_settings(create_dirs=True)
    base = Path(settings.DATA_CACHE_DIR) / BASE_SUBDIR
    base.mkdir(parents=True, exist_ok=True)
    return base


def compute_base_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCVのDataFrameに共通ベース指標を付加して返す。
    必要列: Open, High, Low, Close, Volume
    出力列:
      SMA25/100/150/200, EMA20/50, ATR10/14/40/50, RSI3/14, ROC200, HV20
    """
    if df is None or df.empty:
        return df
    x = df.copy()
    if "Date" in x.columns:
        x["Date"] = pd.to_datetime(x["Date"])
        x = x.sort_values("Date").set_index("Date")

    close = x["Close"].astype(float)
    high = x["High"].astype(float)
    low = x["Low"].astype(float)
    vol = x.get("Volume")

    # SMA/EMA
    x["SMA25"] = close.rolling(25).mean()
    x["SMA100"] = close.rolling(100).mean()
    x["SMA150"] = close.rolling(150).mean()
    x["SMA200"] = close.rolling(200).mean()
    x["EMA20"] = close.ewm(span=20, adjust=False).mean()
    x["EMA50"] = close.ewm(span=50, adjust=False).mean()

    # True Range / ATR
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    x["ATR10"] = tr.rolling(10).mean()
    x["ATR14"] = tr.rolling(14).mean()
    x["ATR40"] = tr.rolling(40).mean()
    x["ATR50"] = tr.rolling(50).mean()

    # RSI 3/14 (Wilder)
    def _rsi(series: pd.Series, window: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
        loss = -delta.clip(upper=0).ewm(alpha=1 / window, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    x["RSI3"] = _rsi(close, 3)
    x["RSI14"] = _rsi(close, 14)

    # ROC200 (%)
    x["ROC200"] = close.pct_change(200) * 100.0

    # HV20 (% 年率)
    ret = np.log(close / close.shift(1))
    x["HV20"] = ret.rolling(20).std() * np.sqrt(252) * 100

    # 補助: 流動性系
    if vol is not None:
        x["DollarVolume20"] = (close * vol).rolling(20).mean()
        x["DollarVolume50"] = (close * vol).rolling(50).mean()

    return x


def base_cache_path(symbol: str) -> Path:
    return _base_dir() / f"{safe_filename(symbol)}.csv"


def save_base_cache(symbol: str, df: pd.DataFrame) -> Path:
    path = base_cache_path(symbol)
    df_reset = df.reset_index() if df.index.name is not None else df
    path.parent.mkdir(parents=True, exist_ok=True)
    df_reset.to_csv(path, index=False)
    return path


def load_base_cache(symbol: str, *, rebuild_if_missing: bool = True) -> Optional[pd.DataFrame]:
    """data_cache/base/{symbol}.csv を優先的に読み込む。
    無ければ旧 `data_cache/{symbol}.csv` から構築・保存して返す（rebuild_if_missing=True）。
    いずれも無ければ None。
    """
    path = base_cache_path(symbol)
    if path.exists():
        try:
            df = pd.read_csv(path, parse_dates=["Date"]) if path.exists() else None
            if df is not None:
                df = df.sort_values("Date").set_index("Date")
            return df
        except Exception:
            pass

    if not rebuild_if_missing:
        return None

    # 旧キャッシュから再構築
    raw = get_cached_data(symbol)
    if raw is None or raw.empty:
        return None
    out = compute_base_indicators(raw)
    save_base_cache(symbol, out)
    return out

