# common/utils.py
import logging
import os

import pandas as pd

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


def get_cached_data(symbol: str, folder: str = "data_cache") -> pd.DataFrame | None:
    """
    キャッシュから銘柄データを読み込む。

    方針:
    - バックテスト/広期間参照の既定は base（存在しない場合は full/rolling から再構築）
    - 互換のため引数 `folder` は維持するが、`data_cache/直下` のCSVは参照しない。
    """
    try:
        # base を優先してロード（無ければ内部で full/rolling から再構築）
        from common.cache_manager import load_base_cache  # 遅延import

        df = load_base_cache(symbol, rebuild_if_missing=True)
        if df is None or df.empty:
            return None
        # 既存呼び出し互換: 大文字カラム/Date index を維持
        if df.index.name != "Date":
            if "Date" in df.columns:
                df = df.sort_values("Date").set_index("Date")
            elif "date" in df.columns:
                x = df.rename(columns={"date": "Date"})
                x["Date"] = pd.to_datetime(x["Date"], errors="coerce")
                df = x.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
        # 列名の大文字化（存在するもののみ）
        rename_map = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "adjusted_close": "AdjClose",
            "adjclose": "AdjClose",
            "volume": "Volume",
        }
        for k, v in list(rename_map.items()):
            if k in df.columns:
                df = df.rename(columns={k: v})
        return df.sort_index()
    except Exception as e:
        print(f"{symbol}: base 経由の読み込み失敗 - {e}")
        return None


def get_manual_data(symbol: str, folder: str = "data_cache") -> pd.DataFrame | None:
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


def resolve_batch_size(total_symbols: int, configured: int) -> int:
    """Return effective batch size.

    If `total_symbols` is 500 以下, バッチサイズを総銘柄数の10% (切り捨て) とし、
    最低でも10件になるよう調整する。それ以外は `configured` をそのまま返す。
    """
    if total_symbols <= 500:
        return max(total_symbols // 10, 10)
    return configured


class BatchSizeMonitor:
    """Monitor batch durations and auto-tune the batch size.

    Parameters
    ----------
    initial : int
        Initial batch size.
    target_time : float, default 60.0
        Desired duration (seconds) for a single batch.
    patience : int, default 3
        Number of consecutive batches to observe before adjusting.
    min_batch_size : int, default 10
        Lower bound for the batch size.
    max_batch_size : int, default 1000
        Upper bound for the batch size.
    """

    def __init__(
        self,
        initial: int,
        target_time: float = 60.0,
        patience: int = 3,
        min_batch_size: int = 10,
        max_batch_size: int = 1000,
    ) -> None:
        self.batch_size = initial
        self.target_time = target_time
        self.patience = patience
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self._history: list[float] = []
        self.logger = logging.getLogger(__name__)

    def update(self, duration: float) -> int:
        """Record batch duration and adjust the size if needed."""
        self._history.append(duration)
        if len(self._history) < self.patience:
            return self.batch_size

        long = all(t > self.target_time for t in self._history)
        short = all(t < self.target_time / 2 for t in self._history)

        if long:
            new_size = max(self.min_batch_size, self.batch_size // 2)
            if new_size != self.batch_size:
                self.logger.info(
                    "Batch too slow: %.2fs avg, reducing size %s -> %s",
                    sum(self._history) / len(self._history),
                    self.batch_size,
                    new_size,
                )
                self.batch_size = new_size
        elif short:
            new_size = min(self.max_batch_size, self.batch_size * 2)
            if new_size != self.batch_size:
                self.logger.info(
                    "Batch fast: %.2fs avg, increasing size %s -> %s",
                    sum(self._history) / len(self._history),
                    self.batch_size,
                    new_size,
                )
                self.batch_size = new_size

        self._history.clear()
        return self.batch_size
