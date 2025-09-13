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


def get_cached_data(symbol: str, folder: str = "data_cache") -> pd.DataFrame:
    """
    キャッシュ済みCSVから銘柄データを読み込む
    """
    safe_symbol = safe_filename(symbol)
    path = os.path.join(folder, f"{safe_symbol}.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, parse_dates=["Date"])
            df.set_index("Date", inplace=True)
            df = df.sort_index()
            return df
        except Exception as e:
            print(f"{symbol}: 読み込み失敗 - {e}")
            return None

    # フォールバック: CacheManager の full キャッシュから読み込む
    try:
        from common.cache_manager import CacheManager  # 遅延importで循環回避
        from config.settings import get_settings

        cm = CacheManager(get_settings(create_dirs=False))
        df2 = cm.read(symbol, "full")
        if df2 is None or df2.empty:
            return None
        # 従来呼び出し互換: 列を大文字・Date index に正規化
        x = df2.copy()
        cols = {c.lower(): c for c in x.columns}
        # date 列の正規化
        if "date" in cols:
            x["Date"] = pd.to_datetime(x["date"])
        elif "Date" in x.columns:
            x["Date"] = pd.to_datetime(x["Date"])
        else:
            return None
        x = x.sort_values("Date").set_index("Date")

        # 価格系列の正規化（存在するもののみ変換）
        rename_map = {}
        if "open" in cols:
            rename_map["open"] = "Open"
        if "high" in cols:
            rename_map["high"] = "High"
        if "low" in cols:
            rename_map["low"] = "Low"
        if "close" in cols:
            rename_map["close"] = "Close"
        if "adjusted_close" in cols:
            rename_map["adjusted_close"] = "AdjClose"
        if "adjclose" in cols:
            rename_map["adjclose"] = "AdjClose"
        if "volume" in cols:
            rename_map["volume"] = "Volume"

        # 小文字→大文字へ（存在する列のみ）
        for k, v in list(rename_map.items()):
            if k in x.columns:
                x.rename(columns={k: v}, inplace=True)

        return x
    except Exception as e:
        logging.getLogger(__name__).warning(
            "CacheManager フォールバック読込に失敗: %s (%s)", symbol, e
        )
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
