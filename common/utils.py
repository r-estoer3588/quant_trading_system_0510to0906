# common/utils.py
import logging
import re
from collections.abc import Callable, Hashable
from pathlib import Path

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


logger = logging.getLogger(__name__)


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


_OHLCV_CANONICAL_NAMES = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
    "adjclose": "AdjClose",
    "adjustedclose": "AdjClose",
}


_NON_ALNUM = re.compile(r"[^a-z0-9]")


def _normalize_ohlcv_key(name: Hashable) -> str | None:
    """Normalize a column label into a lookup key for OHLCV matching."""

    if not isinstance(name, str):
        try:
            name = str(name)
        except Exception:
            return None
    key = _NON_ALNUM.sub("", name.lower())
    return key or None


def _merge_ohlcv_variants(df: pd.DataFrame) -> pd.DataFrame:
    """Return *df* with case-insensitive OHLCV variants coalesced.

    キャッシュソースによっては ``OPEN`` や ``adj close`` のような表記ゆれが
    混在し、重複列の原因となる。本関数では大文字小文字・空白・アンダースコアを
    吸収したうえで各列を統一し、欠損が少ない列を優先的に残す。
    """

    if df is None or getattr(df, "empty", True):
        return df

    original = df.copy()
    column_groups: dict[str, list[tuple[int, str]]] = {}
    for idx, col in enumerate(original.columns):
        key = _normalize_ohlcv_key(col)
        if key is None:
            continue
        canonical = _OHLCV_CANONICAL_NAMES.get(key)
        if canonical is None:
            continue
        column_groups.setdefault(canonical, []).append((idx, col))

    if not column_groups:
        return original

    combined_series_by_index: dict[int, pd.Series] = {}
    target_name_by_index: dict[int, str] = {}
    drop_indices: set[int] = set()

    for canonical, items in column_groups.items():
        if not items:
            continue
        series_candidates: list[tuple[int, str, pd.Series]] = []
        for idx, col in items:
            try:
                col_series = original.iloc[:, idx]
            except Exception:
                continue
            if isinstance(col_series, pd.DataFrame):
                col_series = col_series.iloc[:, 0]
            series_candidates.append((idx, col, pd.Series(col_series)))

        if not series_candidates:
            continue

        best_idx = series_candidates[0][0]
        best_series = series_candidates[0][2].copy()
        best_non_null = int(best_series.notna().sum())
        best_priority = 1 if series_candidates[0][1] == canonical else 0

        for idx, col, series in series_candidates[1:]:
            non_null = int(series.notna().sum())
            priority = 1 if col == canonical else 0
            if (
                non_null > best_non_null
                or (non_null == best_non_null and priority > best_priority)
                or (
                    non_null == best_non_null
                    and priority == best_priority
                    and idx < best_idx
                )
            ):
                best_idx = idx
                best_series = series.copy()
                best_non_null = non_null
                best_priority = priority

        combined = best_series.copy()
        for idx, _col, series in series_candidates:
            if idx == best_idx:
                continue
            try:
                combined = combined.combine_first(series)
            except Exception:
                try:
                    combined = combined.fillna(series)
                except Exception:
                    pass

        display_idx = min(idx for idx, _ in items)
        for idx, _ in items:
            if idx != display_idx:
                drop_indices.add(idx)

        target_name_by_index[display_idx] = canonical
        combined_series_by_index[display_idx] = combined

    result_columns = []
    result_series = []
    for idx, col in enumerate(original.columns):
        if idx in drop_indices:
            continue
        if idx in combined_series_by_index:
            series = combined_series_by_index[idx]
            name = target_name_by_index.get(idx, col)
        else:
            try:
                series = original.iloc[:, idx]
            except Exception:
                continue
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            series = pd.Series(series)
            name = col

        series = pd.Series(series, index=original.index, copy=True)
        series.name = None
        result_columns.append(name)
        result_series.append(series)

    if not result_series:
        return original

    merged = pd.concat(result_series, axis=1)
    merged.columns = result_columns
    merged.index = original.index
    return merged


def drop_duplicate_columns(
    df: pd.DataFrame,
    *,
    log_callback: Callable[[str], None] | None = None,
    context: str | None = None,
) -> pd.DataFrame:
    """Return *df* with duplicate column labels removed.

    When duplicated labels exist, the column whose series contains the most
    non-null values is preserved. Ties fall back to the left-most column in
    order to maintain historical behaviour.

    Parameters
    ----------
    df:
        Source DataFrame that may contain duplicate column names.
    log_callback:
        Optional callable used for logging. When provided, the function will
        emit a human-readable summary once duplicates are resolved. When
        omitted, the module logger is used instead.
    context:
        Optional textual identifier that is prepended to log messages to help
        trace the origin of duplicated columns.
    """

    if df is None or getattr(df, "empty", False):
        return df

    try:
        columns = list(df.columns)
    except Exception:
        return df

    duplicate_positions: dict[Hashable, list[int]] = {}
    for pos, label in enumerate(columns):
        duplicate_positions.setdefault(label, []).append(pos)

    duplicates = {
        label: positions
        for label, positions in duplicate_positions.items()
        if len(positions) > 1
    }
    if not duplicates:
        return df

    keep_mask = [True] * len(columns)
    log_details: list[str] = []

    for label, positions in duplicates.items():
        non_null_counts: list[int] = []
        for pos in positions:
            try:
                series = pd.Series(df.iloc[:, pos])
            except Exception:
                series = pd.Series(dtype="float64")
            non_null_counts.append(int(series.notna().sum()))

        best_pos = max(
            range(len(positions)),
            key=lambda idx: (non_null_counts[idx], -idx),
        )

        for idx, col_pos in enumerate(positions):
            keep_mask[col_pos] = idx == best_pos

        counts_repr = ", ".join(str(count) for count in non_null_counts)
        detail = (
            f"{label!r}×{len(positions)} "
            f"(非NaN数: [{counts_repr}] → keep #{best_pos + 1})"
        )
        log_details.append(detail)

    try:
        deduped = df.loc[:, keep_mask].copy()
    except Exception:
        deduped = df.loc[:, ~df.columns.duplicated()].copy()

    message_prefix = f"{context}: " if context else ""
    message = (
        f"⚠️ {message_prefix}重複カラムを検出し解消しました -> "
        f"{', '.join(log_details)}"
    )

    if log_callback is not None:
        try:
            log_callback(message)
        except Exception:
            logger.warning(message)
    else:
        logger.warning(message)

    return deduped


def get_cached_data(symbol: str, folder: str = "data_cache") -> pd.DataFrame | None:
    """
    キャッシュから銘柄データを読み込む。

    方針:
    - バックテスト/広期間参照の既定は base（存在しない場合は full/rolling から再構築）
    - 互換のため引数 `folder` は維持するが、`data_cache/直下` のCSVは参照しない。
    """
    # 1) base を優先してロード（無ければ内部で full/rolling から再構築）
    df: pd.DataFrame | None = None
    try:
        from common.cache_manager import load_base_cache  # 遅延import

        df = load_base_cache(symbol, rebuild_if_missing=True)
    except Exception:
        df = None

    if df is None or df.empty:
        # 2) 旧CSV へのフォールバック（テスト互換のため保持）
        try:
            path = Path(folder) / f"{safe_filename(symbol)}.csv"
            if path.exists():
                df = pd.read_csv(path)
            else:
                return None
        except Exception:
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
    df = _merge_ohlcv_variants(df)
    return df.sort_index()


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
