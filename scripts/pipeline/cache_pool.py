"""BaseCachePool - Thread-safe cache pool management.

Extracted from run_all_systems_today.py for better modularity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

if TYPE_CHECKING:
    from common.cache_manager import CacheManager


@dataclass(slots=True)
class BaseCachePool:
    """base キャッシュの共有辞書をスレッドセーフに管理する補助クラス。"""

    cache_manager: "CacheManager"
    shared: dict[str, pd.DataFrame] | None = None
    hits: int = 0
    loads: int = 0
    failures: int = 0
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.shared is None:
            self.shared = {}

    def get(
        self,
        symbol: str,
        *,
        rebuild_if_missing: bool = True,
        min_last_date: pd.Timestamp | None = None,
        allowed_recent_dates: set[pd.Timestamp] | None = None,
    ) -> tuple[pd.DataFrame | None, bool]:
        """base キャッシュから銘柄シンボルの DataFrame を取得する。

        Returns (df, from_cache):
            - df: 取得または再構築された DataFrame（存在しなければ None）
            - from_cache: True=共有キャッシュ命中 / False=新規ロード
        """
        from common.cache_manager import load_base_cache

        allowed_set = set(allowed_recent_dates or ())
        if min_last_date is not None:
            try:
                min_norm: pd.Timestamp | None = pd.Timestamp(min_last_date).normalize()
            except Exception:
                min_norm = None
        else:
            min_norm = None

        def _detect_last(frame: pd.DataFrame | None) -> pd.Timestamp | None:
            if frame is None or getattr(frame, "empty", True):
                return None
            try:
                idx_dt = pd.to_datetime(frame.index, errors="coerce")
                if isinstance(idx_dt, pd.DatetimeIndex) and len(idx_dt):
                    last_val = idx_dt[-1]
                    return pd.Timestamp(cast(Any, last_val)).normalize()
            except Exception:
                pass
            try:
                series = frame.get("Date") if frame is not None else None
                if series is None and frame is not None and "date" in frame.columns:
                    series = frame.get("date")
                if series is not None:
                    ser_dt = pd.to_datetime(series, errors="coerce").dropna()
                    if len(ser_dt):
                        return pd.Timestamp(cast(Any, ser_dt.iloc[-1])).normalize()
            except Exception:
                pass
            return None

        with self._lock:
            if self.shared is not None and symbol in self.shared:
                value = self.shared[symbol]
                last_date = _detect_last(value)
                stale = False
                if allowed_set and (last_date is None or last_date not in allowed_set):
                    stale = True
                if not stale and min_norm is not None:
                    if last_date is None or last_date < min_norm:
                        stale = True
                if not stale:
                    self.hits += 1
                    return value, True
                try:
                    if self.shared is not None:
                        self.shared.pop(symbol, None)
                except Exception:
                    pass

        df = load_base_cache(
            symbol,
            rebuild_if_missing=rebuild_if_missing,
            cache_manager=self.cache_manager,
            min_last_date=min_last_date,
            allowed_recent_dates=allowed_set or None,
            prefer_precomputed_indicators=True,
        )

        with self._lock:
            if self.shared is not None and df is not None:
                self.shared[symbol] = df
            self.loads += 1
            if df is None or getattr(df, "empty", True):
                self.failures += 1

        return df, False

    def sync_to(self, target: dict[str, pd.DataFrame] | None) -> None:
        """既存の外部辞書へ共有キャッシュを反映する。"""
        if target is None or self.shared is None or target is self.shared:
            return
        with self._lock:
            try:
                target.update(self.shared)
            except Exception:
                pass

    def snapshot_stats(self) -> dict[str, int]:
        with self._lock:
            size = len(self.shared or {})
            return {
                "hits": self.hits,
                "loads": self.loads,
                "failures": self.failures,
                "size": size,
            }
