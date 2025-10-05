from __future__ import annotations

from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from threading import Thread

import pandas as pd

from common.cache_manager import load_base_cache
from common.utils import get_cached_data


def _load_one(symbol: str, cache_dir: Path) -> tuple[str, pd.DataFrame | None]:
    # 1) 新キャッシュ（base_cache）優先
    try:
        df = load_base_cache(symbol, rebuild_if_missing=True, prefer_precomputed_indicators=False)
        if df is not None and not df.empty:
            return symbol, df
    except Exception:
        pass

    # 2) 旧キャッシュ（CSV）にフォールバック
    try:
        df = get_cached_data(symbol, folder=str(cache_dir))
        return symbol, df
    except Exception:
        return symbol, None


def load_symbols(
    symbols: Iterable[str],
    cache_dir: Path | str = Path("data_cache"),
    max_workers: int = 8,
) -> dict[str, pd.DataFrame]:
    """共通データローダー。
    - base_cache を優先し、無ければ旧CSVキャッシュから読み込む
    - 存在しない/空データは除外
    戻り値: { symbol: DataFrame }
    """
    cache_dir = Path(cache_dir)
    out: dict[str, pd.DataFrame] = {}
    symbols = list(dict.fromkeys(symbols))  # 重複除去 + 順序維持

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_load_one, s, cache_dir) for s in symbols]
        for f in as_completed(futures):
            sym, df = f.result()
            if df is not None and not df.empty:
                out[sym] = df
    return out


def iter_load_symbols(
    symbols: Iterable[str],
    cache_dir: Path | str = Path("data_cache"),
    max_workers: int = 8,
    prefetch: int = 2,
) -> Iterator[tuple[str, pd.DataFrame]]:
    """非同期先読み付きのシンボルローダー。

    背景スレッドで次の銘柄データを読み込み、CPU 計算と I/O を重複させる。

    Args:
        symbols: 読み込む銘柄リスト。
        cache_dir: キャッシュディレクトリ。
        max_workers: 同時読み込みワーカー数。
        prefetch: 先読みキューの最大サイズ。

    Yields:
        (symbol, DataFrame) タプルを逐次返す。
    """

    cache_dir = Path(cache_dir)
    symbols = list(dict.fromkeys(symbols))
    queue: Queue[tuple[str, pd.DataFrame | None] | None] = Queue(max(prefetch, 1))

    def producer() -> None:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_load_one, s, cache_dir) for s in symbols]
            for fut in as_completed(futures):
                queue.put(fut.result())
        queue.put(None)

    Thread(target=producer, daemon=True).start()

    while True:
        item = queue.get()
        if item is None:
            break
        sym, df = item
        if df is not None and not df.empty:
            yield sym, df


def load_price(ticker: str, cache_profile: str = "full") -> pd.DataFrame:
    """CacheManager 経由で株価データを読み込む。"""

    from common.cache_manager import CacheManager
    from config.settings import get_settings

    settings = get_settings(create_dirs=False)
    cm = CacheManager(settings)

    def _profiles(profile: str) -> list[str]:
        raw = (profile or "full").strip()
        if not raw:
            return ["full"]
        parts = [raw.lower()]
        if "/" in raw:
            parts = [p.strip().lower() for p in raw.split("/") if p.strip()]
            if not parts:
                parts = ["full"]
        # rolling/full のような指定では base を間に挟んでフォールバック順を明示する
        if "rolling" in parts and "full" in parts and "base" not in parts:
            try:
                roll_idx = parts.index("rolling")
            except ValueError:
                roll_idx = -1
            insert_at = roll_idx + 1 if roll_idx >= 0 else max(len(parts) - 1, 0)
            parts.insert(insert_at, "base")
        normalized: list[str] = []
        seen: set[str] = set()
        for part in parts:
            if part not in {"rolling", "base", "full"}:
                continue
            if part in seen:
                continue
            normalized.append(part)
            seen.add(part)
        return normalized or ["full"]

    def _normalize_df(df_in: pd.DataFrame | None) -> pd.DataFrame | None:
        if df_in is None or df_in.empty:
            return df_in
        df_work = df_in.copy()
        try:
            if df_work.index.name and str(df_work.index.name).lower() == "date":
                df_work = df_work.reset_index()
        except Exception:
            pass
        rename_map = {}
        for col in list(getattr(df_work, "columns", [])):
            if str(col).lower() == "date":
                rename_map[col] = "date"
        if rename_map:
            df_work = df_work.rename(columns=rename_map)
        if "date" in df_work.columns:
            df_work["date"] = pd.to_datetime(df_work["date"], errors="coerce")
            df_work = (
                df_work.dropna(subset=["date"])
                .sort_values("date")
                .drop_duplicates("date")
                .reset_index(drop=True)
            )
        try:
            df_work.columns = [str(col).lower() for col in df_work.columns]
        except Exception:
            df_work.columns = [str(col) for col in df_work.columns]
        return df_work

    result: pd.DataFrame | None = None
    profiles = _profiles(str(cache_profile))
    for profile in profiles:
        profile_key = profile.strip().lower()
        try:
            if profile_key == "base":
                base_df = load_base_cache(
                    ticker,
                    rebuild_if_missing=True,
                    cache_manager=cm,
                )
                result = _normalize_df(base_df)
            elif profile_key in {"rolling", "full"}:
                result = _normalize_df(cm.read(ticker, profile_key))
            else:
                continue
        except Exception:
            result = None
        if result is not None and not result.empty:
            return result

    return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])


__all__ = ["load_symbols", "load_price", "iter_load_symbols"]
