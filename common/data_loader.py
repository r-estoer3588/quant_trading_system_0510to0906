from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd

from common.cache_manager import load_base_cache
from common.utils import get_cached_data, safe_filename


def _load_one(symbol: str, cache_dir: Path) -> Tuple[str, pd.DataFrame | None]:
    # 1) 新キャッシュ（base_cache）優先
    try:
        df = load_base_cache(symbol, rebuild_if_missing=True)
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
) -> Dict[str, pd.DataFrame]:
    """共通データローダー。
    - base_cache を優先し、無ければ旧CSVキャッシュから読み込む
    - 存在しない/空データは除外
    戻り値: { symbol: DataFrame }
    """
    cache_dir = Path(cache_dir)
    out: Dict[str, pd.DataFrame] = {}
    symbols = list(dict.fromkeys(symbols))  # 重複除去 + 順序維持

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_load_one, s, cache_dir) for s in symbols]
        for f in as_completed(futures):
            sym, df = f.result()
            if df is not None and not df.empty:
                out[sym] = df
    return out


def load_price(ticker: str, cache_profile: str = "full") -> pd.DataFrame:
    """
    cache_profile: "full" | "rolling"
    読み出しは CacheManager 経由に統一。既存仕様で常にDataFrameを返す。
    """
    from config.settings import get_settings
    from common.cache_manager import CacheManager

    settings = get_settings(create_dirs=False)
    cm = CacheManager(settings)
    df = cm.read(ticker, cache_profile)
    if df is None:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    return df


__all__ = ["load_symbols", "load_price"]

