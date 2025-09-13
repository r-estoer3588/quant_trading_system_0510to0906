from __future__ import annotations

from typing import Callable, Dict, Optional, Iterable, Tuple
import time as _t
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from indicators_common import add_indicators


def _ensure_price_columns_upper(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    # 既に大文字があれば尊重し、無い場合のみ小文字から補完
    if "Open" not in x.columns and "open" in x.columns:
        x["Open"] = x["open"]
    if "High" not in x.columns and "high" in x.columns:
        x["High"] = x["high"]
    if "Low" not in x.columns and "low" in x.columns:
        x["Low"] = x["low"]
    if "Close" not in x.columns and "close" in x.columns:
        x["Close"] = x["close"]
    if "Volume" not in x.columns and "volume" in x.columns:
        x["Volume"] = x["volume"]
    return x


def precompute_shared_indicators(
    basic_data: Dict[str, pd.DataFrame],
    *,
    log: Optional[Callable[[str], None]] = None,
    parallel: bool = False,
    max_workers: int | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    basic_data の各 DataFrame に共有インジケータ列を付与して返す。

    - 入力はローリング/ベース由来の最小カラム（小文字）でも可。
    - 価格系カラム（Open/High/Low/Close/Volume）は大文字を補完してから計算。
    - 出力は元の DataFrame に `add_indicators` で追加された列を結合。
    - 既存列は上書きしない方針（同名が存在すればそのまま残す）。
    """
    if not basic_data:
        return basic_data
    out: Dict[str, pd.DataFrame] = {}
    total = len(basic_data)
    start_ts = _t.time()
    CHUNK = 500

    def _calc(sym_df: Tuple[str, pd.DataFrame]) -> Tuple[str, pd.DataFrame]:
        sym, df = sym_df
        try:
            if df is None or getattr(df, "empty", True):
                return sym, df
            work = _ensure_price_columns_upper(df)
            ind_df = add_indicators(work)
            new_cols = [c for c in ind_df.columns if c not in df.columns]
            if new_cols:
                merged = df.copy()
                for c in new_cols:
                    merged[c] = ind_df[c]
                return sym, merged
            return sym, df
        except Exception:
            return sym, df

    if parallel and total >= 1000:
        workers = max_workers or min(32, (total // 1000) + 8)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_calc, item): item[0] for item in basic_data.items()}
            done = 0
            for fut in as_completed(futures):
                sym, res = fut.result()
                out[sym] = res
                done += 1
                if log and (done % CHUNK == 0 or done == total):
                    try:
                        elapsed = max(0.001, _t.time() - start_ts)
                        rate = done / elapsed
                        remain = max(0, total - done)
                        eta_sec = int(remain / rate) if rate > 0 else 0
                        m, s = divmod(eta_sec, 60)
                        log(f"🧮 共有指標 前計算: {done}/{total} | ETA {m}分{s}秒")
                    except Exception:
                        try:
                            log(f"🧮 共有指標 前計算: {done}/{total}")
                        except Exception:
                            pass
    else:
        for idx, item in enumerate(basic_data.items(), start=1):
            sym, res = _calc(item)
            out[sym] = res
            if log and (idx % CHUNK == 0 or idx == total):
                try:
                    elapsed = max(0.001, _t.time() - start_ts)
                    rate = idx / elapsed
                    remain = max(0, total - idx)
                    eta_sec = int(remain / rate) if rate > 0 else 0
                    m, s = divmod(eta_sec, 60)
                    log(f"🧮 共有指標 前計算: {idx}/{total} | ETA {m}分{s}秒")
                except Exception:
                    try:
                        log(f"🧮 共有指標 前計算: {idx}/{total}")
                    except Exception:
                        pass
    return out


__all__ = ["precompute_shared_indicators"]
