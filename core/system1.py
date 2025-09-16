"""System1 core logic.

Provides data preparation, ROC200 ranking, and total-days helpers.
"""

import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from common.utils import get_cached_data, resolve_batch_size, BatchSizeMonitor


def _compute_indicators(
    symbol: str,
    cache_dir: str,
    reuse_indicators: bool,
) -> tuple[str, pd.DataFrame | None]:
    """Compute indicators for a single symbol in a worker process."""
    import os

    df = get_cached_data(symbol)
    if df is None or df.empty:
        return symbol, None

    # 正規化: 日付インデックス・並び順・重複排除・型
    try:
        if "Date" in df.columns:
            idx = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
        else:
            idx = pd.to_datetime(df.index, errors="coerce").normalize()
        df.index = pd.Index(idx)
        df = df[~df.index.isna()].sort_index()
        try:
            if getattr(df.index, "has_duplicates", False):
                df = df[~df.index.duplicated(keep="last")]
        except Exception:
            pass
        # OHLCV を数値化
        for col in ("Open", "High", "Low", "Close", "Volume"):
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                except Exception:
                    pass
    except Exception:
        # 失敗時も最低限のフォールバック
        try:
            df = df.sort_index()
        except Exception:
            pass

    date_series = pd.to_datetime(df.index, errors="coerce").normalize()
    latest_date = date_series.max()
    cache_path = os.path.join(cache_dir, f"{symbol}_{latest_date.date()}.feather")

    if reuse_indicators and os.path.exists(cache_path):
        try:
            cached = pd.read_feather(cache_path)
            if cached is not None and not cached.isnull().any().any():
                # 安全のため、日付インデックスを正規化し、昇順・重複除去
                if "Date" in cached.columns:
                    cached["Date"] = pd.to_datetime(cached["Date"], errors="coerce").dt.normalize()
                    cached = (
                        cached.dropna(subset=["Date"])  # type: ignore[arg-type]
                        .sort_values("Date")
                        .drop_duplicates("Date")
                    )
                    cached = cached.set_index("Date")
                else:
                    try:
                        idx = pd.to_datetime(cached.index, errors="coerce").normalize()
                        cached.index = pd.Index(idx)
                        cached = cached[~cached.index.isna()].sort_index()
                    except Exception:
                        pass
                try:
                    if getattr(cached.index, "has_duplicates", False):
                        cached = cached[~cached.index.duplicated(keep="last")]
                except Exception:
                    pass
                return symbol, cached
        except Exception:
            pass

    df = df.copy()
    df["SMA25"] = df["Close"].rolling(25).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["ROC200"] = df["Close"].pct_change(200) * 100
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift()).abs(),
            (df["Low"] - df["Close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["ATR20"] = tr.rolling(20).mean()
    df["DollarVolume20"] = (df["Close"] * df["Volume"]).rolling(20).mean()

    df["filter"] = (df["Low"] >= 5) & (df["DollarVolume20"] > 50_000_000)
    df["setup"] = df["filter"] & (df["SMA25"] > df["SMA50"])

    latest_df = df[date_series == latest_date]
    try:
        latest_df.reset_index(drop=True).to_feather(cache_path)
    except Exception:
        pass

    return symbol, df


def prepare_data_vectorized_system1(
    raw_data_dict: dict[str, pd.DataFrame] | None,
    progress_callback=None,
    log_callback=None,
    skip_callback=None,
    batch_size: int | None = None,
    reuse_indicators: bool = True,
    *,
    symbols: list[str] | None = None,
    use_process_pool: bool = False,
    max_workers: int | None = None,
    **kwargs,
):
    """System1 indicator computation (UI-agnostic).

    ``use_process_pool`` が True の場合、各シンボルを ProcessPoolExecutor で並列処理し、
    キャッシュを各プロセスが直接読み込む。
    ``raw_data_dict`` が None の場合は ``symbols`` からキャッシュを取得する。
    """
    import os

    cache_dir = "data_cache/indicators_system1_cache"
    os.makedirs(cache_dir, exist_ok=True)

    if use_process_pool:
        if symbols is None:
            symbols = list(raw_data_dict.keys()) if raw_data_dict else []
        total_symbols = len(symbols)
        if batch_size is None:
            try:
                from config.settings import get_settings

                batch_size = get_settings(create_dirs=False).data.batch_size
            except Exception:
                batch_size = 100
            batch_size = resolve_batch_size(total_symbols, batch_size)

        result_dict: dict[str, pd.DataFrame] = {}
        symbol_buffer: list[str] = []
        start_time = time.time()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_compute_indicators, sym, cache_dir, reuse_indicators): sym
                for sym in symbols
            }
            for i, fut in enumerate(as_completed(futures), 1):
                sym, df = fut.result()
                if df is not None:
                    result_dict[sym] = df
                    symbol_buffer.append(sym)

                if progress_callback:
                    try:
                        progress_callback(i, total_symbols)
                    except Exception:
                        pass

                if (i % batch_size == 0 or i == total_symbols) and log_callback:
                    elapsed = time.time() - start_time
                    remaining = (elapsed / i) * (total_symbols - i) if i else 0
                    em, es = divmod(int(elapsed), 60)
                    rm, rs = divmod(int(remaining), 60)
                    joined_syms = ", ".join(symbol_buffer)
                    try:
                        log_callback(
                            f"📊 指標計算: {i}/{total_symbols} 件 完了",
                            f" | 経過: {em}分{es}秒 / 残り: 約 {rm}分{rs}秒\n",
                            f"銘柄: {joined_syms}",
                        )
                    except Exception:
                        pass
                    symbol_buffer.clear()

        return result_dict

    raw_data_dict = raw_data_dict or {}
    total_symbols = len(raw_data_dict)
    if batch_size is None:
        try:
            from config.settings import get_settings

            batch_size = get_settings(create_dirs=False).data.batch_size
        except Exception:
            batch_size = 100
        batch_size = resolve_batch_size(total_symbols, batch_size)
    batch_monitor = BatchSizeMonitor(batch_size)
    processed = 0
    symbol_buffer: list[str] = []
    start_time = time.time()
    batch_start = time.time()
    result_dict: dict[str, pd.DataFrame] = {}

    for sym, df in raw_data_dict.items():
        try:
            # 正規化: 列名の大小文字差や date 列/インデックス差を吸収
            df = df.copy()
            # 1) まず OHLCV の列名を大文字に寄せる（lower な場合をケア）
            rename_map = {}
            for low, up in (
                ("open", "Open"),
                ("high", "High"),
                ("low", "Low"),
                ("close", "Close"),
                ("volume", "Volume"),
            ):
                if low in df.columns and up not in df.columns:
                    rename_map[low] = up
            if rename_map:
                df.rename(columns=rename_map, inplace=True)

            # 2) インデックス（日付）を決定
            idx = None
            if "Date" in df.columns:
                idx = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
            elif "date" in df.columns:
                idx = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
            else:
                # 既存 index が日時ならそれを利用
                try:
                    idx = pd.to_datetime(df.index, errors="coerce").normalize()
                except Exception:
                    idx = None
            # 日付が取れない、または全て欠損ならスキップ
            if idx is None:
                raise ValueError("invalid_date_index")
            try:
                if pd.isna(idx).all():
                    raise ValueError("invalid_date_index")
            except Exception:
                try:
                    if idx.isnull().all():
                        raise ValueError("invalid_date_index")
                except Exception:
                    pass
            df.index = pd.Index(idx)
            df.index.name = "Date"
            # 型・インデックスの健全化（重複や未整列での再インデックスエラー対策）
            for col in ("Open", "High", "Low", "Close", "Volume"):
                if col in df.columns:
                    try:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    except Exception:
                        pass
            df = df.dropna(subset=[c for c in ("High", "Low", "Close") if c in df.columns])
            try:
                df = df.sort_index()
            except Exception:
                pass
            try:
                if getattr(df.index, "has_duplicates", False):
                    df = df[~df.index.duplicated(keep="last")]
            except Exception:
                pass

            # 必須列チェック（S1は Volume を用いるため必須に含める）
            needed = {"Open", "High", "Low", "Close", "Volume"}
            miss = [c for c in needed if c not in df.columns]
            if miss:
                raise ValueError(f"missing_cols:{','.join(miss)}")

            cache_path = os.path.join(cache_dir, f"{sym}.feather")
            cached: pd.DataFrame | None = None
            if reuse_indicators and os.path.exists(cache_path):
                try:
                    cached = pd.read_feather(cache_path)
                    cached["Date"] = pd.to_datetime(cached["Date"], errors="coerce").dt.normalize()
                    cached = (
                        cached.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date")
                    )
                    cached.set_index("Date", inplace=True)
                    try:
                        if getattr(cached.index, "has_duplicates", False):
                            cached = cached[~cached.index.duplicated(keep="last")]
                    except Exception:
                        pass
                except Exception:
                    cached = None

            def _calc_indicators(src: pd.DataFrame) -> pd.DataFrame:
                dst = src.copy()
                dst["SMA25"] = dst["Close"].rolling(25).mean()
                dst["SMA50"] = dst["Close"].rolling(50).mean()
                dst["ROC200"] = dst["Close"].pct_change(200) * 100
                tr = pd.concat(
                    [
                        dst["High"] - dst["Low"],
                        (dst["High"] - dst["Close"].shift()).abs(),
                        (dst["Low"] - dst["Close"].shift()).abs(),
                    ],
                    axis=1,
                ).max(axis=1)
                dst["ATR20"] = tr.rolling(20).mean()
                dst["DollarVolume20"] = (dst["Close"] * dst["Volume"]).rolling(20).mean()
                dst["filter"] = (dst["Low"] >= 5) & (dst["DollarVolume20"] > 50_000_000)
                dst["setup"] = dst["filter"] & (dst["SMA25"] > dst["SMA50"])
                return dst

            if cached is not None and not cached.empty:
                last_date = cached.index.max()
                new_rows = df[df.index > last_date]
                if new_rows.empty:
                    result_df = cached
                else:
                    context_start = last_date - pd.Timedelta(days=200)
                    recompute_src = df[df.index >= context_start]
                    recomputed = _calc_indicators(recompute_src)
                    recomputed = recomputed[recomputed.index > last_date]
                    result_df = pd.concat([cached, recomputed])
                    try:
                        # 正規化してから保存
                        result_df = result_df.sort_index()
                        if getattr(result_df.index, "has_duplicates", False):
                            result_df = result_df[~result_df.index.duplicated(keep="last")]
                        result_df.reset_index().to_feather(cache_path)
                    except Exception:
                        pass
            else:
                result_df = _calc_indicators(df)
                try:
                    result_df.reset_index().to_feather(cache_path)
                except Exception:
                    pass

            result_dict[sym] = result_df
        except Exception as e:
            # 1銘柄の失敗で全体を止めない
            if skip_callback:
                msg = str(e)
                try:
                    skip_callback(sym, f"calc_error:{msg}")
                except Exception:
                    try:
                        skip_callback(f"{sym}: calc_error:{msg}")
                    except Exception:
                        pass
        finally:
            processed += 1
            symbol_buffer.append(sym)
            if progress_callback:
                try:
                    progress_callback(processed, total_symbols)
                except Exception:
                    pass

        cache_path = os.path.join(cache_dir, f"{sym}.feather")
        cached: pd.DataFrame | None = None
        if reuse_indicators and os.path.exists(cache_path):
            try:
                cached = pd.read_feather(cache_path)
                cached["Date"] = pd.to_datetime(cached["Date"], errors="coerce").dt.normalize()
                cached = cached.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date")
                cached.set_index("Date", inplace=True)
                try:
                    if getattr(cached.index, "has_duplicates", False):
                        cached = cached[~cached.index.duplicated(keep="last")]
                except Exception:
                    pass
            except Exception:
                cached = None

        def _calc_indicators(src: pd.DataFrame) -> pd.DataFrame:
            dst = src.copy()
            dst["SMA25"] = dst["Close"].rolling(25).mean()
            dst["SMA50"] = dst["Close"].rolling(50).mean()
            dst["ROC200"] = dst["Close"].pct_change(200) * 100
            tr = pd.concat(
                [
                    dst["High"] - dst["Low"],
                    (dst["High"] - dst["Close"].shift()).abs(),
                    (dst["Low"] - dst["Close"].shift()).abs(),
                ],
                axis=1,
            ).max(axis=1)
            dst["ATR20"] = tr.rolling(20).mean()
            dst["DollarVolume20"] = (dst["Close"] * dst["Volume"]).rolling(20).mean()
            dst["filter"] = (dst["Low"] >= 5) & (dst["DollarVolume20"] > 50_000_000)
            dst["setup"] = dst["filter"] & (dst["SMA25"] > dst["SMA50"])
            return dst

        if cached is not None and not cached.empty:
            last_date = cached.index.max()
            new_rows = df[df.index > last_date]
            if new_rows.empty:
                result_df = cached
            else:
                context_start = last_date - pd.Timedelta(days=200)
                recompute_src = df[df.index >= context_start]
                recomputed = _calc_indicators(recompute_src)
                recomputed = recomputed[recomputed.index > last_date]
                result_df = pd.concat([cached, recomputed])
                try:
                    result_df.reset_index().to_feather(cache_path)
                except Exception:
                    pass
        else:
            result_df = _calc_indicators(df)
            try:
                result_df.reset_index().to_feather(cache_path)
            except Exception:
                pass

        if processed % batch_size == 0 or processed == total_symbols:
            batch_duration = time.time() - batch_start
            batch_size = batch_monitor.update(batch_duration)
            batch_start = time.time()

            if log_callback:
                elapsed = time.time() - start_time
                remaining = (elapsed / processed) * (total_symbols - processed)
                elapsed_min, elapsed_sec = divmod(int(elapsed), 60)
                remain_min, remain_sec = divmod(int(remaining), 60)
                joined_syms = ", ".join(symbol_buffer)
                try:
                    log_callback(
                        f"📊 指標計算: {processed}/{total_symbols} 件 完了",
                        f" | 経過: {elapsed_min}分{elapsed_sec}秒 / ",
                        f"残り: 約 {remain_min}分{remain_sec}秒\n",
                        f"銘柄: {joined_syms}",
                    )
                    log_callback(
                        f"⏱️ バッチ時間: {batch_duration:.2f}秒 | 次バッチサイズ: {batch_size}"
                    )
                except Exception:
                    pass
                symbol_buffer.clear()

    return result_dict


def get_total_days_system1(data_dict):
    """Return the total number of unique dates across prepared data."""
    all_dates = set()
    for df in data_dict.values():
        if df is None or df.empty:
            continue
        if "Date" in df.columns:
            date_series = pd.to_datetime(df["Date"]).dt.normalize()
        else:
            date_series = pd.to_datetime(df.index).normalize()
        all_dates.update(date_series)

    return len(sorted(all_dates))


def generate_roc200_ranking_system1(data_dict: dict, spy_df: pd.DataFrame, **kwargs):
    """Generate daily ROC200 ranking filtered by SPY trend."""
    all_signals = []
    for symbol, df in data_dict.items():
        if "setup" not in df.columns or df["setup"].sum() == 0:
            continue
        # 安全な日付演算: Index に直接加算せず Series で行う
        sig_df = df[df["setup"]][["ROC200", "ATR20", "Open"]].copy()
        sig_df["symbol"] = symbol
        # インデックスを正規化した日時に変換して列として保持
        idx_norm = pd.to_datetime(sig_df.index, errors="coerce").normalize()
        # reset_index 前に明示的に "Date" 列を持たせる（後続のマージ/ソートを安定化）
        sig_df["Date"] = idx_norm
        # entry_date は『翌営業日』に補正（単純な+1日ではなく、当該シンボルの次の取引日）
        try:
            idx = pd.DatetimeIndex(pd.to_datetime(df.index, errors="coerce").normalize())
            base_dates = pd.to_datetime(sig_df["Date"], errors="coerce").dt.normalize()
            pos = idx.searchsorted(base_dates, side="right")
            next_dates = pd.Series(pd.NaT, index=base_dates.index, dtype="datetime64[ns]")
            mask = (pos >= 0) & (pos < len(idx))
            if mask.any():
                # pos は ndarray なのでブールマスクで抽出後に代入
                next_vals = idx[pos[mask]]
                # pandas Index -> numpy -> pandas Series で整形
                next_dates.loc[mask] = pd.to_datetime(next_vals).tz_localize(None)
            sig_df["entry_date"] = next_dates
            # 翌営業日が無い行はドロップ
            sig_df = sig_df.dropna(subset=["entry_date"])  # type: ignore[arg-type]
        except Exception:
            # 失敗時は後段の当日フィルタで除外されるため、そのまま進める
            pass
        # インデックスは不要なので落としてから蓄積
        all_signals.append(sig_df.reset_index(drop=True))

    if not all_signals:
        return {}, pd.DataFrame()

    all_signals_df = pd.concat(all_signals, ignore_index=True)

    # SPY 側の列を整備（大小文字・date 列を堅牢化）
    if "SMA100" not in spy_df.columns:
        try:
            spy_df = spy_df.copy()
            # Close 列が lower な場合の補正
            if "Close" not in spy_df.columns and "close" in spy_df.columns:
                spy_df["Close"] = spy_df["close"]
            spy_df["SMA100"] = spy_df["Close"].rolling(100).mean()
        except Exception:
            pass

    spy = spy_df.copy()
    # date 列の生成
    date_col = None
    if "Date" in spy.columns:
        spy["date"] = pd.to_datetime(spy["Date"], errors="coerce")
        date_col = "date"
    elif "date" in spy.columns:
        spy["date"] = pd.to_datetime(spy["date"], errors="coerce")
        date_col = "date"
    else:
        # index が日時の場吁E
        try:
            idx = pd.to_datetime(spy.index, errors="coerce")
            try:
                cond_any = pd.notna(idx).any()
            except Exception:
                try:
                    cond_any = idx.notna().any()
                except Exception:
                    cond_any = False
            if cond_any:
                spy = spy.reset_index().rename(columns={spy.index.name or "index": "date"})
                spy["date"] = pd.to_datetime(spy["date"], errors="coerce")
                date_col = "date"
        except Exception:
            pass
    # 必要列に絞ってソート
    if date_col is None:
        # date が無空Eならマージ不能なので空返却
        return {}, pd.DataFrame()
    spy = spy[["date", "Close", "SMA100"]].sort_values("date")

    merged = pd.merge_asof(
        all_signals_df.sort_values("Date"),
        spy.rename(columns={"Close": "Close_SPY", "SMA100": "SMA100_SPY"}),
        left_on="Date",
        right_on="date",
    )
    merged = merged[merged["Close_SPY"] > merged["SMA100_SPY"]].copy()

    merged["entry_date_norm"] = merged["entry_date"].dt.normalize()
    grouped = merged.groupby("entry_date_norm")
    total_days = len(grouped)
    start_time = time.time()
    on_progress = kwargs.get("on_progress")
    on_log = kwargs.get("on_log")

    candidates_by_date = {}
    top_n = int(kwargs.get("top_n", 10))
    for i, (date, group) in enumerate(grouped, 1):
        top_df = group.nlargest(top_n, "ROC200")
        candidates_by_date[date] = top_df.to_dict("records")

        if on_progress:
            on_progress(i, total_days, start_time)
        if on_log and (i % 10 == 0 or i == total_days):
            elapsed = time.time() - start_time
            remain = elapsed / i * (total_days - i)
            on_log(
                f"📊 ROC200ランキング: {i}/{total_days} 日処理完了",
                f" | 経過: {int(elapsed // 60)}分{int(elapsed % 60)}秒",
                f" / 残り: 約 {int(remain // 60)}分{int(remain % 60)}秒",
            )

    return candidates_by_date, merged


__all__ = [
    "prepare_data_vectorized_system1",
    "get_total_days_system1",
    "generate_roc200_ranking_system1",
]
