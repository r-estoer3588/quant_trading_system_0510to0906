"""System3 core logic (Long mean-reversion)."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time

import pandas as pd

from common.i18n import tr
from common.utils import (
    BatchSizeMonitor,
    describe_dtype,
    get_cached_data,
    is_today_run,
    resolve_batch_size,
)
from common.utils_spy import resolve_signal_entry_date

# Trading thresholds - Default values for business rules
DEFAULT_ATR_RATIO_THRESHOLD = 0.05  # 5% ATR ratio threshold for filtering

REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
MIN_ROWS = 150


def _rename_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy(deep=False)
    rename_map = {}
    for low, up in (
        ("open", "Open"),
        ("high", "High"),
        ("low", "Low"),
        ("close", "Close"),
        ("volume", "Volume"),
    ):
        if low in x.columns and up not in x.columns:
            rename_map[low] = up
    if rename_map:
        x = x.rename(columns=rename_map)
    return x


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" in df.columns:
        idx = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    elif "date" in df.columns:
        idx = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    else:
        idx = pd.to_datetime(df.index, errors="coerce").normalize()
    if idx is None:
        raise ValueError("invalid_date_index")
    try:
        # pandas >=1.5: pd.isna for Index returns ndarray-like; .all() is valid
        if pd.isna(idx).all():
            raise ValueError("invalid_date_index")
    except Exception:
        # Defensive: if idx has unexpected type
        pass
    x = df.copy()
    x.index = pd.Index(idx)
    x.index.name = "Date"
    x = x[~x.index.isna()]
    try:
        x = x.sort_index()
    except Exception:
        pass
    try:
        if getattr(x.index, "has_duplicates", False):
            x = x[~x.index.duplicated(keep="last")]
    except Exception:
        pass
    return x


def _prepare_source_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("empty_frame")
    x = _rename_ohlcv(df)
    missing = [c for c in REQUIRED_COLUMNS if c not in x.columns]
    if missing:
        raise ValueError(f"missing_cols:{','.join(missing)}")
    x = _normalize_index(x)
    for col in REQUIRED_COLUMNS:
        if col in x.columns:
            x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna(subset=[c for c in ("High", "Low", "Close") if c in x.columns])
    if len(x) < MIN_ROWS:
        raise ValueError("insufficient_rows")
    return x


def _compute_indicators_frame(df: pd.DataFrame) -> pd.DataFrame:
    """プリコンピューテッド指標版：計算除去、早期終了追加"""
    x = df.copy()

    # Required precomputed indicators (lowercase)
    required_indicators = ["sma150", "atr10", "atr_ratio"]
    missing_indicators = [col for col in required_indicators if col not in x.columns]
    if missing_indicators:
        raise RuntimeError(
            f"IMMEDIATE_STOP: System3 missing precomputed indicators {missing_indicators}. Daily signal execution must be stopped."
        )

    # Calculate derived indicators that cannot be precomputed
    # Only calculate Drop3D if not already present (optimization for precomputed data)
    if "Drop3D" not in x.columns:
        x["Drop3D"] = -(x["Close"].pct_change(3))

    # Use precomputed volume average (try dollarvolume50 first, fallback to AvgVolume50)
    if "dollarvolume50" in x.columns:
        volume_condition = x["dollarvolume50"] >= 1_000_000
    elif "AvgVolume50" in x.columns:  # Legacy fallback
        volume_condition = x["AvgVolume50"] >= 1_000_000
    else:
        # Calculate if neither precomputed version exists
        x["AvgVolume50"] = x["Volume"].rolling(50).mean()
        volume_condition = x["AvgVolume50"] >= 1_000_000

    # Use precomputed indicators with lowercase names
    cond_price = x["Low"] >= 1
    cond_volume = volume_condition
    cond_atr = x["atr_ratio"] >= DEFAULT_ATR_RATIO_THRESHOLD
    x["filter"] = cond_price & cond_volume & cond_atr
    cond_close = x["Close"] > x["sma150"]
    cond_drop = x["Drop3D"] >= 0.125
    cond_setup = x["filter"] & cond_close & cond_drop
    x["setup"] = cond_setup.astype(int)
    return x


def _compute_indicators(symbol: str) -> tuple[str, pd.DataFrame | None]:
    """プリコンピューテッド指標版：計算除去、早期終了追加"""
    df = get_cached_data(symbol)
    if df is None or df.empty:
        return symbol, None

    # 子プロセスから親へ簡易進捗を送る（存在すれば）
    try:
        q = globals().get("_PROGRESS_QUEUE")
        if q is not None:
            try:
                q.put((symbol, 0))
            except Exception:
                pass
    except Exception:
        pass

    try:
        prepared = _prepare_source_frame(df)
    except ValueError:
        return symbol, None

    # Early exit: check required precomputed indicators exist
    required_indicators = ["sma150", "atr10", "atr_ratio"]
    missing_indicators = [col for col in required_indicators if col not in prepared.columns]
    if missing_indicators:
        raise RuntimeError(
            f"IMMEDIATE_STOP: System3 missing precomputed indicators {missing_indicators} for {symbol}. Daily signal execution must be stopped."
        )

    try:
        res = _compute_indicators_frame(prepared)
    except Exception:
        return symbol, None

    # 完了を親に伝える
    try:
        q = globals().get("_PROGRESS_QUEUE")
        if q is not None:
            try:
                q.put((symbol, 100))
            except Exception:
                pass
    except Exception:
        pass

    return symbol, res


def prepare_data_vectorized_system3(
    raw_data_dict: dict[str, pd.DataFrame] | None,
    *,
    progress_callback=None,
    log_callback=None,
    batch_size: int | None = None,
    reuse_indicators: bool = True,
    symbols: list[str] | None = None,
    use_process_pool: bool = False,
    max_workers: int | None = None,
    skip_callback=None,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    cache_dir = "data_cache/indicators_system3_cache"
    os.makedirs(cache_dir, exist_ok=True)
    result_dict: dict[str, pd.DataFrame] = {}
    raw_data_dict = raw_data_dict or {}
    if use_process_pool:
        if symbols is None:
            symbols = list(raw_data_dict.keys())
        total = len(symbols)
        if batch_size is None:
            try:
                from config.settings import get_settings

                batch_size = get_settings(create_dirs=False).data.batch_size
            except Exception:
                batch_size = 100
            batch_size = resolve_batch_size(total, batch_size)
        buffer: list[str] = []
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_compute_indicators, s): s for s in symbols}
            for i, fut in enumerate(as_completed(futures), 1):
                sym, df = fut.result()
                if df is not None:
                    result_dict[sym] = df
                    buffer.append(sym)
                if progress_callback:
                    try:
                        progress_callback(i, total)
                    except Exception:
                        pass
                if (i % int(batch_size) == 0 or i == total) and log_callback:
                    elapsed = time.time() - start_time
                    remain = (elapsed / i) * (total - i) if i else 0
                    em, es = divmod(int(elapsed), 60)
                    rm, rs = divmod(int(remain), 60)
                    msg = tr(
                        "📊 indicators progress: {done}/{total} | elapsed: {em}m{es}s / remain: ~{rm}m{rs}s",
                        done=i,
                        total=total,
                        em=em,
                        es=es,
                        rm=rm,
                        rs=rs,
                    )
                    if buffer:
                        # Shorten symbol list when running today's signals to avoid huge logs
                        today_mode = is_today_run()
                        # 当日モードでは銘柄リスト出力はスキップ
                        if not today_mode:
                            if today_mode:
                                sample = ", ".join(buffer[:10])
                                more = len(buffer) - len(buffer[:10])
                                if more > 0:
                                    sample = f"{sample}, ...(+{more} more)"
                                msg += "\n" + tr("symbols: {names}", names=sample)
                            else:
                                msg += "\n" + tr("symbols: {names}", names=", ".join(buffer))
                    try:
                        log_callback(msg)
                    except Exception:
                        pass
                    buffer.clear()
        return result_dict

    total = len(raw_data_dict)
    if batch_size is None:
        try:
            from config.settings import get_settings

            batch_size = get_settings(create_dirs=False).data.batch_size
        except Exception:
            batch_size = 100
        batch_size = resolve_batch_size(total, batch_size)
    start_time = time.time()
    batch_monitor = BatchSizeMonitor(batch_size)
    batch_start = time.time()
    processed, skipped = 0, 0
    buffer: list[str] = []

    def _on_symbol_done(symbol: str | None = None, *, include_in_buffer: bool = False) -> None:
        nonlocal processed, batch_size, batch_start
        if include_in_buffer and symbol:
            buffer.append(symbol)
        processed += 1
        if progress_callback:
            try:
                progress_callback(processed, total)
            except Exception:
                pass

    if (processed % int(batch_size) == 0 or processed == total) and log_callback:
        elapsed = time.time() - start_time
        remain = (elapsed / processed) * (total - processed) if processed else 0
        em, es = divmod(int(elapsed), 60)
        rm, rs = divmod(int(remain), 60)
        msg = tr(
            "📊 indicators progress: {done}/{total} | elapsed: {em}m{es}s / remain: ~{rm}m{rs}s",
            done=processed,
            total=total,
            em=em,
            es=es,
            rm=rm,
            rs=rs,
        )
        if buffer:
            # 当日モードでは銘柄リスト出力はスキップ
            today_mode = is_today_run()
            if not today_mode:
                if today_mode:
                    sample = ", ".join(buffer[:10])
                    more = len(buffer) - len(buffer[:10])
                    if more > 0:
                        sample = f"{sample}, ...(+{more} more)"
                    msg += "\n" + tr("symbols: {names}", names=sample)
                else:
                    msg += "\n" + tr("symbols: {names}", names=", ".join(buffer))
        batch_duration = time.time() - batch_start
        batch_size = batch_monitor.update(batch_duration)
        batch_start = time.time()
        try:
            log_callback(msg)
            log_callback(
                tr(
                    "⏱️ batch time: {sec:.2f}s | next batch size: {size}",
                    sec=batch_duration,
                    size=batch_size,
                )
            )
        except Exception:
            pass
        buffer.clear()

    for sym, df in raw_data_dict.items():
        df = _rename_ohlcv(df)

        # --- 健全性チェック: NaN・型不一致・異常値 ---
        try:
            base_cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
            if base_cols:
                base_nan_rate = df[base_cols].isnull().mean().mean()
            else:
                base_nan_rate = df.isnull().mean().mean() if df.size > 0 else 0.0
            if base_nan_rate >= 0.45:
                msg = f"⚠️ {sym} cache: OHLCV欠損率高 ({base_nan_rate:.2%})"
                if log_callback:
                    log_callback(msg)
                if skip_callback:
                    skip_callback(sym, msg)
                skipped += 1
                _on_symbol_done()
                continue
            if base_nan_rate > 0.20 and log_callback:
                log_callback(f"⚠️ {sym} cache: OHLCV欠損率注意 ({base_nan_rate:.2%})")

            indicator_cols = [
                c
                for c in df.columns
                if c not in base_cols
                and str(c).lower() not in {"date", "symbol"}
                and pd.api.types.is_numeric_dtype(df[c])
            ]
            if indicator_cols:
                indicator_nan_rate = df[indicator_cols].isnull().mean().mean()
                if indicator_nan_rate > 0.60 and log_callback:
                    log_callback(f"⚠️ {sym} cache: 指標NaN率高 ({indicator_nan_rate:.2%})")

            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col in df.columns:
                    series_like = df[col]
                    if not pd.api.types.is_numeric_dtype(series_like):
                        dtype_repr = describe_dtype(series_like)
                        msg = f"⚠️ {sym} cache: {col}型不一致 ({dtype_repr})"
                        if log_callback:
                            log_callback(msg)
                        if skip_callback:
                            skip_callback(sym, msg)
            for col in ["Close", "High", "Low"]:
                if col in df.columns:
                    vals = pd.to_numeric(df[col], errors="coerce")
                    if (vals <= 0).all():
                        msg = f"⚠️ {sym} cache: {col}全て非正値"
                        if log_callback:
                            log_callback(msg)
                        if skip_callback:
                            skip_callback(sym, msg)
        except Exception as e:
            msg = f"⚠️ {sym} cache: 健全性チェック失敗 ({e})"
            if log_callback:
                log_callback(msg)
            if skip_callback:
                skip_callback(sym, msg)
            skipped += 1
            _on_symbol_done()
            continue

        cache_path = os.path.join(cache_dir, f"{sym}.feather")
        cached: pd.DataFrame | None = None
        if reuse_indicators and os.path.exists(cache_path):
            try:
                cached = pd.read_feather(cache_path)
                cached["Date"] = pd.to_datetime(cached["Date"]).dt.normalize()
                cached.set_index("Date", inplace=True)
            except Exception:
                cached = None

        try:
            prepared_df = _prepare_source_frame(df)
        except ValueError as exc:
            skipped += 1
            reason_raw = str(exc)
            if skip_callback:
                reason = "calc_error"
                if reason_raw.startswith("missing_cols:"):
                    reason = reason_raw
                elif "insufficient" in reason_raw:
                    reason = "insufficient_rows"
                try:
                    skip_callback(sym, reason)
                except Exception:
                    try:
                        skip_callback(f"{sym}: {reason}")
                    except Exception:
                        pass
            _on_symbol_done()
            continue

        # プリコンピューテッド指標のみ使用：再計算を完全除去
        try:
            x = prepared_df.copy(deep=False)

            # Check required precomputed indicators early exit
            required_indicators = ["sma150", "atr10", "atr_ratio"]
            missing_indicators = [col for col in required_indicators if col not in x.columns]
            if missing_indicators:
                raise RuntimeError(
                    f"IMMEDIATE_STOP: System3 missing precomputed indicators {missing_indicators} for {sym}. Daily signal execution must be stopped."
                )

            # Only calculate non-precomputable derived indicators
            if "Drop3D" not in x.columns:
                if "Return_3D" in x.columns:
                    try:
                        x["Drop3D"] = -(pd.to_numeric(x["Return_3D"], errors="coerce"))
                    except Exception:
                        close_num = pd.to_numeric(x["Close"], errors="coerce")
                        x["Drop3D"] = -(close_num.pct_change(3))
                else:
                    close_num = pd.to_numeric(x["Close"], errors="coerce")
                    x["Drop3D"] = -(close_num.pct_change(3))

            # System3 filtering logic using precomputed indicators (lowercase)
            cond_price = x["Low"] >= 1

            # Use precomputed volume average
            if "dollarvolume50" in x.columns:
                cond_volume = x["dollarvolume50"] >= 1_000_000
            elif "AvgVolume50" in x.columns:  # Fallback
                cond_volume = x["AvgVolume50"] >= 1_000_000
            else:
                # Last resort: calculate if neither exists
                x["AvgVolume50"] = x["Volume"].rolling(50).mean()
                cond_volume = x["AvgVolume50"] >= 1_000_000

            cond_atr = x["atr_ratio"] >= DEFAULT_ATR_RATIO_THRESHOLD
            x["filter"] = cond_price & cond_volume & cond_atr
            cond_close = x["Close"] > x["sma150"]
            cond_drop = x["Drop3D"] >= 0.125
            x["setup"] = (x["filter"] & cond_close & cond_drop).astype(int)

            result_df = x
            try:
                result_df.reset_index().to_feather(cache_path)
            except Exception:
                pass
            result_dict[sym] = result_df
            _on_symbol_done(sym, include_in_buffer=True)
            continue
        except ValueError as e:
            skipped += 1
            if skip_callback:
                try:
                    msg = str(e).lower()
                    reason = "insufficient_rows" if "insufficient" in msg else "calc_error"
                    skip_callback(sym, reason)
                except Exception:
                    try:
                        skip_callback(f"{sym}: insufficient_rows")
                    except Exception:
                        pass
            _on_symbol_done()
        except Exception:
            skipped += 1
            if skip_callback:
                try:
                    skip_callback(sym, "calc_error")
                except Exception:
                    try:
                        skip_callback(f"{sym}: calc_error")
                    except Exception:
                        pass
            _on_symbol_done()

    if skipped > 0 and log_callback:
        try:
            log_callback(f"⚠️ データ不足/計算失敗でスキップ: {skipped} 件")
        except Exception:
            pass

    return result_dict


def generate_candidates_system3(
    prepared_dict: dict[str, pd.DataFrame],
    *,
    top_n: int = 10,
    progress_callback=None,
    log_callback=None,
    batch_size: int | None = None,
) -> tuple[dict, pd.DataFrame | None]:
    all_signals = []
    total = len(prepared_dict)
    if batch_size is None:
        try:
            from config.settings import get_settings

            batch_size = get_settings(create_dirs=False).data.batch_size
        except Exception:
            batch_size = 100
        batch_size = resolve_batch_size(total, batch_size)
    processed, skipped = 0, 0
    buffer = []
    start_time = time.time()

    for sym, df in prepared_dict.items():
        processed += 1
        if "setup" not in df.columns or not df["setup"].any():
            skipped += 1
            continue
        setup_df = df[df["setup"] == 1].copy()
        setup_df["symbol"] = sym
        # last_price（直近終値）を取得
        last_price = None
        if "Close" in df.columns and not df["Close"].empty:
            last_price = df["Close"].iloc[-1]
        setup_df["entry_price"] = last_price
        base_dates = pd.to_datetime(setup_df.index, errors="coerce").to_series(index=setup_df.index)
        setup_df["entry_date"] = base_dates.map(resolve_signal_entry_date)
        # subset is List[str]; ensure typing consistent for mypy
        setup_df = setup_df.dropna(subset=["entry_date"])
        setup_df = setup_df[["symbol", "entry_date", "Drop3D", "ATR10", "entry_price"]]
        all_signals.append(setup_df)
        buffer.append(sym)

        if progress_callback:
            try:
                progress_callback(processed, total)
            except Exception:
                pass
        if (processed % int(batch_size) == 0 or processed == total) and log_callback:
            elapsed = time.time() - start_time
            remain = (elapsed / processed) * (total - processed) if processed else 0
            em, es = divmod(int(elapsed), 60)
            rm, rs = divmod(int(remain), 60)
            msg = tr(
                "📊 candidates progress: {done}/{total} | elapsed: {em}m{es}s / remain: ~{rm}m{rs}s",
                done=processed,
                total=total,
                em=em,
                es=es,
                rm=rm,
                rs=rs,
            )
            if buffer:
                msg += "\n" + tr("symbols: {names}", names=", ".join(buffer))
            try:
                log_callback(msg)
            except Exception:
                pass
            buffer.clear()

    if log_callback:
        try:
            log_callback(f"✅ 候補銘柄: {len(all_signals)} 件 / ⚠️ 候補対象外銘柄: {skipped} 件")
        except Exception:
            pass

    if not all_signals:
        return {}, None

    all_df = pd.concat(all_signals)
    candidates_by_date = {}
    for date, group in all_df.groupby("entry_date"):
        ranked = group.sort_values("Drop3D", ascending=False).copy()
        total = len(ranked)
        if total == 0:
            candidates_by_date[date] = []
            continue
        ranked.loc[:, "rank"] = list(range(1, total + 1))
        ranked.loc[:, "rank_total"] = total
        limited = ranked.head(int(top_n))
        candidates_by_date[date] = limited.to_dict("records")
    return candidates_by_date, None


def get_total_days_system3(data_dict: dict[str, pd.DataFrame]) -> int:
    all_dates = set()
    for df in data_dict.values():
        if df is None or df.empty:
            continue
        if "Date" in df.columns:
            dates = pd.to_datetime(df["Date"]).dt.normalize()
        elif "date" in df.columns:
            dates = pd.to_datetime(df["date"]).dt.normalize()
        else:
            dates = pd.to_datetime(df.index).normalize()
        all_dates.update(dates)
    return len(all_dates)


__all__ = [
    "prepare_data_vectorized_system3",
    "generate_candidates_system3",
    "get_total_days_system3",
]
