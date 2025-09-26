"""System5 core logic (Long mean-reversion with high ADX).

本ファイルは当日シグナル高速化のために fast-path（既存指標の再利用）を行いつつ、
従来のキャッシュ差分再計算ロジックとの後方互換性を維持する。以前のリファクタ中に
誤ってプロセスプール処理ブロックがファイル先頭へ露出し SyntaxError を誘発していたため、
ここで本来の構造を復元する。
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, SMAIndicator
from ta.volatility import AverageTrueRange

from common.i18n import tr
from common.utils import (
    BatchSizeMonitor,
    describe_dtype,
    get_cached_data,
    is_today_run,
    resolve_batch_size,
)
from common.utils_spy import resolve_signal_entry_date

REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
MIN_ROWS = 150  # System3 が 150, System4 が 200。System5 は 150 日あれば十分。
DEFAULT_ATR_PCT_THRESHOLD = 0.025  # 2.5% を下限のボラティリティ指標閾値とする


def format_atr_pct_threshold_label() -> str:
    """UI / ログ用の閾値ラベルを一元化。scripts/today や today_signals で利用。"""
    return f"> {DEFAULT_ATR_PCT_THRESHOLD:.2%}"


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
        if pd.isna(idx).all():
            raise ValueError("invalid_date_index")
    except Exception:
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
    required_indicators = ["sma100", "atr10", "adx7", "rsi3"]
    missing_indicators = [col for col in required_indicators if col not in x.columns]
    if missing_indicators:
        raise RuntimeError(
            f"IMMEDIATE_STOP: System5 missing precomputed indicators {missing_indicators}. Daily signal execution must be stopped."
        )

    # Use precomputed volume averages
    if "dollarvolume50" in x.columns:
        x["DollarVolume50"] = x["dollarvolume50"]
    else:
        # Fallback: calculate if not precomputed
        vol = x["Volume"] if "Volume" in x.columns else pd.Series(0, index=x.index)
        x["DollarVolume50"] = (x["Close"] * vol).rolling(50).mean()

    if "avgvolume50" in x.columns:
        x["AvgVolume50"] = x["avgvolume50"]
    else:
        # Fallback: calculate if not precomputed
        vol = x["Volume"] if "Volume" in x.columns else pd.Series(0, index=x.index)
        x["AvgVolume50"] = vol.rolling(50).mean()

    # Use precomputed ATR ratio
    if "atr_ratio" in x.columns:
        x["ATR_Pct"] = x["atr_ratio"]
    else:
        # Fallback: calculate from precomputed ATR10
        x["ATR_Pct"] = x["atr10"].div(x["Close"].replace(0, pd.NA))

    # System5 filtering logic using precomputed indicators (lowercase)
    x["filter"] = (
        (x["AvgVolume50"] > 500_000)
        & (x["DollarVolume50"] > 2_500_000)
        & (x["ATR_Pct"] > DEFAULT_ATR_PCT_THRESHOLD)
    )
    x["setup"] = (
        x["filter"] & (x["Close"] > x["sma100"] + x["atr10"]) & (x["adx7"] > 55) & (x["rsi3"] < 50)
    ).astype(int)
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
    except Exception:
        return symbol, None

    # Early exit: check required precomputed indicators exist
    required_indicators = ["sma100", "atr10", "adx7", "rsi3"]
    missing_indicators = [col for col in required_indicators if col not in prepared.columns]
    if missing_indicators:
        raise RuntimeError(
            f"IMMEDIATE_STOP: System5 missing precomputed indicators {missing_indicators} for {symbol}. Daily signal execution must be stopped."
        )

    try:
        res = _compute_indicators_frame(prepared)
    except Exception:
        return symbol, None

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


def prepare_data_vectorized_system5(
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
    cache_dir = "data_cache/indicators_system5_cache"
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
        pool_buffer: list[str] = []  # collect symbols for periodic progress logging
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_compute_indicators, s): s for s in symbols}
            for i, fut in enumerate(as_completed(futures), 1):
                sym, df = fut.result()
                if df is not None:
                    result_dict[sym] = df
                    pool_buffer.append(sym)
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
                        "📊 indicators progress: {done}/{total} | elapsed: {em}m{es}s / "
                        "remain: ~{rm}m{rs}s",
                        done=i,
                        total=total,
                        em=em,
                        es=es,
                        rm=rm,
                        rs=rs,
                    )
                    if pool_buffer:
                        today_mode = is_today_run()
                        if not today_mode:
                            sample = ", ".join(pool_buffer[:10])
                            more = len(pool_buffer) - len(pool_buffer[:10])
                            if more > 0:
                                sample = f"{sample}, ...(+{more} more)"
                            msg += "\n" + tr("symbols: {names}", names=sample)
                    try:
                        log_callback(msg)
                    except Exception:
                        pass
                    pool_buffer.clear()
        return result_dict

    total = len(raw_data_dict)
    if batch_size is None:
        try:
            from config.settings import get_settings

            batch_size = get_settings(create_dirs=False).data.batch_size
        except Exception:
            batch_size = 100
        batch_size = resolve_batch_size(total, batch_size)
    processed, skipped = 0, 0
    # スキップ理由の内訳カウンタ
    skipped_insufficient_rows = 0
    skipped_missing_cols = 0
    skipped_calc_errors = 0
    missing_cols_examples: dict[str, int] = {}
    buffer: list[str] = []  # collect symbols in current batch
    start_time = time.time()
    batch_monitor = BatchSizeMonitor(batch_size)
    batch_start = time.time()

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
            if not today_mode and buffer:
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
            if reason_raw.startswith("missing_cols:"):
                skipped_missing_cols += 1
                missing = reason_raw.split(":", 1)[1]
                for m in missing.split(","):
                    m = m.strip()
                    if m:
                        missing_cols_examples[m] = missing_cols_examples.get(m, 0) + 1
                reason = f"missing_cols:{missing}"
            elif "insufficient" in reason_raw:
                skipped_insufficient_rows += 1
                reason = "insufficient_rows"
            else:
                skipped_calc_errors += 1
                reason = "calc_error"
            if skip_callback:
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

            # Check required precomputed indicators - early exit
            required_indicators = ["sma100", "atr10", "adx7", "rsi3"]
            missing_indicators = [col for col in required_indicators if col not in x.columns]
            if missing_indicators:
                raise RuntimeError(
                    f"IMMEDIATE_STOP: System5 missing precomputed indicators {missing_indicators} for {sym}. Daily signal execution must be stopped."
                )

            # Use precomputed volume averages
            if "dollarvolume50" in x.columns:
                x["DollarVolume50"] = x["dollarvolume50"]
            elif "DollarVolume50" not in x.columns:
                try:
                    vol = x["Volume"] if "Volume" in x.columns else pd.Series(0, index=x.index)
                    x["DollarVolume50"] = (x["Close"] * vol).rolling(50).mean()
                except Exception:
                    pass

            if "avgvolume50" in x.columns:
                x["AvgVolume50"] = x["avgvolume50"]
            elif "AvgVolume50" not in x.columns:
                try:
                    vol = x["Volume"] if "Volume" in x.columns else pd.Series(0, index=x.index)
                    x["AvgVolume50"] = vol.rolling(50).mean()
                except Exception:
                    pass

            # Use precomputed ATR ratio
            if "atr_ratio" in x.columns:
                x["ATR_Pct"] = x["atr_ratio"]
            elif "ATR_Pct" not in x.columns:
                try:
                    close_num = pd.to_numeric(x["Close"], errors="coerce")
                    atr10_ser = x["atr10"]
                    x["ATR_Pct"] = atr10_ser.div(close_num.replace(0, pd.NA))
                except Exception:
                    pass

            # System5 filtering logic using precomputed indicators (lowercase)
            x["filter"] = (
                (x["AvgVolume50"] > 500_000)
                & (x["DollarVolume50"] > 2_500_000)
                & (x["ATR_Pct"] > DEFAULT_ATR_PCT_THRESHOLD)
            )
            close_num = pd.to_numeric(x["Close"], errors="coerce")
            x["setup"] = (
                x["filter"]
                & (close_num > x["sma100"] + x["atr10"])
                & (x["adx7"] > 55)
                & (x["rsi3"] < 50)
            ).astype(int)

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
            skipped_calc_errors += 1
            # insufficient rows の ValueError を分類
            try:
                msg = str(e).lower()
                reason = "insufficient_rows" if "insufficient" in msg else "calc_error"
            except Exception:
                reason = "calc_error"
            if reason == "insufficient_rows":
                skipped_insufficient_rows += 1
                skipped_calc_errors -= 1  # 調整
            if skip_callback:
                try:
                    skip_callback(sym, reason)
                except Exception:
                    try:
                        skip_callback(f"{sym}: {reason}")
                    except Exception:
                        pass
            _on_symbol_done()
        except Exception:
            skipped += 1
            skipped_calc_errors += 1
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
            # 追加の内訳（多い順に表示、ノイズ防止のため上位のみ）
            if skipped_insufficient_rows:
                try:
                    log_callback(f"  ├─ 行数不足(<100): {skipped_insufficient_rows} 件")
                except Exception:
                    pass
            if skipped_missing_cols:
                try:
                    # 欠落列の上位3件
                    top_missing = sorted(
                        missing_cols_examples.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:3]
                    details = ", ".join([f"{k}:{v}" for k, v in top_missing]) if top_missing else ""
                    log_callback(
                        f"  ├─ 必須列欠落: {skipped_missing_cols} 件"
                        + (f" ({details})" if details else "")
                    )
                except Exception:
                    pass
            if skipped_calc_errors:
                try:
                    log_callback(f"  └─ 計算エラー: {skipped_calc_errors} 件")
                except Exception:
                    pass
        except Exception:
            pass
    return result_dict


def generate_candidates_system5(
    prepared_dict: dict[str, pd.DataFrame],
    *,
    top_n: int = 10,
    progress_callback=None,
    log_callback=None,
    batch_size: int | None = None,
) -> tuple[dict, pd.DataFrame | None]:
    candidates_by_date: dict[pd.Timestamp, list] = {}
    total = len(prepared_dict)
    if batch_size is None:
        try:
            from config.settings import get_settings

            batch_size = get_settings(create_dirs=False).data.batch_size
        except Exception:
            batch_size = 100
        batch_size = resolve_batch_size(total, batch_size)
    processed, skipped = 0, 0
    buffer: list[str] = []
    start_time = time.time()

    for sym, df in prepared_dict.items():
        try:
            if "setup" not in df.columns or not df["setup"].any():
                skipped += 1
                continue
            setup_days = df[df["setup"] == 1]
            if setup_days.empty:
                skipped += 1
                continue
            for date, row in setup_days.iterrows():
                ts = pd.to_datetime(pd.Index([date]))[0]
                # last_price（直近終値）を取得
                last_price = None
                if "Close" in df.columns and not df["Close"].empty:
                    last_price = df["Close"].iloc[-1]
                # 翌営業日に補正
                entry_date = resolve_signal_entry_date(ts)
                if pd.isna(entry_date):
                    continue
                rec = {
                    "symbol": sym,
                    "entry_date": entry_date,
                    "ADX7": row["ADX7"],
                    "ATR10": row["ATR10"],
                    "entry_price": last_price,
                }
                candidates_by_date.setdefault(entry_date, []).append(rec)
        except Exception:
            skipped += 1

        processed += 1
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

    limit_n = int(top_n)
    for date in list(candidates_by_date.keys()):
        rows = candidates_by_date.get(date, [])
        if not rows:
            candidates_by_date[date] = []
            continue
        df = pd.DataFrame(rows)
        if df.empty:
            candidates_by_date[date] = []
            continue
        df = df.sort_values("ADX7", ascending=False)
        total = len(df)
        df.loc[:, "rank"] = list(range(1, total + 1))
        df.loc[:, "rank_total"] = total
        limited = df.head(limit_n)
        candidates_by_date[date] = limited.to_dict("records")

    if skipped > 0 and log_callback:
        try:
            log_callback(f"⚠️ 候補抽出中にスキップ: {skipped} 件")
        except Exception:
            pass
    return candidates_by_date, None


def get_total_days_system5(data_dict: dict[str, pd.DataFrame]) -> int:
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
    "prepare_data_vectorized_system5",
    "generate_candidates_system5",
    "get_total_days_system5",
]
