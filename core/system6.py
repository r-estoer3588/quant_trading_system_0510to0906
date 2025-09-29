"""System6 core logic (Short mean-reversion momentum burst)."""

import time

import pandas as pd
from ta.volatility import AverageTrueRange

from common.batch_processing import process_symbols_batch
from common.i18n import tr
from common.structured_logging import MetricsCollector
from common.utils import resolve_batch_size
from common.utils_spy import resolve_signal_entry_date

# System6 configuration constants
MIN_PRICE = 5.0  # 最低価格フィルター（ドル）
MIN_DOLLAR_VOLUME_50 = 10_000_000  # 最低ドルボリューム50日平均（ドル）

SYSTEM6_BASE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
SYSTEM6_FEATURE_COLUMNS = [
    "atr10",
    "dollarvolume50",
    "return_6d",
    "UpTwoDays",
    "filter",
    "setup",
]
SYSTEM6_ALL_COLUMNS = SYSTEM6_BASE_COLUMNS + SYSTEM6_FEATURE_COLUMNS
SYSTEM6_NUMERIC_COLUMNS = ["atr10", "dollarvolume50", "return_6d"]


def _compute_indicators_from_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in SYSTEM6_BASE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")
    x = df.loc[:, SYSTEM6_BASE_COLUMNS].copy()
    if len(x) < 50:
        raise ValueError("insufficient rows")

    # フォールバック使用回数を記録するためのMetricsCollector
    metrics = MetricsCollector()

    try:
        # 🚀 プリコンピューテッド指標を使用（すべての指標を最適化）

        # ATR10
        if "ATR10" in df.columns:
            x["atr10"] = df["ATR10"]
        elif "atr10" in df.columns:
            x["atr10"] = df["atr10"]
        else:
            # フォールバック（通常は実行されない）
            metrics.record_metric("system6_fallback_atr10", 1, "count")
            x["atr10"] = AverageTrueRange(
                x["High"], x["Low"], x["Close"], window=10
            ).average_true_range()

        # DollarVolume50
        if "DollarVolume50" in df.columns:
            x["dollarvolume50"] = df["DollarVolume50"]
        elif "dollarvolume50" in df.columns:
            x["dollarvolume50"] = df["dollarvolume50"]
        else:
            # フォールバック（通常は実行されない）
            metrics.record_metric("system6_fallback_dollarvolume50", 1, "count")
            x["dollarvolume50"] = (x["Close"] * x["Volume"]).rolling(50).mean()

        # Return_6D
        if "Return_6D" in df.columns:
            x["return_6d"] = df["Return_6D"]
        elif "return_6d" in df.columns:
            x["return_6d"] = df["return_6d"]
        else:
            # フォールバック（通常は実行されない）
            metrics.record_metric("system6_fallback_return_6d", 1, "count")
            x["return_6d"] = x["Close"].pct_change(6)

        # UpTwoDays
        if "UpTwoDays" in df.columns:
            x["UpTwoDays"] = df["UpTwoDays"]
        elif "uptwodays" in df.columns:
            x["UpTwoDays"] = df["uptwodays"]
        else:
            # フォールバック（通常は実行されない）
            metrics.record_metric("system6_fallback_uptwodays", 1, "count")
            x["UpTwoDays"] = (x["Close"] > x["Close"].shift(1)) & (
                x["Close"].shift(1) > x["Close"].shift(2)
            )

        # フィルターとセットアップ条件（軽量な論理演算）
        x["filter"] = (x["Low"] >= MIN_PRICE) & (x["dollarvolume50"] > MIN_DOLLAR_VOLUME_50)
        x["setup"] = x["filter"] & (x["return_6d"] > 0.20) & x["UpTwoDays"]

    except Exception as exc:
        raise ValueError(f"calc_error: {type(exc).__name__}: {exc}") from exc

    # データクリーニングと最終的なソート・重複除去（一箇所に統合）
    x = x.dropna(subset=SYSTEM6_NUMERIC_COLUMNS)
    if x.empty:
        raise ValueError("insufficient rows")
    x = x.loc[~x.index.duplicated()].sort_index()
    x.index = pd.to_datetime(x.index).tz_localize(None)
    x.index.name = "Date"
    return x


def prepare_data_vectorized_system6(
    raw_data_dict: dict[str, pd.DataFrame] | None,
    *,
    progress_callback=None,
    log_callback=None,
    skip_callback=None,
    batch_size: int | None = None,
    use_process_pool: bool = False,
    max_workers: int | None = None,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """System6 data preparation using standard batch processing pattern"""

    if not raw_data_dict:
        if log_callback:
            log_callback("System6: No raw data provided, returning empty dict")
        return {}

    target_symbols = list(raw_data_dict.keys())

    if log_callback:
        log_callback(f"System6: Starting processing for {len(target_symbols)} symbols")

    # Create a closure to pass raw_data_dict to the compute function
    def _compute_indicators_with_data(symbol: str) -> tuple[str, pd.DataFrame | None]:
        """Indicator calculation function that uses provided raw data"""
        df = raw_data_dict.get(symbol)
        if df is None or df.empty:
            return symbol, None

        try:
            prepared = _compute_indicators_from_frame(df)
            return symbol, prepared
        except Exception:
            return symbol, None

    # Execute batch processing using standard pattern
    results, error_symbols = process_symbols_batch(
        target_symbols,
        _compute_indicators_with_data,
        batch_size=batch_size,
        use_process_pool=use_process_pool,
        max_workers=max_workers,
        progress_callback=progress_callback,
        log_callback=log_callback,
        skip_callback=skip_callback,
        system_name="System6",
    )

    return results


def generate_candidates_system6(
    prepared_dict: dict[str, pd.DataFrame],
    *,
    top_n: int = 10,
    progress_callback=None,
    log_callback=None,
    skip_callback=None,
    batch_size: int | None = None,
) -> tuple[dict, pd.DataFrame | None]:
    candidates_by_date: dict[pd.Timestamp, list] = {}
    total = len(prepared_dict)

    # MetricsCollector初期化
    metrics = MetricsCollector()

    if batch_size is None:
        try:
            from config.settings import get_settings

            batch_size = get_settings(create_dirs=False).data.batch_size
        except Exception:
            batch_size = 100
        batch_size = resolve_batch_size(total, batch_size)
    start_time = time.time()
    batch_start = time.time()
    processed, skipped = 0, 0
    skipped_missing_cols = 0
    filter_passed = 0  # フィルター条件通過数
    setup_passed = 0  # セットアップ条件通過数
    buffer: list[str] = []

    for sym, df in prepared_dict.items():
        # featherキャッシュの健全性チェック
        if df is None or df.empty:
            if log_callback:
                log_callback(f"[警告] {sym} のデータが空です（featherキャッシュ欠損）")
            skipped += 1
            continue
        missing_cols = [c for c in SYSTEM6_ALL_COLUMNS if c not in df.columns]
        if missing_cols:
            if log_callback:
                log_callback(
                    f"[警告] {sym} のデータに必須列が不足しています: {', '.join(missing_cols)}"
                )
            skipped += 1
            skipped_missing_cols += 1
            continue
        if df[SYSTEM6_NUMERIC_COLUMNS].isnull().any().any():
            if log_callback:
                log_callback(
                    f"[警告] {sym} のデータにNaNが含まれています（featherキャッシュ不完全）"
                )

        # last_price（直近終値）を取得
        last_price = None
        if "Close" in df.columns and not df["Close"].empty:
            last_price = df["Close"].iloc[-1]

        # 統計計算：フィルター通過数とセットアップ通過数をカウント
        if "filter" in df.columns:
            filter_passed += df["filter"].sum()
        if "setup" in df.columns:
            setup_passed += df["setup"].sum()

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
                # 翌営業日に補正
                entry_date = resolve_signal_entry_date(ts)
                if pd.isna(entry_date):
                    continue
                rec = {
                    "symbol": sym,
                    "entry_date": entry_date,
                    "entry_price": last_price,
                    "return_6d": row["return_6d"],
                    "atr10": row["atr10"],
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
        if (processed % batch_size == 0 or processed == total) and log_callback:
            elapsed = time.time() - start_time
            remain = (elapsed / processed) * (total - processed) if processed else 0
            em, es = divmod(int(elapsed), 60)
            rm, rs = divmod(int(remain), 60)

            # System6の詳細統計を計算
            total_candidates = sum(len(cands) for cands in candidates_by_date.values())

            msg = tr(
                "📊 System6 進捗: {done}/{total} | "
                "フィルター通過: {filter_passed}件 | セットアップ通過: {setup_passed}件 | "
                "候補: {candidates}件\n"
                "⏱️ 経過: {em}m{es}s | 残り: ~{rm}m{rs}s",
                done=processed,
                total=total,
                filter_passed=filter_passed,
                setup_passed=setup_passed,
                candidates=total_candidates,
                em=em,
                es=es,
                rm=rm,
                rs=rs,
            )
            if buffer:
                sample = ", ".join(buffer[:10])
                more = len(buffer) - len(buffer[:10])
                if more > 0:
                    sample = f"{sample}, ...(+{more} more)"
                msg += "\n" + tr("🔍 処理中銘柄: {names}", names=sample)
            try:
                log_callback(msg)
            except Exception:
                pass

            # バッチ性能記録
            batch_duration = time.time() - batch_start
            if batch_duration > 0:
                symbols_per_second = len(buffer) / batch_duration
                metrics.record_metric(
                    "system6_candidates_batch_duration", batch_duration, "seconds"
                )
                metrics.record_metric(
                    "system6_candidates_symbols_per_second", symbols_per_second, "rate"
                )

            batch_start = time.time()
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
        df = df.sort_values("return_6d", ascending=False)
        total = len(df)
        df.loc[:, "rank"] = list(range(1, total + 1))
        df.loc[:, "rank_total"] = total
        limited = df.head(limit_n)
        candidates_by_date[date] = limited.to_dict("records")

    # 候補抽出の集計サマリーはログにのみ出力
    if skipped > 0 and log_callback:
        summary_lines = [f"⚠️ 候補抽出中にスキップ: {skipped} 件"]
        if skipped_missing_cols:
            summary_lines.append(f"  └─ 必須列欠落: {skipped_missing_cols} 件")
        try:
            for line in summary_lines:
                log_callback(line)
        except Exception:
            pass

    # 最終メトリクス記録
    total_candidates = sum(len(candidates) for candidates in candidates_by_date.values())
    unique_dates = len(candidates_by_date)
    metrics.record_metric("system6_total_candidates", total_candidates, "count")
    metrics.record_metric("system6_unique_entry_dates", unique_dates, "count")
    metrics.record_metric("system6_processed_symbols_candidates", processed, "count")

    if log_callback:
        try:
            log_callback(
                f"📊 System6 候補生成完了: {total_candidates}件の候補 "
                f"({unique_dates}日分, {processed}シンボル処理)"
            )
        except Exception:
            pass

    return candidates_by_date, None


def get_total_days_system6(data_dict: dict[str, pd.DataFrame]) -> int:
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
    "prepare_data_vectorized_system6",
    "generate_candidates_system6",
    "get_total_days_system6",
]
