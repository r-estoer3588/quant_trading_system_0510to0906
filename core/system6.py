"""System6 core logic (Short mean-reversion momentum burst)."""

import time

import numpy as np
import pandas as pd
from ta.volatility import AverageTrueRange

from common.batch_processing import process_symbols_batch
from common.i18n import tr
from common.structured_logging import MetricsCollector
from common.utils import resolve_batch_size

# System6 configuration constants
MIN_PRICE = 5.0  # 最低価格フィルター（ドル）
MIN_DOLLAR_VOLUME_50 = 10_000_000  # 最低ドルボリューム50日平均（ドル）

# Shared metrics collector to avoid file handle leaks
_metrics = MetricsCollector()

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
    # 柔軟な列名マッピング（大文字・小文字両対応）
    col_mapping = {}
    required_base_cols = ["Open", "High", "Low", "Close", "Volume"]

    for required_col in required_base_cols:
        if required_col in df.columns:
            col_mapping[required_col] = required_col
        elif required_col.lower() in df.columns:
            col_mapping[required_col] = required_col.lower()
        else:
            raise ValueError(f"missing column: {required_col} (or {required_col.lower()})")

    # 必要な列のみを抽出してコピー
    base_cols = [col_mapping[col] for col in required_base_cols]

    # 日付インデックスを決定（列優先・なければ既存インデックス）
    date_series: pd.Series | None = None
    for date_col in ("Date", "date"):
        if date_col in df.columns:
            # 無限値をNaNに変換してから日付変換
            date_col_data = df[date_col].replace([np.inf, -np.inf], np.nan)
            date_series = pd.to_datetime(date_col_data, errors="coerce")
            break

    if date_series is None:
        raw_index = df.index
        if isinstance(raw_index, pd.DatetimeIndex):
            date_series = pd.to_datetime(raw_index, errors="coerce")
        else:
            # 無限値をNaNに変換してから日付変換
            index_data = pd.Series(raw_index).replace([np.inf, -np.inf], np.nan)
            date_series = pd.to_datetime(index_data, errors="coerce")

    if date_series is None:
        raise ValueError("missing date index")

    if isinstance(date_series, (pd.Index, pd.Series)):
        values = date_series.to_numpy()
    else:
        values = pd.to_datetime(date_series, errors="coerce").to_numpy()
    date_series = pd.Series(values, index=df.index)

    if getattr(date_series.dt, "tz", None) is not None:
        date_series = date_series.dt.tz_localize(None)

    valid_mask = date_series.notna()
    if not valid_mask.any():
        raise ValueError("invalid date index")

    if not valid_mask.all():
        date_series = date_series[valid_mask]
        x = df.loc[valid_mask, base_cols].copy()
    else:
        x = df.loc[:, base_cols].copy()

    # 列名を標準化（大文字に統一）
    x.columns = required_base_cols

    if len(x) < 50:
        raise ValueError("insufficient rows")

    # 正規化した日付をインデックスに設定
    x.index = pd.Index(date_series, name="Date")

    try:
        # 🚀 プリコンピューテッド指標を使用（すべての指標を最適化）
        # インデックス対応の問題を回避するため、.valuesを使用してインデックスを無視

        # ATR10
        if "ATR10" in df.columns:
            x["atr10"] = df["ATR10"].values
        elif "atr10" in df.columns:
            x["atr10"] = df["atr10"].values
        else:
            # フォールバック（通常は実行されない）
            _metrics.record_metric("system6_fallback_atr10", 1, "count")
            x["atr10"] = AverageTrueRange(
                x["High"], x["Low"], x["Close"], window=10
            ).average_true_range()

        # DollarVolume50
        if "DollarVolume50" in df.columns:
            x["dollarvolume50"] = df["DollarVolume50"].values
        elif "dollarvolume50" in df.columns:
            x["dollarvolume50"] = df["dollarvolume50"].values
        else:
            # フォールバック（通常は実行されない）
            _metrics.record_metric("system6_fallback_dollarvolume50", 1, "count")
            x["dollarvolume50"] = (x["Close"] * x["Volume"]).rolling(50).mean()

        # Return_6D
        if "Return_6D" in df.columns:
            x["return_6d"] = df["Return_6D"].values
        elif "return_6d" in df.columns:
            x["return_6d"] = df["return_6d"].values
        else:
            # フォールバック（通常は実行されない）
            _metrics.record_metric("system6_fallback_return_6d", 1, "count")
            x["return_6d"] = x["Close"].pct_change(6)

        # UpTwoDays
        if "UpTwoDays" in df.columns:
            x["UpTwoDays"] = df["UpTwoDays"].values
        elif "uptwodays" in df.columns:
            x["UpTwoDays"] = df["uptwodays"].values
        else:
            # フォールバック（通常は実行されない）
            _metrics.record_metric("system6_fallback_uptwodays", 1, "count")
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

    if batch_size is None:
        try:
            from config.settings import get_settings

            batch_size = get_settings(create_dirs=False).data.batch_size
        except Exception:
            batch_size = 100
        # System6では非常に大きなバッチサイズで高速処理（候補抽出は軽い処理）
        batch_size = max(batch_size, 2000)  # 最小2000に設定
        batch_size = resolve_batch_size(total, batch_size)
    start_time = time.time()
    batch_start = time.time()
    processed, skipped = 0, 0
    skipped_missing_cols = 0
    filter_passed = 0  # フィルター条件通過数
    setup_passed = 0  # セットアップ条件通過数
    buffer: list[str] = []

    # 処理開始のログを追加
    if log_callback:
        log_callback(
            f"📊 System6 候補抽出開始: {total}銘柄を処理中... (バッチサイズ: {batch_size})"
        )

    for sym, df in prepared_dict.items():
        # featherキャッシュの健全性チェック
        if df is None or df.empty:
            skipped += 1
            continue
        missing_cols = [c for c in SYSTEM6_ALL_COLUMNS if c not in df.columns]
        if missing_cols:
            skipped += 1
            skipped_missing_cols += 1
            continue
        if df[SYSTEM6_NUMERIC_COLUMNS].isnull().any().any():
            # NaN警告は個別に出力せず、統計のみ記録
            pass

        # last_price（直近終値）を取得
        last_price = None
        if "Close" in df.columns and not df["Close"].empty:
            last_price = df["Close"].iloc[-1]

        # 統計計算：フィルター通過数とセットアップ通過数をカウント（累積日数）
        if "filter" in df.columns:
            filter_passed += df["filter"].sum()  # 全期間でフィルター条件を満たした日数
        if "setup" in df.columns:
            setup_passed += df["setup"].sum()  # 全期間でセットアップ条件を満たした日数

        try:
            if "setup" not in df.columns or not df["setup"].any():
                skipped += 1
                continue
            setup_days = df[df["setup"] == 1]
            if setup_days.empty:
                skipped += 1
                continue
            for date, row in setup_days.iterrows():
                # 日付変換を簡略化（営業日補正なしで高速化）
                if isinstance(date, pd.Timestamp):
                    entry_date = date
                else:
                    entry_date = pd.Timestamp(date)

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
                "フィルター通過: {filter_passed}日 | セットアップ通過: {setup_passed}日 | "
                "候補: {candidates}件\n"
                "⏱️ 経過: {em}m{es}s | 残り: ~{rm}m{rs}s | "
                "スキップ: {skipped}銘柄 (列不足: {missing_cols}銘柄)",
                done=processed,
                total=total,
                filter_passed=filter_passed,
                setup_passed=setup_passed,
                candidates=total_candidates,
                em=em,
                es=es,
                rm=rm,
                rs=rs,
                skipped=skipped,
                missing_cols=skipped_missing_cols,
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
                _metrics.record_metric(
                    "system6_candidates_batch_duration", batch_duration, "seconds"
                )
                _metrics.record_metric(
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
    _metrics.record_metric("system6_total_candidates", total_candidates, "count")
    _metrics.record_metric("system6_unique_entry_dates", unique_dates, "count")
    _metrics.record_metric("system6_processed_symbols_candidates", processed, "count")

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
