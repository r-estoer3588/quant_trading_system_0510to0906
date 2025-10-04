"""System6 core logic (Short mean-reversion momentum burst)."""

import time
from typing import Any

import pandas as pd
from ta.volatility import AverageTrueRange

from common.batch_processing import process_symbols_batch
from common.i18n import tr
from common.structured_logging import MetricsCollector
from common.utils import resolve_batch_size

# System6 configuration constants
MIN_PRICE = 5.0  # 最低価格フィルター（ドル）
MIN_DOLLAR_VOLUME_50 = 10_000_000  # 最低ドルボリューム50日平均（ドル）
HV50_BOUNDS_PERCENT = (10.0, 40.0)
HV50_BOUNDS_FRACTION = (0.10, 0.40)

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
    "hv50",
]
SYSTEM6_ALL_COLUMNS = SYSTEM6_BASE_COLUMNS + SYSTEM6_FEATURE_COLUMNS
SYSTEM6_NUMERIC_COLUMNS = ["atr10", "dollarvolume50", "return_6d", "hv50"]


def _compute_indicators_from_frame(df: pd.DataFrame) -> pd.DataFrame:
    """System6 個別銘柄用の前処理 + 指標利用.

    ポイント:
    1. まずインデックス（日付）を正規化してから列操作
    2. OHLCV を大文字統一
    3. 事前計算済み指標はラベルアラインでそのまま利用（.values 不使用）
    4. 欠損時のみフォールバック計算
    """
    if df is None or df.empty:
        raise ValueError("empty_frame")

    # --- 日付インデックス正規化 ---
    if "Date" in df.columns:
        idx = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    elif "date" in df.columns:
        idx = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    else:
        idx = pd.to_datetime(df.index, errors="coerce").normalize()
    x = df.copy(deep=False)
    x.index = pd.Index(idx, name="Date")
    # 無効日付除去
    x = x[~x.index.isna()]
    if x.empty:
        raise ValueError("invalid date index")
    # 重複除去（最新優先）
    if getattr(x.index, "has_duplicates", False):
        x = x[~x.index.duplicated(keep="last")]
    # ソート
    try:
        x = x.sort_index()
    except Exception:
        pass

    # --- OHLCV リネーム（小文字→大文字） ---
    rename_map: dict[str, str] = {}
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
        try:
            x = x.rename(columns=rename_map)
        except Exception:
            pass

    # 必須列確認
    missing = [c for c in SYSTEM6_BASE_COLUMNS if c not in x.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    # 行数チェック（最低 50 行）
    if len(x) < 50:
        raise ValueError("insufficient rows")

    # --- 指標列追加（ラベルアライン） ---
    try:
        # ATR10
        if "ATR10" in x.columns:
            x["atr10"] = x["ATR10"]
        elif "atr10" in x.columns:
            # 既に小文字形がある場合はそのまま利用
            pass
        else:
            _metrics.record_metric("system6_fallback_atr10", 1, "count")
            x["atr10"] = AverageTrueRange(
                x["High"], x["Low"], x["Close"], window=10
            ).average_true_range()

        # DollarVolume50
        if "DollarVolume50" in x.columns:
            x["dollarvolume50"] = x["DollarVolume50"]
        elif "dollarvolume50" in x.columns:
            pass
        else:
            _metrics.record_metric("system6_fallback_dollarvolume50", 1, "count")
            x["dollarvolume50"] = (x["Close"] * x["Volume"]).rolling(50).mean()

        # Return_6D
        if "Return_6D" in x.columns:
            x["return_6d"] = x["Return_6D"]
        elif "return_6d" in x.columns:
            pass
        else:
            _metrics.record_metric("system6_fallback_return_6d", 1, "count")
            x["return_6d"] = x["Close"].pct_change(6)

        # UpTwoDays
        if "UpTwoDays" in x.columns:
            x["UpTwoDays"] = x["UpTwoDays"]
        elif "uptwodays" in x.columns:
            x["UpTwoDays"] = x["uptwodays"]
        else:
            _metrics.record_metric("system6_fallback_uptwodays", 1, "count")
            x["UpTwoDays"] = (x["Close"] > x["Close"].shift(1)) & (
                x["Close"].shift(1) > x["Close"].shift(2)
            )

        # HV50 (historical volatility)
        hv50_series = None
        if "HV50" in x.columns:
            hv50_series = pd.to_numeric(x["HV50"], errors="coerce")
        elif "hv50" in x.columns:
            hv50_series = pd.to_numeric(x["hv50"], errors="coerce")
        if hv50_series is None:
            _metrics.record_metric("system6_fallback_hv50", 1, "count")
            returns = pd.Series(x["Close"], index=x.index).pct_change()
            hv50_series = returns.rolling(50).std() * (252**0.5) * 100
        x["hv50"] = hv50_series

        hv50_percent = x["hv50"].between(*HV50_BOUNDS_PERCENT)
        hv50_fraction = x["hv50"].between(*HV50_BOUNDS_FRACTION)
        hv50_condition = (hv50_percent | hv50_fraction).fillna(False)

        # フィルターとセットアップ
        x["filter"] = (
            (x["Low"] >= MIN_PRICE)
            & (x["dollarvolume50"] > MIN_DOLLAR_VOLUME_50)
            & hv50_condition
        )
        x["setup"] = x["filter"] & (x["return_6d"] > 0.20) & x["UpTwoDays"]
    except Exception as exc:
        raise ValueError(f"calc_error: {type(exc).__name__}: {exc}") from exc

    # 数値指標の欠損除去
    x = x.dropna(subset=SYSTEM6_NUMERIC_COLUMNS)
    if x.empty:
        raise ValueError("insufficient rows")
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
    latest_only: bool = False,
    include_diagnostics: bool = False,
    diagnostics: dict[str, Any] | None = None,
) -> (
    tuple[dict[pd.Timestamp, dict[str, dict[str, Any]]], pd.DataFrame | None]
    | tuple[
        dict[pd.Timestamp, dict[str, dict[str, Any]]],
        pd.DataFrame | None,
        dict[str, Any],
    ]
):
    """Generate System6 candidates.

    Added fast-path (latest_only=True): O(symbols) processing using only the last row
    of each DataFrame. Returns normalized mapping {date: {symbol: payload}}.
    """
    # diagnostics payload (opt-in)
    if diagnostics is None:
        diagnostics = {
            "ranking_source": None,
            "setup_predicate_count": 0,
            "final_top_n_count": 0,
            "predicate_only_pass_count": 0,
            "mismatch_flag": 0,
        }

    candidates_by_date: dict[pd.Timestamp, list] = {}

    # === Fast Path: latest_only ===
    if latest_only:
        try:
            rows: list[dict[str, Any]] = []
            date_counter: dict[pd.Timestamp, int] = {}
            for sym, df in prepared_dict.items():
                if df is None or df.empty:
                    continue
                if "setup" not in df.columns:
                    continue
                last_row = df.iloc[-1]
                if bool(last_row.get("setup")):
                    diagnostics["setup_predicate_count"] += 1
                else:
                    continue
                # 必要指標取得 (存在しない場合はスキップ)
                return_6d = last_row.get("return_6d")
                if return_6d is None or pd.isna(return_6d):
                    continue
                atr10 = last_row.get("atr10", None)
                dt = df.index[-1]
                date_counter[dt] = date_counter.get(dt, 0) + 1
                entry_price = last_row.get("Close") if "Close" in df else None
                rows.append(
                    {
                        "symbol": sym,
                        "date": dt,
                        "return_6d": return_6d,
                        "atr10": atr10,
                        "entry_price": entry_price,
                    }
                )
            if not rows:
                return ({}, None, diagnostics) if include_diagnostics else ({}, None)
            df_all = pd.DataFrame(rows)
            # 最頻日で揃える（欠落シンボル耐性）
            try:
                mode_date = max(date_counter.items(), key=lambda kv: kv[1])[0]
                df_all = df_all[df_all["date"] == mode_date]
            except Exception:
                pass
            df_all = df_all.sort_values("return_6d", ascending=False, kind="stable")
            df_all = df_all.head(int(top_n)) if top_n else df_all
            # rank 付与（従来互換）
            total = len(df_all)
            df_all.loc[:, "rank"] = list(range(1, total + 1))
            df_all.loc[:, "rank_total"] = total
            normalized: dict[pd.Timestamp, dict[str, dict[str, Any]]] = {}
            for dt_raw, sub in df_all.groupby("date"):
                dt = pd.Timestamp(str(dt_raw))
                symbol_map: dict[str, dict[str, Any]] = {}
                for rec in sub.to_dict("records"):
                    sym_val = rec.get("symbol")
                    if not isinstance(sym_val, str) or not sym_val:
                        continue
                    payload = {
                        k: v for k, v in rec.items() if k not in ("symbol", "date")
                    }
                    symbol_map[sym_val] = payload
                normalized[dt] = symbol_map
            if log_callback:
                try:
                    log_callback(
                        f"System6: latest_only fast-path -> {len(df_all)} candidates (symbols={len(rows)})"
                    )
                except Exception:
                    pass
            diagnostics["final_top_n_count"] = len(df_all)
            diagnostics["ranking_source"] = "latest_only"
            return (
                (normalized, df_all.copy(), diagnostics)
                if include_diagnostics
                else (normalized, df_all.copy())
            )
        except Exception as e:
            if log_callback:
                try:
                    log_callback(f"System6: fast-path failed -> fallback ({e})")
                except Exception:
                    pass
            # fall through to full path
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
                    # 安全な型のみ受け付ける（文字列 / 日付 / 数値インデックス想定）
                    if isinstance(date, (str, int, float)) or hasattr(date, "__str__"):
                        try:
                            entry_date = pd.to_datetime(str(date), errors="coerce")
                            if pd.isna(entry_date):
                                continue
                        except Exception:
                            continue
                    else:
                        continue

                rec = {
                    "symbol": sym,
                    "entry_date": entry_date,
                    "entry_price": last_price,
                    "return_6d": row["return_6d"],
                    "atr10": row["atr10"],
                }
                candidates_by_date.setdefault(entry_date, []).append(rec)
                try:
                    if bool(row.get("setup", False)):
                        diagnostics["setup_predicate_count"] += 1
                except Exception:
                    pass
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
    total_candidates = sum(
        len(candidates) for candidates in candidates_by_date.values()
    )
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

    # Normalize list structure to dict-of-dicts for consistency
    normalized_full: dict[pd.Timestamp, dict[str, dict[str, Any]]] = {}
    for dt, recs in candidates_by_date.items():
        symbol_map: dict[str, dict[str, Any]] = {}
        for rec in recs:
            sym_val = rec.get("symbol") if isinstance(rec, dict) else None
            if not isinstance(sym_val, str) or not sym_val:
                continue
            # rec may contain entry_date; unify key name 'date' for DF compatibility
            payload = {
                k: v for k, v in rec.items() if k not in ("symbol", "entry_date")
            }
            # 保持: 元々 'entry_date' をキー化しているのでそのまま payload にも残す
            payload["entry_date"] = rec.get("entry_date")
            symbol_map[sym_val] = payload
        normalized_full[pd.Timestamp(dt)] = symbol_map
    # diagnostics for full path
    diagnostics["ranking_source"] = diagnostics.get("ranking_source") or "full_scan"
    try:
        last_dt = max(normalized_full.keys()) if normalized_full else None
        diagnostics["final_top_n_count"] = (
            len(normalized_full.get(last_dt, {})) if last_dt is not None else 0
        )
    except Exception:
        diagnostics["final_top_n_count"] = 0

    return (
        (normalized_full, None, diagnostics)
        if include_diagnostics
        else (normalized_full, None)
    )


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
