"""今日のシグナル抽出パイプラインで用いる基礎・指標データローダ群。

run_all_systems_today.py からデータ読み込み責務を分離（責務分割）:
  - basic_data 読み込み（symbol_data / rolling / base 階層対応）
  - indicator_data 読み込み
  - 新鮮度判定・並列処理・進捗レポート

注意: 公開 API は run_all_systems_today.py と互換。
      依存: CacheManager, Settings, pandas, threading（外部 UI コール不含）
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any

import pandas as pd

from common.cache_manager import CacheManager
from common.rate_limited_logging import create_rate_limited_logger

# グローバルレート制限ロガー
_rate_limited_logger = None


def _get_rate_limited_logger():
    """レート制限ロガーを取得。"""
    global _rate_limited_logger
    if _rate_limited_logger is None:
        _rate_limited_logger = create_rate_limited_logger("today_data_loader", 3.0)
    return _rate_limited_logger


__all__ = [
    "_extract_last_cache_date",
    "_recent_trading_days",
    "_build_rolling_from_base",
    "load_basic_data",
    "load_indicator_data",
]

# ----------------------------- データ操作ヘルパ ----------------------------- #


def _extract_last_cache_date(df: pd.DataFrame) -> pd.Timestamp | None:
    """キャッシュデータから最終日付を抽出。"""
    if df is None or getattr(df, "empty", True):
        return None
    for col in ("date", "Date"):
        if col in df.columns:
            try:
                values = pd.to_datetime(df[col].to_numpy(), errors="coerce")
                values = values.dropna()
                if not values.empty:
                    return pd.Timestamp(values[-1]).normalize()
            except Exception:
                continue
    try:
        idx = pd.to_datetime(df.index.to_numpy(), errors="coerce")
        mask = ~pd.isna(idx)
        if mask.any():
            return pd.Timestamp(idx[mask][-1]).normalize()
    except Exception:
        pass
    return None


def _recent_trading_days(today: pd.Timestamp | None, max_back: int) -> list[pd.Timestamp]:
    """今日から最大 max_back 営業日を遡って日付リストを生成。"""
    if today is None:
        return []
    try:
        from common.utils_spy import get_latest_nyse_trading_day
    except ImportError:
        # フォールバック: 単純な日付減算
        dates = []
        current = pd.Timestamp(today).normalize()
        for _i in range(max_back + 1):
            dates.append(current)
            current = current - pd.Timedelta(days=1)
        return dates

    out: list[pd.Timestamp] = []
    seen: set[pd.Timestamp] = set()
    current = pd.Timestamp(today).normalize()
    steps = max(0, int(max_back))
    for _ in range(steps + 1):
        if current in seen:
            break
        out.append(current)
        seen.add(current)
        prev_candidate = get_latest_nyse_trading_day(current - pd.Timedelta(days=1))
        prev_candidate = pd.Timestamp(prev_candidate).normalize()
        if prev_candidate == current:
            break
        current = prev_candidate
    return out


def _build_rolling_from_base(
    symbol: str,
    base_df: pd.DataFrame,
    target_len: int,
    cache_manager: CacheManager | None = None,
) -> pd.DataFrame | None:
    """base キャッシュから rolling 形式（尻尾切り）に変換、必要なら保存。"""
    if base_df is None or getattr(base_df, "empty", True):
        return None
    try:
        work = base_df.copy()
    except Exception:
        work = base_df
    if work.index.name is not None:
        work = work.reset_index()
    if "Date" in work.columns:
        work["date"] = pd.to_datetime(work["Date"].to_numpy(), errors="coerce")
    elif "date" in work.columns:
        work["date"] = pd.to_datetime(work["date"].to_numpy(), errors="coerce")
    else:
        return None
    work = work.dropna(subset=["date"]).sort_values("date")
    col_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "AdjClose": "adjusted_close",
        "Adj Close": "adjusted_close",
        "Volume": "volume",
    }
    try:
        for src, dst in list(col_map.items()):
            if src in work.columns:
                work = work.rename(columns={src: dst})
    except Exception:
        pass
    sliced = work.tail(int(target_len)).reset_index(drop=True)
    if sliced.empty:
        return None
    if cache_manager is not None:
        try:
            cache_manager.write_atomic(sliced, symbol, "rolling")
        except Exception:
            pass
    return sliced


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """列名を大文字OHLCVに統一。"""
    col_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "adj_close": "AdjClose",
        "adjusted_close": "AdjClose",
    }
    try:
        return df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    except Exception:
        return df


# ----------------------------- 基礎データローダ ----------------------------- #


def load_basic_data(
    symbols: list[str],
    cache_manager: CacheManager,
    settings: Any,
    symbol_data: dict[str, pd.DataFrame] | None,
    *,
    today: pd.Timestamp | None = None,
    freshness_tolerance: int | None = None,
    base_cache: dict[str, pd.DataFrame] | None = None,
    log_callback: Callable[[str, bool], None] | None = None,
    ui_log_callback: Callable[[str], None] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    基礎データ（OHLCV + 基本指標）を読み込み。

    読み込み順序:
    1. symbol_data (事前提供)
    2. rolling キャッシュ
    3. base キャッシュから rolling 生成

    Args:
        symbols: 対象シンボル一覧
        cache_manager: キャッシュ管理オブジェクト
        settings: 設定オブジェクト
        symbol_data: 事前ロード済みデータ
        today: 今日の日付（新鮮度判定用）
        freshness_tolerance: 許容される陳腐化日数
        base_cache: ベースキャッシュ (未使用, 互換性維持)
        log_callback: ログコールバック
        ui_log_callback: UI ログコールバック

    Returns:
        {symbol: DataFrame} の辞書
    """

    def _log(msg: str, ui: bool = True) -> None:
        if log_callback:
            log_callback(msg, ui)

    def _emit_ui_log(msg: str) -> None:
        if ui_log_callback:
            ui_log_callback(msg)

    data: dict[str, pd.DataFrame] = {}
    total_syms = len(symbols)
    start_ts = time.perf_counter()
    chunk = 500

    if freshness_tolerance is None:
        try:
            freshness_tolerance = int(settings.cache.rolling.max_staleness_days)
        except Exception:
            freshness_tolerance = 2
    freshness_tolerance = max(0, int(freshness_tolerance))

    # target length 試算は未使用のため削除（以前のロジック残骸）

    stats_lock = Lock()
    stats: dict[str, int] = {}

    def _record_stat(key: str) -> None:
        with stats_lock:
            stats[key] = stats.get(key, 0) + 1

    recent_allowed: set[pd.Timestamp] = set()
    if today is not None and freshness_tolerance >= 0:
        try:
            recent_allowed = {
                pd.Timestamp(d).normalize()
                for d in _recent_trading_days(pd.Timestamp(today), freshness_tolerance)
            }
        except Exception:
            recent_allowed = set()

    gap_probe_days = max(freshness_tolerance + 5, 10)

    def _estimate_gap_days(
        today_dt: pd.Timestamp | None, last_dt: pd.Timestamp | None
    ) -> int | None:
        if today_dt is None or last_dt is None:
            return None
        try:
            recent = _recent_trading_days(pd.Timestamp(today_dt), gap_probe_days)
        except Exception:
            recent = []
        for offset, dt in enumerate(recent):
            if dt == last_dt:
                return offset
        try:
            return max(0, int((pd.Timestamp(today_dt) - pd.Timestamp(last_dt)).days))
        except Exception:
            return None

    def _pick_symbol_data(sym: str) -> pd.DataFrame | None:
        try:
            if not symbol_data or sym not in symbol_data:
                return None
            df = symbol_data.get(sym)
            if df is None or getattr(df, "empty", True):
                return None
            x = df.copy()
            if x.index.name is not None:
                x = x.reset_index()
            if "date" in x.columns:
                x["date"] = pd.to_datetime(x["date"].to_numpy(), errors="coerce")
            elif "Date" in x.columns:
                x["date"] = pd.to_datetime(x["Date"].to_numpy(), errors="coerce")
            else:
                return None
            col_map = {
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adjusted_close",
                "AdjClose": "adjusted_close",
                "Volume": "volume",
            }
            for k, v in list(col_map.items()):
                if k in x.columns:
                    x = x.rename(columns={k: v})
            required = {"date", "close"}
            if not required.issubset(set(x.columns)):
                return None
            x = x.dropna(subset=["date"]).sort_values("date")
            return x
        except Exception:
            return None

    def _normalize_loaded(df: pd.DataFrame | None) -> pd.DataFrame | None:
        if df is None or getattr(df, "empty", True):
            return None
        try:
            if "Date" not in df.columns:
                work = df.copy()
                if "date" in work.columns:
                    work["Date"] = pd.to_datetime(work["date"].to_numpy(), errors="coerce")
                else:
                    work["Date"] = pd.to_datetime(work.index.to_numpy(), errors="coerce")
                df = work
            df["Date"] = pd.to_datetime(df["Date"].to_numpy(), errors="coerce").normalize()
        except Exception:
            pass
        normalized = _normalize_ohlcv(df)
        try:
            fill_cols = [
                c for c in ("Open", "High", "Low", "Close", "Volume") if c in normalized.columns
            ]
            if fill_cols:
                normalized = normalized.copy()
                try:
                    filled = normalized[fill_cols].apply(pd.to_numeric, errors="coerce")
                except Exception:
                    filled = normalized[fill_cols]
                normalized.loc[:, fill_cols] = filled.ffill().bfill()
        except Exception:
            pass
        try:
            if "Date" in normalized.columns:
                normalized = normalized.dropna(subset=["Date"])
        except Exception:
            pass
        return normalized

    env_parallel = (os.environ.get("BASIC_DATA_PARALLEL", "") or "").strip().lower()
    try:
        env_parallel_threshold = int(os.environ.get("BASIC_DATA_PARALLEL_THRESHOLD", "200"))
    except Exception:
        env_parallel_threshold = 200
    if env_parallel in ("1", "true", "yes"):
        use_parallel = total_syms > 1
    elif env_parallel in ("0", "false", "no"):
        use_parallel = False
    else:
        use_parallel = total_syms >= max(0, env_parallel_threshold)

    max_workers: int | None = None
    if use_parallel and total_syms > 0:
        try:
            env_workers = (os.environ.get("BASIC_DATA_MAX_WORKERS", "") or "").strip()
            if env_workers:
                max_workers = int(env_workers)
        except Exception:
            max_workers = None
        if max_workers is None:
            try:
                cfg_workers = getattr(settings.cache.rolling, "load_max_workers", None)
                if cfg_workers:
                    max_workers = int(cfg_workers)
            except Exception:
                pass
        if max_workers is None:
            cpu_count = os.cpu_count() or 4
            max_workers = max(4, cpu_count * 2)
        max_workers = max(1, min(int(max_workers), total_syms))
        if _log:
            _log(f"🧵 基礎データロード並列化: workers={max_workers}")

    def _load_one(sym: str) -> tuple[str, pd.DataFrame | None]:
        try:
            source: str | None = None
            df = _pick_symbol_data(sym)
            rebuild_reason: str | None = None
            last_seen_date: pd.Timestamp | None = None
            # gap_days: 不使用（統計機能簡素化により削除）
            if df is None or getattr(df, "empty", True):
                df = cache_manager.read(sym, "rolling")
            else:
                source = "prefetched"
            if df is None or getattr(df, "empty", True):
                source = None
            if df is None or getattr(df, "empty", True):
                needs_rebuild = True
            else:
                needs_rebuild = False
            if df is not None and not getattr(df, "empty", True) and source is None:
                source = "rolling"
            if df is not None and not getattr(df, "empty", True):
                last_seen_date = _extract_last_cache_date(df)
                if last_seen_date is None:
                    rebuild_reason = rebuild_reason or "missing_date"
                    needs_rebuild = True
                else:
                    last_seen_date = pd.Timestamp(last_seen_date).normalize()
                    if (
                        today is not None
                        and recent_allowed
                        and last_seen_date not in recent_allowed
                    ):
                        rebuild_reason = "stale"
                        # gap_days = _estimate_gap_days(pd.Timestamp(today), last_seen_date)
                        needs_rebuild = True
            if needs_rebuild:
                # 個別ログを抑制（サマリー表示に統合）
                _record_stat("manual_rebuild_required")
                _record_stat("failed")
                return sym, None
            normalized = _normalize_loaded(df)
            if normalized is not None and not getattr(normalized, "empty", True):
                _record_stat(source or "rolling")
                return sym, normalized
            _record_stat("failed")
            return sym, None
        except Exception:
            _record_stat("failed")
            return sym, None

    def _report_progress(done: int) -> None:
        if done <= 0 or chunk <= 0:
            return
        if done % chunk != 0:
            return
        try:
            elapsed = max(0.001, time.perf_counter() - start_ts)
            rate = done / elapsed
            remain = max(0, total_syms - done)
            eta_sec = int(remain / rate) if rate > 0 else 0
            m, s = divmod(eta_sec, 60)
            msg = f"📦 基礎データロード進捗: {done}/{total_syms} | ETA {m}分{s}秒"

            # 進捗ログはDEBUGレベルでレート制限適用
            try:
                rate_logger = _get_rate_limited_logger()
                rate_logger.debug_rate_limited(
                    f"📦 基礎データロード進捗: {done}/{total_syms}",
                    interval=2.0,
                    message_key="基礎データ進捗",
                )
            except Exception:
                pass
            if _emit_ui_log:
                _emit_ui_log(msg)
        except Exception:
            if _log:
                _log(f"📦 基礎データロード進捗: {done}/{total_syms}", ui=False)
            if _emit_ui_log:
                _emit_ui_log(f"📦 基礎データロード進捗: {done}/{total_syms}")

    processed = 0
    if use_parallel and max_workers and total_syms > 1:
        # 新しい並列バッチ読み込みを使用（Phase2最適化）
        try:
            if _log:
                _log(f"🚀 並列バッチ読み込み開始: {total_syms}シンボル, workers={max_workers}")

            def progress_callback_internal(loaded, total):
                nonlocal processed
                processed = loaded
                _report_progress(processed)

            # CacheManagerの並列読み込み機能を活用
            parallel_data = cache_manager.read_batch_parallel(
                symbols=symbols,
                profile="rolling",
                max_workers=max_workers,
                fallback_profile="full",
                progress_callback=progress_callback_internal,
            )

            # 結果を既存のデータフォーマットに合わせて処理
            for sym, df in parallel_data.items():
                if df is not None and not getattr(df, "empty", True):
                    # 既存の_normalize_loadedと同様の処理を適用
                    normalized = _normalize_loaded(df)
                    if normalized is not None and not getattr(normalized, "empty", True):
                        data[sym] = normalized
                        _record_stat("rolling")
                    else:
                        _record_stat("failed")
                else:
                    _record_stat("failed")

            if _log:
                _log(f"✅ 並列バッチ読み込み完了: {len(data)}/{total_syms}件成功")

        except Exception as e:
            # 並列処理失敗時はフォールバック
            if _log:
                _log(f"⚠️ 並列バッチ読み込み失敗、従来処理にフォールバック: {e}")
            data.clear()
            processed = 0
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_load_one, sym): sym for sym in symbols}
                for fut in as_completed(futures):
                    try:
                        sym, df = fut.result()
                    except Exception:
                        sym, df = futures[fut], None
                    if df is not None and not getattr(df, "empty", True):
                        data[sym] = df
                    processed += 1
                    _report_progress(processed)
    else:
        for sym in symbols:
            sym, df = _load_one(sym)
            if df is not None and not getattr(df, "empty", True):
                data[sym] = df
            processed += 1
            _report_progress(processed)

    try:
        total_elapsed = max(0.0, time.perf_counter() - start_ts)
        total_int = int(total_elapsed)
        m, s = divmod(total_int, 60)
        done_msg = f"📦 基礎データロード完了: {len(data)}/{total_syms} | 所要 {m}分{s}秒" + (
            " | 並列=ON" if use_parallel and max_workers else " | 並列=OFF"
        )
        if _log:
            _log(done_msg)
        if _emit_ui_log:
            _emit_ui_log(done_msg)
    except Exception:
        if _log:
            _log(f"📦 基礎データロード完了: {len(data)}/{total_syms}")
        if _emit_ui_log:
            _emit_ui_log(f"📦 基礎データロード完了: {len(data)}/{total_syms}")

    try:
        summary_map = {
            "prefetched": "事前供給",
            "rolling": "rolling再利用",
            "manual_rebuild_required": "手動対応",
            "failed": "失敗",
        }
        summary_parts = [
            f"{label}={stats.get(key, 0)}" for key, label in summary_map.items() if stats.get(key)
        ]
        if summary_parts:
            try:
                rate_logger = _get_rate_limited_logger()
                rate_logger.debug_rate_limited(
                    "📊 基礎データロード内訳: " + " / ".join(summary_parts),
                    interval=5.0,
                    message_key="基礎データ内訳",
                )
            except Exception:
                pass
    except Exception:
        pass

    return data


# ----------------------------- 指標データローダ ----------------------------- #


def load_indicator_data(
    symbols: list[str],
    cache_manager: CacheManager,
    settings: Any,
    symbol_data: dict[str, pd.DataFrame] | None,
    *,
    log_callback: Callable[[str, bool], None] | None = None,
    ui_log_callback: Callable[[str], None] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    指標データ（事前計算済み指標含む完全なデータセット）を読み込み。

    Args:
        symbols: 対象シンボル一覧
        cache_manager: キャッシュ管理オブジェクト
        settings: 設定オブジェクト
        symbol_data: 事前ロード済みデータ
        log_callback: ログコールバック
        ui_log_callback: UI ログコールバック

    Returns:
        {symbol: DataFrame} の辞書
    """

    def _log(msg: str, ui: bool = True) -> None:
        if log_callback:
            log_callback(msg, ui)

    def _emit_ui_log(msg: str) -> None:
        if ui_log_callback:
            ui_log_callback(msg)

    data: dict[str, pd.DataFrame] = {}
    total_syms = len(symbols)
    start_ts = time.time()
    chunk = 500

    # 個別銘柄ごとの "⛔ rolling未整備" ログは冗長になるため既定で抑制し、
    # ループ終了後にサマリーのみを出力する方針に変更。
    # 旧挙動を復活させたい場合は環境変数 ROLLING_MISSING_VERBOSE=1 を設定。
    missing_symbols: list[str] = []
    # 理由別カウンタ (生成失敗/長さ不足など) を収集し最終サマリーに載せる
    missing_reasons: dict[str, int] = {}
    verbose_missing = os.environ.get("ROLLING_MISSING_VERBOSE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    for idx, sym in enumerate(symbols, start=1):
        try:
            df = None
            try:
                if symbol_data and sym in symbol_data:
                    df = symbol_data.get(sym)
                    if df is not None and not df.empty:
                        x = df.copy()
                        if x.index.name is not None:
                            x = x.reset_index()
                        if "date" in x.columns:
                            x["date"] = pd.to_datetime(x["date"].to_numpy(), errors="coerce")
                        elif "Date" in x.columns:
                            x["date"] = pd.to_datetime(x["Date"].to_numpy(), errors="coerce")
                        col_map = {
                            "Open": "open",
                            "High": "high",
                            "Low": "low",
                            "Close": "close",
                            "Adj Close": "adjusted_close",
                            "AdjClose": "adjusted_close",
                            "Volume": "volume",
                        }
                        for k, v in list(col_map.items()):
                            if k in x.columns:
                                x = x.rename(columns={k: v})
                        required = {"date", "close"}
                        if required.issubset(set(x.columns)):
                            x = x.dropna(subset=["date"]).sort_values("date")
                            df = x
                        else:
                            df = None
                    else:
                        df = None
            except Exception:
                df = None
            if df is None or df.empty:
                df = cache_manager.read(sym, "rolling")

            try:
                target_len = int(
                    settings.cache.rolling.base_lookback_days + settings.cache.rolling.buffer_days
                )
            except Exception:
                target_len = 300  # デフォルト

            needs_rebuild = df is None or getattr(df, "empty", True)
            if needs_rebuild:
                # 理由はまとめて使わないが、将来の詳細集約用途に保持するならタプル拡張可
                if df is None or getattr(df, "empty", True):
                    reason_desc = "rolling未生成"
                else:
                    try:
                        reason_desc = f"len={len(df)}/{target_len}"
                    except Exception:
                        reason_desc = "行数不足"
                missing_symbols.append(sym)
                missing_reasons[reason_desc] = missing_reasons.get(reason_desc, 0) + 1
                if verbose_missing and _log:
                    from common.cache_warnings import get_rolling_issue_aggregator

                    agg = get_rolling_issue_aggregator()
                    # CacheManager 側で "missing_rolling" が既に記録されている場合は二重報告を抑止
                    if not agg.has_issue("missing_rolling", sym):
                        _log(
                            f"⛔ rolling未整備: {sym} ({reason_desc}) → 手動更新を実行してください",
                            ui=False,
                        )
                continue

            if df is not None and not df.empty:
                try:
                    if "Date" not in df.columns:
                        if "date" in df.columns:
                            df = df.copy()
                            df["Date"] = pd.to_datetime(df["date"].to_numpy(), errors="coerce")
                        else:
                            df = df.copy()
                            df["Date"] = pd.to_datetime(df.index.to_numpy(), errors="coerce")
                    df["Date"] = pd.to_datetime(df["Date"].to_numpy(), errors="coerce").normalize()
                except Exception:
                    pass
                df = _normalize_ohlcv(df)
                data[sym] = df
        except Exception:
            continue

        if total_syms > 0 and idx % chunk == 0:
            try:
                elapsed = max(0.001, time.time() - start_ts)
                rate = idx / elapsed
                remain = max(0, total_syms - idx)
                eta_sec = int(remain / rate) if rate > 0 else 0
                m, s = divmod(eta_sec, 60)
                msg = f"🧮 指標データロード進捗: {idx}/{total_syms} | ETA {m}分{s}秒"

                # 進捗ログはDEBUGレベルでレート制限適用
                try:
                    rate_logger = _get_rate_limited_logger()
                    rate_logger.debug_rate_limited(
                        f"🧮 指標データロード進捗: {idx}/{total_syms}",
                        interval=2.0,
                        message_key="指標データ進捗",
                    )
                except Exception:
                    pass
                if _emit_ui_log:
                    _emit_ui_log(msg)
            except Exception:
                if _log:
                    _log(f"🧮 指標データロード進捗: {idx}/{total_syms}", ui=False)
                if _emit_ui_log:
                    _emit_ui_log(f"🧮 指標データロード進捗: {idx}/{total_syms}")

    # ループ終了後に missing のサマリーをバッチ表示
    if missing_symbols and _log:
        try:
            total_missing = len(missing_symbols)
            # 10%刻み（最低1件）で分割して見やすさ確保
            batch_size = max(1, int(total_missing * 0.1))
            for i in range(0, total_missing, batch_size):
                batch = missing_symbols[i : i + batch_size]
                symbols_str = ", ".join(batch)
                _log(
                    f"⚠️ rolling未整備 ({i+1}〜{min(i+batch_size, total_missing)}/{total_missing}): {symbols_str}",
                    ui=False,
                )
            # 理由別分布を整形
            if missing_reasons:
                try:
                    reason_parts = [
                        f"{k}={v}"
                        for k, v in sorted(missing_reasons.items(), key=lambda x: (-x[1], x[0]))
                    ]
                    reason_str = " / ".join(reason_parts)
                except Exception:
                    reason_str = ""
            else:
                reason_str = ""
            base_summary = f"💡 rolling未整備の計{total_missing}銘柄は自動的にスキップされました（base/full_backupからの再試行は不要）"
            if reason_str:
                base_summary += f" | 内訳: {reason_str}"
            _log(base_summary, ui=False)
        except Exception:
            pass

    try:
        total_elapsed = max(0.0, time.time() - start_ts)
        total_int = int(total_elapsed)
        m, s = divmod(total_int, 60)
        done_msg = f"🧮 指標データロード完了: {len(data)}/{total_syms} | 所要 {m}分{s}秒"
        if _log:
            _log(done_msg)
        if _emit_ui_log:
            _emit_ui_log(done_msg)
    except Exception:
        if _log:
            _log(f"🧮 指標データロード完了: {len(data)}/{total_syms}")
        if _emit_ui_log:
            _emit_ui_log(f"🧮 指標データロード完了: {len(data)}/{total_syms}")

    return data
