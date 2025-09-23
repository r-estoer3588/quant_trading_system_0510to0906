"""Extract rolling window data with indicators from full backup cache.

このスクリプトは ``data_cache/full_backup`` に保存されたフル履歴データを
読み込み、ローリング用キャッシュ ``data_cache/rolling`` を 330 日分
（設定値に基づく）へ再構築します。出力時には各戦略で利用する主要
インジケーター（ATR/SMA/RSI/ADX など）を事前計算して保存します。

直接 CLI から実行できるほか、``extract_rolling_from_full`` 関数を通じて
テストや他スクリプトから再利用することも可能です。
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import sys
from typing import Any
import concurrent.futures
import time
import json

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from indicators_common import add_indicators  # noqa: E402
import pandas as pd  # noqa: E402  ディレクトリ解決後にインポート

from common.cache_manager import CacheManager  # noqa: E402
from common.symbol_universe import build_symbol_universe_from_settings  # noqa: E402
from common.symbols_manifest import MANIFEST_FILENAME, load_symbol_manifest  # noqa: E402
from common.utils import safe_filename  # noqa: E402
from config.settings import get_settings  # noqa: E402

# json already imported at top

LOGGER = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".csv", ".parquet", ".feather"}


@dataclass
class ExtractionStats:
    """集計結果を保持するデータクラス。"""

    total_symbols: int = 0
    processed_symbols: int = 0
    updated_symbols: int = 0
    skipped_no_data: int = 0
    errors: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_symbols": self.total_symbols,
            "processed_symbols": self.processed_symbols,
            "updated_symbols": self.updated_symbols,
            "skipped_no_data": self.skipped_no_data,
            "errors": dict(self.errors),
        }


def _log_message(message: str, log: Callable[[str], None] | None) -> None:
    # If an external logging callable is provided (e.g. console printer),
    # use it and avoid emitting the same message via the module logger to
    # prevent duplicate lines in logs. If no external logger is provided,
    # fall back to the module logger.
    if log:
        try:
            log(message)
        except Exception:  # pragma: no cover - ログが失敗しても続行
            pass
        return
    LOGGER.info(message)


def _normalize_positive_int(value: Any | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _discover_symbols(full_dir: Path) -> list[str]:
    """Detect available symbols from the full backup directory."""

    symbols: set[str] = set()
    for path in full_dir.glob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if path.name.startswith("_"):
            continue
        stem = path.stem.strip()
        if stem:
            symbols.add(stem)
    return sorted(symbols)


def _round_numeric_columns(df: pd.DataFrame, decimals: int | None) -> pd.DataFrame:
    """数値列を ``decimals`` 桁に丸めた DataFrame を返す。"""

    if decimals is None:
        return df
    try:
        dec = int(decimals)
    except (TypeError, ValueError):
        return df
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return df
    rounded = df.copy()
    try:
        rounded[numeric.columns] = numeric.round(dec)
    except Exception:
        return df
    return rounded


def _prepare_rolling_frame(df: pd.DataFrame, target_days: int) -> pd.DataFrame | None:
    """Normalize full-history dataframe and compute indicators for rolling cache."""

    if df is None or getattr(df, "empty", True):
        return None

    try:
        work = df.copy()
    except Exception:  # pragma: no cover - defensive fallback
        work = pd.DataFrame(df)

    if "date" not in work.columns:
        if "Date" in work.columns:
            work = work.rename(columns={"Date": "date"})
        else:
            try:
                idx_series = pd.to_datetime(work.index, errors="coerce")
            except Exception:
                idx_series = None
            if idx_series is None or idx_series.isna().all():
                return None
            work = work.reset_index(drop=True)
            work["date"] = idx_series

    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"])  # 不正日付を除外
    if work.empty:
        return None
    work = work.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)

    calc = work.copy()
    calc["Date"] = pd.to_datetime(calc["date"], errors="coerce").dt.normalize()

    # Upper-case OHLCV columns for indicator calculation
    col_pairs = (
        ("open", "Open"),
        ("high", "High"),
        ("low", "Low"),
        ("close", "Close"),
        ("volume", "Volume"),
    )
    for src, dst in col_pairs:
        if src in calc.columns and dst not in calc.columns:
            calc[dst] = calc[src]

    if "AdjClose" not in calc.columns:
        for cand in ("adjusted_close", "adj_close", "adjclose"):
            if cand in calc.columns:
                calc["AdjClose"] = calc[cand]
                break

    required = {"Open", "High", "Low", "Close"}
    if required - set(calc.columns):
        missing = ",".join(sorted(required - set(calc.columns)))
        raise ValueError(f"missing_price_columns:{missing}")

    # 指標計算に必要な過去データを確保するための lookback margin
    try:
        settings = get_settings(create_dirs=True)
        lookback_margin = int(getattr(settings.cache, "indicator_lookback_margin", 200))
    except Exception:
        lookback_margin = 200

    # add_indicators に渡す前に、target_days に加えて余分な過去を含める
    # これにより ROC200 等の長期指標が tail 部分で適切に計算される
    if target_days > 0 and lookback_margin > 0:
        prefetch_days = int(target_days) + int(lookback_margin)
        calc_for_ind = calc.copy().tail(prefetch_days)
    else:
        calc_for_ind = calc

    enriched = add_indicators(calc_for_ind)

    enriched["date"] = pd.to_datetime(enriched.get("date", enriched.get("Date")), errors="coerce")
    enriched = enriched.drop(columns=["Date"], errors="ignore")
    enriched = enriched.dropna(subset=["date"]).sort_values("date")
    if target_days > 0:
        enriched = enriched.tail(int(target_days))
    enriched = enriched.reset_index(drop=True)

    cols = ["date"] + [c for c in enriched.columns if c != "date"]
    return enriched.loc[:, cols]


def _process_symbol_worker(args: tuple) -> tuple[str, bool, str | None]:
    """Worker function run in a separate process.

    Returns (symbol, success_flag, message). message is None on success,
    or 'no_data' / error message on failure.
    """
    symbol, target_days, round_decimals, nan_warnings = args
    try:
        settings = get_settings(create_dirs=True)
        cm = CacheManager(settings)
        try:
            full_df = cm.read(symbol, "full")
        except Exception as exc:
            return (symbol, False, f"read_error:{exc}")
        if full_df is None or getattr(full_df, "empty", True):
            return (symbol, False, "no_data")
        enriched = _prepare_rolling_frame(full_df, target_days)
        if enriched is None or getattr(enriched, "empty", True):
            return (symbol, False, "no_data")
        if round_decimals is not None:
            try:
                enriched = _round_numeric_columns(enriched, round_decimals)
            except Exception:
                pass
        try:
            cm.write_atomic(enriched, symbol, "rolling")
        except Exception as exc:
            return (symbol, False, f"write_error:{exc}")
        return (symbol, True, None)
    except Exception as exc:
        return (symbol, False, f"{type(exc).__name__}:{exc}")


def _resolve_symbol_universe(
    cache_manager: CacheManager,
    symbols: Iterable[str] | None,
    log: Callable[[str], None] | None,
) -> list[str]:
    if symbols is not None:
        return [s for s in (sym.strip() for sym in symbols) if s]

    manifest_symbols = load_symbol_manifest(cache_manager.full_dir)
    if manifest_symbols:
        try:
            msg = (
                f"ℹ️ cache_daily_data マニフェスト({MANIFEST_FILENAME}) から "
                f"{len(manifest_symbols)} 銘柄を読み込みました"
            )
            _log_message(msg, log)
        except Exception:
            _log_message(
                f"ℹ️ cache_daily_data マニフェスト({MANIFEST_FILENAME}) を読み込みました",
                log,
            )

        available = _discover_symbols(cache_manager.full_dir)
        available_set = {sym.upper() for sym in available}
        filtered = [sym for sym in manifest_symbols if sym.upper() in available_set]

        if filtered:
            missing = len(manifest_symbols) - len(filtered)
            if missing:
                _log_message(
                    (
                        f"ℹ️ full_backup に未存在の {missing} 銘柄を除外し "
                        f"{len(filtered)} 銘柄を処理対象とします"
                    ),
                    log,
                )
            return filtered

        if available:
            _log_message(
                (
                    "⚠️ マニフェスト銘柄が full_backup に存在しないため "
                    f"full_backup を走査した {len(available)} 銘柄を利用します"
                ),
                log,
            )
            return available

        _log_message("⚠️ full_backup ディレクトリから処理対象を検出できませんでした", log)
        return []

    # cache_daily_data と同一ロジックで銘柄集合を構築
    try:
        settings = getattr(cache_manager, "settings", None)
        fetched = build_symbol_universe_from_settings(settings, logger=LOGGER)
    except Exception as exc:  # pragma: no cover - ログのみ
        _log_message(f"⚠️ NASDAQ/EODHD ユニバース取得に失敗: {exc}", log)
        fetched = []

    if fetched:
        safe_symbols = list(dict.fromkeys(safe_filename(sym) for sym in fetched))
        available = _discover_symbols(cache_manager.full_dir)
        if available:
            available_set = {sym.upper() for sym in available}
            filtered = [sym for sym in safe_symbols if sym.upper() in available_set]
            missing = len(safe_symbols) - len(filtered)
            if missing:
                _log_message(
                    (
                        f"ℹ️ NASDAQ/EODHD ユニバース {len(safe_symbols)} 件のうち "
                        f"{missing} 件が full_backup に存在しないため除外します"
                    ),
                    log,
                )
            if filtered:
                return filtered
            # 取得したユニバースと full_backup に重複が無い場合は
            # full_backup を走査した銘柄を利用する（テスト互換性のため）
            _log_message(
                (
                    "⚠️ NASDAQ/EODHD ユニバースが full_backup に存在しないため "
                    f"full_backup を走査した {len(available)} 銘柄を利用します"
                ),
                log,
            )
            return available

        _log_message(
            f"ℹ️ NASDAQ/EODHD ユニバース {len(safe_symbols)} 銘柄を処理対象とします",
            log,
        )
        return safe_symbols

    discovered = _discover_symbols(cache_manager.full_dir)
    _log_message(
        (f"ℹ️ マニフェスト未検出のため full_backup を走査して {len(discovered)} 銘柄を検出しました"),
        log,
    )
    return discovered


def extract_rolling_from_full(
    cache_manager: CacheManager,
    *,
    symbols: Iterable[str] | None = None,
    target_days: int | None = None,
    max_symbols: int | None = None,
    log: Callable[[str], None] | None = None,
    nan_warnings: bool = False,
    workers: int | None = None,
    adaptive: bool = True,
) -> ExtractionStats:
    """Extract rolling window slices from full backup cache and persist them.

    ``max_symbols`` can be used to cap the number of symbols processed.  When
    not provided explicitly the method falls back to
    ``cache_manager.rolling_cfg.max_symbols`` if it is configured with a
    positive integer value.
    """

    if target_days is None:
        try:
            target_days = int(
                cache_manager.rolling_cfg.base_lookback_days + cache_manager.rolling_cfg.buffer_days
            )
        except Exception:
            target_days = 330
    target_days = max(1, int(target_days))

    symbol_list = _resolve_symbol_universe(cache_manager, symbols, log)

    stats = ExtractionStats(total_symbols=len(symbol_list))

    if not symbol_list:
        _log_message("対象シンボルが見つかりませんでした。", log)
        return stats

    _log_message(
        f"🔁 rolling 再構築を開始: {len(symbol_list)} 銘柄 | 期間={target_days}営業日", log
    )

    try:
        # tests may provide a SimpleNamespace without nested attributes; fall back safely
        round_decimals = getattr(
            getattr(cache_manager, "rolling_cfg", None), "round_decimals", None
        )
        if round_decimals is None:
            settings_obj = getattr(cache_manager, "settings", None)
            cache_obj = getattr(settings_obj, "cache", None)
            round_decimals = getattr(cache_obj, "round_decimals", None)
    except Exception:
        round_decimals = None

    # Determine initial worker count preference
    cfg_workers = getattr(getattr(cache_manager, "rolling_cfg", None), "workers", None)
    # If explicit workers passed to function, it takes precedence
    if workers is None:
        workers = cfg_workers

    # Serial fallback if workers not specified
    if workers is None:
        # keep original sequential behavior
        for idx, symbol in enumerate(symbol_list, start=1):
            stats.processed_symbols += 1
            try:
                full_df = cache_manager.read(symbol, "full")
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                stats.errors[symbol] = message
                _log_message(f"⚠️ {symbol}: full 読み込みに失敗 ({message})", log)
                continue

            if full_df is None or getattr(full_df, "empty", True):
                stats.skipped_no_data += 1
                _log_message(f"⏭️ {symbol}: full データ無しのためスキップ", log)
                continue

            try:
                enriched = _prepare_rolling_frame(full_df, target_days)
            except Exception as exc:  # pragma: no cover - logging only
                message = f"{type(exc).__name__}: {exc}"
                stats.errors[symbol] = message
                _log_message(f"⚠️ {symbol}: インジ計算に失敗 ({message})", log)
                continue

            if enriched is None or getattr(enriched, "empty", True):
                stats.skipped_no_data += 1
                _log_message(f"⏭️ {symbol}: 有効なローリングデータ無し", log)
                continue

            try:
                enriched = _round_numeric_columns(enriched, round_decimals)
                cache_manager.write_atomic(enriched, symbol, "rolling")
            except Exception as exc:  # pragma: no cover - logging only
                message = f"{type(exc).__name__}: {exc}"
                stats.errors[symbol] = message
                _log_message(f"⚠️ {symbol}: rolling 書き込みに失敗 ({message})", log)
                continue

            stats.updated_symbols += 1
            if idx % 100 == 0 or idx == len(symbol_list):
                _log_message(f"✅ 進捗: {idx}/{len(symbol_list)} 銘柄処理完了", log)
    else:
        # Parallel execution with adaptive concurrency control
        try:
            workers = int(workers)
        except Exception:
            workers = 0

        # establish sensible bounds
        cpu = os.cpu_count() or 1
        max_possible = max(1, min(32, int(cpu * 2), len(symbol_list)))
        if workers and workers > 0:
            initial_workers = int(workers)
        else:
            settings_obj = getattr(cache_manager, "settings", None)
            cache_obj = getattr(settings_obj, "cache", None)
            rolling_obj = getattr(cache_obj, "rolling", None)
            try:
                initial_workers = int(getattr(rolling_obj, "workers", 4) or 4)
            except Exception:
                initial_workers = 4
        current_workers = max(1, min(initial_workers, max_possible))

        _log_message(
            (
                f"ℹ️ 並列処理: 初期ワーカー={current_workers} "
                f"最大ワーカー={max_possible} 適応型={'有効' if adaptive else '無効'}"
            ),
            log,
        )

        args_list = [(symbol, target_days, round_decimals, nan_warnings) for symbol in symbol_list]

        # prepare progress output file
        try:
            settings_obj = getattr(cache_manager, "settings", None)
            cache_obj = getattr(settings_obj, "cache", None)
            rolling_obj = getattr(cache_obj, "rolling", None)
            report_seconds = int(getattr(rolling_obj, "adaptive_report_seconds", 10) or 10)
        except Exception:
            report_seconds = 10

        logs_dir_candidate = (
            getattr(cache_manager.settings.outputs, "logs_dir", None)
            or getattr(cache_manager.settings, "LOGS_DIR", None)
            or "logs"
        )
        logs_dir_path = Path(str(logs_dir_candidate))
        try:
            logs_dir_path.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        progress_path = logs_dir_path / "rolling_progress.json"

        # create executor with upper bound; we will control active submissions
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_possible) as exe:
            next_idx = 0
            active: dict[concurrent.futures.Future, tuple[str, float]] = {}

            # adaptive measurement
            window_durations: list[float] = []
            window_count = 8
            prev_throughput = None

            while stats.processed_symbols < len(symbol_list):
                # submit tasks until reaching current_workers
                while len(active) < current_workers and next_idx < len(args_list):
                    args = args_list[next_idx]
                    fut = exe.submit(_process_symbol_worker, args)
                    active[fut] = (args[0], time.time())
                    next_idx += 1

                if not active:
                    break

                done, _ = concurrent.futures.wait(
                    active.keys(), return_when=concurrent.futures.FIRST_COMPLETED
                )
                for fut in done:
                    symbol, start_ts = active.pop(fut)
                    stats.processed_symbols += 1
                    end_ts = time.time()
                    duration = max(0.0001, end_ts - start_ts)
                    window_durations.append(duration)
                    # keep window size bounded
                    if len(window_durations) > window_count:
                        window_durations.pop(0)

                    try:
                        sym, ok, message = fut.result()
                    except Exception as exc:
                        stats.errors[symbol] = str(exc)
                        _log_message(f"⚠️ {symbol}: worker 例外 ({exc})", log)
                        continue

                    if not ok:
                        if message == "no_data":
                            stats.skipped_no_data += 1
                            _log_message(f"⏭️ {symbol}: full データ無しのためスキップ", log)
                        else:
                            stats.errors[symbol] = message or "error"
                            _log_message(f"⚠️ {symbol}: 処理失敗 ({message})", log)
                    else:
                        stats.updated_symbols += 1

                # write progress JSON periodically
                try:
                    now_ts = int(time.time())
                    if (
                        not progress_path.exists()
                        or now_ts - int(progress_path.stat().st_mtime) >= report_seconds
                    ):
                        prog = {
                            "total": stats.total_symbols,
                            "processed": stats.processed_symbols,
                            "updated": stats.updated_symbols,
                            "skipped": stats.skipped_no_data,
                            "errors": len(stats.errors),
                            "current_workers": current_workers,
                            "recent_window_seconds": [round(d, 3) for d in window_durations],
                            "timestamp": now_ts,
                        }
                        try:
                            with open(progress_path, "w", encoding="utf-8") as pf:
                                json.dump(prog, pf, ensure_ascii=False)
                        except Exception:
                            pass
                except Exception:
                    pass

                # report progress periodically
                if stats.processed_symbols % 100 == 0 or stats.processed_symbols == len(
                    symbol_list
                ):
                    _log_message(
                        f"✅ 進捗: {stats.processed_symbols}/{len(symbol_list)} 銘柄処理完了",
                        log,
                    )

                # adaptive adjustment: evaluate throughput over window
                if adaptive and len(window_durations) >= max(4, window_count // 2):
                    window_time = sum(window_durations)
                    if window_time <= 0:
                        continue
                    throughput = len(window_durations) / window_time
                    # try small adjustments: increase or decrease by 1
                    if prev_throughput is None:
                        prev_throughput = throughput
                    else:
                        # if throughput improved notably, try increasing workers
                        if throughput > prev_throughput * 1.02 and current_workers < max_possible:
                            current_workers += 1
                            _log_message(f"ℹ️ ワーカー数を増やします -> {current_workers}", log)
                            prev_throughput = throughput
                        # if throughput degraded notably, decrease workers
                        elif throughput < prev_throughput * 0.98 and current_workers > 1:
                            current_workers = max(1, current_workers - 1)
                            _log_message(f"ℹ️ ワーカー数を減らします -> {current_workers}", log)
                            prev_throughput = throughput
                        else:
                            # small/no change, keep current
                            prev_throughput = throughput

    _log_message(
        "✅ rolling 再構築完了: "
        + f"対象={stats.total_symbols} | 更新={stats.updated_symbols} | "
        + f"欠損={stats.skipped_no_data} | エラー={len(stats.errors)}",
        log,
    )
    return stats


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="full_backup から rolling を再構築し主要インジケーターを付与",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="処理対象シンボル（未指定時は cache_daily_data マニフェスト/全銘柄）",
    )
    parser.add_argument(
        "--target-days",
        type=int,
        help="ローリングに保持する営業日数（既定: 設定値 base+buffer）",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        help="処理上限銘柄数（0 以下で無制限。既定: 設定値 rolling.max_symbols）",
    )
    parser.add_argument(
        "--nan-warnings",
        action="store_true",
        help="指標 NaN 警告を有効化（既定: 無効、ログ抑止）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="並列ワーカー数の上限（未指定で設定値またはデフォルトを使用）",
    )
    parser.add_argument(
        "--no-adaptive",
        action="store_true",
        help="適応的ワーカー調整を無効化（既定: 有効）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    parser = _build_parser()
    args = parser.parse_args(argv)

    settings = get_settings(create_dirs=True)
    cache_manager = CacheManager(settings)

    def _console_log(msg: str) -> None:
        LOGGER.info(msg)

    stats = extract_rolling_from_full(
        cache_manager,
        symbols=args.symbols,
        target_days=args.target_days,
        max_symbols=args.max_symbols,
        log=_console_log,
        nan_warnings=bool(getattr(args, "nan_warnings", False)),
        workers=getattr(args, "workers", None),
        adaptive=(not bool(getattr(args, "no_adaptive", False))),
    )

    if stats.errors:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
