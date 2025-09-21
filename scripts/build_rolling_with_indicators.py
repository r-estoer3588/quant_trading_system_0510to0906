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
from pathlib import Path
import sys
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd  # noqa: E402  ディレクトリ解決後にインポート

from common.cache_manager import CacheManager  # noqa: E402
from common.symbol_universe import build_symbol_universe_from_settings  # noqa: E402
from common.symbols_manifest import (  # noqa: E402
    MANIFEST_FILENAME,
    load_symbol_manifest,
)
from common.utils import safe_filename  # noqa: E402
from config.settings import get_settings  # noqa: E402
from indicators_common import add_indicators  # noqa: E402

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
    if log:
        try:
            log(message)
        except Exception:  # pragma: no cover - ログが失敗しても続行
            pass
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
    work = (
        work.sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )

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

    enriched = add_indicators(calc)

    enriched["date"] = pd.to_datetime(
        enriched.get("date", enriched.get("Date")), errors="coerce"
    )
    enriched = enriched.drop(columns=["Date"], errors="ignore")
    enriched = enriched.dropna(subset=["date"]).sort_values("date")
    if target_days > 0:
        enriched = enriched.tail(int(target_days))
    enriched = enriched.reset_index(drop=True)

    cols = ["date"] + [c for c in enriched.columns if c != "date"]
    return enriched.loc[:, cols]


def _resolve_symbol_universe(
    cache_manager: CacheManager,
    symbols: Iterable[str] | None,
    log: Callable[[str], None] | None,
) -> list[str]:
    if symbols is not None:
        return [s for s in (sym.strip() for sym in symbols) if s]

    manifest_symbols = load_symbol_manifest(cache_manager.full_dir)
    if manifest_symbols:
        _log_message(
            (
                "ℹ️ cache_daily_data マニフェスト({file}) から "
                "{count} 銘柄を読み込みました"
            ).format(file=MANIFEST_FILENAME, count=len(manifest_symbols)),
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
                        "ℹ️ full_backup に未存在の {missing} 銘柄を除外し "
                        "{count} 銘柄を処理対象とします"
                    ).format(missing=missing, count=len(filtered)),
                    log,
                )
            return filtered

        if available:
            _log_message(
                (
                    "⚠️ マニフェスト銘柄が full_backup に存在しないため "
                    "full_backup を走査した {count} 銘柄を利用します"
                ).format(count=len(available)),
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
            filtered = [
                sym for sym in safe_symbols if sym.upper() in available_set
            ]
            missing = len(safe_symbols) - len(filtered)
            if missing:
                _log_message(
                    (
                        "ℹ️ NASDAQ/EODHD ユニバース {total} 件のうち "
                        "{missing} 件が full_backup に存在しないため除外します"
                    ).format(total=len(safe_symbols), missing=missing),
                    log,
                )
            if filtered:
                return filtered
        _log_message(
            f"ℹ️ NASDAQ/EODHD ユニバース {len(safe_symbols)} 銘柄を処理対象とします",
            log,
        )
        return safe_symbols

    discovered = _discover_symbols(cache_manager.full_dir)
    _log_message(
        (
            "ℹ️ マニフェスト未検出のため full_backup を走査して "
            "{count} 銘柄を検出しました"
        ).format(count=len(discovered)),
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
                cache_manager.rolling_cfg.base_lookback_days
                + cache_manager.rolling_cfg.buffer_days
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
            cache_manager.write_atomic(enriched, symbol, "rolling")
        except Exception as exc:  # pragma: no cover - logging only
            message = f"{type(exc).__name__}: {exc}"
            stats.errors[symbol] = message
            _log_message(f"⚠️ {symbol}: rolling 書き込みに失敗 ({message})", log)
            continue

        stats.updated_symbols += 1
        if idx % 100 == 0 or idx == len(symbol_list):
            _log_message(
                f"✅ 進捗: {idx}/{len(symbol_list)} 銘柄処理完了", log
            )

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
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    parser = _build_parser()
    args = parser.parse_args(argv)

    settings = get_settings(create_dirs=True)
    cache_manager = CacheManager(settings)

    def _console_log(msg: str) -> None:
        print(msg, flush=True)

    stats = extract_rolling_from_full(
        cache_manager,
        symbols=args.symbols,
        target_days=args.target_days,
        max_symbols=args.max_symbols,
        log=_console_log,
    )

    if stats.errors:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
