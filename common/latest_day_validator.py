"""最新営業日チェック: rolling キャッシュが直近営業日のデータを持っているか検証."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from common.cache_manager import CacheManager
from config.settings import get_settings


def _is_derivative_like(symbol: str) -> bool:
    """デリバティブ銘柄らしい suffix (W/R/U 系) を持つか判定."""
    pattern = r"[A-Z]{1,4}(W{1,2}|WS|WT|R|RT|U|UN)$"
    return bool(re.match(pattern, symbol.upper()))


def _classify_stale_reason(
    symbol: str,
    expected_date: pd.Timestamp,
    rolling_df: pd.DataFrame | None,
    full_df: pd.DataFrame | None,
) -> str:
    """rolling が stale な理由を分類."""
    if _is_derivative_like(symbol):
        return "derivative_suffix"

    # full cache に expected_date が存在するか確認
    full_has_expected = False
    rolling_has_expected = False
    full_date_parseable = False
    rolling_date_parseable = False

    # Full cache の日付確認
    if full_df is not None and not full_df.empty:
        date_col = None
        for c in ["date", "Date", "DATE"]:
            if c in full_df.columns:
                date_col = c
                break
        if date_col:
            try:
                full_dates = pd.to_datetime(
                    full_df[date_col], errors="coerce"
                ).dt.normalize()
                full_date_parseable = full_dates.notna().any()
                full_has_expected = expected_date in full_dates.values
            except Exception:
                pass

    # Rolling cache の日付確認
    if rolling_df is not None and not rolling_df.empty:
        date_col = None
        for c in ["date", "Date", "DATE"]:
            if c in rolling_df.columns:
                date_col = c
                break
        if date_col:
            try:
                rolling_dates = pd.to_datetime(
                    rolling_df[date_col], errors="coerce"
                ).dt.normalize()
                rolling_date_parseable = rolling_dates.notna().any()
                rolling_has_expected = expected_date in rolling_dates.values
            except Exception:
                pass

    # 日付解析失敗の検出
    if not full_date_parseable and not rolling_date_parseable:
        return "date_parse_failure"

    # Expected date の有無によるサブ分類
    if not full_has_expected and not rolling_has_expected:
        return "both_missing_date"
    if full_has_expected and not rolling_has_expected:
        return "rolling_missing_date"
    if not full_has_expected:
        return "no_expected_in_full"

    # rolling の最終行が全て NaN か確認
    if rolling_df is not None and not rolling_df.empty:
        last_row = rolling_df.iloc[-1]
        price_cols = ["Open", "High", "Low", "Close", "open", "high", "low", "close"]
        present_price_cols = [c for c in price_cols if c in rolling_df.columns]
        if present_price_cols:
            all_nan = last_row[present_price_cols].isna().all()
            if all_nan:
                return "latest_row_all_nan"

    # full cache が rolling より新しい日付を持っているか確認
    if (
        full_df is not None
        and not full_df.empty
        and rolling_df is not None
        and not rolling_df.empty
    ):
        date_col_full = None
        date_col_rolling = None
        for c in ["date", "Date", "DATE"]:
            if c in full_df.columns:
                date_col_full = c
            if c in rolling_df.columns:
                date_col_rolling = c
        if date_col_full and date_col_rolling:
            try:
                full_last = pd.to_datetime(
                    full_df[date_col_full], errors="coerce"
                ).max()
                rolling_last = pd.to_datetime(
                    rolling_df[date_col_rolling], errors="coerce"
                ).max()
                if (
                    pd.notna(full_last)
                    and pd.notna(rolling_last)
                    and full_last > rolling_last
                ):
                    return "full_newer_than_rolling"
            except Exception:
                pass

    return "unknown"


def validate_latest_trading_day(
    symbols: list[str],
    expected_date: pd.Timestamp,
    *,
    cache_manager: CacheManager | None = None,
    log_callback: Callable[[str], None] | None = None,
    rolling_data: dict[str, pd.DataFrame] | None = None,
    tolerance_days: int | None = None,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """
    全銘柄の rolling キャッシュ最終日が expected_date と一致するかチェック。

    Args:
        symbols: 対象銘柄リスト
        expected_date: 期待される最新営業日（正規化済み）
        cache_manager: CacheManager インスタンス（None なら内部で生成）
        log_callback: ログ出力コールバック

    Returns:
        (valid_symbols, stale_details)
        - valid_symbols: 最新営業日データを持つ銘柄リスト
        - stale_details: dict with symbol -> reason/dates
    """
    if cache_manager is None:
        settings = get_settings()
        cache_manager = CacheManager(settings)

    # 許容遅延日数（営業日ベース）。未指定なら設定値/既定値にフォールバック。
    try:
        if tolerance_days is None:
            # settings.cache.rolling.max_staleness_days と揃える（存在しない場合は 1）
            cache_cfg = getattr(settings, "cache", None)
            rolling_cfg = getattr(cache_cfg, "rolling", None)
            tolerance_days = int(getattr(rolling_cfg, "max_staleness_days", 1))
    except Exception:
        tolerance_days = 1
    tolerance_days = max(0, int(tolerance_days or 0))

    # 許容される最新営業日の集合を構築（expected_date 当日を含め、過去 tolerance_days 営業日ぶん）
    allowed_recent: set[pd.Timestamp] = set()
    try:
        # pandas_market_calendars が使えない場合はカレンダー日でフォールバック
        start = pd.Timestamp(expected_date).normalize()
        allowed_recent.add(start)
        # 営業日カレンダーで後退
        try:
            import pandas_market_calendars as mcal  # type: ignore

            nyse = mcal.get_calendar("NYSE")
            sched = nyse.schedule(
                start_date=start - pd.Timedelta(days=max(5, tolerance_days * 3)),
                end_date=start,
            )
            trading_days = pd.to_datetime(sched.index).normalize().tolist()
            # 直近側から tolerance_days 件を拾う
            trading_days = [pd.Timestamp(d).normalize() for d in trading_days]
            for d in reversed(trading_days):
                allowed_recent.add(d)
                if len(allowed_recent) >= tolerance_days + 1:
                    break
        except Exception:
            # フォールバック: カレンダー日で単純に後退
            for i in range(1, tolerance_days + 1):
                allowed_recent.add((start - pd.Timedelta(days=i)).normalize())
    except Exception:
        allowed_recent = {pd.Timestamp(expected_date).normalize()}

    def _log(msg: str) -> None:
        if log_callback:
            log_callback(msg)
        else:
            logger = logging.getLogger(__name__)
            logger.info(msg)

    valid_symbols = []
    stale_details: dict[str, dict[str, Any]] = {}

    for symbol in symbols:
        try:
            # 可能なら事前に読み込んだ rolling データを再利用（I/O 削減）
            if rolling_data is not None:
                rolling_df = rolling_data.get(symbol)
            else:
                rolling_df = cache_manager.read(symbol, "rolling")
        except Exception:
            rolling_df = None

        if rolling_df is None or rolling_df.empty:
            stale_details[symbol] = {
                "reason": "no_rolling_cache",
                "rolling_last": None,
                "full_last": None,
            }
            continue

        # rolling の最終日を取得
        date_col = None
        for c in ["date", "Date", "DATE"]:
            if c in rolling_df.columns:
                date_col = c
                break

        if not date_col:
            # date 列が index の場合
            if isinstance(rolling_df.index, pd.DatetimeIndex):
                rolling_last = rolling_df.index.max()
            else:
                try:
                    rolling_last = pd.to_datetime(
                        rolling_df.index, errors="coerce"
                    ).max()
                except Exception:
                    rolling_last = pd.Timestamp("NaT")
        else:
            rolling_last = pd.to_datetime(rolling_df[date_col], errors="coerce").max()

        if pd.isna(rolling_last):
            stale_details[symbol] = {
                "reason": "no_valid_date",
                "rolling_last": None,
                "full_last": None,
            }
            continue

        rolling_last_normalized = pd.Timestamp(rolling_last).normalize()

        # 許容範囲判定: 期待日と一致、または期待日以前で allowed_recent に含まれる場合は有効
        if (
            rolling_last_normalized == expected_date
            or rolling_last_normalized in allowed_recent
            or rolling_last_normalized > expected_date  # 将来日付（より新しい）も許容
        ):
            valid_symbols.append(symbol)
        else:
            # stale - 理由を分類
            try:
                full_df = cache_manager.read(symbol, "full_backup")
            except Exception:
                full_df = None

            reason = _classify_stale_reason(symbol, expected_date, rolling_df, full_df)

            # full cache の最終日も記録
            full_last = None
            if full_df is not None and not full_df.empty:
                full_date_col = None
                for c in ["date", "Date", "DATE"]:
                    if c in full_df.columns:
                        full_date_col = c
                        break
                if full_date_col:
                    full_last = pd.to_datetime(
                        full_df[full_date_col], errors="coerce"
                    ).max()

            stale_details[symbol] = {
                "reason": reason,
                "rolling_last": (
                    rolling_last_normalized.date()
                    if pd.notna(rolling_last_normalized)
                    else None
                ),
                "full_last": full_last.date() if pd.notna(full_last) else None,
            }

    _log(f"✅ 最新営業日データを持つ銘柄: {len(valid_symbols)}/{len(symbols)}")
    if tolerance_days:
        try:
            base_date = pd.Timestamp(expected_date).date()
            msg = f"ℹ️ 許容遅延: 過去 {int(tolerance_days)} 営業日までを有効とみなしています (基準日: {base_date})"
            _log(msg)
        except Exception:
            pass
    if stale_details:
        _log(f"⚠️  除外された銘柄: {len(stale_details)}")

    return valid_symbols, stale_details


def get_exclusion_stats(stale_details: dict[str, dict[str, Any]]) -> dict[str, int]:
    """除外理由別の統計を集計."""
    reason_counts: dict[str, int] = {}
    for details in stale_details.values():
        reason = details.get("reason", "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return reason_counts


def save_excluded_symbols_csv(
    stale_details: dict[str, dict[str, Any]],
    expected_date: pd.Timestamp,
    output_dir: Path | str = "logs",
) -> Path | None:
    """除外銘柄の詳細を CSV 保存."""
    if not stale_details:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d")
    output_path = output_dir / f"excluded_symbols_{timestamp_str}.csv"

    rows = []
    for symbol, details in stale_details.items():
        rows.append(
            {
                "symbol": symbol,
                "expected_date": expected_date.date(),
                "rolling_last": details.get("rolling_last"),
                "full_last": details.get("full_last"),
                "reason": details.get("reason", "unknown"),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    logger = logging.getLogger(__name__)
    logger.info(f"除外銘柄の詳細を保存: {output_path}")

    return output_path


__all__ = [
    "validate_latest_trading_day",
    "save_excluded_symbols_csv",
    "get_exclusion_stats",
]
