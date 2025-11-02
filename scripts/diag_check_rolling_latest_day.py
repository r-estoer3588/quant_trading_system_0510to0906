from __future__ import annotations

"""
診断: rolling キャッシュの最終日が「直近NY営業日」かをチェックして集計します。

出力:
- コンソールに集計（総数 / 直近でない数 / 基準日 / 代表例）
- logs/rolling_latest_day_check_YYYYMMDD.csv に詳細
    （symbol,last_date,expected_date,diff_days）

前提:
- 既存の設定/キャッシュを利用。外部APIは呼びません。
- utils_spy が使えない場合は、rolling最終日の「最頻値」を基準日とするフォールバックを使います。
"""

import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

from common.cache_manager import CacheManager
from common.symbol_universe import build_symbol_universe_from_settings
from config.settings import get_settings


def _detect_date_column(df: pd.DataFrame) -> str | None:
    for c in ("date", "Date"):
        if c in df.columns:
            return c
    return None


def _expected_base_day(cm: CacheManager, symbols: Iterable[str]) -> pd.Timestamp:
    """当日シグナルと同等の基準日を取得。不可なら rolling の最終日モードで代替。"""
    try:
        from common.utils_spy import (
            get_latest_nyse_trading_day,
            get_signal_target_trading_day,
        )

        entry_day = get_signal_target_trading_day()
        base_day = pd.Timestamp(
            get_latest_nyse_trading_day(entry_day - pd.Timedelta(days=1))
        ).normalize()
        return base_day
    except Exception:
        # フォールバック: rolling最終日の最頻値
        last_days: list[pd.Timestamp] = []
        for sym in symbols:
            try:
                df = cm.read(sym, "rolling")
            except Exception:
                df = None
            if df is None or getattr(df, "empty", True):
                continue
            dcol = _detect_date_column(df)
            if not dcol:
                continue
            last = pd.to_datetime(df[dcol], errors="coerce").max()
            if pd.isna(last):
                continue
            last_days.append(pd.Timestamp(last).normalize())
        if last_days:
            mode = pd.Series(last_days).mode()
            return mode.iloc[0] if not mode.empty else pd.Timestamp("1970-01-01")
        return pd.Timestamp("1970-01-01")


def main() -> int:
    settings = get_settings(create_dirs=True)
    cm = CacheManager(settings)
    symbols = list(build_symbol_universe_from_settings(settings))

    expected = _expected_base_day(cm, symbols)

    total = 0
    non_latest = 0
    samples: list[tuple[str, str]] = []
    rows: list[dict[str, str | int]] = []

    for sym in symbols:
        try:
            df = cm.read(sym, "rolling")
        except Exception:
            df = None
        if df is None or getattr(df, "empty", True):
            continue

        dcol = _detect_date_column(df)
        if not dcol:
            continue

        last = pd.to_datetime(df[dcol], errors="coerce").max()
        if pd.isna(last):
            continue

        total += 1
        last_norm = pd.Timestamp(last).normalize()
        is_latest = last_norm == expected
        if not is_latest:
            non_latest += 1
            if len(samples) < 12:
                samples.append((sym, str(last_norm.date())))
        rows.append(
            {
                "symbol": sym,
                "last_date": str(last_norm.date()),
                "expected_date": str(expected.date()),
                "diff_days": int((expected - last_norm).days),
            }
        )

    print("============================================================")
    print(f"[集計] rollingに存在        : {total} 銘柄")
    print(f"[集計] 直近営業日でない銘柄: {non_latest} 銘柄")
    print(f"[基準] expected(base_day)  : {str(expected.date())}")
    if samples:
        print("[例] 最大12件まで表示:")
        for s, d in samples:
            print(f"  - {s}: last={d}, expected={expected.date()}")

    out_dir = Path("logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rolling_latest_day_check_{expected.strftime('%Y%m%d')}.csv"
    pd.DataFrame(rows).sort_values(["diff_days", "symbol"]).to_csv(
        out_path, index=False, encoding="utf-8"
    )
    print("------------------------------------------------------------")
    print(f"[保存] {out_path} に詳細を書き出しました（差分日数や対象銘柄の一覧）。")

    # 終了コード: 問題があれば1、なければ0
    return 0 if non_latest == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
