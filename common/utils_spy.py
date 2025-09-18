from __future__ import annotations

import os
from datetime import time as dtime
from pathlib import Path
from collections.abc import Iterable
import sys

import pandas as pd
import pandas_market_calendars as mcal
import streamlit as st
from ta.trend import SMAIndicator

from common.i18n import tr
from config.settings import get_settings


def _ui_enabled() -> bool:
    """Return True when running under Streamlit UI.

    Heuristic only: avoid querying Streamlit runtime context directly to prevent
    'missing ScriptRunContext' warnings. Prefer environment flag or command hint.
    """
    try:
        v = (os.getenv("STREAMLIT_SERVER_ENABLED") or "").strip().lower()
        if v in {"1", "true", "yes"}:
            return True
    except Exception:
        pass
    try:
        argv = " ".join(sys.argv).lower()
        if "streamlit" in argv:
            return True
    except Exception:
        pass
    return False


def _st_emit(kind: str, *args, **kwargs) -> None:
    """Safely emit Streamlit UI calls only when UI context is active.

    In CLI/batch runs, this becomes a no-op to avoid ScriptRunContext warnings.
    """
    if not _ui_enabled():
        return
    try:
        fn = getattr(st, kind, None)
        if callable(fn):
            fn(*args, **kwargs)
    except Exception:
        # Silently ignore UI errors in non-critical paths
        return


def _candidate_spy_paths(root: Path) -> list[Path]:
    """SPY.csv を探す候補パスを返す。

    （バックテストなど広期間が必要な場面では base を優先し、
     当日シグナルなどでは rolling を優先する設計だが、
     本関数は単純な候補列挙のみを行う）
    大文字小文字の違いも吸収する（クロスプラットフォームのため）。
    """

    def _case_insensitive_find(dir_path: Path, name: str) -> Path | None:
        try:
            if not dir_path.exists():
                return None
            for fn in os.listdir(dir_path):
                if fn.lower() == name.lower():
                    return dir_path / fn
        except Exception:
            return None
        return None

    names = ("SPY.csv",)
    dirs: Iterable[Path] = (
        root / "base",
        root / "full",
        root / "rolling",
    )
    out: list[Path] = []
    for d in dirs:
        for nm in names:
            p = _case_insensitive_find(d, nm)
            if p is not None:
                out.append(p)
    return out


def _read_daily_csv_any_datecol(path: Path) -> pd.DataFrame:
    """日足CSVを 'Date' または 'date' いずれでも読み込めるようにする。"""
    df = pd.read_csv(path)
    date_col = None
    if "Date" in df.columns:
        date_col = "Date"
    elif "date" in df.columns:
        date_col = "date"
    if date_col is None:
        raise ValueError("date/Date 列が見つかりません")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    return df


def get_spy_data_cached_v2(folder: str = "data_cache", mode: str = "backtest"):
    """
    SPY.csv をキャッシュから読み込む。
    - mode="backtest": data_cache/base → data_cache/full_backup の順（rolling は探索しない）
    - mode="today": data_cache/rolling → data_cache/base → data_cache/full_backup の順
    - キャッシュが見つからない場合はエラーを返す
    """
    try:
        settings = get_settings(create_dirs=True)
        root = Path(settings.DATA_CACHE_DIR)
        full_dir = Path(getattr(settings.cache, "full_dir", root / "full_backup"))
        rolling_dir = Path(getattr(settings.cache, "rolling_dir", root / "rolling"))
    except Exception:
        root = Path(folder)
        full_dir = root / "full_backup"
        rolling_dir = root / "rolling"

    # 探索順を mode で切り替え
    base_dir = root / "base"
    mode_lower = str(mode).lower()
    if mode_lower == "today":
        search_dirs: list[Path] = [rolling_dir, base_dir, full_dir]
    else:
        # backtest: rolling は探索しない
        search_dirs = [base_dir, full_dir]

    def _find_case_insensitive(d: Path, name: str) -> Path | None:
        try:
            if not d.exists():
                return None
            for fn in os.listdir(d):
                if fn.lower() == name.lower():
                    return d / fn
        except Exception:
            return None
        return None

    path: Path | None = None
    for d in search_dirs:
        p = _find_case_insensitive(d, "SPY.csv")
        if p is not None:
            path = p
            break
    if path is None or not path.exists():
        _st_emit(
            "error", tr("❌ SPY.csv が見つかりません (base/full_backup/rolling を確認)")
        )
        return None

    # backtest 時は full_backup の存在を必須とし、無ければエラーメッセージを表示
    if mode_lower != "today":
        has_full_backup = _find_case_insensitive(full_dir, "SPY.csv") is not None
        if not has_full_backup:
            try:
                _st_emit(
                    "error",
                    tr(
                        "⚠ SPY の full_backup が存在しません: {p}",
                        p=str(full_dir / "SPY.csv"),
                    ),
                )
            except Exception:
                pass

    try:
        df = _read_daily_csv_any_datecol(Path(path))

        # 直近情報の表示（UIが無い場面では無視される）
        try:
            _st_emit(
                "write", tr("✅ SPYキャッシュ最終日: {d}", d=str(df.index[-1].date()))
            )
        except Exception:
            pass

        # NYSE 最新営業日
        today = pd.Timestamp.today().normalize()
        latest_trading_day = get_latest_nyse_trading_day(today)
        try:
            _st_emit(
                "write", tr("🗓️ 直近のNYSE営業日: {d}", d=str(latest_trading_day.date()))
            )
        except Exception:
            pass

        # 1つ前の営業日（当日営業時間帯の影響を避ける）
        try:
            nyse = mcal.get_calendar("NYSE")
            sched = nyse.schedule(
                start_date=today - pd.Timedelta(days=7),
                end_date=today,
            )
            valid = pd.to_datetime(sched.index).normalize()
            prev_trading_day = valid[-2] if len(valid) >= 2 else latest_trading_day
        except Exception:
            prev_trading_day = latest_trading_day

        # 米東部時間を取得
        try:
            ny_time = pd.Timestamp.now(tz="America/New_York").time()
        except Exception:
            ny_time = dtime(18, 0)

        # 古い場合は警告のみ表示
        if df.index[-1].normalize() < prev_trading_day and ny_time >= dtime(18, 0):
            try:
                _st_emit("warning", tr("⚠ SPYキャッシュが古い可能性があります"))
            except Exception:
                pass
        else:
            try:
                _st_emit("write", tr("✅ SPYキャッシュは有効"))
            except Exception:
                pass

        return df

    except Exception as e:
        _st_emit("error", tr("❌ SPY読み込み失敗: {e}", e=str(e)))
        return None


def _normalize_to_naive_day(ts: pd.Timestamp | None) -> pd.Timestamp:
    """Normalize timestamp to tz-naive midnight."""

    if ts is None:
        ts = pd.Timestamp.today()
    else:
        ts = pd.Timestamp(ts)
    tz = getattr(ts, "tzinfo", None)
    if tz is not None:
        try:
            ts = ts.tz_convert(None)
        except (TypeError, ValueError, AttributeError):
            try:
                ts = ts.tz_localize(None)
            except Exception:
                ts = pd.Timestamp(ts.to_pydatetime().replace(tzinfo=None))
    return ts.normalize()


def get_latest_nyse_trading_day(today: pd.Timestamp | None = None) -> pd.Timestamp:
    nyse = mcal.get_calendar("NYSE")
    today_naive = _normalize_to_naive_day(today)
    sched = nyse.schedule(
        start_date=today_naive - pd.Timedelta(days=7),
        end_date=today_naive + pd.Timedelta(days=1),
    )
    valid_days = pd.to_datetime(sched.index).normalize()
    return valid_days[valid_days <= today_naive].max()


def get_next_nyse_trading_day(current: pd.Timestamp | None = None) -> pd.Timestamp:
    """NY証券取引所の翌営業日を返す。"""

    nyse = mcal.get_calendar("NYSE")
    current_naive = _normalize_to_naive_day(current)
    sched = nyse.schedule(
        start_date=current_naive,
        end_date=current_naive + pd.Timedelta(days=10),
    )
    valid_days = pd.to_datetime(sched.index).normalize()
    future_days = valid_days[valid_days > current_naive]
    if future_days.empty:
        raise ValueError("No upcoming NYSE trading day found")
    return future_days.min()


def resolve_signal_entry_date(base_date) -> pd.Timestamp | pd.NaT:
    """シグナル日から翌営業日（取引予定日）を算出する。

    - base_date が欠損・変換不可の場合は NaT を返す。
    - get_next_nyse_trading_day の結果は常に tz-naive な日付に正規化する。
    """

    if base_date is None or (isinstance(base_date, float) and pd.isna(base_date)):
        return pd.NaT
    try:
        ts = pd.Timestamp(base_date)
    except Exception:
        return pd.NaT
    if pd.isna(ts):
        return pd.NaT
    ts = _normalize_to_naive_day(ts)
    try:
        entry_candidate = get_next_nyse_trading_day(ts)
    except Exception:
        return pd.NaT
    if pd.isna(entry_candidate):
        return pd.NaT
    entry_ts = pd.Timestamp(entry_candidate)
    try:
        entry_ts = entry_ts.tz_localize(None)
    except Exception:
        pass
    return entry_ts.normalize()


def get_spy_data_cached(folder: str = "data_cache"):
    """
    旧バージョンの SPY キャッシュ読み込み関数。
    - v2 リゾルバを使用（backtest: base→full_backup, today: rolling→base→full_backup）
    """
    # v2 実装へ委譲（探索順切替は v2 に集約）
    try:
        return get_spy_data_cached_v2(folder)
    except Exception:
        # v2 側に一本化するため、従来の data_cache 直下フォールバックは廃止
        return get_spy_data_cached_v2(folder)


def get_spy_with_indicators(spy_df=None):
    """
    SPY に SMA100 / SMA200 を付与（フィルターは戦略側で判定する）
    """
    if spy_df is None:
        # より堅牢な v2 を使用
        spy_df = get_spy_data_cached_v2()
    if spy_df is not None and not getattr(spy_df, "empty", True):
        # Close 列名のゆらぎに対応（close/AdjClose/adjusted_close 等）
        if "Close" not in spy_df.columns:
            if "close" in spy_df.columns:
                spy_df["Close"] = spy_df["close"]
            elif "AdjClose" in spy_df.columns:
                spy_df["Close"] = spy_df["AdjClose"]
            elif "adjclose" in spy_df.columns:
                spy_df["Close"] = spy_df["adjclose"]
            elif "adjusted_close" in spy_df.columns:
                spy_df["Close"] = spy_df["adjusted_close"]
            else:
                try:
                    _st_emit(
                        "warning",
                        tr(
                            "❗SPYの終値列が見つかりません: {cols}",
                            cols=str(list(spy_df.columns)),
                        ),
                    )
                except Exception:
                    pass
                return spy_df

        spy_df["SMA100"] = SMAIndicator(
            pd.to_numeric(spy_df["Close"], errors="coerce"),
            window=100,
        ).sma_indicator()
        spy_df["SMA200"] = SMAIndicator(
            pd.to_numeric(spy_df["Close"], errors="coerce"),
            window=200,
        ).sma_indicator()
    return spy_df
