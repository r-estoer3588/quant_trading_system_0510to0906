from __future__ import annotations

import os
import subprocess
from datetime import time as dtime
from pathlib import Path
from typing import Iterable

import pandas as pd
import pandas_market_calendars as mcal
import streamlit as st
from ta.trend import SMAIndicator

from common.i18n import tr
from config.settings import get_settings


def _candidate_spy_paths(root: Path) -> list[Path]:
    """SPY.csv を探す候補パスを返す。

    優先順位: data_cache/base > data_cache/full > data_cache/rolling > data_cache 直下。
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
        root,
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


def get_spy_data_cached_v2(folder: str = "data_cache"):
    """
    SPY.csv をキャッシュから読み込み、古ければ recover_spy_cache.py を呼んで更新。
    - 直近のNYSE営業日とキャッシュ最終日を比較
    - 米東部時間 18:00 以降かつキャッシュが古い場合のみ自動更新
    - 探索は data_cache/base, data_cache/full, data_cache/rolling, data_cache の順
    """
    try:
        settings = get_settings(create_dirs=True)
        root = Path(settings.DATA_CACHE_DIR)
    except Exception:
        root = Path(folder)

    candidates = _candidate_spy_paths(root)
    path: Path | None = candidates[0] if candidates else None
    if path is None or not path.exists():
        legacy = Path(folder) / "SPY.csv"
        if legacy.exists():
            path = legacy
        else:
            st.error(tr("❌ SPY.csv が存在しません"))
            return None

    try:
        df = _read_daily_csv_any_datecol(Path(path))

        # 直近情報の表示（UIが無い場面では無視される）
        try:
            st.write(tr("✅ SPYキャッシュ最終日: {d}", d=str(df.index[-1].date())))
        except Exception:
            pass

        # NYSE 最新営業日
        today = pd.Timestamp.today().normalize()
        latest_trading_day = get_latest_nyse_trading_day(today)
        try:
            st.write(tr("🗓️ 直近のNYSE営業日: {d}", d=str(latest_trading_day.date())))
        except Exception:
            pass

        # 1つ前の営業日（当日営業時間帯の影響を避ける）
        try:
            nyse = mcal.get_calendar("NYSE")
            sched = nyse.schedule(start_date=today - pd.Timedelta(days=7), end_date=today)
            valid = sched.index.normalize()
            prev_trading_day = valid[-2] if len(valid) >= 2 else latest_trading_day
        except Exception:
            prev_trading_day = latest_trading_day

        # 米東部時間を取得
        try:
            ny_time = pd.Timestamp.now(tz="America/New_York").time()
        except Exception:
            ny_time = dtime(18, 0)

        # 古い場合は自動更新（米東部18:00以降）
        if df.index[-1].normalize() < prev_trading_day and ny_time >= dtime(18, 0):
            try:
                st.warning(tr("⚠ SPYキャッシュが古いため自動更新します.."))
            except Exception:
                pass
            try:
                result = subprocess.run(
                    ["python", "recover_spy_cache.py"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.stdout:
                    try:
                        st.text(result.stdout)
                    except Exception:
                        pass
                if result.stderr:
                    try:
                        st.error(result.stderr)
                    except Exception:
                        pass
            except Exception as e:
                st.error(tr("SPY自動更新失敗: {e}", e=str(e)))
                return None

            # 更新後に読み込み（保存先は base/full のため候補を再解決）
            new_candidates = _candidate_spy_paths(root)
            path2 = new_candidates[0] if new_candidates else path
            if path2.exists():
                try:
                    df = _read_daily_csv_any_datecol(Path(path2))
                    try:
                        st.success(tr("✅ SPYキャッシュ更新済: {d}", d=str(df.index[-1].date())))
                    except Exception:
                        pass
                except Exception as e:
                    st.error(tr("❌ SPY更新後の読み込みに失敗: {e}", e=str(e)))
                    return None
            else:
                st.error(tr("❌ 更新後もSPY.csvが存在しません"))
                return None
        else:
            try:
                st.write(tr("✅ SPYキャッシュは有効"))
            except Exception:
                pass

        return df

    except Exception as e:
        st.error(tr("❌ SPY読み込み失敗: {e}", e=str(e)))
        return None


def get_latest_nyse_trading_day(today: pd.Timestamp | None = None) -> pd.Timestamp:
    nyse = mcal.get_calendar("NYSE")
    if today is None:
        today = pd.Timestamp.today().normalize()
    sched = nyse.schedule(
        start_date=today - pd.Timedelta(days=7),
        end_date=today + pd.Timedelta(days=1),
    )
    valid_days = sched.index.normalize()
    return valid_days[valid_days <= today].max()


def get_spy_data_cached(folder: str = "data_cache"):
    """
    SPY.csv をキャッシュから読み込み、古ければ recover_spy_cache.py を呼んで更新。
    - Streamlit UI で最小限のメッセージを表示
    - 直近のNYSE営業日とキャッシュ最終日を比較
    - 米東部時間 18:00 以降かつキャッシュが古い場合のみ自動更新
    """
    path = os.path.join(folder, "SPY.csv")
    if not os.path.exists(path):
        st.error(tr("❌ SPY.csv が存在しません"))
        return None

    try:
        df = pd.read_csv(path, parse_dates=["Date"])
        if "Date" not in df.columns:
            st.error(tr("❌ 'Date' 列が存在しません"))
            return None
        df.set_index("Date", inplace=True)
        df = df.sort_index()

        # 直近情報の表示（UIが無い環境では無視される）
        try:
            st.write(tr("✅ SPYキャッシュ最終日: {d}", d=str(df.index[-1].date())))
        except Exception:
            pass

        # NYSE 最新営業日
        today = pd.Timestamp.today().normalize()
        latest_trading_day = get_latest_nyse_trading_day(today)
        try:
            st.write(tr("🗓️ 直近のNYSE営業日: {d}", d=str(latest_trading_day.date())))
        except Exception:
            pass

        # 1つ前の営業日（当日営業時間帯の影響を避ける）
        try:
            nyse = mcal.get_calendar("NYSE")
            sched = nyse.schedule(start_date=today - pd.Timedelta(days=7), end_date=today)
            valid = sched.index.normalize()
            prev_trading_day = valid[-2] if len(valid) >= 2 else latest_trading_day
        except Exception:
            prev_trading_day = latest_trading_day

        # 米東部時間を取得
        try:
            ny_time = pd.Timestamp.now(tz="America/New_York").time()
        except Exception:
            ny_time = dtime(18, 0)

        # 古い場合は自動更新（米東部18:00以降）
        if df.index[-1].normalize() < prev_trading_day and ny_time >= dtime(18, 0):
            try:
                st.warning(tr("⚠ SPYキャッシュが古いため自動更新します..."))
            except Exception:
                pass
            try:
                result = subprocess.run(
                    ["python", "recover_spy_cache.py"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.stdout:
                    try:
                        st.text(result.stdout)
                    except Exception:
                        pass
                if result.stderr:
                    try:
                        st.error(result.stderr)
                    except Exception:
                        pass
            except Exception as e:
                st.error(tr("SPY自動更新失敗: {e}", e=str(e)))
                return None

            # 更新後再読み込み
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path, parse_dates=["Date"])
                    df.set_index("Date", inplace=True)
                    df = df.sort_index()
                    try:
                        st.success(tr("✅ SPYキャッシュ更新後: {d}", d=str(df.index[-1].date())))
                    except Exception:
                        pass
                except Exception as e:
                    st.error(tr("❌ SPY更新後の読み込みに失敗: {e}", e=str(e)))
                    return None
            else:
                st.error(tr("❌ 更新後もSPY.csvが存在しません"))
                return None
        else:
            try:
                st.write(tr("✅ SPYキャッシュは有効"))
            except Exception:
                pass

        return df

    except Exception as e:
        st.error(tr("❌ SPY読み込み失敗: {e}", e=str(e)))
        return None


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
                    st.warning(tr("❗SPYの終値列が見つかりません: {cols}", cols=str(list(spy_df.columns))))
                except Exception:
                    pass
                return spy_df

        spy_df["SMA100"] = SMAIndicator(pd.to_numeric(spy_df["Close"], errors="coerce"), window=100).sma_indicator()
        spy_df["SMA200"] = SMAIndicator(pd.to_numeric(spy_df["Close"], errors="coerce"), window=200).sma_indicator()
    return spy_df

