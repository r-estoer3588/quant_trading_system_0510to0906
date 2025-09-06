from __future__ import annotations

import os
import subprocess
from datetime import time as dtime

import pandas as pd
import pandas_market_calendars as mcal
import streamlit as st
from ta.trend import SMAIndicator

from common.i18n import tr


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
        spy_df = get_spy_data_cached()
    if spy_df is not None and not getattr(spy_df, "empty", True):
        spy_df["SMA100"] = SMAIndicator(spy_df["Close"], window=100).sma_indicator()
        spy_df["SMA200"] = SMAIndicator(spy_df["Close"], window=200).sma_indicator()
    return spy_df

