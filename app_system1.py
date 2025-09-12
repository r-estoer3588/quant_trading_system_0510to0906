"""System1 Streamlit アプリ."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd
import streamlit as st
import requests
from config.settings import get_settings

from common.cache_utils import save_prepared_data_cache
from common.equity_curve import save_equity_curve
from common.i18n import language_selector, load_translations_from_dir, tr
from common.notifier import Notifier, now_jst_str
from common.performance_summary import summarize as summarize_perf
from common.ui_components import (
    clean_date_column,
    display_roc200_ranking,
    run_backtest_app,
    save_signal_and_trade_logs,
    show_signal_trade_summary,
)
import common.ui_patch  # noqa: F401
from common.utils_spy import get_spy_with_indicators
from strategies.system1_strategy import System1Strategy

# Notifier は存在しない環境もあるため安全にフォールバック
try:  # noqa: WPS501
    from common.notifier import get_notifiers_from_env  # type: ignore
except Exception:  # pragma: no cover

    def get_notifiers_from_env():  # type: ignore
        return []


# 翻訳辞書ロードと言語選択（統合 UI 内ではスキップ）
load_translations_from_dir(Path(__file__).parent / "translations")
if not st.session_state.get("_integrated_ui", False):
    language_selector()


SYSTEM_NAME = "System1"
DISPLAY_NAME = "システム1"

strategy: System1Strategy = System1Strategy()
notifiers: list[Notifier] = get_notifiers_from_env()


# --- 追加: SPY データ救済用ユーティリティ ---------------------------------
def _resolve_spy_csv_path() -> Path:
    """SPY.csv の保存先（既定: data/SPY.csv）を返す。"""
    p = Path("data")
    p.mkdir(exist_ok=True)
    return p / "SPY.csv"


def _download_spy_data(save_path: Path, years: int = 15) -> pd.DataFrame | None:
    """EODHD API を利用して SPY 日足データを取得し CSV 保存する。
    years: 取得年数（過去 n 年分）
    戻り値: 成功時 DataFrame / 失敗時 None
    """
    try:
        settings = get_settings()  # type: ignore
        api_key = getattr(settings, "eodhd_api_key", None) or os.getenv("EODHD_API_KEY")  # type: ignore
    except Exception:
        api_key = os.getenv("EODHD_API_KEY")  # type: ignore
    if not api_key:
        return None

    try:
        end = pd.Timestamp.utcnow().normalize()
        start = end - pd.DateOffset(years=years)
        url = (
            "https://eodhd.com/api/eod/SPY.US"
            f"?from={start:%Y-%m-%d}&to={end:%Y-%m-%d}&period=d&fmt=json&api_token={api_key}"
        )
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if not data:
            return None
        df = pd.DataFrame(data)
        # 想定フィールド: date, open, high, low, close, adjusted_close, volume
        # 列存在チェックしつつリネーム
        rename_map = {
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "adjusted_close": "Adj Close",
            "volume": "Volume",
        }
        # 欠損列はスキップ
        df = df[[c for c in rename_map if c in df.columns]].rename(columns=rename_map)
        if "Date" not in df.columns or df.empty:
            return None
        df.sort_values("Date", inplace=True)
        df.to_csv(save_path, index=False)
        return df
    except Exception:
        return None


# ---------------------------------------------------------------------------


def run_tab(spy_df: pd.DataFrame | None = None, ui_manager: object | None = None) -> None:
    st.header(tr(f"{DISPLAY_NAME} — ロング・トレンド＋ハイ・モメンタム 候補銘柄ランキング"))

    spy_df = spy_df if spy_df is not None else get_spy_with_indicators()
    if spy_df is None or getattr(spy_df, "empty", True):
        st.error(tr("SPYデータの取得に失敗しました。キャッシュを更新してください"))
        with st.expander("SPYデータ自動ダウンロード (EODHD API 利用)", expanded=True):
            st.write(
                "SPY.csv が存在しないためバックテストを実行できません。"
                " 以下のボタンで EODHD API から最新データを取得して再読み込みします。"
            )
            col1, col2 = st.columns(2)
            with col1:
                years = st.number_input("取得年数", 5, 30, 15, 1)
            with col2:
                do_dl = st.button("SPYデータをダウンロード / 更新", type="primary")
            if do_dl:
                path = _resolve_spy_csv_path()
                with st.spinner("SPYデータ取得中 (EODHD)..."):
                    raw = _download_spy_data(path, years=years)
                if raw is None or raw.empty:
                    st.warning("ダウンロードに失敗しました (APIキー未設定/通信失敗/レスポンス空)")
                    return
                st.success(f"保存しました: {path}")
                # 取得後に再トライ
                new_df = get_spy_with_indicators()
                if new_df is None or getattr(new_df, "empty", True):
                    st.warning("再読み込みに失敗しました。必要ならアプリを再起動してください。")
                    return
                spy_df = new_df
            else:
                return
    _rb = cast(
        tuple[
            pd.DataFrame | None,
            pd.DataFrame | None,
            dict[str, pd.DataFrame] | None,
            float,
            object | None,
        ],
        run_backtest_app(
            strategy,
            system_name=SYSTEM_NAME,
            limit_symbols=10,
            spy_df=spy_df,
            ui_manager=ui_manager,
        ),
    )
    results_df, merged_df, data_dict, capital, _ = _rb

    if results_df is not None and merged_df is not None:
        daily_df = clean_date_column(merged_df, col_name="Date")
        display_roc200_ranking(daily_df, title=f"📊 {DISPLAY_NAME} 日別ROC200ランキング")

        signal_summary_df = show_signal_trade_summary(
            merged_df, results_df, SYSTEM_NAME, display_name=DISPLAY_NAME
        )
        with st.expander(tr("取引ログ・保存ファイル"), expanded=False):
            save_signal_and_trade_logs(signal_summary_df, results_df, SYSTEM_NAME, capital)
        if data_dict is not None:
            save_prepared_data_cache(data_dict, SYSTEM_NAME)

        summary, df2 = summarize_perf(results_df, capital)
        # 統合タブと同じ算出（ピーク資産比の%）で表示
        max_dd = (
            float(df2["drawdown"].min())
            if "drawdown" in df2.columns
            else float(summary.max_drawdown)
        )
        try:
            max_dd_pct = float((df2["drawdown"] / (float(capital) + df2["cum_max"])).min() * 100)
        except Exception:
            max_dd_pct = (max_dd / capital * 100) if capital else 0.0
        stats: dict[str, Any] = {
            "総リターン": f"{summary.total_return:.2f}",
            "最大DD": f"{max_dd:.2f} ({max_dd_pct:.2f}%)",
            "Sharpe": f"{summary.sharpe:.2f}",
            "実施日時": now_jst_str(),
            "銘柄数": len(data_dict) if data_dict else 0,
            "開始資金": int(capital),
        }
        # 画面上にも DD と DD% を表示（統合サマリーと同様）
        # show_results 側で統一表示するため、ここでのメトリクス表示は不要
        try:
            pass
        except Exception:
            pass
        # 年別サマリー（%表記）
        try:
            equity = float(capital) + df2["cumulative_pnl"].astype(float)
            equity.index = pd.to_datetime(df2["exit_date"])  # type: ignore
            daily_eq = equity.resample("D").last().ffill()
            ys = daily_eq.resample("Y").first()
            ye = daily_eq.resample("Y").last()
            yearly_df = pd.DataFrame(
                {
                    "year": pd.to_datetime(ye.index).year,
                    "pnl": (ye - ys).round(2).values,
                    "return_pct": ((ye / ys - 1) * 100).values,
                }
            )
            # 年次サマリーは共通の show_results で統一表示
            _ = yearly_df  # keep computed but unused to avoid lints
        except Exception:
            pass
        try:
            if hasattr(summary, "profit_factor"):
                stats["PF"] = f"{summary.profit_factor:.2f}"
            if hasattr(summary, "win_rate"):
                stats["勝率(%)"] = f"{summary.win_rate:.2f}"
        except Exception:
            pass

        ranking: list[dict[str, Any] | str] = []
        try:
            last_date = pd.to_datetime(daily_df["Date"]).max()
            cols = {c.lower(): c for c in daily_df.columns}
            roc_col = cols.get("roc200")
            vol_col = cols.get("volume") or cols.get("vol")
            if roc_col:
                today = daily_df[pd.to_datetime(daily_df["Date"]) == last_date]
                today = today.sort_values(roc_col, ascending=False).head(10)
                for _, r in today.iterrows():
                    item: dict[str, Any] = {"symbol": str(r.get("symbol"))}
                    val: Any = r.get(roc_col)
                    try:
                        if val is None:
                            item["roc"] = val
                        else:
                            item["roc"] = float(val)
                    except Exception:
                        item["roc"] = val
                    if vol_col is not None:
                        item["volume"] = r.get(vol_col)
                    ranking.append(item)
            elif "symbol" in results_df.columns:
                ranking = [str(s) for s in results_df["symbol"].head(10)]
        except Exception:
            if "symbol" in results_df.columns:
                ranking = [str(s) for s in results_df["symbol"].head(10)]

        img_path, img_url = save_equity_curve(results_df, capital, SYSTEM_NAME)
        period = ""
        if "entry_date" in results_df.columns and "exit_date" in results_df.columns:
            start = pd.to_datetime(results_df["entry_date"]).min()
            end = pd.to_datetime(results_df["exit_date"]).max()
            period = f"{start:%Y-%m-%d}〜{end:%Y-%m-%d}"

        notify_key = f"{SYSTEM_NAME}_notify_backtest"
        if st.session_state.get(notify_key, False):
            sent = False
            for n in notifiers:
                try:
                    mention: str | None = (
                        "channel" if getattr(n, "platform", None) == "slack" else None
                    )
                    if hasattr(n, "send_backtest_ex"):
                        n.send_backtest_ex(
                            "system1",
                            period,
                            stats,
                            ranking,
                            image_url=img_url,
                            mention=mention,
                        )
                    else:
                        ranking_list_str = [
                            (
                                x
                                if isinstance(x, str)
                                else str(getattr(x, "get", lambda *_: "?")("symbol"))
                            )
                            for x in ranking
                        ]
                        n.send_backtest("system1", period, stats, ranking_list_str)
                    sent = True
                except Exception:
                    continue
            if sent:
                st.success(tr("通知を送信しました"))
            else:
                st.warning(tr("通知の送信に失敗しました"))

    elif results_df is None and merged_df is None:
        prev_res = st.session_state.get(f"{SYSTEM_NAME}_results_df")
        prev_merged = st.session_state.get(f"{SYSTEM_NAME}_merged_df")
        prev_cap = st.session_state.get(f"{SYSTEM_NAME}_capital_saved")
        if prev_res is not None and prev_merged is not None:
            daily_df = clean_date_column(prev_merged, col_name="Date")
            display_roc200_ranking(
                daily_df, title=f"📊 {DISPLAY_NAME} 日別ROC200ランキング（保存済み）"
            )
            _ = show_signal_trade_summary(
                prev_merged, prev_res, SYSTEM_NAME, display_name=DISPLAY_NAME
            )
            try:
                from common.ui_components import show_results

                show_results(prev_res, prev_cap or 0.0, SYSTEM_NAME, key_context="prev")
            except Exception:
                pass


if __name__ == "__main__":
    import sys

    if "streamlit" not in sys.argv[0]:
        run_tab()
