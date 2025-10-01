"""System1 Streamlit アプリ."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast

import pandas as pd
import streamlit as st

import common.ui_patch  # noqa: F401
from common.i18n import language_selector, load_translations_from_dir, tr
from common.notifier import Notifier, now_jst_str
from common.performance_summary import summarize as summarize_perf
from common.price_chart import save_price_chart
from common.ui_components import (
    clean_date_column,
    display_roc200_ranking,
    run_backtest_app,
    save_signal_and_trade_logs,
    show_signal_trade_summary,
)
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


def run_tab(
    spy_df: pd.DataFrame | None = None,
    ui_manager: object | None = None,
) -> None:
    st.header(tr(f"{DISPLAY_NAME} — ロング・トレンド＋ハイ・モメンタム 候補銘柄ランキング"))

    spy_df = spy_df if spy_df is not None else get_spy_with_indicators()
    if spy_df is None or getattr(spy_df, "empty", True):
        st.error(tr("SPYデータの取得に失敗しました。キャッシュを更新してください"))
        new_df = get_spy_with_indicators()
        if new_df is None or getattr(new_df, "empty", True):
            st.warning("再読み込みに失敗しました。必要ならアプリを再起動してください。")
            return
        spy_df = new_df
    run_start = time.time()
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
    elapsed = time.time() - run_start
    results_df, merged_df, data_dict, capital, _ = _rb

    if results_df is not None and merged_df is not None:
        daily_df = clean_date_column(merged_df, col_name="Date")
        display_roc200_ranking(daily_df, title=f"📊 {DISPLAY_NAME} 日別ROC200ランキング")

        signal_summary_df = show_signal_trade_summary(
            merged_df, results_df, SYSTEM_NAME, display_name=DISPLAY_NAME
        )
        with st.expander(tr("取引ログ・保存ファイル"), expanded=False):
            save_signal_and_trade_logs(signal_summary_df, results_df, SYSTEM_NAME, capital)
        # Prepared data cache save removed (deprecated feature)

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
            "処理時間": f"{elapsed:.2f}s",
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
            ys = daily_eq.resample("YE").first()
            ye = daily_eq.resample("YE").last()
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

        period = ""
        if "entry_date" in results_df.columns and "exit_date" in results_df.columns:
            start = pd.to_datetime(results_df["entry_date"]).min()
            end = pd.to_datetime(results_df["exit_date"]).max()
            period = f"{start:%Y-%m-%d}〜{end:%Y-%m-%d}"
        chart_url = None
        if not results_df.empty and "symbol" in results_df.columns:
            try:
                top_sym = results_df.sort_values("pnl", ascending=False)["symbol"].iloc[0]
                _, chart_url = save_price_chart(str(top_sym), trades=results_df)
            except Exception:
                chart_url = None

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
                            image_url=chart_url,
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
