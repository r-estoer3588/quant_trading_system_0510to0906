from __future__ import annotations
from pathlib import Path
from typing import Any, cast
import time

import pandas as pd
import streamlit as st

from common.cache_utils import save_prepared_data_cache
from common.equity_curve import save_equity_curve
from common.i18n import language_selector, load_translations_from_dir, tr
from common.notifier import Notifier, get_notifiers_from_env, now_jst_str
from common.performance_summary import summarize as summarize_perf
from common.ui_components import (
    run_backtest_app,
    save_signal_and_trade_logs,
    show_signal_trade_summary,
)
from common.ui_manager import UIManager
from common.logging_utils import log_with_progress
import common.ui_patch  # noqa: F401
from strategies.system5_strategy import System5Strategy

# 翻訳辞書ロード + 言語選択
load_translations_from_dir(Path(__file__).parent / "translations")
if not st.session_state.get("_integrated_ui", False):
    language_selector()


SYSTEM_NAME = "System5"
DISPLAY_NAME = "システム5"

strategy: System5Strategy = System5Strategy()
notifiers: list[Notifier] = get_notifiers_from_env()


def display_adx_ranking(
    candidates_by_date,
    years: int = 5,
    top_n: int = 100,
):
    if not candidates_by_date:
        st.warning(tr("ADX7ランキングデータがありません"))
        return
    rows: list[dict[str, Any]] = []
    total = len(candidates_by_date)
    progress = st.progress(0)
    log_area = st.empty()
    start = time.time()
    for i, (date, cands) in enumerate(candidates_by_date.items(), 1):
        for c in cands:
            rows.append(
                {
                    "Date": date,
                    "symbol": c.get("symbol"),
                    "ADX7": c.get("ADX7"),
                }
            )
        log_with_progress(
            i,
            total,
            start,
            prefix="ADX7ランキング",
            log_func=log_area.write,
            progress_func=progress.progress,
            unit=tr("days"),
        )
    progress.empty()
    log_area.write(tr("ADX7ランキング完了"))
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])  # type: ignore[arg-type]
    start_date = pd.Timestamp.now() - pd.DateOffset(years=years)
    df = df[df["Date"] >= start_date]
    df["ADX7_Rank"] = df.groupby("Date")["ADX7"].rank(
        ascending=False,
        method="first",
    )
    df = df.sort_values(["Date", "ADX7_Rank"], ascending=[True, True])
    df = df.groupby("Date").head(top_n)
    title = tr(
        "{display_name} ADX7 ランキング（直近{years}年 / 上位{top_n}銘柄）",
        display_name=DISPLAY_NAME,
        years=years,
        top_n=top_n,
    )
    with st.expander(title, expanded=False):
        st.dataframe(
            df.reset_index(drop=True)[["Date", "ADX7_Rank", "symbol", "ADX7"]],
            hide_index=False,
        )


def run_tab(ui_manager: UIManager | None = None) -> None:
    st.header(
        tr(
            "{display_name} バックテスト（ロング・ミーンリバージョン＋ADXフィルター）",
            display_name=DISPLAY_NAME,
        )
    )
    ui_base: UIManager = (
        ui_manager.system(SYSTEM_NAME) if ui_manager else UIManager().system(SYSTEM_NAME)
    )
    fetch_phase = ui_base.phase("fetch", title=tr("データ取得"))
    ind_phase = ui_base.phase("indicators", title=tr("インジケーター計算"))
    cand_phase = ui_base.phase("candidates", title=tr("候補選定"))
    # 通知トグルは共通UI(run_backtest_app)内に配置して順序を統一
    notify_key = f"{SYSTEM_NAME}_notify_backtest"
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
            limit_symbols=100,
            ui_manager=ui_base,
        ),
    )
    results_df, _, data_dict, capital, candidates_by_date = _rb
    fetch_phase.log_area.write(tr("データ取得完了"))
    ind_phase.log_area.write(tr("インジケーター計算完了"))
    cand_phase.log_area.write(tr("候補選定完了"))
    if results_df is not None and candidates_by_date is not None:
        display_adx_ranking(candidates_by_date)
        summary_df = show_signal_trade_summary(
            data_dict,
            results_df,
            SYSTEM_NAME,
            display_name=DISPLAY_NAME,
        )
        with st.expander(tr("取引ログ・保存ファイル"), expanded=False):
            save_signal_and_trade_logs(
                summary_df,
                results_df,
                SYSTEM_NAME,
                capital,
            )
        if data_dict is not None:
            save_prepared_data_cache(data_dict, SYSTEM_NAME)
        summary, df2 = summarize_perf(results_df, capital)
        try:
            _max_dd = float(df2["drawdown"].min())
        except Exception:
            _max_dd = float(getattr(summary, "max_drawdown", 0.0))
        try:
            _dd_pct = float((df2["drawdown"] / (float(capital) + df2["cum_max"])).min() * 100)
        except Exception:
            _dd_pct = 0.0
        stats: dict[str, Any] = {
            "総リターン": f"{summary.total_return:.2f}",
            "最大DD": f"{_max_dd:.2f} ({_dd_pct:.2f}%)",
            "Sharpe": f"{summary.sharpe:.2f}",
            "実施日時": now_jst_str(),
            "銘柄数": len(data_dict) if data_dict else 0,
            "開始資金": int(capital),
        }
        # メトリクスは共通 show_results で統一表示
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
                    "pnl": (ye - ys).values,
                    "return_pct": ((ye / ys - 1) * 100).values,
                }
            )
            # 年次サマリーは共通の show_results で統一表示
            _ = yearly_df
        except Exception:
            pass
        ranking: list[str] = (
            [str(s) for s in results_df["symbol"].head(10)]
            if "symbol" in results_df.columns
            else []
        )
        period = ""
        if "entry_date" in results_df.columns and "exit_date" in results_df.columns:
            start = pd.to_datetime(results_df["entry_date"]).min()
            end = pd.to_datetime(results_df["exit_date"]).max()
            period = f"{start:%Y-%m-%d}〜{end:%Y-%m-%d}"
        _img_path, _img_url = save_equity_curve(
            results_df,
            capital,
            SYSTEM_NAME,
        )
        if st.session_state.get(notify_key, False):
            sent = False
            for n in notifiers:
                try:
                    _mention: str | None = (
                        "channel" if getattr(n, "platform", None) == "slack" else None
                    )
                    if hasattr(n, "send_backtest_ex"):
                        n.send_backtest_ex(
                            "system5",
                            period,
                            stats,
                            ranking,
                            image_url=_img_url,
                            mention=_mention,
                        )
                    else:
                        n.send_backtest("system5", period, stats, ranking)
                    sent = True
                except Exception:
                    continue
            if sent:
                st.success(tr("通知を送信しました"))
            else:
                st.warning(tr("通知の送信に失敗しました"))
    else:
        # フォールバック表示（セッション保存から復元）
        prev_res = st.session_state.get(f"{SYSTEM_NAME}_results_df")
        prev_cands = st.session_state.get(f"{SYSTEM_NAME}_candidates_by_date")
        prev_data = st.session_state.get(f"{SYSTEM_NAME}_prepared_dict")
        prev_cap = st.session_state.get(f"{SYSTEM_NAME}_capital_saved")
        if prev_res is not None and prev_cands is not None:
            display_adx_ranking(prev_cands)
            _ = show_signal_trade_summary(
                prev_data,
                prev_res,
                SYSTEM_NAME,
                display_name=DISPLAY_NAME,
            )
            try:
                from common.ui_components import show_results

                show_results(
                    prev_res,
                    prev_cap or 0.0,
                    SYSTEM_NAME,
                    key_context="prev",
                )
            except Exception:
                pass


if __name__ == "__main__":
    import sys

    if "streamlit" not in sys.argv[0]:
        run_tab()
