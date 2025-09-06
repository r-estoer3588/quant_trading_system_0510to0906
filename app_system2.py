from pathlib import Path
from typing import cast

import pandas as pd
import streamlit as st

from common.cache_utils import save_prepared_data_cache
from common.equity_curve import save_equity_curve
from common.i18n import language_selector, load_translations_from_dir, tr
from common.notifier import Notifier, get_notifiers_from_env
from common.performance_summary import summarize as summarize_perf
from common.ui_components import (
    run_backtest_app,
    save_signal_and_trade_logs,
    show_signal_trade_summary,
)
from common.ui_manager import UIManager
import common.ui_patch  # noqa: F401
from strategies.system2_strategy import System2Strategy

# 翻訳辞書ロードと言語選択
load_translations_from_dir(Path(__file__).parent / "translations")
if not st.session_state.get("_integrated_ui", False):
    language_selector()


# 戦略インスタンス
strategy: System2Strategy = System2Strategy()
notifiers: list[Notifier] = get_notifiers_from_env()


def display_adx7_ranking(
    candidates_by_date,
    years: int = 5,
    top_n: int = 100,
    title: str = "📊 System2 日別 ADX7 ランキング（直近{years}年 / 上位{top_n}銘柄）",
):
    """System2 固有のADX7日別ランキングを表示する。"""
    if not candidates_by_date:
        st.warning("ADX7ランキングが空です")
        return

    rows = []
    for date, cands in candidates_by_date.items():
        for c in cands:
            rows.append({"Date": date, "symbol": c.get("symbol"), "ADX7": c.get("ADX7")})
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])  # type: ignore[arg-type]

    start_date = pd.Timestamp.now() - pd.DateOffset(years=years)
    df = df[df["Date"] >= start_date]

    df["ADX7_Rank"] = df.groupby("Date")["ADX7"].rank(ascending=False, method="first")
    df = df.sort_values(["Date", "ADX7_Rank"], ascending=[True, True])
    df = df.groupby("Date").head(top_n)

    with st.expander(title.format(years=years, top_n=top_n), expanded=False):
        st.dataframe(
            df.reset_index(drop=True)[["Date", "ADX7_Rank", "symbol", "ADX7"]],
            hide_index=False,
        )


def run_tab(ui_manager: UIManager | None = None) -> None:
    st.header("System2 バックテスト（ショートRSIスパイク + ADX傾きランキング）")
    ui: UIManager = ui_manager or UIManager()
    # 通知トグルは共通UI(run_backtest_app)内に配置して順序を統一
    notify_key = "System2_notify_backtest"
    _rb = cast(
        tuple[
            pd.DataFrame | None,
            pd.DataFrame | None,
            dict[str, pd.DataFrame] | None,
            float,
            object | None,
        ],
        run_backtest_app(strategy, system_name="System2", limit_symbols=100, ui_manager=ui),
    )
    results_df, _, data_dict, capital, candidates_by_date = _rb
    # 実行直後の表示・保存
    if results_df is not None and candidates_by_date is not None:
        display_adx7_ranking(candidates_by_date)
        summary_df = show_signal_trade_summary(data_dict, results_df, "System2")
        with st.expander(tr("取引ログ・保存ファイル"), expanded=False):
            save_signal_and_trade_logs(summary_df, results_df, "System2", capital)
        if data_dict is not None:
            save_prepared_data_cache(data_dict, "System2")
        summary, df2 = summarize_perf(results_df, capital)
        try:
            _max_dd = float(df2["drawdown"].min())
        except Exception:
            _max_dd = float(getattr(summary, "max_drawdown", 0.0))
        try:
            _dd_pct = float((df2["drawdown"] / (float(capital) + df2["cum_max"])).min() * 100)
        except Exception:
            _dd_pct = 0.0
        stats: dict[str, str] = {
            "総リターン": f"{summary.total_return:.2f}",
            "最大DD": f"{_max_dd:.2f} ({_dd_pct:.2f}%)",
            "Sharpe": f"{summary.sharpe:.2f}",
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
                    "year": ye.index.year,
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
        # equity image and mention for Slack
        _img_path, _img_url = save_equity_curve(results_df, capital, "System2")
        if st.session_state.get(notify_key, False):
            sent = False
            for n in notifiers:
                try:
                    _mention: str | None = (
                        "channel" if getattr(n, "platform", None) == "slack" else None
                    )
                    if hasattr(n, "send_backtest_ex"):
                        n.send_backtest_ex(
                            "system2",
                            period,
                            stats,
                            ranking,
                            image_url=_img_url,
                            mention=_mention,
                        )
                    else:
                        n.send_backtest("system2", period, stats, ranking)
                    sent = True
                except Exception:
                    continue
            if sent:
                st.success(tr("通知を送信しました"))
            else:
                st.warning(tr("通知の送信に失敗しました"))
    # リラン時のフォールバック表示（セッションから復元）
    elif results_df is None and candidates_by_date is None:
        prev_res = st.session_state.get("System2_results_df")
        prev_cands = st.session_state.get("System2_candidates_by_date")
        prev_data = st.session_state.get("System2_prepared_dict")
        prev_cap = st.session_state.get("System2_capital_saved")
        if prev_res is not None and prev_cands is not None:
            display_adx7_ranking(prev_cands)
            _ = show_signal_trade_summary(prev_data, prev_res, "System2")
            try:
                from common.ui_components import show_results

                show_results(prev_res, prev_cap or 0.0, "System2", key_context="prev")
            except Exception:
                pass


if __name__ == "__main__":
    import sys

    if "streamlit" not in sys.argv[0]:
        run_tab()
