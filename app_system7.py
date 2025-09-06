import streamlit as st
import common.ui_patch  # noqa: F401
import pandas as pd
from strategies.system7_strategy import System7Strategy
from common.ui_components import (
    run_backtest_app,
    show_signal_trade_summary,
    save_signal_and_trade_logs,
)
from common.cache_utils import save_prepared_data_cache
from common.ui_manager import UIManager
from pathlib import Path
from common.i18n import tr, load_translations_from_dir, language_selector
from common.performance_summary import summarize as summarize_perf
from common.notifier import get_notifiers_from_env
from common.equity_curve import save_equity_curve
import os

# Load translations and (optionally) show language selector
load_translations_from_dir(Path(__file__).parent / "translations")
if not st.session_state.get("_integrated_ui", False):
    language_selector()

strategy = System7Strategy()
notifiers = get_notifiers_from_env()


def run_tab(single_mode=None, ui_manager=None):
    st.header(tr("System7 バックテスト（カタストロフィー・ヘッジ / SPYのみ）"))
    single_mode = st.checkbox(tr("単体モード（資金100%を使用）"), value=False)

    ui = ui_manager or UIManager()
    # 通知トグルは共通UI(run_backtest_app)内に配置して順序を統一
    notify_key = "System7_notify_backtest"
    results_df, _, data_dict, capital, candidates_by_date = run_backtest_app(
        strategy,
        system_name="System7",
        limit_symbols=1,
        ui_manager=ui,
        single_mode=single_mode,
    )

    if st.checkbox(tr("チェック: インジケーターの確認"), value=False):
        if data_dict:
            for sym, df in data_dict.items():
                st.write(tr("例: 2020年02月〜03月"))
                st.dataframe(df.loc["2020-02-01":"2020-03-31"])  # 確認用
        else:
            st.info(tr("データが取得できていないため表示できません。バックテストを先に実行してください。"))

    if results_df is not None and candidates_by_date is not None:
        summary_df = show_signal_trade_summary(data_dict, results_df, "System7")
        with st.expander(tr("取引ログ・保存ファイル"), expanded=False):
            save_signal_and_trade_logs(summary_df, results_df, "System7", capital)
        save_prepared_data_cache(data_dict, "System7")
        summary, df2 = summarize_perf(results_df, capital)
        try:
            _max_dd = float(df2["drawdown"].min())
        except Exception:
            _max_dd = float(getattr(summary, "max_drawdown", 0.0))
        try:
            _dd_pct = float((df2["drawdown"] / (float(capital) + df2["cum_max"])) .min() * 100)
        except Exception:
            _dd_pct = 0.0
        stats = {
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
        ranking = (
            [str(s) for s in results_df["symbol"].head(10)]
            if "symbol" in results_df.columns
            else []
        )
        period = ""
        if "entry_date" in results_df.columns and "exit_date" in results_df.columns:
            start = pd.to_datetime(results_df["entry_date"]).min()
            end = pd.to_datetime(results_df["exit_date"]).max()
            period = f"{start:%Y-%m-%d}〜{end:%Y-%m-%d}"
        _img_path, _img_url = save_equity_curve(results_df, capital, "System7")
        if st.session_state.get(notify_key, False):
            sent = False
            for n in notifiers:
                try:
                    _mention = "channel" if getattr(n, "platform", None) == "slack" else None
                    if hasattr(n, "send_backtest_ex"):
                        n.send_backtest_ex("system7", period, stats, ranking, image_url=_img_url, mention=_mention)
                    else:
                        n.send_backtest("system7", period, stats, ranking)
                    sent = True
                except Exception:
                    continue
            if sent:
                st.success(tr("通知を送信しました"))
            else:
                st.warning(tr("通知の送信に失敗しました"))
    else:
        # Fallback view from session state
        prev_res = st.session_state.get("System7_results_df")
        prev_data = st.session_state.get("System7_prepared_dict")
        prev_cap = st.session_state.get("System7_capital_saved")
        if prev_res is not None:
            _ = show_signal_trade_summary(prev_data, prev_res, "System7")
            try:
                from common.ui_components import show_results
                show_results(prev_res, prev_cap or 0.0, "System7", key_context="prev")
            except Exception:
                pass


if __name__ == "__main__":
    import sys
    if "streamlit" not in sys.argv[0]:
        run_tab()
