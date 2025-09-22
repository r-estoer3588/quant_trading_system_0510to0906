from __future__ import annotations

from pathlib import Path
import time
from typing import Any, cast

import pandas as pd
import streamlit as st

from common.cache_utils import save_prepared_data_cache
from common.i18n import language_selector, load_translations_from_dir, tr
from common.notifier import Notifier, get_notifiers_from_env, now_jst_str
from common.performance_summary import summarize as summarize_perf
from common.price_chart import save_price_chart
from common.ui_components import (
    run_backtest_app,
    save_signal_and_trade_logs,
    show_signal_trade_summary,
)
from common.ui_manager import UIManager
import common.ui_patch  # noqa: F401
from strategies.system7_strategy import System7Strategy

# Load translations and (optionally) show language selector
load_translations_from_dir(Path(__file__).parent / "translations")
if not st.session_state.get("_integrated_ui", False):
    language_selector()

SYSTEM_NAME = "System7"
DISPLAY_NAME = "システム7"

strategy: System7Strategy = System7Strategy()
notifiers: list[Notifier] = get_notifiers_from_env()


def run_tab(
    single_mode: bool | None = None,
    ui_manager: UIManager | None = None,
) -> None:
    """System7 タブを描画し、バックテストを実行する。"""
    st.header(
        tr(
            "{display_name} バックテスト（カタストロフィー・ヘッジ / SPYのみ）",
            display_name=DISPLAY_NAME,
        )
    )
    single_mode = st.checkbox(tr("単体モード（資金100%を使用）"), value=False)

    ui_base: UIManager = (
        ui_manager.system(SYSTEM_NAME) if ui_manager else UIManager().system(SYSTEM_NAME)
    )
    fetch_phase = ui_base.phase("fetch", title=tr("データ取得"))
    ind_phase = ui_base.phase("indicators", title=tr("インジケーター計算"))
    cand_phase = ui_base.phase("candidates", title=tr("候補選定"))
    # 通知トグルは共通UI(run_backtest_app)内に配置して順序を統一
    notify_key = f"{SYSTEM_NAME}_notify_backtest"
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
            limit_symbols=1,
            ui_manager=ui_base,
            single_mode=single_mode,
        ),
    )
    elapsed = time.time() - run_start
    results_df, _, data_dict, capital, candidates_by_date = _rb
    fetch_phase.log_area.write(tr("データ取得完了"))
    ind_phase.log_area.write(tr("インジケーター計算完了"))
    cand_phase.log_area.write(tr("候補選定完了"))

    if results_df is not None and candidates_by_date is not None:
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
            "処理時間": f"{elapsed:.2f}s",
        }

        # --- 追加: 資産推移と実行日数チェック（資金不足の可能性を判定） ---
        # 累積PnL から時系列の equity を作成（capital + cumulative_pnl）
        equity = (float(capital) + df2["cumulative_pnl"].astype(float)).astype(float)
        # 最小資産と最終資産
        min_equity = float(equity.min())
        final_equity = float(equity.iloc[-1]) if len(equity) > 0 else float(capital)
        stats["最小資産"] = f"{min_equity:.2f}"
        stats["最終資産"] = f"{final_equity:.2f}"

        # 実行日数（カレンダー日数）と取引記録数
        if "entry_date" in df2.columns and "exit_date" in df2.columns:
            start_dt = pd.to_datetime(df2["entry_date"]).min()
            end_dt = pd.to_datetime(df2["exit_date"]).max()
            total_calendar_days = (end_dt - start_dt).days + 1
            trade_records = len(df2)
            stats["実行日数"] = f"{total_calendar_days}日 (取引記録: {trade_records})"
        else:
            stats["実行日数"] = f"{len(df2)} レコード"

        # 日付インデックスを持つ equity Series を作成
        if "exit_date" in df2.columns:
            eq_series = pd.Series(equity.values, index=pd.to_datetime(df2["exit_date"]))
        elif "entry_date" in df2.columns:
            eq_series = pd.Series(equity.values, index=pd.to_datetime(df2["entry_date"]))
        else:
            eq_series = pd.Series(equity.values)
        # インデックスをソートして日次にリサンプル（取引の無い日も埋める）
        try:
            eq_series = eq_series.sort_index()
            # 日次にリサンプルして直近の値で補完（取引が無い日は直近の資産を保持）
            daily_eq = eq_series.resample("D").last().ffill()
        except Exception:
            # リサンプル不可ならそのまま使用
            daily_eq = eq_series

        # 常に要約を見える化（見落とし防止）
        try:
            st.metric(label=tr("最小資産"), value=f"{min_equity:.2f}")
            st.metric(label=tr("最終資産"), value=f"{final_equity:.2f}")
            st.caption(tr("※詳細は下の展開部を確認してください"))
        except Exception:
            pass

        # --- 常時表示: 資産推移の可視化（ラインチャート + 直近テーブル） ---
        try:
            if daily_eq is None or len(daily_eq) == 0:
                st.info(tr("資産推移データが存在しません。取引記録や累積PnL を確認してください。"))
            else:
                with st.expander(tr("資産推移（直近）"), expanded=False):
                    try:
                        st.line_chart(daily_eq)
                    except Exception:
                        # line_chartが失敗したらテーブルで代替
                        df_tbl = pd.DataFrame(
                            {
                                "date": pd.to_datetime(daily_eq.index),
                                "equity": daily_eq.values,
                            }
                        )
                        st.dataframe(df_tbl.tail(30).sort_values("date"))
                    # 直近30行を表示（表示用）
                    try:
                        df_recent = pd.DataFrame(
                            {
                                "date": pd.to_datetime(daily_eq.index),
                                "equity": daily_eq.values,
                            }
                        )
                        st.dataframe(df_recent.tail(30).sort_values("date"))
                    except Exception:
                        pass
        except Exception:
            pass
        # --- /常時表示ここまで ---

        # 資金が0以下になった日一覧（日次で判定）
        try:
            zero_days = daily_eq[daily_eq <= 0]
            if not zero_days.empty:
                first_zero_date = pd.to_datetime(zero_days.index[0])
                zero_count = len(zero_days)
                stats["資金尽きた日"] = f"{first_zero_date:%Y-%m-%d} (件数: {zero_count})"
                st.error(
                    tr(
                        "バックテスト中に資金が0以下になった日があります: {d} (件数: {n})",
                        d=f"{first_zero_date:%Y-%m-%d}",
                        n=zero_count,
                    )
                )
                # 自動展開してユーザーに見せる（問題ありなら展開）
                with st.expander(tr("資金が0以下だった日付一覧"), expanded=True):
                    df_log = pd.DataFrame(
                        {
                            "date": pd.to_datetime(zero_days.index),
                            "equity": zero_days.values,
                        }
                    )
                    df_log = df_log.sort_values("date")
                    st.dataframe(df_log)
                    try:
                        try:
                            from config.settings import get_settings

                            settings2 = get_settings(create_dirs=True)
                            round_dec = getattr(settings2.cache, "round_decimals", None)
                        except Exception:
                            round_dec = None
                        try:
                            from common.cache_manager import round_dataframe

                            out_df = round_dataframe(df_log, round_dec)
                        except Exception:
                            out_df = df_log
                        csv = out_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label=tr("資金尽きた日一覧をCSVでダウンロード"),
                            data=csv,
                            file_name=f"{SYSTEM_NAME}_zero_equity_log.csv",
                            mime="text/csv",
                        )
                    except Exception:
                        pass
        except Exception:
            # 日次判定失敗でも先に進める
            pass

        # 初期資金の閾値（例: 10%）を下回った日一覧（日次で判定）
        try:
            threshold = float(capital) * 0.1
            low_days = daily_eq[daily_eq <= threshold]
            if not low_days.empty:
                first_low_date = pd.to_datetime(low_days.index[0])
                low_count = len(low_days)
                stats["資金10%未満日"] = f"{first_low_date:%Y-%m-%d} (件数: {low_count})"
                st.warning(
                    tr(
                        (
                            "最終資産あるいは途中で初期資金の10%を下回った日があります: {d} "
                            "(件数: {n})"
                        ),
                        d=f"{first_low_date:%Y-%m-%d}",
                        n=low_count,
                    )
                )
                # 自動展開してユーザーに見せる（問題ありなら展開）
                with st.expander(tr("資金が10%未満だった日付一覧"), expanded=True):
                    df_low = pd.DataFrame(
                        {
                            "date": pd.to_datetime(low_days.index),
                            "equity": low_days.values,
                        }
                    )
                    df_low = df_low.sort_values("date")
                    st.dataframe(df_low)
                    try:
                        try:
                            from config.settings import get_settings

                            settings2 = get_settings(create_dirs=True)
                            round_dec = getattr(settings2.cache, "round_decimals", None)
                        except Exception:
                            round_dec = None
                        try:
                            from common.cache_manager import round_dataframe

                            out_df = round_dataframe(df_low, round_dec)
                        except Exception:
                            out_df = df_low
                        csv2 = out_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label=tr("資金10%未満日一覧をCSVでダウンロード"),
                            data=csv2,
                            file_name=f"{SYSTEM_NAME}_low_equity_log.csv",
                            mime="text/csv",
                        )
                    except Exception:
                        pass
        except Exception:
            pass
        # --- 追加ここまで ---

        # メトリクスは共通 show_results で統一表示
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
            _ = yearly_df
        except Exception:
            pass
        ranking: list[str] = (
            [str(s) for s in results_df["symbol"].head(10)]
            if "symbol" in results_df.columns
            else []
        )
        period: str = ""
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
        if st.session_state.get(notify_key, False):
            sent = False
            for n in notifiers:
                try:
                    _mention: str | None = (
                        "channel" if getattr(n, "platform", None) == "slack" else None
                    )
                    if hasattr(n, "send_backtest_ex"):
                        n.send_backtest_ex(
                            SYSTEM_NAME.lower(),
                            period,
                            stats,
                            ranking,
                            image_url=chart_url,
                            mention=_mention,
                        )
                    else:
                        n.send_backtest(
                            SYSTEM_NAME.lower(),
                            period,
                            stats,
                            ranking,
                        )
                    sent = True
                except Exception:
                    continue
            if sent:
                st.success(tr("通知を送信しました"))
            else:
                st.warning(tr("通知の送信に失敗しました"))
    else:
        # Fallback view from session state
        prev_res = st.session_state.get(f"{SYSTEM_NAME}_results_df")
        prev_data = st.session_state.get(f"{SYSTEM_NAME}_prepared_dict")
        prev_cap = st.session_state.get(f"{SYSTEM_NAME}_capital_saved")
        if prev_res is not None:
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
