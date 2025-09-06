from __future__ import annotations

import os

import streamlit as st

from common.equity_curve import save_equity_curve
from common.i18n import tr
from common.notifier import now_jst_str
# Notifier は型ヒント用途のみ。実体は app 側で生成・注入する。
from typing import Any as Notifier  # forward alias for type hints
from common.performance_summary import summarize as summarize_perf
from common.ui_bridge import prepare_backtest_data_ui as _prepare_ui
from common.ui_bridge import run_backtest_with_logging_ui as _run_ui
from common.ui_manager import UIManager
from common.utils_spy import get_spy_data_cached, get_spy_with_indicators
from scripts.tickers_loader import get_all_tickers


def _show_sys_result(df, capital):
    if df is None or getattr(df, "empty", True):
        st.info(tr("no trades"))
        return
    summary, df2 = summarize_perf(df, capital)
    d = summary.to_dict()
    cols = st.columns(6)
    # 統合タブと同じ算出式（ピーク資産比の%）
    try:
        dd_pct = (df2["drawdown"] / (capital + df2["cum_max"])) .min() * 100
    except Exception:
        dd_pct = 0.0
    cols[0].metric(tr("trades"), d.get("trades"))
    cols[1].metric(tr("total pnl"), f"{d.get('total_return', 0):.2f}")
    cols[2].metric(tr("win rate (%)"), f"{d.get('win_rate', 0):.2f}")
    cols[3].metric("PF", f"{d.get('profit_factor', 0):.2f}")
    cols[4].metric("Sharpe", f"{d.get('sharpe', 0):.2f}")
    cols[5].metric(
        tr("max drawdown"),
        f"{d.get('max_drawdown', 0):.2f}",
        f"{dd_pct:.2f}%",
    )
    st.dataframe(df)


def render_integrated_tab(settings, notifier: Notifier) -> None:
    """統合バックテストタブの描画"""
    st.subheader(tr("Integrated Backtest (Systems 1-7)"))
    from common.holding_tracker import display_holding_heatmap, generate_holding_matrix
    from common.integrated_backtest import (
        DEFAULT_ALLOCATIONS,
        build_system_states,
        run_integrated_backtest,
    )

    capital_i = st.number_input(
        tr("capital (USD)"),
        min_value=1000,
        value=int(settings.ui.default_capital),
        step=1000,
        key="integrated_capital",
    )
    all_tickers = get_all_tickers()
    limit_i = st.number_input(
        tr("symbol limit"),
        min_value=50,
        max_value=len(all_tickers),
        value=min(500, len(all_tickers)),
        step=50,
        key="integrated_limit",
    )
    use_all = st.checkbox(tr("use all symbols"), key="integrated_all")
    colA, colB = st.columns(2)
    with colA:
        allow_gross = st.checkbox(
            tr("allow gross leverage (sum cost can exceed capital)"),
            value=False,
            key="integrated_gross",
        )
    with colB:
        st.caption(tr("allocation is fixed: long 1/3/4/5: each 25%, short 2:40%,6:40%,7:20%"))
    colL, colS = st.columns(2)
    with colL:
        long_share = st.slider(
            tr("long bucket share (%)"),
            min_value=0,
            max_value=100,
            value=50,
            step=5,
            key="integrated_long_share",
        )
    with colS:
        st.caption(tr("short bucket share = 100% - long"))
    short_share = 100 - int(long_share)
    notify_key_i = "Integrated_notify_backtest"
    if notify_key_i not in st.session_state:
        st.session_state[notify_key_i] = True
    _label_i = tr("バックテスト結果を通知する（Webhook）")
    try:
        if hasattr(st, "toggle"):
            st.toggle(_label_i, key=notify_key_i)
        else:
            st.checkbox(_label_i, key=notify_key_i)
        if not (os.getenv("DISCORD_WEBHOOK_URL") or os.getenv("SLACK_WEBHOOK_URL")):
            st.caption(tr("Webhook URL が未設定です（.env を確認）"))
    except Exception:
        pass

    run_btn_i = st.button(tr("run integrated"))

    if run_btn_i:
        symbols = all_tickers if use_all else all_tickers[: int(limit_i)]
        spy_base = get_spy_with_indicators(get_spy_data_cached())

        ui = UIManager().system("Integrated", title=tr("Integrated"))
        prep_phase = ui.phase("prepare", title=tr("prepare all systems"))
        prep_phase.info(tr("preparing per-system data / candidates..."))

        states = build_system_states(
            symbols,
            spy_df=spy_base,
            ui_bridge_prepare=_prepare_ui,
            ui_manager=ui,
        )

        import pandas as _pd

        sig_counts = {
            s.name: int(sum(len(v) for v in s.candidates_by_date.values())) for s in states
        }
        st.write(tr("signals per system:"))
        st.dataframe(_pd.DataFrame([sig_counts]))

        sim = ui.phase("simulate", title=tr("simulate integrated"))
        sim.info(tr("running integrated engine..."))

        # 進捗更新用のコールバック
        def _on_progress(i: int, total: int, start):
            try:
                sim.progress_bar.progress(0 if not total else i / total)
            except Exception:
                pass

        trades_df, _sig = run_integrated_backtest(
            states,
            capital_i,
            allocations=DEFAULT_ALLOCATIONS,
            long_share=float(long_share) / 100.0,
            short_share=float(short_share) / 100.0,
            allow_gross_leverage=allow_gross,
            on_progress=_on_progress,
        )

        # 終了時にプログレスバーを消す
        try:
            sim.progress_bar.empty()
        except Exception:
            pass

        st.markdown("---")
        st.subheader(tr("Integrated Summary"))
        if trades_df is not None and not trades_df.empty:
            summary, df2 = summarize_perf(trades_df, capital_i)
            d = summary.to_dict()
            d.update({
                "実施日時": now_jst_str(),
                "銘柄数": len(symbols),
                "開始資金": int(capital_i),
            })
            cols = st.columns(6)
            try:
                dd_pct = (df2["drawdown"] / (capital_i + df2["cum_max"])).min() * 100
            except Exception:
                dd_pct = 0.0
            cols[0].metric(tr("trades"), d.get("trades"))
            cols[1].metric(tr("total pnl"), f"{d.get('total_return', 0):.2f}")
            cols[2].metric(tr("win rate (%)"), f"{d.get('win_rate', 0):.2f}")
            cols[3].metric("PF", f"{d.get('profit_factor', 0):.2f}")
            cols[4].metric("Sharpe", f"{d.get('sharpe', 0):.2f}")
            cols[5].metric(
                tr("max drawdown"),
                f"{d.get('max_drawdown', 0):.2f}",
                f"{dd_pct:.2f}%",
            )
            st.dataframe(df2)

            try:
                equity = capital_i + df2["cumulative_pnl"]
                equity.index = _pd.to_datetime(df2["exit_date"])
                daily_eq = equity.resample("D").last().ffill()
                year_start = daily_eq.resample("Y").first()
                year_end = daily_eq.resample("Y").last()
                yearly_df = _pd.DataFrame(
                    {
                        "year": year_end.index.year,
                        "pnl": (year_end - year_start).values,
                        "return_pct": ((year_end / year_start - 1) * 100).values,
                    }
                )
                st.subheader(tr("yearly summary"))
                # 百分率として1桁で表示（例: 468.9% / -63.6%）
                st.dataframe(yearly_df.style.format({"return_pct": "{:.1f}%"}))
            except Exception:
                pass

            with st.expander("holdings heatmap", expanded=False):
                matrix = generate_holding_matrix(df2)
                display_holding_heatmap(matrix, title="Integrated - holdings heatmap")

            _ts_i = _pd.Timestamp.now().strftime("%Y-%m-%d_%H%M")
            st.download_button(
                label=tr("download integrated trades CSV"),
                data=df2.to_csv(index=False).encode("utf-8"),
                file_name=f"integrated_trades_{_ts_i}_{int(capital_i)}.csv",
                mime="text/csv",
                key="download_integrated_csv",
            )
            # Save equity curve image for integrated results
            _img_path, _img_url = save_equity_curve(df2, capital_i, "Integrated")
            _title = tr("Integrated Summary")
            _mention = "channel" if os.getenv("SLACK_WEBHOOK_URL") else None
            # Use unified sender with mention support if available
            if st.session_state.get(notify_key_i, False):
                try:
                    if hasattr(notifier, "send_with_mention"):
                        notifier.send_with_mention(
                            _title,
                            "",
                            fields=d,
                            image_url=_img_url,
                            image_path=_img_path,
                            mention=_mention,
                        )
                    else:
                        notifier.send(_title, "", fields=d, image_url=_img_url)
                    st.success(tr("通知を送信しました"))
                except Exception:
                    try:
                        notifier.send_summary(
                            "integrated",
                            "daily",
                            _pd.Timestamp.now().strftime("%Y-%m-%d"),
                            d,
                            image_url=_img_url,
                        )
                        st.success(tr("通知を送信しました"))
                    except Exception:
                        st.warning(tr("通知の送信に失敗しました"))
        else:
            st.info(tr("no trades in integrated run"))


def render_batch_tab(settings, logger, notifier: Notifier | None = None) -> None:
    """バッチバックテストタブの描画"""
    st.subheader(tr("Batch Backtest / Summary"))
    _mode_options = {
        "Backtest": tr("Backtest"),
        "Today": tr("Today signals"),
    }
    _mode_label = st.radio(
        tr("mode"),
        list(_mode_options.values()),
        index=0,
        horizontal=True,
        key="batch_mode",
    )
    mode = "Backtest" if _mode_label == _mode_options["Backtest"] else "Today"
    capital = st.number_input(
        tr("capital (USD)"),
        min_value=1000,
        value=int(settings.ui.default_capital),
        step=1000,
    )
    # 銘柄数と上限/全選択オプション
    all_tickers = get_all_tickers()
    max_allowed = len(all_tickers)
    limit_symbols = st.number_input(
        tr("symbol limit"),
        min_value=50,
        max_value=max_allowed,
        value=min(500, max_allowed),
        step=50,
    )
    use_all = st.checkbox(tr("use all symbols"), key="batch_all")
    run_btn = st.button(
        tr("run batch") if mode == "Backtest" else tr("run today signals"),
        key="run_batch" if mode == "Backtest" else "run_today",
    )

    if mode != "Backtest":
        if run_btn:
            from scripts.run_all_systems_today import compute_today_signals

            symbols = all_tickers if use_all else all_tickers[: int(limit_symbols)]
            with st.spinner(tr("running today signals...")):
                final_df, per_system = compute_today_signals(
                    symbols,
                    capital_long=float(capital),
                    capital_short=float(capital),
                    save_csv=False,
                )
            if final_df is None or final_df.empty:
                st.info(tr("no results"))
            else:
                st.dataframe(final_df, use_container_width=True)
            with st.expander("Per-system details"):
                for name, df in per_system.items():
                    st.markdown(f"#### {name}")
                    if df is None or df.empty:
                        st.write("(empty)")
                    else:
                        st.dataframe(df, use_container_width=True)
        return

    log_tail_lines = st.number_input(
        tr("max log lines shown per system"),
        min_value=10,
        max_value=10000,
        value=500,
        step=50,
        key="batch_log_tail_n",
    )

    saved_df = st.session_state.get("Batch_all_trades_df")
    saved_summary = st.session_state.get("Batch_summary_dict")
    saved_capital = st.session_state.get("Batch_capital")
    if saved_df is not None:
        st.markdown("---")
        st.subheader(tr("Saved Batch Results (persisted)"))
        if isinstance(saved_summary, dict):
            cols = st.columns(6)
            # 可能なら保存DFからピーク比のDD%を再計算
            try:
                _cap = float(saved_capital or 0)
                dd_pct_saved = (
                    (saved_df["drawdown"] / (_cap + saved_df["cum_max"])) .min() * 100
                )
            except Exception:
                dd_pct_saved = 0.0
            cols[0].metric(tr("trades"), saved_summary.get("trades"))
            cols[1].metric(tr("total pnl"), f"{saved_summary.get('total_return', 0):.2f}")
            cols[2].metric(tr("win rate (%)"), f"{saved_summary.get('win_rate', 0):.2f}")
            cols[3].metric("PF", f"{saved_summary.get('profit_factor', 0):.2f}")
            cols[4].metric("Sharpe", f"{saved_summary.get('sharpe', 0):.2f}")
            cols[5].metric(
                tr("max drawdown"),
                f"{saved_summary.get('max_drawdown', 0):.2f}",
                f"{dd_pct_saved:.2f}%",
            )
        st.dataframe(saved_df)
        import pandas as _pd

        _ts = _pd.Timestamp.now().strftime("%Y-%m-%d_%H%M")
        st.download_button(
            label=tr("download saved batch trades CSV"),
            data=saved_df.to_csv(index=False).encode("utf-8"),
            file_name=f"batch_trades_saved_{_ts}_{int(saved_capital or 0)}.csv",
            mime="text/csv",
            key="download_saved_batch_csv",
        )
        if st.button(tr("save saved batch CSV to disk"), key="save_saved_batch_to_disk"):
            out_dir = os.path.join("results_csv", "batch")
            os.makedirs(out_dir, exist_ok=True)
            trades_path = os.path.join(
                out_dir, f"batch_trades_saved_{_ts}_{int(saved_capital or 0)}.csv"
            )
            saved_df.to_csv(trades_path, index=False)
            if isinstance(saved_summary, dict):
                sum_df = _pd.DataFrame([saved_summary])
                sum_path = os.path.join(
                    out_dir, f"batch_summary_saved_{_ts}_{int(saved_capital or 0)}.csv"
                )
                sum_df.to_csv(sum_path, index=False)
            st.success(tr("saved to {out_dir}", out_dir=out_dir))
        if st.button(tr("clear saved batch results"), key="clear_saved_batch"):
            for k in ["Batch_all_trades_df", "Batch_summary_dict", "Batch_capital"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.experimental_rerun()

    st.markdown("---")
    st.subheader(tr("Saved Per-System Logs"))
    any_logs = False
    for i in range(1, 8):
        sys_name = f"System{i}"
        logs = st.session_state.get(f"{sys_name}_debug_logs")
        if logs:
            any_logs = True
            with st.expander(f"{sys_name} logs", expanded=False):
                tail = list(map(str, logs))[-int(log_tail_lines) :]
                st.text("\n".join(tail))
    if not any_logs:
        st.info(tr("no saved logs yet"))

    if run_btn:
        symbols = all_tickers if use_all else all_tickers[: int(limit_symbols)]
        spy_df = get_spy_with_indicators(get_spy_data_cached())

        overall = []
        sys_progress = st.progress(0)
        sys_log = st.empty()
        total_sys = 7
        done_sys = 0
        batch_ui = UIManager()

        for i in range(1, 8):
            sys_name = f"System{i}"
            sys_log.text(f"{sys_name}: starting...")
            try:
                mod = __import__(
                    f"strategies.system{i}_strategy",
                    fromlist=[f"System{i}Strategy"],
                )  # type: ignore
                cls = getattr(mod, f"System{i}Strategy")
                strat = cls()

                sys_ui = batch_ui.system(sys_name, title=sys_name)
                prepared, cands, merged = _prepare_ui(
                    strat,
                    symbols if sys_name != "System7" else ["SPY"],
                    system_name=sys_name,
                    spy_df=spy_df,
                    ui_manager=sys_ui,
                )
                if cands is None:
                    sys_log.text(f"{sys_name}: no candidates (skip)")
                    done_sys += 1
                    sys_progress.progress(done_sys / total_sys)
                    continue

                sys_log.text(f"{sys_name}: running...")
                res = _run_ui(
                    strat,
                    prepared,
                    cands,
                    capital,
                    system_name=sys_name,
                    ui_manager=sys_ui,
                )
                if res is not None and not res.empty:
                    res["system"] = sys_name
                    overall.append(res)
                    try:
                        with sys_ui.container:
                            st.success(f"{sys_name}: 完了（取引 {len(res)} 件）")
                    except Exception:
                        pass
                    with sys_ui.container.expander(f"{sys_name} result", expanded=False):
                        _show_sys_result(res, capital)  # type: ignore  # noqa: F821
                else:
                    with sys_ui.container:
                        st.info(f"{sys_name}: 取引なし")
                    try:
                        sys_log.text(f"{sys_name}: done (no trades)")
                    except Exception:
                        pass
            except Exception as e:  # noqa: BLE001
                logger.exception("%s error", sys_name)
                st.exception(e)
            finally:
                done_sys += 1
                sys_progress.progress(done_sys / total_sys)
                try:
                    if done_sys <= total_sys:
                        sys_log.text(f"{sys_name}: done")
                except Exception:
                    pass

        st.markdown("---")
        st.subheader(tr("All systems summary"))
        if overall:
            import pandas as pd

            all_df = pd.concat(overall, ignore_index=True)
            summary, all_df2 = summarize_perf(all_df, capital)
            cols = st.columns(6)
            d = summary.to_dict()
            d.update({
                "実施日時": now_jst_str(),
                "銘柄数": len(symbols),
                "開始資金": int(capital),
            })
            cols[0].metric(tr("trades"), d.get("trades"))
            cols[1].metric(tr("total pnl"), f"{d.get('total_return', 0):.2f}")
            cols[2].metric(tr("win rate (%)"), f"{d.get('win_rate', 0):.2f}")
            cols[3].metric("PF", f"{d.get('profit_factor', 0):.2f}")
            cols[4].metric("Sharpe", f"{d.get('sharpe', 0):.2f}")
            try:
                dd_pct_overall = (
                    (all_df2["drawdown"] / (capital + all_df2["cum_max"])) .min() * 100
                )
            except Exception:
                dd_pct_overall = 0.0
            cols[5].metric(
                tr("max drawdown"),
                f"{d.get('max_drawdown', 0):.2f}",
                f"{dd_pct_overall:.2f}%",
            )
            st.dataframe(all_df2)

            _ts2 = pd.Timestamp.now().strftime("%Y-%m-%d_%H%M")
            st.download_button(
                label=tr("download batch trades CSV"),
                data=all_df2.to_csv(index=False).encode("utf-8"),
                file_name=f"batch_trades_{_ts2}_{int(capital)}.csv",
                mime="text/csv",
                key="download_batch_csv_current",
            )
            if st.button(tr("save batch CSV to disk"), key="save_batch_to_disk_current"):
                out_dir = os.path.join("results_csv", "batch")
                os.makedirs(out_dir, exist_ok=True)
                trades_path = os.path.join(out_dir, f"batch_trades_{_ts2}_{int(capital)}.csv")
                all_df2.to_csv(trades_path, index=False)
                sum_df = pd.DataFrame([d])
                sum_path = os.path.join(out_dir, f"batch_summary_{_ts2}_{int(capital)}.csv")
                sum_df.to_csv(sum_path, index=False)
                st.success(tr("saved to {out_dir}", out_dir=out_dir))

            st.session_state["Batch_all_trades_df"] = all_df2
            st.session_state["Batch_summary_dict"] = d
            st.session_state["Batch_capital"] = capital

            # Optional notification for batch summary with equity image
            if notifier is not None:
                _img_path, _img_url = save_equity_curve(all_df2, capital, "Batch")
                _title = tr("Batch Backtest / Summary")
                _mention = "channel" if os.getenv("SLACK_WEBHOOK_URL") else None
                try:
                    if hasattr(notifier, "send_with_mention"):
                        notifier.send_with_mention(
                            _title, "", fields=d, image_url=_img_url, mention=_mention
                        )
                    else:
                        notifier.send(_title, "", fields=d, image_url=_img_url)
                except Exception:
                    pass

            try:
                import matplotlib.pyplot as _plt

                st.markdown("---")
                st.subheader("システム別 資金推移（サマリー）")
                eq_map = {}
                for df_sys in overall:
                    try:
                        df_tmp = df_sys.copy()
                        df_tmp["exit_date"] = pd.to_datetime(df_tmp["exit_date"])  # type: ignore
                        df_tmp = df_tmp.sort_values("exit_date")
                        equity = float(capital) + df_tmp["pnl"].cumsum()
                        daily = equity.rename(df_tmp["system"].iloc[0]).copy()
                        daily_df = daily.to_frame()
                        daily_df.index = df_tmp["exit_date"].values
                        daily_df = daily_df.resample("D").last().ffill()
                        eq_map[daily.name] = daily_df.iloc[:, 0]
                    except Exception:
                        continue
                if eq_map:
                    eq_df = pd.DataFrame(eq_map)
                    _plt.figure(figsize=(10, 4))
                    for col in eq_df.columns:
                        _plt.plot(eq_df.index, eq_df[col], label=col)
                    _plt.legend()
                    _plt.xlabel(tr("date"))
                    _plt.ylabel("Equity (USD)")
                    st.pyplot(_plt)
            except Exception:
                pass
        else:
            st.info(tr("no results"))

        st.markdown("---")
        st.subheader(tr("Per-System Logs (latest)"))
        any_logs2 = False
        for i in range(1, 8):
            sys_name = f"System{i}"
            logs = st.session_state.get(f"{sys_name}_debug_logs")
            if logs:
                any_logs2 = True
                with st.expander(f"{sys_name} logs", expanded=False):
                    tail2 = list(map(str, logs))[-int(log_tail_lines) :]
                    st.text("\n".join(tail2))
        if not any_logs2:
            st.info(tr("no logs to show"))
