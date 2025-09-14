from __future__ import annotations

import os
import time

import streamlit as st

from common.equity_curve import save_equity_curve
from common.i18n import tr

# Notifier は型ヒント用途のみ。実体は app 側で生成・注入する。
from typing import Any as Notifier  # forward alias for type hints
from common.performance_summary import summarize as summarize_perf
from common.ui_bridge import prepare_backtest_data_ui as _prepare_ui
from common.ui_bridge import run_backtest_with_logging_ui as _run_ui
from common.ui_manager import UIManager
from common.utils_spy import get_spy_data_cached, get_spy_with_indicators
from scripts.tickers_loader import get_all_tickers


def render_metrics_tab(settings) -> None:
    import pandas as _pd
    import streamlit as st
    from pathlib import Path

    st.subheader(tr("Daily Metrics"))
    try:
        results_dir = Path(settings.RESULTS_DIR)
    except Exception:
        results_dir = Path("results_csv")
    metrics_fp = results_dir / "daily_metrics.csv"
    if not metrics_fp.exists():
        st.info(tr("metrics csv not found: {p}").format(p=str(metrics_fp)))
        return
    try:
        df = _pd.read_csv(metrics_fp)
    except Exception as e:
        st.warning(f"failed to read metrics: {e}")
        return
    if df.empty:
        st.info(tr("no metrics yet"))
        return
    # normalize date
    try:
        df["date"] = _pd.to_datetime(df["date"]).dt.date
    except Exception:
        pass
    systems = sorted(df["system"].dropna().unique())
    col1, col2, col3 = st.columns(3)
    with col1:
        sel_metric = st.selectbox("metric", ["candidates", "prefilter_pass"], index=0)
    with col2:
        sel_systems = st.multiselect("systems", systems, default=systems)
    with col3:
        chart_type = st.selectbox("chart", ["line", "bar"], index=0)

    work = df[df["system"].isin(sel_systems)].copy()
    pivot = work.pivot_table(
        index="date", columns="system", values=sel_metric, aggfunc="sum"
    ).fillna(0)
    st.caption(tr("daily {m} by system").format(m=sel_metric))
    try:
        if chart_type == "line":
            st.line_chart(pivot)
        else:
            st.bar_chart(pivot)
    except Exception:
        st.dataframe(pivot)
    st.markdown("---")
    st.caption(tr("raw metrics"))
    st.dataframe(df.sort_values(["date", "system"]))


def _show_sys_result(df, capital):
    if df is None or getattr(df, "empty", True):
        st.info(tr("no trades"))
        return
    summary, df2 = summarize_perf(df, capital)
    d = summary.to_dict()
    cols = st.columns(6)
    # 統合タブと同じ算出式（ピーク資産比の%）
    try:
        dd_pct = (df2["drawdown"] / (capital + df2["cum_max"])).min() * 100
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
        st.caption(
            tr("allocation is fixed: long 1/3/4/5: each 25%, short 2:40%,6:40%,7:20%")
        )
        try:
            # 表示用に現在の設定配分も添える
            def _norm_map(d: dict[str, float], default_map: dict[str, float]):
                try:
                    f = {k: float(v) for k, v in (d or {}).items() if float(v) > 0}
                    s = sum(f.values())
                    return (
                        {k: v / s for k, v in (f or default_map).items()}
                        if s > 0
                        else default_map
                    )
                except Exception:
                    return default_map

            la = getattr(settings.ui, "long_allocations", {}) or {}
            sa = getattr(settings.ui, "short_allocations", {}) or {}
            la_n = _norm_map(
                la, {"system1": 0.25, "system3": 0.25, "system4": 0.25, "system5": 0.25}
            )
            sa_n = _norm_map(sa, {"system2": 0.40, "system6": 0.40, "system7": 0.20})

            def _fmt(d: dict[str, float]):
                try:
                    items = [f"{k}:{v:.0%}" for k, v in d.items()]
                    return ", ".join(items)
                except Exception:
                    return ""

            st.caption(f"settings long=({_fmt(la_n)}), short=({_fmt(sa_n)})")
        except Exception:
            pass
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
        try:
            import logging as _logging

            _logging.getLogger(__name__).info(
                "[integrated] target symbols: %d (e.g., %s)",
                len(symbols),
                ", ".join(symbols[:10]) + ("..." if len(symbols) > 10 else ""),
            )
        except Exception:
            pass
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
            s.name: int(sum(len(v) for v in s.candidates_by_date.values()))
            for s in states
        }
        st.write(tr("signals per system:"))
        st.dataframe(_pd.DataFrame([sig_counts]))
        try:
            import logging as _logging

            _logging.getLogger(__name__).info(
                "[integrated] signals per system: %s",
                {k: int(v) for k, v in sig_counts.items()},
            )
        except Exception:
            pass

        sim = ui.phase("simulate", title=tr("simulate integrated"))
        sim.info(tr("running integrated engine..."))

        # 進捗更新用のコールバック（時刻 + 分秒を表示）
        sim_prog_txt = st.empty()

        def _on_progress(i: int, total: int, start):
            try:
                sim.progress_bar.progress(0 if not total else i / total)
            except Exception:
                pass
            try:
                elapsed = max(0, time.time() - (start or time.time()))
                m, s = divmod(int(elapsed), 60)
                now = time.strftime("%H:%M:%S")
                sim_prog_txt.text(f"[{now} | {m}分{s}秒] integrated {i}/{total}")
            except Exception:
                pass

        # 設定から配分マップを構築（System1..System7 キー、長短それぞれ正規化）
        def _canon(k: str) -> str:
            s = str(k)
            try:
                if s.lower().startswith("system"):
                    num = "".join(ch for ch in s if ch.isdigit())
                    return f"System{num}" if num else s.title()
                if s.isdigit():
                    return f"System{s}"
                return s
            except Exception:
                return s

        def _norm_map(d: dict[str, float], default_map: dict[str, float]):
            try:
                f = {k: float(v) for k, v in (d or {}).items() if float(v) > 0}
                s = sum(f.values())
                if s <= 0:
                    f = default_map
                    s = sum(f.values())
                return {_canon(k): v / s for k, v in f.items()}
            except Exception:
                s = sum(default_map.values())
                return {_canon(k): v / s for k, v in default_map.items()}

        la = getattr(settings.ui, "long_allocations", {}) or {}
        sa = getattr(settings.ui, "short_allocations", {}) or {}
        alloc_map_long = _norm_map(
            la, {"system1": 0.25, "system3": 0.25, "system4": 0.25, "system5": 0.25}
        )
        alloc_map_short = _norm_map(
            sa, {"system2": 0.40, "system6": 0.40, "system7": 0.20}
        )
        alloc_map = {**alloc_map_long, **alloc_map_short}

        trades_df, _sig = run_integrated_backtest(
            states,
            capital_i,
            allocations=alloc_map or DEFAULT_ALLOCATIONS,
            long_share=float(long_share) / 100.0,
            short_share=float(short_share) / 100.0,
            allow_gross_leverage=allow_gross,
            on_progress=_on_progress,
        )
        try:
            import logging as _logging

            _logging.getLogger(__name__).info(
                "[integrated] result trades=%d",
                0 if trades_df is None else len(trades_df),
            )
        except Exception:
            pass

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
            d.update(
                銘柄数=len(symbols),
                開始資金=int(capital_i),
            )
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
                import numpy as np

                equity = _pd.Series(
                    np.array(df2["cumulative_pnl"].values, dtype=float)
                    + float(capital_i),
                    index=_pd.to_datetime(df2["exit_date"]),
                )
                daily_eq = equity.resample("D").last().ffill()
                year_start = daily_eq.resample("Y").first()
                year_end = daily_eq.resample("Y").last()
                yearly_df = _pd.DataFrame(
                    {
                        "年": year_end.index.to_series().dt.year.values,
                        "損益": (year_end - year_start).round(2).values,
                        "リターン(%)": ((year_end / year_start - 1) * 100).values,
                    }
                )
                st.subheader(tr("yearly summary"))
                # 百分率として1桁で表示（例: 468.9% / -63.6%）、pnlは小数第2位
                st.dataframe(
                    yearly_df.style.format({"損益": "{:.2f}", "リターン(%)": "{:.1f}%"})
                )
                # 月次サマリー
                month_start = daily_eq.resample("M").first()
                month_end = daily_eq.resample("M").last()
                monthly_df = _pd.DataFrame(
                    {
                        "月": month_end.index.to_series().dt.strftime("%Y-%m").values,
                        "損益": (month_end - month_start).round(2).values,
                        "リターン(%)": ((month_end / month_start - 1) * 100).values,
                    }
                )
                st.subheader(tr("monthly summary"))
                st.dataframe(
                    monthly_df.style.format(
                        {"損益": "{:.2f}", "リターン(%)": "{:.1f}%"}
                    )
                )
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
    if mode == "Backtest":
        capital = st.number_input(
            tr("capital (USD)"),
            min_value=1000,
            value=int(settings.ui.default_capital),
            step=1000,
        )
    else:
        if "batch_cap_long" not in st.session_state:
            st.session_state["batch_cap_long"] = 2000
        if "batch_cap_short" not in st.session_state:
            st.session_state["batch_cap_short"] = 2000

        col1, col2 = st.columns(2)
        with col1:
            cap_long = st.number_input(
                tr("capital long (USD)"),
                min_value=0,
                step=100,
                key="batch_cap_long",
            )
        with col2:
            cap_short = st.number_input(
                tr("capital short (USD)"),
                min_value=0,
                step=100,
                key="batch_cap_short",
            )
        # Ensure cap_long and cap_short are always defined
        cap_long = st.session_state.get("batch_cap_long", 2000)
        cap_short = st.session_state.get("batch_cap_short", 2000)

        from common import broker_alpaca as ba

        def _fetch_balances() -> None:
            try:
                client = ba.get_client(paper=True)
                acct = client.get_account()
                bp = None
                try:
                    bp = float(
                        getattr(acct, "buying_power", None)
                        or getattr(acct, "cash", None)
                        or 0.0
                    )
                except Exception:
                    bp = None
                if bp:
                    half = round(float(bp) / 2.0, 2)
                    st.session_state["batch_cap_long"] = half
                    st.session_state["batch_cap_short"] = half
                    st.session_state["batch_fetch_msg"] = (
                        "success",
                        f"Set long/short to {half} each",
                    )
                else:
                    st.session_state["batch_fetch_msg"] = (
                        "warning",
                        tr("could not read buying_power/cash"),
                    )
            except Exception as e:  # noqa: BLE001
                st.session_state["batch_fetch_msg"] = ("error", f"Alpaca error: {e}")
            st.session_state["batch_should_rerun"] = True  # rerunフラグを立てる

        st.button(tr("Fetch Alpaca balances"), on_click=_fetch_balances)

        # rerunフラグが立っていれば rerun
        if st.session_state.pop("batch_should_rerun", False):
            st.rerun()

        _msg = st.session_state.pop("batch_fetch_msg", None)
        if _msg:
            lvl, txt = _msg
            getattr(st, lvl)(txt)

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
    use_parallel = st.checkbox(tr("use parallel processing"), key="batch_parallel")
    run_btn = st.button(
        tr("run batch") if mode == "Backtest" else tr("run today signals"),
        key="run_batch" if mode == "Backtest" else "run_today",
    )

    if mode != "Backtest":
        if run_btn:
            from scripts.run_all_systems_today import compute_today_signals

            symbols = all_tickers if use_all else all_tickers[: int(limit_symbols)]

            # log area
            if "batch_today_logs" not in st.session_state:
                st.session_state["batch_today_logs"] = []
            log_box = st.empty()

            # progress表示
            prog = st.progress(0)
            prog_txt = st.empty()
            start = time.time()

            def _ui_log(msg: str) -> None:
                try:
                    elapsed = max(0, time.time() - start)
                    m, s = divmod(int(elapsed), 60)
                    now = time.strftime("%H:%M:%S")
                    line = f"[{now} | {m}分{s}秒] {str(msg)}"
                    st.session_state["batch_today_logs"].append(line)
                    log_box.code("\n".join(st.session_state["batch_today_logs"]))
                except Exception:
                    pass

            def _progress(i: int, total: int, name: str) -> None:
                try:
                    prog.progress(0 if not total else i / total)
                    elapsed = max(0, time.time() - start)
                    m, s = divmod(int(elapsed), 60)
                    if i < total:
                        prog_txt.text(f"{name} {i}/{total} | 経過: {m}分{s}秒")
                    else:
                        prog_txt.text(f"{m}分{s}秒: done")
                except Exception:
                    pass

            # Ensure cap_long and cap_short are always defined before use
            cap_long = st.session_state.get("batch_cap_long", 2000)
            cap_short = st.session_state.get("batch_cap_short", 2000)
            with st.spinner(tr("running today signals...")):
                final_df, per_system = compute_today_signals(
                    symbols,
                    capital_long=float(cap_long),
                    capital_short=float(cap_short),
                    save_csv=False,
                    log_callback=_ui_log,
                    progress_callback=_progress,
                    parallel=use_parallel,
                )

            if final_df is None or final_df.empty:
                st.info(tr("no results"))
            else:
                # --- 結論から表示: 発注銘柄リスト ---
                st.subheader(tr("Order list"))
                st.dataframe(final_df, use_container_width=True)
                csv = final_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=tr("Download Final CSV"),
                    data=csv,
                    file_name="today_signals_final.csv",
                    mime="text/csv",
                )

                # --- 内訳表示 ---
                long_syms = final_df[final_df["side"] == "long"]["symbol"].tolist()
                short_syms = final_df[final_df["side"] == "short"]["symbol"].tolist()
                col_ls_1, col_ls_2 = st.columns(2)
                with col_ls_1:
                    st.markdown(tr("Long symbols"))
                    st.write(", ".join(long_syms) if long_syms else "-")
                with col_ls_2:
                    st.markdown(tr("Short symbols"))
                    st.write(", ".join(short_syms) if short_syms else "-")

                st.markdown(tr("Orders by system"))
                st.dataframe(
                    final_df.groupby("system")["symbol"].count().rename("count"),
                    use_container_width=True,
                )

                # 資金推移
                import pandas as pd

                cap_long = st.session_state.get("batch_cap_long", 2000)
                cap_short = st.session_state.get("batch_cap_short", 2000)
                total_capital = float(cap_long) + float(cap_short)
                cap_df = final_df.sort_values("entry_date")[
                    ["entry_date", "symbol", "position_value"]
                ].copy()
                cap_df["entry_date"] = pd.to_datetime(cap_df["entry_date"])
                cap_df["capital_after"] = (
                    total_capital - cap_df["position_value"].cumsum()
                )
                st.markdown(tr("Capital progression"))
                st.dataframe(cap_df, use_container_width=True)

                # 候補一覧（ロング/ショート）
                from common.today_signals import LONG_SYSTEMS, SHORT_SYSTEMS

                with st.expander(tr("Long system candidates"), expanded=False):
                    for name, df in per_system.items():
                        if name.lower() not in LONG_SYSTEMS:
                            continue
                        st.markdown(f"#### {name}")
                        if df is None or df.empty:
                            st.write("(empty)")
                        else:
                            _tmp = df.copy()
                            _tmp["setup"] = (
                                ~_tmp[["entry_price", "stop_price"]].isna().any(axis=1)
                            ).map(lambda x: "⭐" if x else "")
                            st.dataframe(_tmp, use_container_width=True)

                with st.expander(tr("Short system candidates"), expanded=False):
                    for name, df in per_system.items():
                        if name.lower() not in SHORT_SYSTEMS:
                            continue
                        st.markdown(f"#### {name}")
                        if df is None or df.empty:
                            st.write("(empty)")
                        else:
                            _tmp = df.copy()
                            _tmp["setup"] = (
                                ~_tmp[["entry_price", "stop_price"]].isna().any(axis=1)
                            ).map(lambda x: "⭐" if x else "")
                            st.dataframe(_tmp, use_container_width=True)

                # ログのCSV保存ボタン
                logs = st.session_state.get("batch_today_logs", [])
                if logs:
                    log_csv = "\n".join(logs).encode("utf-8")
                    st.download_button(
                        label=tr("download log CSV"),
                        data=log_csv,
                        file_name="today_logs.csv",
                        mime="text/csv",
                    )
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
                    saved_df["drawdown"] / (_cap + saved_df["cum_max"])
                ).min() * 100
            except Exception:
                dd_pct_saved = 0.0
            cols[0].metric(tr("trades"), saved_summary.get("trades"))
            cols[1].metric(
                tr("total pnl"), f"{saved_summary.get('total_return', 0):.2f}"
            )
            cols[2].metric(
                tr("win rate (%)"), f"{saved_summary.get('win_rate', 0):.2f}"
            )
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
        if st.button(
            tr("save saved batch CSV to disk"), key="save_saved_batch_to_disk"
        ):
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
            st.rerun()

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

        # Ensure capital is always defined
        capital = locals().get("capital", None)
        if capital is None:
            # If not defined, try to get from session_state (for Today mode)
            capital = st.session_state.get("batch_cap_long", 0) + st.session_state.get(
                "batch_cap_short", 0
            )

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
                    with sys_ui.container.expander(
                        f"{sys_name} result", expanded=False
                    ):
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

            # 各DataFrameにsystem列がなければ追加
            for idx, df in enumerate(overall):
                if "system" not in df.columns:
                    df["system"] = f"System{idx+1}"

            all_df = pd.concat(overall, ignore_index=True)
            summary, all_df2 = summarize_perf(all_df, capital)
            cols = st.columns(6)
            d = summary.to_dict()
            # d["実施日時"] = now_jst_str()  # Removed to avoid type error
            d["銘柄数"] = len(symbols)
            d["開始資金"] = int(capital)
            cols[0].metric(tr("trades"), d.get("trades"))
            cols[1].metric(tr("total pnl"), f"{d.get('total_return', 0):.2f}")
            cols[2].metric(tr("win rate (%)"), f"{d.get('win_rate', 0):.2f}")
            cols[3].metric("PF", f"{d.get('profit_factor', 0):.2f}")
            cols[4].metric("Sharpe", f"{d.get('sharpe', 0):.2f}")
            try:
                dd_pct_overall = (
                    all_df2["drawdown"] / (capital + all_df2["cum_max"])
                ).min() * 100
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
            if st.button(
                tr("save batch CSV to disk"), key="save_batch_to_disk_current"
            ):
                out_dir = os.path.join("results_csv", "batch")
                os.makedirs(out_dir, exist_ok=True)
                trades_path = os.path.join(
                    out_dir, f"batch_trades_{_ts2}_{int(capital)}.csv"
                )
                all_df2.to_csv(trades_path, index=False)
                sum_df = pd.DataFrame([d])
                sum_path = os.path.join(
                    out_dir, f"batch_summary_{_ts2}_{int(capital)}.csv"
                )
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
                        df_tmp["exit_date"] = pd.to_datetime(
                            df_tmp["exit_date"]
                        )  # type: ignore[arg-type]
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
                    st.pyplot(_plt.gcf())
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
        if not any_logs2:
            st.info(tr("no logs to show"))
