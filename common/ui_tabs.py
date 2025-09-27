from __future__ import annotations

import os
import time
from pathlib import Path

# Notifier は型ヒント用途のみ。実体は app 側で生成・注入する。
from typing import Any as Notifier  # forward alias for type hints

import streamlit as st

from common.cache_format import round_dataframe
from common.equity_curve import save_equity_curve
from common.i18n import tr
from common.performance_summary import summarize as summarize_perf
from common.ui_bridge import prepare_backtest_data_ui as _prepare_ui
from common.ui_bridge import run_backtest_with_logging_ui as _run_ui
from common.ui_manager import UIManager
from common.utils_spy import get_spy_data_cached, get_spy_with_indicators
from config.settings import get_settings
from scripts.tickers_loader import get_all_tickers


def render_positions_tab(settings, notifier: Notifier | None = None) -> None:
    from pathlib import Path as _Path

    import pandas as _pd
    import streamlit as st

    from common import broker_alpaca as _ba
    from common.alpaca_order import submit_exit_orders_df as _submit_exits
    from common.profit_protection import evaluate_positions as _eval

    st.subheader(tr("Positions / Orders"))
    colL, colR = st.columns(2)
    with colL:
        paper = st.checkbox("ペーパートレード", value=True, key="pos_tab_paper")
    with colR:
        st.caption(".env の ALPACA_PAPER と独立。ここは明示設定です。")

    # Account summary (buying power, cash, type, status)
    st.markdown("---")
    st.subheader("口座サマリー / 買付余力")
    # session keys for account info
    st.session_state.setdefault("pos_tab_acct_type", None)
    st.session_state.setdefault("pos_tab_multiplier", None)
    st.session_state.setdefault("pos_tab_shorting_enabled", None)
    st.session_state.setdefault("pos_tab_status", None)
    st.session_state.setdefault("pos_tab_buying_power", None)
    st.session_state.setdefault("pos_tab_cash", None)

    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("ℹ️ 口座サマリーを取得/更新"):
            try:
                client = _ba.get_client(paper=paper)
                acct = client.get_account()
                st.session_state["pos_tab_acct_type"] = getattr(
                    acct, "account_type", None
                )
                st.session_state["pos_tab_multiplier"] = getattr(
                    acct, "multiplier", None
                )
                st.session_state["pos_tab_shorting_enabled"] = getattr(
                    acct, "shorting_enabled", None
                )
                st.session_state["pos_tab_status"] = getattr(acct, "status", None)
                bp_raw = getattr(acct, "buying_power", None)
                if bp_raw is None:
                    bp_raw = getattr(acct, "cash", None)
                try:
                    st.session_state["pos_tab_buying_power"] = (
                        float(bp_raw) if bp_raw is not None else None
                    )
                except Exception:
                    st.session_state["pos_tab_buying_power"] = None
                try:
                    st.session_state["pos_tab_cash"] = float(
                        getattr(acct, "cash", None) or 0.0
                    )
                except Exception:
                    st.session_state["pos_tab_cash"] = None
                st.success("口座情報を更新しました")
            except Exception as e:  # noqa: BLE001
                st.error(f"口座情報の取得に失敗: {e}")
    with colB:
        # derived account type
        mult = st.session_state.get("pos_tab_multiplier")
        try:
            mult_f = float(mult) if mult is not None else None
        except Exception:
            mult_f = None
        derived_type = (
            "Margin"
            if (mult_f is not None and mult_f > 1.0)
            else ("Cash" if mult_f is not None else "不明")
        )
        acct_type = st.session_state.get("pos_tab_acct_type")
        status = st.session_state.get("pos_tab_status")
        st.caption(
            f"種別(推定): {derived_type} / status: {status if status is not None else '-'}"
        )
        if acct_type is not None or mult_f is not None:
            st.caption(
                f"詳細: account_type={acct_type}, "
                f"multiplier={mult_f if mult_f is not None else '-'}"
            )
    with colC:
        bp = st.session_state.get("pos_tab_buying_power")
        cash = st.session_state.get("pos_tab_cash")
        bp_txt = f"${bp:,.2f}" if isinstance(bp, (int | float)) else "未取得"
        cash_txt = f"${cash:,.2f}" if isinstance(cash, (int | float)) else "未取得"
        st.metric("買付余力 (Buying Power)", bp_txt)
        st.caption(f"Cash: {cash_txt}")

    # Refresh positions
    if st.button("🔄 ポジション取得"):
        try:
            client = _ba.get_client(paper=paper)
            positions = client.get_all_positions()
            st.session_state["positions_df_tab"] = _eval(positions)
            st.success("取得しました")
        except Exception as e:  # noqa: BLE001
            st.error(f"取得失敗: {e}")

    df_pos = st.session_state.get("positions_df_tab")
    if isinstance(df_pos, _pd.DataFrame) and not df_pos.empty:
        st.dataframe(df_pos, use_container_width=True)
        # Selection for exits
        syms = df_pos["symbol"].astype(str).tolist()
        sel = st.multiselect("手仕舞い対象シンボル", syms, default=[])
        if sel:
            qty_map = (
                df_pos.set_index("symbol")["qty"].astype(int).to_dict()
                if "qty" in df_pos.columns
                else {s: 0 for s in sel}
            )
            side_map = (
                df_pos.set_index("symbol")["side"].astype(str).str.lower().to_dict()
                if "side" in df_pos.columns
                else {s: "long" for s in sel}
            )
            # Today close (MOC)
            if st.button("本日引け（CLS）で手仕舞い"):
                rows = [
                    {
                        "symbol": s,
                        "qty": int(qty_map.get(s, 0)),
                        "position_side": side_map.get(s, "long"),
                        "system": "",
                        "when": "today_close",
                    }
                    for s in sel
                    if int(qty_map.get(s, 0)) > 0
                ]
                res = _submit_exits(
                    _pd.DataFrame(rows), paper=paper, tif="CLS", notify=True
                )
                if res is not None and not res.empty:
                    st.dataframe(res, use_container_width=True)
            # Plan tomorrow open/close
            col_o, col_c = st.columns(2)
            with col_o:
                if st.button("明日寄り（OPG）で手仕舞いを予約"):
                    _plan = _Path("data/planned_exits.jsonl")
                    _plan.parent.mkdir(parents=True, exist_ok=True)
                    import json as _json

                    with _plan.open("a", encoding="utf-8") as f:
                        for s in sel:
                            if int(qty_map.get(s, 0)) <= 0:
                                continue
                            f.write(
                                _json.dumps(
                                    {
                                        "symbol": s,
                                        "qty": int(qty_map.get(s, 0)),
                                        "position_side": side_map.get(s, "long"),
                                        "system": "",
                                        "when": "tomorrow_open",
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                    st.success("予約を書き込みました（tomorrow_open）")
            with col_c:
                if st.button("明日引け（CLS）で手仕舞いを予約"):
                    _plan = _Path("data/planned_exits.jsonl")
                    _plan.parent.mkdir(parents=True, exist_ok=True)
                    import json as _json

                    with _plan.open("a", encoding="utf-8") as f:
                        for s in sel:
                            if int(qty_map.get(s, 0)) <= 0:
                                continue
                            f.write(
                                _json.dumps(
                                    {
                                        "symbol": s,
                                        "qty": int(qty_map.get(s, 0)),
                                        "position_side": side_map.get(s, "long"),
                                        "system": "",
                                        "when": "tomorrow_close",
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                    st.success("予約を書き込みました（tomorrow_close）")

    st.markdown("---")
    st.subheader("予約の実行 / 注文管理")
    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("⏱️ 寄り（OPG）予約を今すぐ実行"):
            try:
                from schedulers.next_day_exits import submit_planned_exits as _exec

                df = _exec("open")
                if df is not None and not df.empty:
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("実行対象はありません")
            except Exception as e:  # noqa: BLE001
                st.error(f"実行失敗: {e}")
    with colB:
        if st.button("⏱️ 引け（CLS）予約を今すぐ実行"):
            try:
                from schedulers.next_day_exits import submit_planned_exits as _exec

                df = _exec("close")
                if df is not None and not df.empty:
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("実行対象はありません")
            except Exception as e:  # noqa: BLE001
                st.error(f"実行失敗: {e}")
    with colC:
        if st.button("未約定注文をすべてキャンセル"):
            try:
                client = _ba.get_client(paper=paper)
                _ba.cancel_all_orders(client)
                st.success("キャンセルを送信しました")
            except Exception as e:  # noqa: BLE001
                st.error(f"キャンセル失敗: {e}")

    # Planned exits viewer/editor
    st.markdown("---")
    st.subheader("予約一覧（編集）")
    import json as _json
    from pathlib import Path as _Path

    _plan = _Path("data/planned_exits.jsonl")
    plans: list[dict] = []
    if _plan.exists():
        try:
            for line in _plan.read_text(encoding="utf-8").splitlines():
                try:
                    plans.append(_json.loads(line))
                except Exception:
                    continue
        except Exception:
            pass
    if plans:
        import pandas as _pd

        df_pl = _pd.DataFrame(plans)
        st.dataframe(df_pl, use_container_width=True)
        sel_to_remove = st.multiselect(
            "削除する予約（symbol when で選択）",
            [f"{r.get('symbol')} | {r.get('when')}" for r in plans],
            default=[],
        )
        col_rm1, col_rm2 = st.columns(2)
        with col_rm1:
            if st.button("選択した予約を削除"):
                new_plans = []
                keys = set(sel_to_remove)
                for r in plans:
                    key = f"{r.get('symbol')} | {r.get('when')}"
                    if key in keys:
                        continue
                    new_plans.append(r)
                try:
                    _plan.write_text(
                        "\n".join(_json.dumps(x, ensure_ascii=False) for x in new_plans)
                        + ("\n" if new_plans else ""),
                        encoding="utf-8",
                    )
                    st.success("削除しました")
                except Exception as e:  # noqa: BLE001
                    st.error(f"削除失敗: {e}")
        with col_rm2:
            if st.button("全予約をクリア"):
                try:
                    _plan.unlink(missing_ok=True)
                    st.success("クリアしました")
                except Exception as e:  # noqa: BLE001
                    st.error(f"クリア失敗: {e}")
    else:
        st.info("予約はありません")

    # Open orders list + individual cancel
    st.markdown("---")
    st.subheader("未約定注文一覧")
    try:
        client = _ba.get_client(paper=st.session_state.get("pos_tab_paper", True))
        orders = client.get_orders(status="open")
        rows = []
        for o in orders:
            rows.append(
                {
                    "id": getattr(o, "id", None),
                    "symbol": getattr(o, "symbol", None),
                    "side": getattr(o, "side", None),
                    "qty": getattr(o, "qty", None),
                    "type": getattr(o, "type", None),
                    "tif": getattr(o, "time_in_force", None),
                    "status": getattr(o, "status", None),
                    "submitted_at": getattr(o, "submitted_at", None),
                }
            )
        if rows:
            import pandas as _pd

            df_o = _pd.DataFrame(rows)
            st.dataframe(df_o, use_container_width=True)
            ids = [str(r.get("id")) for r in rows if r.get("id")]
            sel_ids = st.multiselect("キャンセルする order_id", ids, default=[])
            if st.button("選択した注文をキャンセル"):
                ok = 0
                for oid in sel_ids:
                    try:
                        client.cancel_order_by_id(oid)
                        ok += 1
                    except Exception:
                        pass
                st.success(f"{ok} 件キャンセルを送信しました")
        else:
            st.info("未約定注文はありません")
    except Exception as e:  # noqa: BLE001
        st.warning(f"未約定注文の取得に失敗: {e}")


def render_metrics_tab(settings) -> None:
    from pathlib import Path

    import pandas as _pd
    import streamlit as st

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

    # リアルタイム進捗表示セクション
    with st.expander("🔄 Real-time Progress Monitor", expanded=False):
        progress_container = st.empty()
        auto_refresh = st.checkbox("Auto-refresh (every 1 sec)", value=False)

        if auto_refresh:
            # Use session state to track progress polling
            if "progress_poll_count" not in st.session_state:
                st.session_state.progress_poll_count = 0

            # Import render_digest_log from app_integrated
            try:
                import app_integrated

                logs_dir = Path(settings.LOGS_DIR)
                progress_log = logs_dir / "progress_today.jsonl"
                app_integrated.render_digest_log(progress_log, progress_container)

                # Auto-refresh mechanism
                st.session_state.progress_poll_count += 1
                if st.session_state.progress_poll_count % 100 == 0:  # Reduce frequency
                    import time as time_module

                    time_module.sleep(0.1)
                    st.rerun()
                else:
                    # Use a timer-based approach for smooth updates
                    import time as time_module

                    time_module.sleep(1)
                    st.rerun()

            except ImportError:
                progress_container.warning(
                    "Progress monitoring not available (app_integrated not found)"
                )
            except Exception as e:
                progress_container.error(f"Progress monitoring error: {e}")
        else:
            # Manual refresh button
            if st.button("🔄 Refresh Progress"):
                try:
                    import app_integrated

                    logs_dir = Path(settings.LOGS_DIR)
                    progress_log = logs_dir / "progress_today.jsonl"
                    app_integrated.render_digest_log(progress_log, progress_container)
                except Exception as e:
                    progress_container.error(f"Failed to refresh progress: {e}")

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
        if not (os.getenv("DISCORD_WEBHOOK_URL") or os.getenv("SLACK_BOT_TOKEN")):
            st.caption(tr("Webhook/Bot 設定が未設定です（.env を確認）"))
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
                year_start = daily_eq.resample("YE").first()
                year_end = daily_eq.resample("YE").last()
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
                month_start = daily_eq.resample("ME").first()
                month_end = daily_eq.resample("ME").last()
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
            try:
                try:
                    settings2 = get_settings(create_dirs=True)
                    round_dec = getattr(settings2.cache, "round_decimals", None)
                except Exception:
                    round_dec = None
                try:
                    out_df = round_dataframe(df2, round_dec)
                except Exception:
                    out_df = df2
                st.download_button(
                    label=tr("download integrated trades CSV"),
                    data=out_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"integrated_trades_{_ts_i}_{int(capital_i)}.csv",
                    mime="text/csv",
                    key="download_integrated_csv",
                )
            except Exception:
                pass
            # Save equity curve image for integrated results
            _img_path, _img_url = save_equity_curve(df2, capital_i, "Integrated")
            _title = tr("Integrated Summary")
            _mention = "channel" if os.getenv("SLACK_BOT_TOKEN") else None
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

    if mode != "Backtest":
        # SPY ゲート状態を表示
        st.markdown("---")
        st.subheader("SPY Market Gate Status")
        try:
            spy_df = get_spy_with_indicators(get_spy_data_cached())
            if spy_df is not None and not spy_df.empty:
                last = spy_df.iloc[-1]
                close = last.get("Close", 0)
                sma100 = last.get("SMA100", 0)
                gate_ok = close > sma100
                status = (
                    "✅ OPEN (SPY > SMA100)"
                    if gate_ok
                    else "❌ CLOSED (SPY <= SMA100) - System1/4_TRDlist is 0"
                )
                st.metric(
                    "SPY Gate", status, f"Close: {close:.2f}, SMA100: {sma100:.2f}"
                )
            else:
                st.warning("SPY data not available")
        except Exception as e:
            st.error(f"Failed to check SPY gate: {e}")

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
                    msg_str = str(msg)
                    skip_keywords = (
                        "進捗",
                        "インジケーター",
                        "indicator",
                        "indicators",
                        "指標計算",
                        "共有指標",
                        "バッチ時間",
                        "batch time",
                        "候補抽出",
                        "候補日数",
                        "銘柄:",
                        "📊 インジケーター計算",
                        "📊 候補抽出",
                        "⏱️ バッチ時間",
                    )
                    if any(k in msg_str for k in skip_keywords):
                        return
                    elapsed = max(0, time.time() - start)
                    m, s = divmod(int(elapsed), 60)
                    now = time.strftime("%H:%M:%S")
                    line = f"[{now} | {m}分{s}秒] {msg_str}"
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
                try:
                    try:
                        settings2 = get_settings(create_dirs=True)
                        round_dec = getattr(settings2.cache, "round_decimals", None)
                    except Exception:
                        round_dec = None
                    try:
                        out_df = round_dataframe(final_df, round_dec)
                    except Exception:
                        out_df = final_df
                    csv = out_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label=tr("Download Final CSV"),
                        data=csv,
                        file_name="today_signals_final.csv",
                        mime="text/csv",
                    )
                except Exception:
                    try:
                        csv = final_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label=tr("Download Final CSV"),
                            data=csv,
                            file_name="today_signals_final.csv",
                            mime="text/csv",
                        )
                    except Exception:
                        pass

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
        try:
            try:
                settings2 = get_settings(create_dirs=True)
                round_dec = getattr(settings2.cache, "round_decimals", None)
            except Exception:
                round_dec = None
            try:
                out_df = round_dataframe(saved_df, round_dec)
            except Exception:
                out_df = saved_df
            st.download_button(
                label=tr("download saved batch trades CSV"),
                data=out_df.to_csv(index=False).encode("utf-8"),
                file_name=f"batch_trades_saved_{_ts}_{int(saved_capital or 0)}.csv",
                mime="text/csv",
                key="download_saved_batch_csv",
            )
        except Exception:
            try:
                st.download_button(
                    label=tr("download saved batch trades CSV"),
                    data=saved_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"batch_trades_saved_{_ts}_{int(saved_capital or 0)}.csv",
                    mime="text/csv",
                    key="download_saved_batch_csv",
                )
            except Exception:
                pass
        if st.button(
            tr("save saved batch CSV to disk"), key="save_saved_batch_to_disk"
        ):
            out_dir = os.path.join("results_csv", "batch")
            os.makedirs(out_dir, exist_ok=True)
            trades_path = os.path.join(
                out_dir, f"batch_trades_saved_{_ts}_{int(saved_capital or 0)}.csv"
            )
            try:
                try:
                    settings2 = get_settings(create_dirs=True)
                    round_dec = getattr(settings2.cache, "round_decimals", None)
                except Exception:
                    round_dec = None
                try:
                    out_df = round_dataframe(saved_df, round_dec)
                except Exception:
                    out_df = saved_df
                out_df.to_csv(trades_path, index=False)
            except Exception:
                try:
                    saved_df.to_csv(trades_path, index=False)
                except Exception:
                    pass
            if isinstance(saved_summary, dict):
                sum_df = _pd.DataFrame([saved_summary])
                sum_path = os.path.join(
                    out_dir, f"batch_summary_saved_{_ts}_{int(saved_capital or 0)}.csv"
                )
                try:
                    try:
                        settings2 = get_settings(create_dirs=True)
                        round_dec = getattr(settings2.cache, "round_decimals", None)
                    except Exception:
                        round_dec = None
                    try:
                        out_sum = round_dataframe(sum_df, round_dec)
                    except Exception:
                        out_sum = sum_df
                    out_sum.to_csv(sum_path, index=False)
                except Exception:
                    try:
                        sum_df.to_csv(sum_path, index=False)
                    except Exception:
                        pass
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
                )
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
                        _show_sys_result(res, capital)  # noqa: F821
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
                    df["system"] = f"System{idx + 1}"

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
            try:
                try:
                    settings2 = get_settings(create_dirs=True)
                    round_dec = getattr(settings2.cache, "round_decimals", None)
                except Exception:
                    round_dec = None
                try:
                    out_df = round_dataframe(all_df2, round_dec)
                except Exception:
                    out_df = all_df2
                st.download_button(
                    label=tr("download batch trades CSV"),
                    data=out_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"batch_trades_{_ts2}_{int(capital)}.csv",
                    mime="text/csv",
                    key="download_batch_csv_current",
                )
            except Exception:
                try:
                    st.download_button(
                        label=tr("download batch trades CSV"),
                        data=all_df2.to_csv(index=False).encode("utf-8"),
                        file_name=f"batch_trades_{_ts2}_{int(capital)}.csv",
                        mime="text/csv",
                        key="download_batch_csv_current",
                    )
                except Exception:
                    pass
            if st.button(
                tr("save batch CSV to disk"), key="save_batch_to_disk_current"
            ):
                out_dir = os.path.join("results_csv", "batch")
                os.makedirs(out_dir, exist_ok=True)
                trades_path = os.path.join(
                    out_dir, f"batch_trades_{_ts2}_{int(capital)}.csv"
                )
                try:
                    try:
                        settings2 = get_settings(create_dirs=True)
                        round_dec = getattr(settings2.cache, "round_decimals", None)
                    except Exception:
                        round_dec = None
                    try:
                        out_df = round_dataframe(all_df2, round_dec)
                    except Exception:
                        out_df = all_df2
                    out_df.to_csv(trades_path, index=False)
                except Exception:
                    try:
                        all_df2.to_csv(trades_path, index=False)
                    except Exception:
                        pass
                sum_df = pd.DataFrame([d])
                sum_path = os.path.join(
                    out_dir, f"batch_summary_{_ts2}_{int(capital)}.csv"
                )
                try:
                    try:
                        settings2 = get_settings(create_dirs=True)
                        round_dec = getattr(settings2.cache, "round_decimals", None)
                    except Exception:
                        round_dec = None
                    try:
                        out_sum = round_dataframe(sum_df, round_dec)
                    except Exception:
                        out_sum = sum_df
                    out_sum.to_csv(sum_path, index=False)
                except Exception:
                    try:
                        sum_df.to_csv(sum_path, index=False)
                    except Exception:
                        pass
                st.success(tr("saved to {out_dir}", out_dir=out_dir))

            st.session_state["Batch_all_trades_df"] = all_df2
            st.session_state["Batch_summary_dict"] = d
            st.session_state["Batch_capital"] = capital

            # Optional notification for batch summary with equity image
            if notifier is not None:
                _img_path, _img_url = save_equity_curve(all_df2, capital, "Batch")
                _title = tr("Batch Backtest / Summary")
                _mention = "channel" if os.getenv("SLACK_BOT_TOKEN") else None
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
                        df_tmp["exit_date"] = pd.to_datetime(df_tmp["exit_date"])
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


def render_cache_health_tab(settings) -> None:
    """
    Cache健全性とrolling cache分析を行うタブを描画する。
    """
    st.title("🩺 Cache Health Dashboard")
    st.write("rolling cacheの健全性と整備状況を監視・分析します。")

    # タブ内でサブタブを作成
    subtab1, subtab2, subtab3 = st.tabs(
        ["🔍 基本ヘルスチェック", "🎯 システム別カバレッジ", "💡 推奨アクション"]
    )

    with subtab1:
        st.write("### Cache基本状況")
        from common.ui_components import display_cache_health_dashboard

        display_cache_health_dashboard()

    with subtab2:
        st.write("### システム別カバレッジ分析")
        from common.ui_components import display_system_cache_coverage

        display_system_cache_coverage()

    with subtab3:
        st.write("### 推奨アクションと改善提案")

        # 分析実行ボタン
        if st.button("🔍 詳細分析実行", key="cache_analysis_for_recommendations"):
            from common.cache_manager import CacheManager
            from common.ui_components import display_cache_recommendations
            from config.settings import get_settings

            try:
                settings = get_settings(create_dirs=True)
                cache_manager = CacheManager(settings)
                analysis_result = cache_manager.analyze_rolling_gaps()

                # 推奨アクションを表示
                display_cache_recommendations(analysis_result)

            except Exception as e:
                st.error(f"分析エラー: {str(e)}")
        else:
            st.info("上のボタンをクリックして詳細分析を実行してください。")
