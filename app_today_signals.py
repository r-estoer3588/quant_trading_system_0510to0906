from __future__ import annotations

import io
import pandas as pd
import streamlit as st

from config.settings import get_settings
from common import broker_alpaca as ba
from scripts.run_all_systems_today import compute_today_signals
from common.universe import (
    build_universe_from_cache,
    save_universe_file,
    load_universe_file,
)
from common.notifier import create_notifier


st.set_page_config(page_title="本日のシグナル", layout="wide")
st.title("📈 本日のシグナル（全システム）")

settings = get_settings(create_dirs=True)
notifier = create_notifier(platform="slack", fallback=True)

with st.sidebar:
    st.header("ユニバース")
    universe = load_universe_file()
    if not universe:
        universe = build_universe_from_cache(limit=None)
        save_universe_file(universe)
    all_syms = universe

    # テスト用10銘柄 or 全銘柄選択
    test_mode = st.checkbox("テスト用（10銘柄のみ）", value=False)
    syms = all_syms[:10] if test_mode else all_syms

    st.write(f"銘柄数: {len(syms)}")
    st.write(", ".join(syms[:10]) + (" ..." if len(syms) > 10 else ""))

    st.header("資産")
    # Alpacaから取得した資産のみを使う
    if "today_cap_long" not in st.session_state:
        st.session_state["today_cap_long"] = 0.0
    if "today_cap_short" not in st.session_state:
        st.session_state["today_cap_short"] = 0.0

    # --- 削除: 青いinfo表示（資産の現在値） ---
    # st.info(f"long資産: {st.session_state['today_cap_long']:.2f} / short資産: {st.session_state['today_cap_short']:.2f}")

    # Alpacaから取得してフォームに反映
    if st.button("🔍 Alpacaから資産取得してフォームに反映"):
        try:
            client = ba.get_client(paper=True)
            acct = client.get_account()
            bp_raw = getattr(acct, "buying_power", None)
            if bp_raw is None:
                bp_raw = getattr(acct, "cash", None)
            if bp_raw is not None:
                bp = float(bp_raw)
                st.session_state["today_cap_long"] = round(bp / 2.0, 2)
                st.session_state["today_cap_short"] = round(bp / 2.0, 2)
                st.success(
                    f"long資産/short資産を{st.session_state['today_cap_long']}ずつに設定（buying_powerの半分={bp}）"
                )
            else:
                st.warning("Alpaca口座情報: buying_power/cashが取得できません")
        except Exception as e:
            st.error(f"Alpaca資産取得エラー: {e}")

    # 資産入力フォーム
    st.session_state["today_cap_long"] = st.number_input(
        "long資産 (USD)",
        min_value=0.0,
        step=100.0,
        value=float(st.session_state["today_cap_long"]),
        key="today_cap_long_input",
    )
    st.session_state["today_cap_short"] = st.number_input(
        "short資産 (USD)",
        min_value=0.0,
        step=100.0,
        value=float(st.session_state["today_cap_short"]),
        key="today_cap_short_input",
    )

    st.header("CSV保存")
    save_csv = st.checkbox("CSVをsignals_dirに保存", value=False)

    st.header("Alpaca自動発注")
    paper_mode = st.checkbox("ペーパートレードを使用", value=True)
    retries = st.number_input("リトライ回数", min_value=0, max_value=5, value=2)
    delay = st.number_input("遅延（秒）", min_value=0.0, step=0.5, value=0.5)
    poll_status = st.checkbox("注文状況を10秒ポーリング", value=False)
    do_trade = st.checkbox("Alpacaで自動発注", value=False)

    # キャッシュクリアボタン
    if st.button("キャッシュクリア"):
        st.cache_data.clear()
        st.success("キャッシュをクリアしました")

if st.button("▶ 本日のシグナル実行", type="primary"):
    # prepare live log display
    if "_today_logs" not in st.session_state:
        st.session_state["_today_logs"] = []
    log_box = st.empty()

    def _ui_log(msg: str) -> None:
        try:
            st.session_state["_today_logs"].append(str(msg))
            # show as code block for monospaced output
            log_box.code("\n".join(st.session_state["_today_logs"]))
        except Exception:
            pass

    with st.spinner("実行中..."):
        final_df, per_system = compute_today_signals(
            syms,
            capital_long=float(st.session_state["today_cap_long"]),
            capital_short=float(st.session_state["today_cap_short"]),
            save_csv=save_csv,
            log_callback=_ui_log,
        )

    for name, df in per_system.items():
        syms2 = df["symbol"].tolist() if df is not None and not df.empty else []
        notifier.send_signals(name, syms2)

    st.subheader("最終選定銘柄")
    if final_df is None or final_df.empty:
        st.info("本日のシグナルはありません。")
    else:
        st.dataframe(final_df, use_container_width=True)
        csv = final_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "最終CSVをダウンロード", data=csv, file_name="today_signals_final.csv"
        )

        # Alpaca 自動発注（任意）
        if do_trade:
            st.divider()
            st.subheader("Alpaca自動発注結果")
            try:
                client = ba.get_client(paper=paper_mode)
            except Exception as e:
                st.error(f"Alpaca接続エラー: {e}")
                client = None
            results = []
            if client is not None:
                # システムごとに注文タイプとTIFを自動決定
                system_order_type = {
                    "system1": "market",
                    "system3": "market",
                    "system4": "market",
                    "system5": "market",
                    "system2": "limit",
                    "system6": "limit",
                    "system7": "limit",
                }
                tif = "DAY"  # 米国市場の当日限り
                # 重複注文防止: symbol, system, entry_date で一意に
                unique_orders = {}
                for _, r in final_df.iterrows():
                    key = (
                        str(r.get("symbol")),
                        str(r.get("system")).lower(),
                        str(r.get("entry_date")),
                    )
                    if key in unique_orders:
                        continue
                    unique_orders[key] = r

                for key, r in unique_orders.items():
                    sym = str(r.get("symbol"))
                    qty = int(r.get("shares") or 0)
                    side = "buy" if str(r.get("side")).lower() == "long" else "sell"
                    system = str(r.get("system")).lower()
                    order_type = system_order_type.get(system, "market")
                    entry_price_raw = r.get("entry_price")
                    if (
                        order_type == "limit"
                        and entry_price_raw is not None
                        and entry_price_raw != ""
                    ):
                        try:
                            limit_price = float(entry_price_raw)
                        except (TypeError, ValueError):
                            limit_price = None
                    else:
                        limit_price = None
                    if not sym or qty <= 0:
                        continue
                    try:
                        order = ba.submit_order_with_retry(
                            client,
                            sym,
                            qty,
                            side=side,
                            order_type=order_type,
                            limit_price=limit_price,
                            time_in_force=tif,
                            retries=int(retries),
                            backoff_seconds=float(max(0.0, delay)),
                            rate_limit_seconds=float(max(0.0, delay)),
                            log_callback=_ui_log,
                        )
                        results.append(
                            {
                                "symbol": sym,
                                "side": side,
                                "qty": qty,
                                "order_id": getattr(order, "id", None),
                                "status": getattr(order, "status", None),
                                "system": system,
                                "order_type": order_type,
                                "time_in_force": tif,
                                "entry_date": r.get("entry_date"),
                            }
                        )
                    except Exception as e:
                        results.append(
                            {
                                "symbol": sym,
                                "side": side,
                                "qty": qty,
                                "error": str(e),
                                "system": system,
                                "order_type": order_type,
                                "time_in_force": tif,
                                "entry_date": r.get("entry_date"),
                            }
                        )
            if results:
                st.dataframe(pd.DataFrame(results), use_container_width=True)
                notifier.send_trade_report("integrated", results)
                if poll_status and any(r.get("order_id") for r in results):
                    st.info("注文状況を10秒間ポーリングします...")
                    import time

                    order_ids = [
                        r.get("order_id") for r in results if r.get("order_id")
                    ]
                    end = time.time() + 10
                    last = {}
                    while time.time() < end:
                        status_map = ba.get_orders_status_map(client, order_ids)
                        if status_map != last:
                            st.write(status_map)
                            last = status_map
                        time.sleep(1.0)

    with st.expander("システム別詳細"):
        # Ensure results is always defined
        results = []
        # Ensure client is defined for polling order status
        try:
            client = ba.get_client(paper=paper_mode)
        except Exception as e:
            st.error(f"Alpaca接続エラー: {e}")
            client = None
        for name, df in per_system.items():
            st.markdown(f"#### {name}")
            if df is None or df.empty:
                st.write("(空)")
            else:
                st.dataframe(df, use_container_width=True)
                csv2 = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    f"{name}のCSVをダウンロード",
                    data=csv2,
                    file_name=f"signals_{name}.csv",
                )

                # Debug: show per-symbol reason text for why it was selected
                if "reason" in df.columns:
                    with st.expander(f"{name} - 選定理由", expanded=False):
                        for _, row in df.iterrows():
                            sym = row.get("symbol")
                            reason = row.get("reason")
                            st.markdown(f"- **{sym}**: {reason}")
                if "reason" in df.columns:
                    with st.expander(f"{name} - selection reasons", expanded=False):
                        for _, row in df.iterrows():
                            sym = row.get("symbol")
                            reason = row.get("reason")
                            st.markdown(f"- **{sym}**: {reason}")
                st.dataframe(pd.DataFrame(results), use_container_width=True)
                notifier.send_trade_report("integrated", results)
                if (
                    poll_status
                    and any(r.get("order_id") for r in results)
                    and client is not None
                ):
                    st.info("注文状況を10秒間ポーリングします...")
                    import time

                    order_ids = [
                        r.get("order_id") for r in results if r.get("order_id")
                    ]
                    order_ids = [
                        r.get("order_id") for r in results if r.get("order_id")
                    ]
                    end = time.time() + 10
                    last = {}
                    while time.time() < end:
                        status_map = ba.get_orders_status_map(client, order_ids)
                        if status_map != last:
                            st.write(status_map)
                            last = status_map
                        time.sleep(1.0)

    with st.expander("システム別詳細"):
        for name, df in per_system.items():
            st.markdown(f"#### {name}")
            if df is None or df.empty:
                st.write("(空)")
            else:
                # show dataframe (includes reason column if available)
                st.dataframe(df, use_container_width=True)
                csv2 = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    f"{name}のCSVをダウンロード",
                    data=csv2,
                    file_name=f"signals_{name}.csv",
                    key=f"{name}_download_csv",  # ← ここを追加
                )

                # Debug: show per-symbol reason text for why it was selected
                if "reason" in df.columns:
                    with st.expander(f"{name} - 選定理由", expanded=False):
                        for _, row in df.iterrows():
                            sym = row.get("symbol")
                            reason = row.get("reason")
                            st.markdown(f"- **{sym}**: {reason}")
                if "reason" in df.columns:
                    with st.expander(f"{name} - selection reasons", expanded=False):
                        for _, row in df.iterrows():
                            sym = row.get("symbol")
                            reason = row.get("reason")
                            st.markdown(f"- **{sym}**: {reason}")
                            sym = row.get("symbol")
                            reason = row.get("reason")
                            st.markdown(f"- **{sym}**: {reason}")
