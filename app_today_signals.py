from __future__ import annotations

import io
import pandas as pd
import streamlit as st

from config.settings import get_settings
from common import broker_alpaca as ba
from scripts.run_all_systems_today import compute_today_signals
from common.universe import build_universe_from_cache, save_universe_file, load_universe_file
from common.notifier import create_notifier


st.set_page_config(page_title="Today Signals", layout="wide")
st.title("📈 Today Signals (All Systems)")

settings = get_settings(create_dirs=True)
notifier = create_notifier(platform="slack", fallback=True)

with st.sidebar:
    st.header("Universe")
    if st.button("🔁 Rebuild Universe (cached)"):
        syms = build_universe_from_cache(limit=None)
        path = save_universe_file(syms)
        st.success(f"Universe updated: {path} ({len(syms)} symbols)")

    universe = load_universe_file()
    if not universe:
        universe = build_universe_from_cache(limit=None)
        save_universe_file(universe)
    default_syms = universe
    syms_text = st.text_area(
        "Symbols (comma/space separated)",
        value=", ".join(default_syms),
        height=100,
    )
    syms = [
        s.strip().upper()
        for s in syms_text.replace("\n", ",").replace(" ", ",").split(",")
        if s.strip()
    ]

    st.header("Budgets")
    col1, col2 = st.columns(2)
    # use session_state so we can update values programmatically (e.g., fetch from Alpaca)
    if "cap_long_input" not in st.session_state:
        st.session_state["cap_long_input"] = float(
            getattr(settings.backtest, "initial_capital", 2000.0) or 2000.0
        )
    if "cap_short_input" not in st.session_state:
        st.session_state["cap_short_input"] = float(
            getattr(settings.backtest, "initial_capital", 2000.0) or 2000.0
        )

    with col1:
        cap_long = st.number_input(
            "Capital Long ($)",
            min_value=0.0,
            step=100.0,
            value=st.session_state["cap_long_input"],
            key="cap_long_input",
        )
    with col2:
        cap_short = st.number_input(
            "Capital Short ($)",
            min_value=0.0,
            step=100.0,
            value=st.session_state["cap_short_input"],
            key="cap_short_input",
        )

    save_csv = st.checkbox("Save CSV to signals_dir", value=False)

    st.header("Filters")
    only_long = st.checkbox("Trade only LONG", value=False)
    only_short = st.checkbox("Trade only SHORT", value=False)
    top_per_system = st.number_input(
        "Top N per system", min_value=0, step=1, value=0, help="0 for unlimited"
    )

    st.header("Alpaca Auto-Trade")
    do_trade = st.checkbox("Auto submit via Alpaca", value=False)
    order_type = st.selectbox("Order Type", ["market", "limit"], index=0)
    tif = st.selectbox("Time In Force", ["GTC", "DAY"], index=0)
    paper_mode = st.checkbox("Use Paper Trading", value=True)
    retries = st.number_input("Retries", min_value=0, max_value=5, value=2)
    delay = st.number_input("Delay (sec)", min_value=0.0, step=0.5, value=0.5)
    poll_status = st.checkbox("Poll order status (10s)", value=False)

    # Alpaca: fetch balances and populate inputs
    if st.button("🔍 Fetch Alpaca Balances into inputs"):
        try:
            client = ba.get_client(paper=paper_mode)
            acct = client.get_account()
            bp = None
            try:
                bp = float(getattr(acct, "buying_power", None) or getattr(acct, "cash", None))
            except Exception:
                bp = None
            if bp is None:
                st.warning("Alpaca account info: buying_power/cash not available")
            else:
                # split into long/short equally
                half = round(max(0.0, float(bp)) / 2.0, 2)
                st.session_state["cap_long_input"] = half
                st.session_state["cap_short_input"] = half
                st.success(f"Set Capital Long/Short to {half} each (half of buying_power={bp})")
        except Exception as e:
            st.error(f"Alpaca 残高取得エラー: {e}")

if st.button("▶ Run Today Signals", type="primary"):
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

    with st.spinner("Running..."):
        final_df, per_system = compute_today_signals(
            syms,
            capital_long=cap_long,
            capital_short=cap_short,
            save_csv=save_csv,
            log_callback=_ui_log,
        )

    for name, df in per_system.items():
        syms2 = df["symbol"].tolist() if df is not None and not df.empty else []
        notifier.send_signals(name, syms2)

    st.subheader("Final Picks")
    if final_df is None or final_df.empty:
        st.info("No signals today.")
    else:
        # filters
        filtered = final_df.copy()
        if "side" in filtered.columns:
            if only_long and not only_short:
                filtered = filtered[filtered["side"].str.lower() == "long"]
            if only_short and not only_long:
                filtered = filtered[filtered["side"].str.lower() == "short"]
        if top_per_system and top_per_system > 0 and "system" in filtered.columns:
            by = ["system"] + (["side"] if "side" in filtered.columns else [])
            filtered = filtered.groupby(by, as_index=False, group_keys=False).head(
                int(top_per_system)
            )

        st.dataframe(filtered, use_container_width=True)
        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("Download Final CSV", data=csv, file_name="today_signals_final.csv")

        # Alpaca 自動発注（任意）
        if do_trade:
            st.divider()
            st.subheader("Alpaca Auto-Trade Result")
            try:
                client = ba.get_client(paper=paper_mode)
            except Exception as e:
                st.error(f"Alpaca接続エラー: {e}")
                client = None
            results = []
            if client is not None:
                for _, r in filtered.iterrows():
                    sym = str(r.get("symbol"))
                    qty = int(r.get("shares") or 0)
                    side = "buy" if str(r.get("side")).lower() == "long" else "sell"
                    if not sym or qty <= 0:
                        continue
                    # Safely parse entry_price only when using limit orders and a value is present
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
                            }
                        )
                    except Exception as e:
                        results.append(
                            {
                                "symbol": sym,
                                "side": side,
                                "qty": qty,
                                "error": str(e),
                            }
                        )
            if results:
                st.dataframe(pd.DataFrame(results), use_container_width=True)
                notifier.send_trade_report("integrated", results)
                if poll_status and any(r.get("order_id") for r in results):
                    st.info("Polling order status for 10 seconds...")
                    import time

                    order_ids = [r.get("order_id") for r in results if r.get("order_id")]
                    end = time.time() + 10
                    last = {}
                    while time.time() < end:
                        status_map = ba.get_orders_status_map(client, order_ids)
                        if status_map != last:
                            st.write(status_map)
                            last = status_map
                        time.sleep(1.0)

    with st.expander("Per-system details"):
        for name, df in per_system.items():
            st.markdown(f"#### {name}")
            if df is None or df.empty:
                st.write("(empty)")
            else:
                # show dataframe (includes reason column if available)
                st.dataframe(df, use_container_width=True)
                csv2 = df.to_csv(index=False).encode("utf-8")
                st.download_button(f"Download {name}", data=csv2, file_name=f"signals_{name}.csv")

                # Debug: show per-symbol reason text for why it was selected
                if "reason" in df.columns:
                    with st.expander(f"{name} - selection reasons", expanded=False):
                        for _, row in df.iterrows():
                            sym = row.get("symbol")
                            reason = row.get("reason")
                            st.markdown(f"- **{sym}**: {reason}")
