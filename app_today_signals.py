from __future__ import annotations

import time
from typing import Any

import pandas as pd
import streamlit as st

# Streamlit checkbox の重複ID対策（key未指定時に自動で一意キーを付与）
try:
    if not hasattr(st, "_orig_checkbox"):
        st._orig_checkbox = st.checkbox  # type: ignore[attr-defined]

        def _unique_checkbox(label, *args, **kwargs):
            if "key" not in kwargs:
                base = f"auto_cb_{abs(hash(str(label))) % 10**8}"
                count_key = f"_{base}_cnt"
                try:
                    cnt = int(st.session_state.get(count_key, 0)) + 1
                except Exception:
                    cnt = 1
                st.session_state[count_key] = cnt
                kwargs["key"] = f"{base}_{cnt}"
            return st._orig_checkbox(  # type: ignore[attr-defined]
                label,
                *args,
                **kwargs,
            )

        st.checkbox = _unique_checkbox  # type: ignore[attr-defined]
except Exception:
    # 失敗しても従来動作のまま進める
    pass

from common import broker_alpaca as ba
from common import universe as univ
from common.data_loader import load_price
from common.notifier import create_notifier
from common.profit_protection import evaluate_positions
from config.settings import get_settings
from scripts.run_all_systems_today import compute_today_signals

st.set_page_config(page_title="本日のシグナル", layout="wide")
st.title("📈 本日のシグナル（全システム）")

settings = get_settings(create_dirs=True)
notifier = create_notifier(platform="slack", fallback=True)

with st.sidebar:
    st.header("ユニバース")
    universe = univ.load_universe_file()
    if not universe:
        universe = univ.build_universe_from_cache(limit=None)
        univ.save_universe_file(universe)
    all_syms = universe

    # 任意の件数でユニバースを制限するテスト用オプション
    limit_max = max(1, len(all_syms))
    test_limit = st.number_input(
        "テスト用の銘柄数 (0は制限なし)",
        min_value=0,
        max_value=limit_max,
        value=0,
        step=1,
    )
    syms = all_syms[: int(test_limit)] if test_limit else all_syms

    st.write(f"銘柄数: {len(syms)}")
    st.write(", ".join(syms[:10]) + (" ..." if len(syms) > 10 else ""))

    st.header("資産")
    # Alpacaから取得した資産のみを使う
    if "today_cap_long" not in st.session_state:
        st.session_state["today_cap_long"] = 0.0
    if "today_cap_short" not in st.session_state:
        st.session_state["today_cap_short"] = 0.0

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
                    f"long資産/short資産を{st.session_state['today_cap_long']}ずつに設定"
                    f"（buying_powerの半分={bp}）"
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

    # 注文状況を10秒ポーリングとは？
    # → Alpacaに注文を送信した後、注文IDのステータス（filled, canceled等）を10秒間、
    #    1秒ごとに取得・表示する機能です。
    # これにより、注文が約定したかどうかをリアルタイムで確認できます。

    # キャッシュクリアボタン
    if st.button("キャッシュクリア"):
        st.cache_data.clear()
        st.success("キャッシュをクリアしました")

    if st.button("全注文キャンセル"):
        try:
            client = ba.get_client(paper=paper_mode)
            ba.cancel_all_orders(client)
            st.success("すべての未約定注文をキャンセルしました")
        except Exception as e:
            st.error(f"注文キャンセルエラー: {e}")

st.subheader("保有ポジションと利益保護判定")
if st.button("🔍 Alpacaから保有ポジション取得"):
    try:
        client = ba.get_client(paper=paper_mode)
        positions = client.get_all_positions()
        st.session_state["positions_df"] = evaluate_positions(positions)
        st.success("ポジションを取得しました")
    except Exception as e:
        st.error(f"ポジション取得エラー: {e}")

if "positions_df" in st.session_state:
    df_pos = st.session_state["positions_df"]
    if df_pos.empty:
        st.info("保有ポジションはありません。")
    else:
        st.dataframe(df_pos, use_container_width=True)

if st.button("▶ 本日のシグナル実行", type="primary"):
    # 指標ごとに必要な日数（＋10%余裕）を定義
    indicator_days = {
        "ROC200": int(200 * 1.1),
        "SMA25": int(25 * 1.1),
        "ATR20": int(20 * 1.1),
        "ADX7": int(7 * 1.1),
        "RETURN6": int(6 * 1.1),
        "Drop3D": int(3 * 1.1),
        "Return6D": int(6 * 1.1),
        # 必要に応じて追加
    }

    # 必要な最大日数を算出（全システムで使う指標の最大値）
    max_days = max(indicator_days.values())

    # 開始時刻を記録
    start_time = time.time()
    # 進捗表示用の領域（1行上書き）
    progress_area = st.empty()

    def _ui_log(msg: str) -> None:
        try:
            elapsed = time.time() - start_time
            progress_area.text(f"[{elapsed:6.1f}s] {msg}")
        except Exception:
            pass

    # 必要な日数分だけデータをロードする関数例
    def load_minimal_history(symbol: str, days: int) -> pd.DataFrame:
        # 指標ごとに必要な日数（＋10%余裕）を明示リストで定義
        indicator_days = {
            "SMA25": int(25 * 1.1),
            "SMA50": int(50 * 1.1),
            "SMA100": int(100 * 1.1),
            "SMA150": int(150 * 1.1),
            "SMA200": int(200 * 1.1),
            "ATR3": int(50 * 1.1),  # 3ATRは最大50日分必要と仮定
            "ATR1.5": int(40 * 1.1),
            "ATR1": int(10 * 1.1),
            "ATR2.5": int(10 * 1.1),
            "ATR": int(50 * 1.1),  # 最大50日分
            "ADX7": int(7 * 1.1),
            "ADX7_High": int(7 * 1.1),
            "RETURN6": int(6 * 1.1),
            "Return6D": int(6 * 1.1),
            "return_pct": int(200 * 1.1),  # 総リターンは最大200日分
            "Drop3D": int(3 * 1.1),
        }
        # 最大必要日数を算出
        max_days = max(indicator_days.values())
        # 銘柄ごとのヒストリカルCSVを最大必要日数分だけロード
        try:
            df = load_price(symbol, cache_profile="rolling")
            data = df.tail(max_days)
        except Exception:
            data = pd.DataFrame()
        return data

    # シグナル計算時に必要な日数分だけデータを渡すようにcompute_today_signalsへ
    with st.spinner("実行中... (経過時間表示あり)"):
        # 必要な日数分だけデータをロードして渡す（例: dictで渡す）
        symbol_data = {sym: load_minimal_history(sym, max_days) for sym in syms}
        final_df, per_system = compute_today_signals(
            syms,
            capital_long=float(st.session_state["today_cap_long"]),
            capital_short=float(st.session_state["today_cap_short"]),
            save_csv=save_csv,
            log_callback=_ui_log,
            symbol_data=symbol_data,  # 追加: 必要日数分だけのデータ
        )

    # DataFrameのインデックスをリセットしてf1などの疑似インデックスを排除
    final_df = final_df.reset_index(drop=True)
    per_system = {name: df.reset_index(drop=True) for name, df in per_system.items()}

    # 処理終了時に総経過時間を表示
    total_elapsed = time.time() - start_time
    st.info(f"総経過時間: {total_elapsed:.1f}秒")

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
            "最終CSVをダウンロード",
            data=csv,
            file_name="today_signals_final.csv",
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

                for _key, r in unique_orders.items():
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
                    # estimate price for notification purposes
                    price_val = None
                    try:
                        if entry_price_raw is not None and entry_price_raw != "":
                            price_val = float(entry_price_raw)
                    except (TypeError, ValueError):
                        price_val = None
                    if limit_price is not None:
                        price_val = limit_price
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
                                "price": price_val,
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
                                "price": price_val,
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

                    order_ids: list[str] = []
                    for r in results:
                        oid = r.get("order_id")
                        if oid:
                            order_ids.append(oid)
                    end = time.time() + 10
                    last: dict[str, Any] = {}
                    while time.time() < end:
                        status_map = ba.get_orders_status_map(client, order_ids)
                        if status_map != last:
                            if status_map:  # 空でなければ表示
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
                    key=f"{name}_download_csv",
                )

                # Debug: show per-symbol reason text for why it was selected
                if "reason" in df.columns:
                    with st.expander(f"{name} - 選定理由", expanded=False):
                        for _, row in df.iterrows():
                            sym = row.get("symbol")
                            reason = row.get("reason")
                            st.markdown(f"- **{sym}**: {reason}")
                    with st.expander(f"{name} - selection reasons", expanded=False):
                        for _, row in df.iterrows():
                            sym = row.get("symbol")
                            reason = row.get("reason")
                            st.markdown(f"- **{sym}**: {reason}")
