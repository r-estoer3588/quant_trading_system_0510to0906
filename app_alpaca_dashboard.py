"""Simple Streamlit dashboard to display current Alpaca account status.

The page fetches the account and open positions from the Alpaca API via
``common.broker_alpaca``.  Styling is applied with a small CSS snippet to keep
the layout clean and easy to read.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from common import broker_alpaca as ba


def _inject_css() -> None:
    """Inject modern CSS for a stylish dashboard."""
    st.markdown(
        """
        <style>
        body, .stApp {
            background: #181c24 !important;
            color: #f5f6fa !important;
            font-family: 'Segoe UI', 'Meiryo', 'sans-serif';
        }
        .main {
            background: #181c24 !important;
        }
        h1, h2, h3, h4 {
            color: #f5f6fa !important;
            font-weight: 700;
            letter-spacing: 1px;
        }
        h1 {
            font-size: 2.6rem !important;
            margin-top: 0.5em !important;
            margin-bottom: 0.5em !important;
            padding-left: 0.2em !important;
        }
        .alpaca-card {
            background: #23283a;
            border-radius: 1rem;
            box-shadow: 0 2px 16px rgba(0,0,0,0.12);
            padding: 1.2rem 1.2rem 1.2rem 1.2rem;
            margin-bottom: 1.5rem;
        }
        .alpaca-metric {
            background: #23283a;
            border-radius: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.10);
            padding: 1.2rem 1rem;
            margin: 0.5rem;
            text-align: center;
        }
        .alpaca-metric .metric-label {
            font-size: 1.1rem;
            color: #a5b1c2;
            margin-bottom: 0.5rem;
        }
        .alpaca-metric .metric-value {
            font-size: 2.6rem;
            font-weight: 700;
            color: #f5f6fa;
        }
        .stDataFrame {
            background: #23283a !important;
            border-radius: 1rem !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.10) !important;
            padding: 0.5rem !important;
            margin-top: 0 !important;
        }
        .stInfo {
            background: #23283a !important;
            color: #a5b1c2 !important;
            border-radius: 1rem;
            padding: 1rem;
        }
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
        }
        /* 不要なバーを消す */
        .css-1dp5vir.e1fqkh3o3 {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _fetch_account_and_positions() -> tuple[object, object, object]:
    """Retrieve client, account and open positions using the Alpaca client."""
    client = ba.get_client()
    account = client.get_account()
    positions = client.get_all_positions()
    return client, account, positions


def _days_held(entry_dt: pd.Timestamp | str | datetime | None) -> int | None:
    """Calculate holding days from entry date."""
    if entry_dt is None:
        return None
    # 既に pandas.Timestamp の場合はそのまま使用し、そうでなければパースを試みる
    if isinstance(entry_dt, pd.Timestamp):
        dt = entry_dt
    else:
        try:
            dt = pd.to_datetime(entry_dt, errors="coerce")
            if pd.isna(dt):
                return None
        except Exception:
            return None
    today = pd.Timestamp(datetime.utcnow()).normalize()
    return int((today - dt.normalize()).days)


def _fetch_entry_dates(client, symbols: list[str]) -> dict[str, pd.Timestamp]:
    """Fetch entry dates for symbols via account activities."""
    out: dict[str, pd.Timestamp] = {}
    for sym in symbols:
        try:
            acts = client.get_activities(symbol=sym, activity_types="FILL")
        except Exception:
            continue
        # key が None を返すと型チェックでエラーになるため空文字列をデフォルトにする
        for act in sorted(acts, key=lambda a: getattr(a, "transaction_time", "")):
            t = getattr(act, "transaction_time", None)
            if t:
                out[sym] = pd.Timestamp(t)
                break
    return out


def _positions_to_df(positions, client=None) -> pd.DataFrame:
    """Convert positions to DataFrame and append holding days and exit hints.

    client が指定されない場合はエントリー日の取得をスキップする。
    """
    symbols = [getattr(p, "symbol", "") for p in positions]
    entry_map = _fetch_entry_dates(client, symbols) if client else {}

    mapping_path = Path("data/symbol_system_map.json")
    symbol_map: dict[str, str] = {}
    if mapping_path.exists():
        try:
            symbol_map = json.loads(mapping_path.read_text())
        except Exception:
            symbol_map = {}

    hold_limits = {"system2": 2, "system3": 3, "system5": 6, "system6": 3}

    records: list[dict[str, object]] = []
    for pos in positions:
        sym = getattr(pos, "symbol", "")
        held = _days_held(entry_map.get(sym))
        system = symbol_map.get(sym, "unknown").lower()
        limit = hold_limits.get(system)
        exit_hint = (
            f"{limit}日経過で手仕舞い" if held is not None and limit and held >= limit else ""
        )
        records.append(
            {
                "銘柄": sym,
                "数量": getattr(pos, "qty", ""),
                "平均取得単価": getattr(pos, "avg_entry_price", ""),
                "現在値": getattr(pos, "current_price", ""),
                "含み損益": getattr(pos, "unrealized_pl", ""),
                "保有日数": held if held is not None else "-",
                "経過日手仕舞い": exit_hint,
            }
        )
    return pd.DataFrame(records)


def _group_by_system(
    df: pd.DataFrame,
    symbol_map: dict[str, str],
) -> dict[str, pd.DataFrame]:
    """銘柄をシステムにマッピングして資金配分を計算する."""
    if df.empty:
        return {}

    work = df.copy()
    work["評価額"] = work["数量"].astype(float) * work["現在値"].astype(float)
    work["system"] = work["銘柄"].map(symbol_map).fillna("unknown")

    grouped: dict[str, pd.DataFrame] = {}
    for system, g in work.groupby("system"):
        grouped[system] = g[["銘柄", "評価額"]]
    return grouped


def main() -> None:
    """Run the Streamlit dashboard."""
    st.set_page_config(page_title="Alpaca Dashboard", layout="wide")
    st.markdown(
        (
            "<h1 style='margin-bottom:0.5em; margin-top:0.5em; padding-left:0.2em;'>"
            "Alpaca "
            '<span style="color:#00b894;">現在状況</span>'
            "</h1>"
        ),
        unsafe_allow_html=True,
    )
    _inject_css()

    try:
        client, account, positions = _fetch_account_and_positions()
    except Exception as exc:  # pragma: no cover - network or credential errors
        st.error(f"データ取得に失敗しました: {exc}")
        return

    st.markdown("<div class='alpaca-card'>", unsafe_allow_html=True)
    cols = st.columns(3)
    with cols[0]:
        st.markdown(
            (
                f"<div class='alpaca-metric'>"
                f"<div class='metric-label'>総資産</div>"
                f"<div class='metric-value'>{getattr(account, 'equity', '-')}</div>"
                f"</div>"
            ),
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            (
                f"<div class='alpaca-metric'>"
                f"<div class='metric-label'>現金</div>"
                f"<div class='metric-value'>{getattr(account, 'cash', '-')}</div>"
                f"</div>"
            ),
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            (
                f"<div class='alpaca-metric'>"
                f"<div class='metric-label'>余力</div>"
                f"<div class='metric-value'>"
                f"{getattr(account, 'buying_power', '-')}</div>"
                f"</div>"
            ),
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<h2 style='margin-top:2em;'>保有ポジション</h2>", unsafe_allow_html=True)
    st.markdown("<div class='alpaca-card'>", unsafe_allow_html=True)
    pos_df = _positions_to_df(positions, client)
    if pos_df.empty:
        st.info("ポジションはありません。")
    else:
        st.dataframe(pos_df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # --- system ごとの円グラフ表示 ---
    mapping_path = Path("data/symbol_system_map.json")
    if not pos_df.empty and mapping_path.exists():
        try:
            symbol_map = json.loads(mapping_path.read_text())
        except Exception:
            st.info("symbol_system_map.json の読み込みに失敗しました。")
        else:
            grouped = _group_by_system(pos_df, symbol_map)
            for system, g in grouped.items():
                st.markdown(f"<h3>{system} 資金配分</h3>", unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.pie(g["評価額"], labels=g["銘柄"], autopct="%1.1f%%")
                ax.set_aspect("equal")
                st.pyplot(fig)
    elif mapping_path.exists():
        st.info("ポジションがないため円グラフを表示できません。")
    else:
        st.info("data/symbol_system_map.json が見つかりません。")


if __name__ == "__main__":  # pragma: no cover - UI entry point
    main()
