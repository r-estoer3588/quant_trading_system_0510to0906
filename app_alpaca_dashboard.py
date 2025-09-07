"""Simple Streamlit dashboard to display current Alpaca account status.

The page fetches the account and open positions from the Alpaca API via
``common.broker_alpaca``.  Styling is applied with a small CSS snippet to keep
the layout clean and easy to read.
"""

from __future__ import annotations

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


def _fetch_account_and_positions() -> tuple[object, object]:
    """Retrieve account and open positions using the Alpaca client."""
    client = ba.get_client()
    account = client.get_account()
    positions = client.get_all_positions()
    return account, positions


def _positions_to_df(positions) -> pd.DataFrame:
    """Convert list of position objects to ``pandas.DataFrame`` with Japanese columns."""
    records: list[dict[str, str]] = []
    for pos in positions:
        records.append(
            {
                "銘柄": getattr(pos, "symbol", ""),
                "数量": getattr(pos, "qty", ""),
                "平均取得単価": getattr(pos, "avg_entry_price", ""),
                "現在値": getattr(pos, "current_price", ""),
                "含み損益": getattr(pos, "unrealized_pl", ""),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    """Run the Streamlit dashboard."""
    st.set_page_config(page_title="Alpaca Dashboard", layout="wide")
    st.markdown(
        "<h1 style='margin-bottom:0.5em; margin-top:0.5em; padding-left:0.2em;'>Alpaca <span style=\"color:#00b894;\">現在状況</span></h1>",
        unsafe_allow_html=True,
    )
    _inject_css()

    try:
        account, positions = _fetch_account_and_positions()
    except Exception as exc:  # pragma: no cover - network or credential errors
        st.error(f"データ取得に失敗しました: {exc}")
        return

    st.markdown("<div class='alpaca-card'>", unsafe_allow_html=True)
    cols = st.columns(3)
    with cols[0]:
        st.markdown(
            f"<div class='alpaca-metric'><div class='metric-label'>総資産</div><div class='metric-value'>{getattr(account, 'equity', '-')}</div></div>",
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            f"<div class='alpaca-metric'><div class='metric-label'>現金</div><div class='metric-value'>{getattr(account, 'cash', '-')}</div></div>",
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            f"<div class='alpaca-metric'><div class='metric-label'>余力</div><div class='metric-value'>{getattr(account, 'buying_power', '-')}</div></div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<h2 style='margin-top:2em;'>保有ポジション</h2>", unsafe_allow_html=True
    )
    st.markdown("<div class='alpaca-card'>", unsafe_allow_html=True)
    pos_df = _positions_to_df(positions)
    if pos_df.empty:
        st.info("ポジションはありません。")
    else:
        st.dataframe(pos_df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":  # pragma: no cover - UI entry point
    main()
