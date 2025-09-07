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
    """Inject minimal CSS for a cleaner look."""
    st.markdown(
        """
        <style>
        .alpaca-metric {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #f0f2f6;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _fetch_account_and_positions() -> tuple[object, list[object]]:
    """Retrieve account and open positions using the Alpaca client."""
    client = ba.get_client()
    account = client.get_account()
    positions = client.get_all_positions()
    return account, positions


def _positions_to_df(positions: list[object]) -> pd.DataFrame:
    """Convert list of position objects to ``pandas.DataFrame``."""
    records: list[dict[str, str]] = []
    for pos in positions:
        records.append(
            {
                "symbol": getattr(pos, "symbol", ""),
                "qty": getattr(pos, "qty", ""),
                "avg_entry_price": getattr(pos, "avg_entry_price", ""),
                "current_price": getattr(pos, "current_price", ""),
                "unrealized_pl": getattr(pos, "unrealized_pl", ""),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    """Run the Streamlit dashboard."""
    st.set_page_config(page_title="Alpaca Dashboard", layout="wide")
    st.title("Alpaca 現在状況")
    _inject_css()

    try:
        account, positions = _fetch_account_and_positions()
    except Exception as exc:  # pragma: no cover - network or credential errors
        st.error(f"データ取得に失敗しました: {exc}")
        return

    cols = st.columns(3)
    cols[0].metric("総資産", getattr(account, "equity", "-"))
    cols[1].metric("現金", getattr(account, "cash", "-"))
    cols[2].metric("余力", getattr(account, "buying_power", "-"))

    st.subheader("保有ポジション")
    pos_df = _positions_to_df(positions)
    if pos_df.empty:
        st.info("ポジションはありません。")
    else:
        st.dataframe(pos_df, use_container_width=True)


if __name__ == "__main__":  # pragma: no cover - UI entry point
    main()
