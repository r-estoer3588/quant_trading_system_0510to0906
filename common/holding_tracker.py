import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import streamlit as st


def generate_holding_matrix(
    results_df: pd.DataFrame,
    trade_progress_callback=None,
    matrix_progress_callback=None,
) -> pd.DataFrame:
    holding_dict = defaultdict(set)
    all_trades = list(results_df.iterrows())
    total_trades = len(all_trades)

    # フェーズ1: トレード処理
    for i, (_, row) in enumerate(all_trades, 1):
        current_date = pd.to_datetime(row["entry_date"])
        end_date = pd.to_datetime(row["exit_date"])
        while current_date <= end_date:
            holding_dict[current_date.date()].add(row["symbol"])
            current_date += pd.Timedelta(days=1)

        if trade_progress_callback and (i % 10 == 0 or i == total_trades):
            trade_progress_callback(i, total_trades)

    # フェーズ2: マトリクス生成
    all_dates = sorted(holding_dict.keys())
    all_symbols = sorted(set(results_df["symbol"].unique()))
    holding_matrix = pd.DataFrame(index=all_dates, columns=all_symbols)

    total_steps = len(all_dates)
    for j, date in enumerate(all_dates, 1):
        for sym in all_symbols:
            holding_matrix.loc[date, sym] = 1 if sym in holding_dict[date] else 0

        if matrix_progress_callback:
            ratio = j / total_steps
            # 1%進んだタイミング or 最後の処理で更新
            if int(ratio * 100) != int((j - 1) / total_steps * 100) or j == total_steps:
                matrix_progress_callback(j, total_steps)
    return holding_matrix.fillna(0).astype(int)


def display_holding_heatmap(
    matrix: pd.DataFrame, title: str = "日別保有ヒートマップ"
) -> None:
    """
    Streamlitで保有銘柄のヒートマップを表示。
    - matrix: generate_holding_matrixの出力
    - title: 表示タイトル
    """
    st.subheader(title)

    # 表示行数が多い場合の制限（例: 最大100行）
    max_rows = 100
    if len(matrix) > max_rows:
        st.info(f"表示行数を制限中（最新{max_rows}日分）")
        matrix = matrix.tail(max_rows)

    fig, ax = plt.subplots(figsize=(12, max(4, len(matrix) // 3)))
    sns.heatmap(matrix, cmap="Greens", cbar=False, linewidths=0.5, linecolor="gray")
    ax.set_xlabel("銘柄")
    ax.set_ylabel("日付")
    ax.set_title(title)
    st.pyplot(fig)


def download_holding_csv(
    matrix: pd.DataFrame, filename: str = "holding_status.csv"
) -> None:
    """
    保有銘柄の遷移をCSV形式でダウンロード提供。
    """
    csv = matrix.to_csv().encode("utf-8")
    st.download_button(
        "保有銘柄の遷移をCSVで保存", data=csv, file_name=filename, mime="text/csv"
    )
