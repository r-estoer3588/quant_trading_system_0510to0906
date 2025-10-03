from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

try:
    from common.cache_format import round_dataframe
except Exception:

    def round_dataframe(df: pd.DataFrame, decimals: int | None) -> pd.DataFrame:
        if decimals is None:
            return df
        try:
            return df.copy().round(int(decimals))
        except Exception:
            try:
                return df.round(int(decimals))
            except Exception:
                return df


try:
    from config.settings import get_settings
except Exception:
    get_settings = None


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
    # 事前に 0 で埋めた int8 型マトリクスを確保することで、後段の fillna → downcast に伴う
    # FutureWarning (pandas の object -> int 暗黙ダウンキャスト非推奨) を完全に回避する。
    # これにより列ごとの個別型推論と逐次代入も不要になりパフォーマンスも僅かに改善。
    holding_matrix = pd.DataFrame(0, index=all_dates, columns=all_symbols, dtype="int8")

    total_steps = len(all_dates)
    for j, date in enumerate(all_dates, 1):
        # その日の保有銘柄だけ 1 に更新（非保有は既に 0 初期化済み）
        held = holding_dict[date]
        if held:
            # .loc/at で個別更新より高速なベクトル代入を使用
            holding_matrix.loc[date, list(held)] = 1

        if matrix_progress_callback:
            ratio = j / total_steps
            # 1%進んだタイミング or 最後の処理で更新
            if int(ratio * 100) != int((j - 1) / total_steps * 100) or j == total_steps:
                matrix_progress_callback(j, total_steps)
    # すでに int8 で確定しているので追加処理不要
    return holding_matrix


def display_holding_heatmap(matrix: pd.DataFrame, title: str = "日別保有ヒートマップ") -> None:
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


def download_holding_csv(matrix: pd.DataFrame, filename: str = "holding_status.csv") -> None:
    """
    保有銘柄の遷移をCSV形式でダウンロード提供。
    """
    try:
        if get_settings:
            settings = get_settings(create_dirs=False)
            round_dec = getattr(settings.cache, "round_decimals", None)
        else:
            round_dec = None
    except Exception:
        round_dec = None
    try:
        matrix_to_write = round_dataframe(matrix, round_dec)
    except Exception:
        matrix_to_write = matrix
    csv = matrix_to_write.to_csv().encode("utf-8")
    st.download_button("保有銘柄の遷移をCSVで保存", data=csv, file_name=filename, mime="text/csv")
