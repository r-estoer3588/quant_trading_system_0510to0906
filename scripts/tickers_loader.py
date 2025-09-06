# tickers_loader.py
import pandas as pd
import requests
import zipfile
import io
from datetime import datetime
from pathlib import Path
import streamlit as st
import time
import os

FAILED_LIST = "eodhd_failed_symbols.csv"


def load_failed_symbols():
    if os.path.exists(FAILED_LIST):
        # CSVの1列目のみをリスト化（headerなしの場合はheader=None指定）
        return set(pd.read_csv(FAILED_LIST, header=None)[0].astype(str).str.upper())
    return set()


# 日次キャッシュ（24時間）


@st.cache_data(ttl=86400)
def get_all_tickers():
    nasdaq_url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
    other_url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt"

    def load_tickers(url):
        try:
            df = pd.read_csv(url, sep="|")
            df = df[df.columns[0:-1]]
            return df
        except Exception as e:
            print(f"ティッカー取得失敗: {url} -> {e}")
            return pd.DataFrame()

    nasdaq_df = load_tickers(nasdaq_url)
    other_df = load_tickers(other_url)

    all_symbols = pd.concat(
        [
            nasdaq_df[["Symbol"]],
            other_df[["ACT Symbol"]].rename(columns={"ACT Symbol": "Symbol"}),
        ],
        ignore_index=True,
    )

    all_symbols = all_symbols.dropna().drop_duplicates().reset_index(drop=True)
    symbols_list = all_symbols["Symbol"].astype(str).str.upper().tolist()

    # ブラックリスト除外
    failed_symbols = load_failed_symbols()
    symbols_filtered = [s for s in symbols_list if s not in failed_symbols]

    print(f"ブラックリスト除外: {len(symbols_list) - len(symbols_filtered)}件")
    return symbols_filtered


def update_ticker_list(output_path: str | Path | None = None) -> Path:
    """Fetch latest ticker list and save to cache dir.

    Parameters
    ----------
    output_path: Optional path to save CSV. Defaults to
        ``settings.data.cache_dir / 'tickers.csv'``.

    Returns
    -------
    Path
        Path to the saved ticker CSV.
    """
    from config.settings import get_settings
    from tools.notify_signals import _post_webhook

    settings = get_settings(create_dirs=True)
    out = Path(output_path) if output_path else Path(settings.data.cache_dir) / "tickers.csv"

    tickers = get_all_tickers()
    prev: set[str] = set()
    if out.exists():
        try:
            prev = set(pd.read_csv(out)["Symbol"].astype(str).str.upper())
        except Exception:
            prev = set()
    df = pd.DataFrame({"Symbol": tickers})
    df.to_csv(out, index=False)

    new = sorted(set(tickers) - prev)
    removed = sorted(prev - set(tickers))
    text = f"Ticker update: {len(tickers)} symbols ( +{len(new)} / -{len(removed)} )"
    if new:
        text += f"\nAdded: {', '.join(new[:10])}"
    if removed:
        text += f"\nRemoved: {', '.join(removed[:10])}"
    for env in ("TEAMS_WEBHOOK_URL", "SLACK_WEBHOOK_URL", "DISCORD_WEBHOOK_URL"):
        url = os.getenv(env)
        if url:
            _post_webhook(url, text)

    return out


# System1用のフィルター関数（例）
def filter_symbols_by_system1(data_dict):
    result = {}
    total = len(data_dict)
    debug_mode = st.session_state.get("system1_debug", False)

    start_time = time.time()
    last_log_time = start_time

    for i, (symbol, df) in enumerate(data_dict.items(), 1):
        if df is None or df.empty or len(df) < 50:
            if debug_mode:
                st.write(f"{symbol}: データ不足で除外")
            continue
        df = df.dropna()
        if df.empty:
            if debug_mode:
                st.write(f"{symbol}: dropna後にデータなし")
            continue
        try:
            latest = df.iloc[-1]
        except IndexError:
            if debug_mode:
                st.write(f"{symbol}: df.dropna()後にデータなし")
            continue

        # 各条件チェックを表示
        close_ok = latest["Close"] > 5
        volume_ok = latest.get("DollarVolume20", 0) > 50_000_000
        trend_ok = latest.get("SMA25", 0) > latest.get("SMA50", 0)
        if debug_mode:
            close_val = float(latest.get("Close", float("nan")))
            dollar_vol = float(latest.get("DollarVolume20", 0))
            sma25 = float(latest.get("SMA25", 0))
            sma50 = float(latest.get("SMA50", 0))
            msg = (
                f"{symbol}: Close={close_val:.2f} ({close_ok}), "
                f"Volume={dollar_vol:.0f} ({volume_ok}), "
                f"Trend={sma25:.2f}>{sma50:.2f} ({trend_ok})"
            )
            st.write(msg)

        if close_ok and volume_ok and trend_ok:
            result[symbol] = df

        # 全体進捗を定期的に出力
        current_time = time.time()
        if current_time - last_log_time > 3:  # 3秒おきに更新
            elapsed = current_time - start_time
            avg_time = elapsed / i
            remaining = avg_time * (total - i)
            mins, secs = divmod(remaining, 60)
            st.write(
                f"進捗: {i}/{total} | 経過時間: {elapsed:.1f}秒 | 推定残り: {int(mins)}分{int(secs)}秒"
            )
            last_log_time = current_time

    total_elapsed = time.time() - start_time
    st.write(f"✅ フィルター処理完了：{total}件中 {len(result)}件通過 | 総処理時間: {total_elapsed:.1f}秒")

    return result


# テスト実行用
if __name__ == "__main__":
    tickers = get_all_tickers()
    print(f"取得ティッカー数: {len(tickers)}")
    print(tickers[:10])
