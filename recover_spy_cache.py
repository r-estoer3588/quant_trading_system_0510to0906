"""SPY の日足データを取得しローカルキャッシュに保存するスクリプト."""

# ruff: noqa: I001
import os
import sys

from dotenv import load_dotenv
import pandas as pd
import requests

import common  # noqa: F401

# .envからAPIキー取得
load_dotenv()
API_KEY = os.getenv("EODHD_API_KEY")


def fetch_and_cache_spy_from_eodhd(folder="data_cache"):
    symbol = "SPY"
    url = f"https://eodhistoricaldata.com/api/eod/{symbol}.US?api_token={API_KEY}&period=d&fmt=json"
    path = os.path.join(folder, f"{symbol}.csv")

    try:
        print(f"[INFO] Fetching URL: {url}")
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            msg = "❌ データが空です。APIキーまたはリクエスト制限を確認してください。"
            print(msg, file=sys.stderr)
            return

        df = pd.DataFrame(data)
        print(f"[INFO] 取得件数: {len(df)}")

        df["date"] = pd.to_datetime(df["date"])
        df = df.rename(
            columns={
                "date": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "adjusted_close": "AdjClose",
                "volume": "Volume",
            }
        )

        os.makedirs(folder, exist_ok=True)
        df.to_csv(path, index=False)

        print(f"✅ SPY.csv を保存しました: {path}")

    except Exception as e:
        msg = f"❌ 例外が発生しました: {e}"
        print(msg, file=sys.stderr)


if __name__ == "__main__":
    fetch_and_cache_spy_from_eodhd()
