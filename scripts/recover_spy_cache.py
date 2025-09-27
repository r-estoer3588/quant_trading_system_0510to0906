"""SPY の日足データを取得し CacheManager を使って保存するスクリプト."""

# ruff: noqa: I001
import argparse
import sys

from dotenv import load_dotenv
import pandas as pd
import requests

import common  # noqa: F401

from common.cache_manager import CacheManager
from config.settings import get_settings

# .envからAPIキー取得
load_dotenv()


def fetch_and_cache_spy_from_eodhd(folder=None, group=None):
    """EODHD API から SPY データを取得し CacheManager で保存する。"""
    symbol = "SPY"

    # 設定とCacheManager初期化
    try:
        settings = get_settings(create_dirs=True)
        cache_manager = CacheManager(settings)
        api_key = settings.EODHD_API_KEY
        if not api_key:
            print("❌ EODHD_API_KEY が設定されていません", file=sys.stderr)
            return
    except Exception as e:
        print(f"❌ 設定またはCacheManager初期化に失敗: {e}", file=sys.stderr)
        return

    # API呼び出し用に小文字変換
    api_symbol = symbol.lower()
    url = f"https://eodhistoricaldata.com/api/eod/{api_symbol}.US?api_token={api_key}&period=d&fmt=json"

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

        # 正規化: 日付列を作り、標準的な列名にする
        df["date"] = pd.to_datetime(df["date"])

        # 列名の正規化（CacheManagerが期待する形式）
        rename_map = {
            "date": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "volume": "volume",
        }

        # adjusted_close を close として扱う（優先）
        if "adjusted_close" in df.columns:
            rename_map["adjusted_close"] = "close"
            if "close" in df.columns:
                # 元の close は raw_close として保持
                df["raw_close"] = df["close"]
        elif "close" in df.columns:
            rename_map["close"] = "close"

        df = df.rename(columns=rename_map)

        # 必須列の確認
        required_cols = {"date", "open", "high", "low", "close", "volume"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            print(f"❌ 必須列が不足: {missing}", file=sys.stderr)
            return

        # 日付でソート
        df = df.sort_values("date").reset_index(drop=True)

        # CacheManagerのupsert_bothを使用して保存
        # これにより full_backup, base, rolling すべてに適切な指標付きで保存される
        try:
            cache_manager.upsert_both(symbol, df)
            print("✅ SPY データを CacheManager 経由で保存しました")
            print("   - full_backup: 全指標付きデータ")
            print("   - base: ベース指標付きデータ")
            print("   - rolling: 直近300+30日のデータ")
        except Exception as e:
            print(f"❌ CacheManager による保存に失敗: {e}", file=sys.stderr)

    except Exception as e:
        msg = f"❌ 例外が発生しました: {e}"
        print(msg, file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SPY の日足を取得し CacheManager でキャッシュへ保存"
    )
    parser.add_argument(
        "--out", dest="out", default=None, help="非推奨: CacheManager が設定から自動決定します"
    )
    parser.add_argument(
        "--group",
        choices=["full_backup", "base", "rolling"],
        default=None,
        help="非推奨: CacheManager が自動的に全グループに保存します",
    )
    args = parser.parse_args()

    if args.out or args.group:
        print("⚠️  --out と --group オプションは CacheManager により自動処理されるため無視されます")

    fetch_and_cache_spy_from_eodhd()
