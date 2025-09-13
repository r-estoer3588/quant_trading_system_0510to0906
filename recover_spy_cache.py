"""SPY の日足データを取得しローカルキャッシュに保存するスクリプト."""

# ruff: noqa: I001
import os
import sys

from dotenv import load_dotenv
import pandas as pd
import requests
import argparse

import common  # noqa: F401

# .envからAPIキー取得
load_dotenv()
API_KEY = os.getenv("EODHD_API_KEY")


def resolve_cache_dir() -> str:
    """キャッシュ保存先を解決する。
    優先順位: 環境変数(QUANT_CACHE_DIR/CACHE_DIR/DATA_CACHE_DIR) > common.cache_manager.CACHE_DIR > このリポジトリ配下のdata_cache
    """
    for key in ("QUANT_CACHE_DIR", "CACHE_DIR", "DATA_CACHE_DIR"):
        v = os.getenv(key)
        if v:
            return v
    try:
        from common.cache_manager import CACHE_DIR as CM_CACHE_DIR  # type: ignore

        return CM_CACHE_DIR
    except Exception:
        pass
    return os.path.join(os.path.dirname(__file__), "data_cache")


def resolve_cache_group(root: str) -> str:
    """保存グループを解決する（新方針: full_backup のみ）。"""
    return "full_backup"


def append_group(folder: str, group: str) -> str:
    tail = os.path.basename(os.path.normpath(folder)).lower()
    if tail in ("base", "full", "full_backup"):
        if tail == "full_backup":
            return folder
        # base/full の場合は full_backup に付け替える
        return os.path.join(os.path.dirname(folder), "full_backup")
    return os.path.join(folder, "full_backup")


def resolve_target_dirs(folder: str) -> list[str]:
    """与えられたフォルダから full_backup の保存先パスのみを返す。"""
    norm = os.path.normpath(folder)
    tail = os.path.basename(norm).lower()
    root = os.path.dirname(norm) if tail in ("base", "full", "full_backup") else norm
    return [os.path.join(root, "full_backup")]


def fetch_and_cache_spy_from_eodhd(folder=None, group=None):
    symbol = "SPY"
    if folder is None:
        folder = resolve_cache_dir()
    # 新方針: full_backup のみに保存する
    target_dirs = resolve_target_dirs(folder)
    url = f"https://eodhistoricaldata.com/api/eod/{symbol}.US?api_token={API_KEY}&period=d&fmt=json"

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

        df["date"] = pd.to_datetime(df["date"])  # 小文字のままにする
        df = df.rename(
            columns={
                "date": "date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "adjusted_close": "AdjClose",
                "volume": "Volume",
            }
        )

        for d in target_dirs:
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"{symbol}.csv")
            df.to_csv(path, index=False)
            print(f"✅ SPY.csv を保存しました: {path}")

    except Exception as e:
        msg = f"❌ 例外が発生しました: {e}"
        print(msg, file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPY の日足を取得しキャッシュへ保存")
    parser.add_argument(
        "--out", dest="out", default=None, help="保存ルートディレクトリ(未指定時は自動解決)"
    )
    parser.add_argument(
        "--group",
        choices=["full_backup"],
        default=None,
        help="保存グループ(full_backup)。未指定時は full_backup",
    )
    args = parser.parse_args()
    fetch_and_cache_spy_from_eodhd(folder=args.out, group=args.group)
