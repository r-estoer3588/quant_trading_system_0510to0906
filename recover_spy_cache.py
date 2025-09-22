"""SPY の日足データを取得しローカルキャッシュに保存するスクリプト."""

# ruff: noqa: I001
import argparse
import os
import sys

from dotenv import load_dotenv
import pandas as pd
import requests

import common  # noqa: F401

from indicators_common import add_indicators
from common.cache_manager import compute_base_indicators, round_dataframe
from config.settings import get_settings

# .envからAPIキー取得
load_dotenv()
API_KEY = os.getenv("EODHD_API_KEY")


def resolve_cache_dir() -> str:
    """キャッシュ保存先を解決する。
    優先順位:
        1. 環境変数(QUANT_CACHE_DIR/CACHE_DIR/DATA_CACHE_DIR)
        2. common.cache_manager.CACHE_DIR
        3. このリポジトリ配下の `data_cache`
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

        # 正規化: 日付列を作り、指標計算に適した列名にする
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

        # full_backup 用: 主要インジケーターを事前計算して保存
        try:
            full_enriched = add_indicators(df.copy())
        except Exception:
            full_enriched = df.copy()

        # 小数丸め設定を取得
        try:
            settings = get_settings(create_dirs=True)
            round_dec = getattr(settings.cache, "round_decimals", None)
        except Exception:
            round_dec = None

        # フルは target_dirs に保存（lowercase date カラムにしておく）
        full_reset = full_enriched.reset_index()
        # add_indicators may produce 'Date' column or keep 'date'
        if "Date" in full_reset.columns and "date" not in full_reset.columns:
            full_reset = full_reset.rename(columns={"Date": "date"})
        full_reset = full_reset.rename(columns={c: str(c).lower() for c in full_reset.columns})
        try:
            full_reset = round_dataframe(full_reset, round_dec)
        except Exception:
            # fall back: numeric-only rounding if round_dec is an int
            try:
                if isinstance(round_dec, int):
                    numeric_cols = full_reset.select_dtypes(include=["number"]).columns
                    full_reset[numeric_cols] = full_reset[numeric_cols].round(round_dec)
            except Exception:
                pass

        for d in target_dirs:
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"{symbol}.csv")
            try:
                full_reset.to_csv(path, index=False)
                print(f"✅ SPY full_backup を保存しました: {path}")
            except Exception as e:
                print(f"❌ full 保存失敗: {e}", file=sys.stderr)

        # base キャッシュ（主要指標のうち base に相当する列）を作成して base に保存
        try:
            base_df = compute_base_indicators(full_enriched)
        except Exception:
            base_df = None

        if base_df is not None and not base_df.empty:
            base_reset = base_df.reset_index()
            try:
                base_reset = round_dataframe(base_reset, round_dec)
            except Exception:
                try:
                    if isinstance(round_dec, int):
                        numeric_cols = base_reset.select_dtypes(include=["number"]).columns
                        base_reset[numeric_cols] = base_reset[numeric_cols].round(round_dec)
                except Exception:
                    pass
            # base は full_backup の親ディレクトリにある 'base' サブディレクトリへ保存
            for d in target_dirs:
                root = os.path.dirname(os.path.normpath(d))
                base_dir = os.path.join(root, "base")
                os.makedirs(base_dir, exist_ok=True)
                base_path = os.path.join(base_dir, f"{symbol}.csv")
                try:
                    base_reset.to_csv(base_path, index=False)
                    print(f"✅ SPY base を保存しました: {base_path}")
                except Exception as e:
                    print(f"❌ base 保存失敗: {e}", file=sys.stderr)

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
