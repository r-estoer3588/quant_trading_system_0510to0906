"""SPY の日足データを取得しローカルキャッシュに保存するスクリプト."""

# ruff: noqa: I001
import argparse
import os
import sys

from dotenv import load_dotenv
import pandas as pd
import requests

import common  # noqa: F401

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
    """与えられたフォルダから full_backup、base、rolling の保存先パスのみを返す。"""
    norm = os.path.normpath(folder)
    tail = os.path.basename(norm).lower()
    root = os.path.dirname(norm) if tail in ("base", "full", "full_backup", "rolling") else norm
    return [
        os.path.join(root, "full_backup"),
        os.path.join(root, "base"),
        os.path.join(root, "rolling"),
    ]


def fetch_and_cache_spy_from_eodhd(folder=None, group=None):
    symbol = "SPY"
    if folder is None:
        folder = resolve_cache_dir()
    # 新方針: full_backup のみに保存する
    target_dirs = resolve_target_dirs(folder)
    # API呼び出し用に小文字変換
    api_symbol = symbol.lower()
    url = f"https://eodhistoricaldata.com/api/eod/{api_symbol}.US?api_token={API_KEY}&period=d&fmt=json"

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
        rename_map: dict[str, str] = {
            "date": "date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "volume": "Volume",
        }
        if "adjusted_close" in df.columns:
            # 調整後終値を優先して Close として扱う。元の close は補助列として保持。
            rename_map["adjusted_close"] = "Close"
            if "close" in df.columns:
                rename_map["close"] = "CloseRaw"
        elif "close" in df.columns:
            rename_map["close"] = "Close"

        df = df.rename(columns=rename_map)

        # base キャッシュを作成
        try:
            base_df = compute_base_indicators(df)
        except Exception:
            base_df = None

        if base_df is not None and not base_df.empty:
            # rolling を作成
            settings = None
            try:
                settings = get_settings(create_dirs=True)
                rolling_len = (
                    settings.cache.rolling.base_lookback_days + settings.cache.rolling.buffer_days
                )
                rolling_df = base_df.sort_values("date").tail(rolling_len).copy()
            except Exception:
                rolling_df = base_df.copy()

            # 小数丸め設定を取得
            round_dec = getattr(settings.cache, "round_decimals", None) if settings else None

            # 保存
            for d in target_dirs:
                os.makedirs(d, exist_ok=True)
                if "rolling" in d:
                    reset_df = rolling_df.reset_index()
                    data_type = "rolling"
                else:
                    reset_df = base_df.reset_index()
                    data_type = os.path.basename(d)

                if "Date" in reset_df.columns and "date" not in reset_df.columns:
                    reset_df = reset_df.rename(columns={"Date": "date"})
                reset_df = reset_df.rename(columns={c: str(c).lower() for c in reset_df.columns})
                try:
                    reset_df = round_dataframe(reset_df, round_dec)
                except Exception:
                    try:
                        if isinstance(round_dec, int):
                            numeric_cols = reset_df.select_dtypes(include=["number"]).columns
                            reset_df[numeric_cols] = reset_df[numeric_cols].round(round_dec)
                    except Exception:
                        pass

                path = os.path.join(d, f"{symbol}.csv")
                try:
                    reset_df.to_csv(path, index=False)
                    print(f"✅ SPY {data_type} を保存しました: {path}")
                except Exception as e:
                    print(f"❌ {data_type} 保存失敗: {e}", file=sys.stderr)
        else:
            print("❌ base_df が作成できなかったため、保存をスキップします", file=sys.stderr)

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
        choices=["full_backup", "base", "rolling"],
        default=None,
        help="保存グループ(full_backup, base, rolling)。未指定時は full_backup",
    )
    args = parser.parse_args()
    fetch_and_cache_spy_from_eodhd(folder=args.out, group=args.group)
