import os
import pandas as pd
import requests
from dotenv import load_dotenv
from datetime import datetime
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings  # noqa: E402
from common.cache_manager import CacheManager  # noqa: E402

load_dotenv()
API_KEY = os.getenv("EODHD_API_KEY")


def fetch_bulk_last_day():
    url = f"https://eodhistoricaldata.com/api/eod-bulk-last-day/US?api_token={API_KEY}&fmt=json"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        print("Error fetching bulk data:", r.status_code)
        return None
    return pd.DataFrame(r.json())


def append_to_cache(df: pd.DataFrame, cm: CacheManager) -> tuple[int, int]:
    """
    取得した1日分のデータを CacheManager の full/rolling にインクリメンタル反映する。
    戻り値: (対象銘柄数, upsert 成功数)
    """
    if df is None or df.empty:
        return 0, 0
    # 必要列を小文字に統一し、'date' を datetime 化
    cols_map = {
        "code": "code",
        "date": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adjusted_close": "adjusted_close",
        "volume": "volume",
    }
    # 列名大文字・混在対策
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    df = df.rename(columns={k: v for k, v in cols_map.items() if k in df.columns})
    if "date" not in df.columns:
        return 0, 0
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])  # 不正日時を除外
    total = 0
    updated = 0
    # シンボル列（code）が無い場合は更新対象なし
    if "code" not in df.columns:
        return 0, 0
    for sym, g in df.groupby("code"):
        total += 1
        # 1日分（複数行が来てもそのまま渡せる）
        keep_cols = [
            "date",
            "open",
            "high",
            "low",
            "close",
            "adjusted_close",
            "volume",
        ]
        cols_exist = [c for c in keep_cols if c in g.columns]
        if not cols_exist:
            continue
        rows = g[cols_exist].copy()
        try:
            sym_norm = str(sym).upper().strip()
            if sym_norm.endswith(".US"):
                sym_norm = sym_norm.rsplit(".", 1)[0]
            cm.upsert_both(sym_norm, rows)
            updated += 1
        except Exception as e:
            print(f"{sym}: upsert error - {e}")
    # rolling のメンテナンス
    try:
        cm.prune_rolling_if_needed(anchor_ticker="SPY")
    except Exception:
        pass
    return total, updated


def main():
    if not API_KEY:
        print("EODHD_API_KEY が未設定です (.env を確認)")
        return
    data = fetch_bulk_last_day()
    if data is None or data.empty:
        print("No data to update.")
        return

    settings = get_settings(create_dirs=True)
    cm = CacheManager(settings)
    total, updated = append_to_cache(data, cm)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"✅ {now} | 対象: {total} 銘柄 / 更新: {updated} 銘柄（full/rolling へ反映）")


if __name__ == "__main__":
    main()
