from __future__ import annotations

import csv
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

# 親ディレクトリ（リポジトリ ルート）を import パスに追加して、直下モジュール `indicators_common.py` を解決可能にする
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from indicators_common import add_indicators  # noqa: E402


# -----------------------------
# 設定/環境
# -----------------------------

# .env から API キー等を取り込む（プロジェクトルートの .env）
load_dotenv(dotenv_path=r".env")

try:
    from config.settings import get_settings

    _settings = get_settings(create_dirs=True)
    LOG_DIR = Path(_settings.LOGS_DIR)
    DATA_CACHE_DIR = Path(_settings.DATA_CACHE_DIR)
    THREADS_DEFAULT = int(_settings.THREADS_DEFAULT)
    REQUEST_TIMEOUT = int(_settings.REQUEST_TIMEOUT)
    DOWNLOAD_RETRIES = int(_settings.DOWNLOAD_RETRIES)
    API_THROTTLE_SECONDS = float(_settings.API_THROTTLE_SECONDS)
    API_BASE = str(_settings.API_EODHD_BASE).rstrip("/")
    API_KEY = _settings.EODHD_API_KEY or os.getenv("EODHD_API_KEY", "")
except Exception:
    # フォールバック（settings が読めない場合）
    LOG_DIR = Path(os.path.dirname(__file__)) / "logs"
    DATA_CACHE_DIR = Path(os.path.dirname(__file__)) / ".." / "data_cache"
    THREADS_DEFAULT = int(os.getenv("THREADS_DEFAULT", 8))
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 10))
    DOWNLOAD_RETRIES = int(os.getenv("DOWNLOAD_RETRIES", 3))
    API_THROTTLE_SECONDS = float(os.getenv("API_THROTTLE_SECONDS", 1.5))
    API_BASE = os.getenv("API_EODHD_BASE", "https://eodhistoricaldata.com").rstrip("/")
    API_KEY = os.getenv("EODHD_API_KEY", "")

LOG_DIR.mkdir(parents=True, exist_ok=True)
DATA_CACHE_DIR = DATA_CACHE_DIR.resolve()
DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# ロギング
# -----------------------------

logging.basicConfig(
    filename=str(LOG_DIR / "cache_log.txt"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# -----------------------------
# ブラックリスト（クールダウン: 月単位）
# -----------------------------

FAILED_LIST_PATH = LOG_DIR / "eodhd_failed_symbols.csv"
LEGACY_FAILED_LIST = Path(__file__).resolve().parents[1] / "eodhd_failed_symbols.csv"


@dataclass
class FailedEntry:
    symbol: str
    last_failed_at: datetime  # 失敗日
    count: int = 1


def _parse_date(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s)
    except Exception:
        try:
            return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return datetime.utcnow()


def _migrate_legacy_failed_if_needed() -> None:
    """リポジトリ直下の旧 CSV（シンボルのみ）を logs/ に移行する。
    旧形式: 1列（symbol）
    新形式: 3列（symbol,last_failed_at,count）
    """
    if LEGACY_FAILED_LIST.exists() and not FAILED_LIST_PATH.exists():
        symbols = []
        try:
            with open(LEGACY_FAILED_LIST, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        symbols.append(s.upper())
        except Exception:
            pass

        now = datetime.utcnow().isoformat()
        FAILED_LIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(FAILED_LIST_PATH, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["symbol", "last_failed_at", "count"])  # header
                for s in sorted(set(symbols)):
                    writer.writerow([s, now, 1])
        except Exception:
            pass


def _load_failed_map() -> Dict[str, FailedEntry]:
    """CSV から失敗情報を読み込む。"""
    _migrate_legacy_failed_if_needed()
    entries: Dict[str, FailedEntry] = {}
    if not FAILED_LIST_PATH.exists():
        return entries

    try:
        df = pd.read_csv(FAILED_LIST_PATH)
        # 新形式（ヘッダあり）
        if set(df.columns.str.lower()) >= {"symbol", "last_failed_at"}:
            for _, row in df.iterrows():
                sym = str(row["symbol"]).upper().strip()
                if not sym:
                    continue
                last_dt = _parse_date(str(row["last_failed_at"]))
                cnt = int(row.get("count", 1) or 1)
                entries[sym] = FailedEntry(sym, last_dt, cnt)
            return entries
        # 旧形式（1列のみ）
        else:
            now = datetime.utcnow()
            for s in df.iloc[:, 0].astype(str).str.upper():
                s = s.strip()
                if s:
                    entries[s] = FailedEntry(s, now, 1)
            return entries
    except Exception:
        # CSV が壊れている等の場合は空扱い
        return {}


def _save_failed_map(entries: Dict[str, FailedEntry]) -> None:
    FAILED_LIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for e in entries.values():
        rows.append([e.symbol, e.last_failed_at.isoformat(), int(e.count)])
    rows.sort(key=lambda r: r[0])
    with open(FAILED_LIST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["symbol", "last_failed_at", "count"])  # header
        writer.writerows(rows)


def load_monthly_blacklist() -> Set[str]:
    """当月に失敗した銘柄を集合で返す（同一月はスキップ）。"""
    m = _load_failed_map()
    now = datetime.utcnow()
    skip: Set[str] = set()
    for sym, e in m.items():
        if e.last_failed_at.year == now.year and e.last_failed_at.month == now.month:
            skip.add(sym)
    return skip


def update_failed_symbols(failed: Iterable[str]) -> None:
    """失敗銘柄を更新（当月の失敗日時を上書き、回数をインクリメント）。"""
    failed_set = {str(s).upper().strip() for s in failed if str(s).strip()}
    if not failed_set:
        return
    m = _load_failed_map()
    now = datetime.utcnow()
    for s in failed_set:
        if s in m:
            e = m[s]
            e.last_failed_at = now
            e.count = int(e.count) + 1
        else:
            m[s] = FailedEntry(s, now, 1)
    _save_failed_map(m)


def remove_recovered_symbols(succeeded: Iterable[str]) -> None:
    """成功した銘柄はブラックリストから削除。"""
    suc_set = {str(s).upper().strip() for s in succeeded if str(s).strip()}
    if not suc_set:
        return
    m = _load_failed_map()
    changed = False
    for s in list(suc_set):
        if s in m:
            del m[s]
            changed = True
    if changed:
        _save_failed_map(m)


# -----------------------------
# データ取得
# -----------------------------

def get_all_symbols() -> List[str]:
    urls = [
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    ]
    symbols: Set[str] = set()
    for url in urls:
        try:
            r = requests.get(url, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            lines = r.text.splitlines()
            for line in lines[1:]:
                if "|" in line:
                    parts = line.split("|")
                    if parts[0].isalpha():
                        symbols.add(parts[0].upper())
        except Exception as e:
            logging.error(f"取得失敗: {url} - {e}")
    return sorted(symbols)


def get_with_retry(url: str, retries: int = DOWNLOAD_RETRIES, delay: float = 2.0):
    for i in range(max(1, retries)):
        try:
            r = requests.get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                return r
            logging.warning(f"ステータスコード {r.status_code} - {url}")
        except Exception as e:
            logging.warning(f"試行{i+1}回目のエラー: {e}")
        time.sleep(delay)
    return None


def get_eodhd_data(symbol: str) -> pd.DataFrame | None:
    url = f"{API_BASE}/api/eod/{symbol}.US?api_token={API_KEY}&period=d&fmt=json"
    r = get_with_retry(url)
    if r is None:
        return None
    try:
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            logging.warning(f"{symbol}: 空または無効なJSON応答")
            return None
        df = pd.DataFrame(data)
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
        df.set_index("Date", inplace=True)
        df = df.sort_index()
        return df
    except Exception as e:
        logging.error(f"{symbol}: データ整形中のエラー - {e}")
        return None


RESERVED_WORDS = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
}


def safe_filename(symbol: str) -> str:
    # Windows 予約語を避ける（大文字小文字無視）
    if symbol.upper() in RESERVED_WORDS:
        return symbol + "_RESV"
    return symbol


def cache_single(symbol: str, output_dir: Path) -> Tuple[str, bool, bool]:
    """指定シンボルをキャッシュ。
    戻り値: (message, used_api, success)
    """
    safe_symbol = safe_filename(symbol)
    filepath = output_dir / f"{safe_symbol}.csv"
    if filepath.exists():
        mod_time = datetime.fromtimestamp(filepath.stat().st_mtime)
        if mod_time.date() == datetime.today().date():
            return (f"{symbol}: already cached", False, True)
    df = get_eodhd_data(symbol)
    if df is not None and not df.empty:
        df = add_indicators(df)
        df.to_csv(filepath)
        return (f"{symbol}: saved", True, True)
    else:
        return (f"{symbol}: failed to fetch", True, False)


def cache_data(symbols: List[str], output_dir: Path | str = DATA_CACHE_DIR, max_workers: int | None = None) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_workers = int(max_workers or THREADS_DEFAULT)

    # 当月ブラックリストに該当する銘柄をスキップ
    monthly_blacklist = load_monthly_blacklist()
    symbols_to_fetch = [s for s in symbols if s.upper() not in monthly_blacklist]
    skipped_due_to_cooldown = len(symbols) - len(symbols_to_fetch)

    failed: List[str] = []
    succeeded: List[str] = []

    results_list: List[Tuple[str, str, bool]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(cache_single, symbol, output_dir): symbol for symbol in symbols_to_fetch}
        for i, future in enumerate(as_completed(futures)):
            msg, used_api, ok = future.result()
            symbol = futures[future]
            results_list.append((symbol, msg, used_api))
            logging.info(msg)
            print(f"[{i}] {msg}")
            if not ok:
                failed.append(symbol)
            else:
                succeeded.append(symbol)
            if used_api:
                time.sleep(API_THROTTLE_SECONDS)

    # ブラックリスト更新/回復削除
    if failed:
        update_failed_symbols(failed)
    if succeeded:
        remove_recovered_symbols(succeeded)

    # 統計の出力
    cached_count = sum(1 for _, _, used_api in results_list if not used_api)
    api_count = sum(1 for _, _, used_api in results_list if used_api)
    print(
        f"✅ キャッシュ済み: {cached_count}件, API使用: {api_count}件, 失敗: {len(failed)}件, クールダウン除外: {skipped_due_to_cooldown}件"
    )


def _cli_main() -> None:
    # symbols = get_all_symbols()[:3]  # 簡易テスト用
    symbols = get_all_symbols()
    print(f"{len(symbols)}銘柄を取得します（クールダウン月次ブラックリスト適用後に除外）")
    cache_data(symbols, output_dir=DATA_CACHE_DIR)
    print("データのキャッシュが完了しました。")


if __name__ == "__main__":
    _cli_main()

