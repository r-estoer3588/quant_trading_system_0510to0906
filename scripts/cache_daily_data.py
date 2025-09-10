from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import shutil
import sys
import time

from dotenv import load_dotenv
import pandas as pd
import requests


def _migrate_root_csv_to_full() -> None:
    """レガシーな CSV キャッシュを ``CacheManager`` の構成へ移行する。

    旧バージョンでは ``data_cache/`` や ``data_cache_recent/`` 直下に
    シンボルごとの CSV を配置していた。現在は ``CacheManager`` により
    ``data_cache/full/`` と ``data_cache/rolling/`` に整理されているため、
    既存ファイルがあればこの関数で移動する。移行に失敗してもログを
    出力するのみで処理を継続する。
    """

    global DATA_CACHE_DIR, DATA_CACHE_RECENT_DIR

    try:
        full_dir = cm.full_dir
        rolling_dir = cm.rolling_dir
    except Exception:  # pragma: no cover - セットアップ失敗時は移行不要
        return

    def _move_csv(src_dir: Path, dest_dir: Path) -> Path:
        if src_dir == dest_dir:
            return dest_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        for src in src_dir.glob("*.csv"):
            dest = dest_dir / src.name
            if dest.exists():
                continue
            try:
                src.rename(dest)
            except Exception:  # pragma: no cover - Windows などで rename 失敗
                try:
                    shutil.move(str(src), str(dest))
                except Exception as e:  # pragma: no cover - logging only
                    logging.warning("移行失敗: %s -> %s (%s)", src, dest, e)
        return dest_dir

    DATA_CACHE_DIR = _move_csv(DATA_CACHE_DIR, full_dir)
    DATA_CACHE_RECENT_DIR = _move_csv(DATA_CACHE_RECENT_DIR, rolling_dir)


# 親ディレクトリ（リポジトリ ルート）を import パスに追加して、
# 直下モジュール `indicators_common.py` を解決可能にする
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from indicators_common import add_indicators  # noqa: E402

from common.cache_manager import CacheManager  # noqa: E402

# -----------------------------
# 設定/環境
# -----------------------------

# .env から API キー等を取り込む（プロジェクトルートの .env）
load_dotenv(dotenv_path=r".env")

try:
    from config.settings import get_settings

    _settings = get_settings(create_dirs=True)
    cm = CacheManager(_settings)
    LOG_DIR = Path(_settings.LOGS_DIR)
    DATA_CACHE_DIR = Path(_settings.DATA_CACHE_DIR)
    DATA_CACHE_RECENT_DIR = Path(_settings.DATA_CACHE_RECENT_DIR)
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
    DATA_CACHE_RECENT_DIR = Path(os.path.dirname(__file__)) / ".." / "data_cache_recent"
    THREADS_DEFAULT = int(os.getenv("THREADS_DEFAULT", 8))
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 10))
    DOWNLOAD_RETRIES = int(os.getenv("DOWNLOAD_RETRIES", 3))
    API_THROTTLE_SECONDS = float(os.getenv("API_THROTTLE_SECONDS", 1.5))
    API_BASE = os.getenv("API_EODHD_BASE", "https://eodhistoricaldata.com").rstrip("/")
    API_KEY = os.getenv("EODHD_API_KEY", "")

LOG_DIR.mkdir(parents=True, exist_ok=True)
DATA_CACHE_DIR = DATA_CACHE_DIR.resolve()
DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
DATA_CACHE_RECENT_DIR = DATA_CACHE_RECENT_DIR.resolve()
DATA_CACHE_RECENT_DIR.mkdir(parents=True, exist_ok=True)

RECENT_DAYS = 240


# -----------------------------
# ロギング
# -----------------------------

logging.basicConfig(
    filename=str(LOG_DIR / "cache_log.txt"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

if os.getenv("SKIP_CACHE_MIGRATION") != "1":
    _migrate_root_csv_to_full()


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
            return datetime.now(timezone.utc)


def _migrate_legacy_failed_if_needed() -> None:
    """リポジトリ直下の旧 CSV（シンボルのみ）を logs/ に移行する。
    旧形式: 1列（symbol）
    新形式: 3列（symbol,last_failed_at,count）
    """
    symbols = []
    if LEGACY_FAILED_LIST.exists() and not FAILED_LIST_PATH.exists():
        try:
            with open(LEGACY_FAILED_LIST, encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        symbols.append(s.upper())
        except Exception:
            pass

    now = datetime.now(timezone.utc).isoformat()
    FAILED_LIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(FAILED_LIST_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["symbol", "last_failed_at", "count"])  # header
            for s in sorted(set(symbols)):
                writer.writerow([s, now, 1])
    except Exception:
        pass


def _load_failed_map() -> dict[str, FailedEntry]:
    """CSV から失敗情報を読み込む。"""
    _migrate_legacy_failed_if_needed()
    entries: dict[str, FailedEntry] = {}
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
            now = datetime.now(timezone.utc)
            for s in df.iloc[:, 0].astype(str).str.upper():
                s = s.strip()
                if s:
                    entries[s] = FailedEntry(s, now, 1)
            return entries
    except Exception:
        # CSV が壊れている等の場合は空扱い
        return {}


def _save_failed_map(entries: dict[str, FailedEntry]) -> None:
    FAILED_LIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for e in entries.values():
        rows.append([e.symbol, e.last_failed_at.isoformat(), int(e.count)])
    rows.sort(key=lambda r: r[0])
    with open(FAILED_LIST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["symbol", "last_failed_at", "count"])  # header
        writer.writerows(rows)


def load_monthly_blacklist() -> set[str]:
    """当月に失敗した銘柄を集合で返す（同一月はスキップ）。"""
    m = _load_failed_map()
    now = datetime.now(timezone.utc)
    skip: set[str] = set()
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
    now = datetime.now(timezone.utc)
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


def get_all_symbols() -> list[str]:
    urls = [
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    ]
    symbols: set[str] = set()
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


def cache_single(
    symbol: str,
    output_dir: Path,
    recent_dir: Path | None = None,
    recent_days: int = RECENT_DAYS,
) -> tuple[str, bool, bool]:
    """指定シンボルをキャッシュ。
    戻り値: (message, used_api, success)
    """
    safe_symbol = safe_filename(symbol)
    filepath = output_dir / f"{safe_symbol}.csv"
    recentpath = recent_dir / f"{safe_symbol}.csv" if recent_dir else None

    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    if recent_dir is not None and not isinstance(recent_dir, Path):
        recent_dir = Path(recent_dir)
    if filepath.exists():
        mod_time = datetime.fromtimestamp(filepath.stat().st_mtime)
        if mod_time.date() == datetime.today().date():
            if recentpath:
                if recent_dir is not None:
                    recent_dir.mkdir(parents=True, exist_ok=True)
                if not recentpath.exists():
                    try:
                        existing_df = pd.read_csv(filepath)
                        existing_df.tail(recent_days).to_csv(recentpath, index=False)
                    except Exception as e:  # pragma: no cover - logging only
                        logging.warning(
                            "%s: failed to write recent cache from existing data - %s",
                            symbol,
                            e,
                        )
            return (f"{symbol}: already cached", False, True)
    df = get_eodhd_data(symbol)
    if df is not None and not df.empty:
        df = add_indicators(df)
        df_reset = df.reset_index().rename(columns=str.lower)
        df_reset.to_csv(filepath, index=False)
        if recentpath:
            if recent_dir is not None:
                recent_dir.mkdir(parents=True, exist_ok=True)
            df_reset.tail(recent_days).to_csv(recentpath, index=False)
        return (f"{symbol}: saved", True, True)
    else:
        return (f"{symbol}: failed to fetch", True, False)


def cache_data(
    symbols: list[str],
    output_dir: Path | str = DATA_CACHE_DIR,
    recent_dir: Path | None = DATA_CACHE_RECENT_DIR,
    recent_days: int = RECENT_DAYS,
    max_workers: int | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if recent_dir is not None:
        recent_dir = Path(recent_dir)
        recent_dir.mkdir(parents=True, exist_ok=True)

    max_workers = int(max_workers or THREADS_DEFAULT)

    # 当月ブラックリストに該当する銘柄をスキップ
    monthly_blacklist = load_monthly_blacklist()
    symbols_to_fetch = [s for s in symbols if s.upper() not in monthly_blacklist]
    skipped_due_to_cooldown = len(symbols) - len(symbols_to_fetch)

    failed: list[str] = []
    succeeded: list[str] = []

    results_list: list[tuple[str, str, bool]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                cache_single,
                symbol,
                output_dir,
                recent_dir,
                recent_days,
            ): symbol
            for symbol in symbols_to_fetch
        }
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

    cm.prune_rolling_if_needed(anchor_ticker="SPY")

    # 統計の出力
    cached_count = sum(1 for _, _, used_api in results_list if not used_api)
    api_count = sum(1 for _, _, used_api in results_list if used_api)
    print(
        f"✅ キャッシュ済み: {cached_count}件, API使用: {api_count}件, "
        f"失敗: {len(failed)}件, クールダウン除外: {skipped_due_to_cooldown}件"
    )


def _cli_main() -> None:
    # symbols = get_all_symbols()[:3]  # 簡易テスト用
    symbols = get_all_symbols()
    print(f"{len(symbols)}銘柄を取得します（クールダウン月次ブラックリスト適用後に除外）")
    cache_data(symbols, output_dir=DATA_CACHE_DIR, recent_dir=DATA_CACHE_RECENT_DIR)
    print("データのキャッシュが完了しました。")


if __name__ == "__main__":
    _cli_main()
