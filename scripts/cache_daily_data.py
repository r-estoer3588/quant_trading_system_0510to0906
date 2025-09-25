from __future__ import annotations

import argparse
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import queue
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import shutil
import sys
import time
import threading
from typing import TYPE_CHECKING, Literal

from dotenv import load_dotenv
import pandas as pd
import requests
from requests.adapters import HTTPAdapter

if TYPE_CHECKING:
    pass


def _migrate_root_csv_to_full() -> None:
    """レガシーな CSV キャッシュを ``CacheManager`` の構成へ移行する。

    旧バージョンでは ``data_cache/`` や ``data_cache_recent/`` 直下に
    シンボルごとの CSV を配置していた。現在は ``CacheManager`` により
    ``data_cache/full_backup/`` と ``data_cache/base/`` に整理されているため、
    既存ファイルがあればこの関数で移動する。移行に失敗してもログを
    出力するのみで処理を継続する。
    """

    global DATA_CACHE_DIR, BASE_CACHE_DIR

    try:
        full_dir = cm.full_dir
        base_dir = BASE_CACHE_DIR
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
    if LEGACY_RECENT_DIR is not None:
        BASE_CACHE_DIR = _move_csv(LEGACY_RECENT_DIR, base_dir)


# 親ディレクトリ（リポジトリ ルート）を import パスに追加して、
# 直下モジュール `indicators_common.py` を解決可能にする
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from indicators_common import add_indicators  # noqa: E402

from common.cache_manager import CacheManager, compute_base_indicators  # noqa: E402

try:
    from common.cache_manager import round_dataframe  # type: ignore # noqa: E402
except ImportError:  # pragma: no cover - tests may stub cache_manager

    def round_dataframe(df: pd.DataFrame, decimals: int | None) -> pd.DataFrame:
        if decimals is None:
            return df
        try:
            decimals_int = int(decimals)
        except Exception:
            return df
        try:
            return df.copy().round(decimals_int)
        except Exception:
            try:
                return df.round(decimals_int)
            except Exception:
                return df


from common.symbol_universe import build_symbol_universe  # noqa: E402
from common.symbols_manifest import save_symbol_manifest  # noqa: E402

CacheUpdateInterrupted: type[BaseException] | None
try:  # Local import guard for optional bulk updater
    from scripts.update_from_bulk_last_day import (
        run_bulk_update,
        CacheUpdateInterrupted as _CacheUpdateInterrupted,
    )
except Exception:  # pragma: no cover - unavailable in constrained envs
    run_bulk_update = None
    CacheUpdateInterrupted = None
else:
    CacheUpdateInterrupted = _CacheUpdateInterrupted


def _attempt_bulk_refresh(symbols: list[str] | None, progress_interval: int = 500):
    """Try to run the optional bulk updater if available.

    Returns whatever the bulk updater returns, or None if unavailable or on
    error. This mirrors the previous behavior expected by callers.
    """
    if run_bulk_update is None:
        return None
    try:
        # 進捗表示用のコールバック
        def progress_callback(processed: int, total: int, updated: int) -> None:
            interval = max(1, int(progress_interval or 500))
            if processed % interval == 0 or processed == total:
                print(
                    f"📊 Bulk進捗: {processed}/{total} 銘柄処理済み (更新: {updated})",
                    flush=True,
                )

        # run_bulk_update expects a CacheManager instance as first arg
        # and accepts `universe=` for filtering by symbols
        print(
            f"🚀 Bulk更新を開始します: 対象={len(symbols) if symbols is not None else '全銘柄'} "
            f"(進捗表示間隔={progress_interval}件)",
            flush=True,
        )
        return run_bulk_update(
            cm,
            universe=symbols,
            fetch_universe=False,
            progress_callback=progress_callback,
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        if CacheUpdateInterrupted is not None and isinstance(exc, CacheUpdateInterrupted):
            raise
        return None


def _report_bulk_interrupt(exc: BaseException, total_symbols: int) -> None:
    """ユーザーによる Bulk 更新の中断状況を標準出力へ記録する。"""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    processed = 0
    updated = 0
    total_for_report = max(total_symbols, 0)
    if CacheUpdateInterrupted is not None and isinstance(exc, CacheUpdateInterrupted):
        processed = getattr(exc, "processed", 0)
        updated = getattr(exc, "updated", 0)
        total_for_report = max(total_for_report, processed)

    print("🛑 Bulk 更新がユーザーにより中断されました。", flush=True)
    summary = (
        f"   ↳ {timestamp} 時点 | 処理済み: {processed}/{total_for_report} 銘柄 / "
        f"更新済み: {updated} 銘柄"
    )
    print(summary, flush=True)


BASE_SUBDIR_NAME = "base"

CACHE_ROUND_DECIMALS: int | None = None

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
    LEGACY_RECENT_DIR = Path(_settings.DATA_CACHE_RECENT_DIR)
    BASE_CACHE_DIR = Path(_settings.DATA_CACHE_DIR) / BASE_SUBDIR_NAME
    CACHE_ROUND_DECIMALS = getattr(_settings.cache, "round_decimals", None)
    THREADS_DEFAULT = int(_settings.THREADS_DEFAULT)
    REQUEST_TIMEOUT = int(_settings.REQUEST_TIMEOUT)
    DOWNLOAD_RETRIES = int(_settings.DOWNLOAD_RETRIES)
    API_THROTTLE_SECONDS = float(_settings.API_THROTTLE_SECONDS)
    API_BASE = str(_settings.API_EODHD_BASE).rstrip("/")
    API_KEY = _settings.EODHD_API_KEY or os.getenv("EODHD_API_KEY", "")
    ROUND_DECIMALS = getattr(_settings.cache, "round_decimals", None)
except Exception:
    # フォールバック（settings が読めない場合）
    LOG_DIR = Path(os.path.dirname(__file__)) / "logs"
    DATA_CACHE_DIR = Path(os.path.dirname(__file__)) / ".." / "data_cache"
    LEGACY_RECENT_DIR = Path(os.path.dirname(__file__)) / ".." / "data_cache_recent"
    BASE_CACHE_DIR = Path(os.path.dirname(__file__)) / ".." / "data_cache" / BASE_SUBDIR_NAME
    THREADS_DEFAULT = int(os.getenv("THREADS_DEFAULT", 8))
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 10))
    DOWNLOAD_RETRIES = int(os.getenv("DOWNLOAD_RETRIES", 3))
    API_THROTTLE_SECONDS = float(os.getenv("API_THROTTLE_SECONDS", 1.5))
    API_BASE = os.getenv(
        "API_EODHD_BASE",
        "https://eodhistoricaldata.com",
    ).rstrip("/")
    API_KEY = os.getenv("EODHD_API_KEY", "")
    ROUND_DECIMALS = None

LOG_DIR.mkdir(parents=True, exist_ok=True)
DATA_CACHE_DIR = DATA_CACHE_DIR.resolve()
DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
try:
    LEGACY_RECENT_DIR = LEGACY_RECENT_DIR.resolve()
except Exception:
    LEGACY_RECENT_DIR = None
BASE_CACHE_DIR = BASE_CACHE_DIR.resolve()
BASE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_session_lock = threading.Lock()
_requests_session: requests.Session | None = None


def _get_requests_session() -> requests.Session:
    global _requests_session
    if _requests_session is None:
        with _session_lock:
            if _requests_session is None:
                pool_size = max(4, int(THREADS_DEFAULT) * 2)
                session = requests.Session()
                adapter = HTTPAdapter(
                    pool_connections=pool_size,
                    pool_maxsize=pool_size,
                )
                session.mount("https://", adapter)
                session.mount("http://", adapter)
                _requests_session = session
    return _requests_session


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
# スロットリング制御
# -----------------------------


class _ThrottleController:
    """共有レートリミッタ。
    ``configure`` で設定された待機時間を元に ``wait`` が次リクエストまで
    のスリープ時間を決定する。429 が返った場合などに ``backoff`` を呼ぶと
    一時的に待機時間を延長する。
    """

    def __init__(self, throttle_seconds: float) -> None:
        self._lock = threading.Lock()
        self._delay = max(0.0, float(throttle_seconds))
        self._next_time = 0.0
        self._block_until = 0.0

    def configure(self, throttle_seconds: float, concurrency_scale: int = 1) -> float:
        """レートリミッタの間隔を更新し、実効遅延を返す。"""
        delay = max(0.0, float(throttle_seconds))
        scale = max(1, int(concurrency_scale))
        if delay > 0 and scale > 1:
            delay /= scale
        with self._lock:
            self._delay = delay
            now = time.monotonic()
            self._next_time = now
            if self._block_until < now:
                self._block_until = now
            return self._delay

    def wait(self) -> None:
        while True:
            with self._lock:
                delay = self._delay
                if delay <= 0:
                    return
                now = time.monotonic()
                block_wait = self._block_until - now
                if block_wait > 0:
                    wait = block_wait
                else:
                    wait = self._next_time - now
                if wait <= 0:
                    self._next_time = now + delay
                    return
            time.sleep(min(delay, wait))

    def backoff(self, seconds: float) -> None:
        """レート制限違反時にバックオフ時間を設定し、ログに記録する。"""
        if seconds <= 0:
            return
        with self._lock:
            target = time.monotonic() + float(seconds)
            if target > self._block_until:
                self._block_until = target
        logging.warning(f"レート制限バックオフ: {seconds:.1f}秒待機")

    def current_delay(self) -> float:
        with self._lock:
            return self._delay


_throttle_controller = _ThrottleController(API_THROTTLE_SECONDS)


def _configure_api_throttle(
    concurrency_scale: int = 1, throttle_seconds: float | None = None
) -> float:
    """Fetch ワーカー数に応じて API レート制限を調整する。"""
    throttle = API_THROTTLE_SECONDS if throttle_seconds is None else throttle_seconds
    return _throttle_controller.configure(throttle, concurrency_scale)


def _throttle_api_call() -> None:
    """API 呼び出し前に共有レートリミッタへ待機を指示する。"""
    _throttle_controller.wait()


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
    try:
        symbols = build_symbol_universe(
            API_BASE,
            API_KEY,
            timeout=REQUEST_TIMEOUT,
            logger=logging.getLogger(__name__),
        )
    except Exception as exc:  # pragma: no cover - ネットワーク異常時は空集合
        logging.error("銘柄ユニバースの取得に失敗: %s", exc)
        return []

    logging.info("NASDAQ/EODHD フィルタ後の銘柄数: %s", len(symbols))
    return symbols


def get_with_retry(url: str, retries: int = DOWNLOAD_RETRIES, delay: float = 2.0):
    session = _get_requests_session()
    for i in range(max(1, retries)):
        sleep_for = delay
        try:
            _throttle_api_call()
            r = session.get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                sleep_for = max(delay, API_THROTTLE_SECONDS) * (i + 1)
                logging.warning("429 Too Many Requests (%s/%s) - %s", i + 1, retries, url)
                _throttle_controller.backoff(sleep_for)
            else:
                logging.warning(f"ステータスコード {r.status_code} - {url}")
        except Exception as e:
            logging.warning(f"試行{i + 1}回目のエラー: {e}")
        if sleep_for > 0:
            time.sleep(sleep_for)
    return None


def get_eodhd_data(symbol: str) -> pd.DataFrame | None:
    # API呼び出し用に小文字変換（内部管理は大文字のまま）
    api_symbol = symbol.lower()
    url = f"{API_BASE}/api/eod/{api_symbol}.US?api_token={API_KEY}&period=d&fmt=json"
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


@dataclass(slots=True)
class CacheResult:
    """キャッシュ処理の結果を格納するデータクラス。"""

    symbol: str
    message: str
    used_api: bool
    success: bool


@dataclass(slots=True)
class CacheJob:
    """キャッシュ処理ジョブを格納するデータクラス。"""

    symbol: str
    safe_symbol: str
    filepath: Path
    basepath: Path | None
    df: pd.DataFrame | None
    mode: Literal["skip", "save_full", "rebuild_base", "error"]
    message: str
    used_api: bool
    success: bool

    def to_result(self) -> CacheResult:
        return CacheResult(self.symbol, self.message, self.used_api, self.success)


def safe_filename(symbol: str) -> str:
    # Windows 予約語を避ける（大文字小文字無視）
    if symbol.upper() in RESERVED_WORDS:
        return symbol + "_RESV"
    return symbol


def _prepare_cache_job(
    symbol: str,
    output_dir: Path,
    base_dir: Path | None = None,
) -> CacheJob:
    """指定シンボルのキャッシュジョブを準備する。

    既存データのチェックとAPI取得の必要性を判断し、適切なモードを設定する。
    """
    output_dir = Path(output_dir)
    base_dir = Path(base_dir) if base_dir is not None else None
    safe_symbol = safe_filename(symbol)
    filepath = output_dir / f"{safe_symbol}.csv"
    basepath = base_dir / f"{safe_symbol}.csv" if base_dir else None

    today = datetime.today().date()
    if filepath.exists():
        mod_time = datetime.fromtimestamp(filepath.stat().st_mtime)
        if mod_time.date() == today:
            if basepath and not basepath.exists():
                existing_df = None
                try:
                    existing_df = pd.read_csv(filepath)
                    base_df = compute_base_indicators(existing_df)
                except Exception as exc:  # pragma: no cover - logging only
                    logging.warning(
                        "%s: 既存データからのbase再構築に失敗 (%s)",
                        symbol,
                        exc,
                    )
                    base_df = None
                if base_df is not None and not base_df.empty:
                    return CacheJob(
                        symbol=symbol,
                        safe_symbol=safe_symbol,
                        filepath=filepath,
                        basepath=basepath,
                        df=existing_df,
                        mode="rebuild_base",
                        message=f"{symbol}: already cached",
                        used_api=False,
                        success=True,
                    )
            return CacheJob(
                symbol=symbol,
                safe_symbol=safe_symbol,
                filepath=filepath,
                basepath=basepath,
                df=None,
                mode="skip",
                message=f"{symbol}: already cached",
                used_api=False,
                success=True,
            )

    df = get_eodhd_data(symbol)
    if df is not None and not df.empty:
        return CacheJob(
            symbol=symbol,
            safe_symbol=safe_symbol,
            filepath=filepath,
            basepath=basepath,
            df=df,
            mode="save_full",
            message=f"{symbol}: saved",
            used_api=True,
            success=True,
        )
    return CacheJob(
        symbol=symbol,
        safe_symbol=safe_symbol,
        filepath=filepath,
        basepath=basepath,
        df=None,
        mode="error",
        message=f"{symbol}: failed to fetch",
        used_api=True,
        success=False,
    )


def _process_cache_job(job: CacheJob) -> CacheResult:
    """キャッシュジョブを処理し、結果を返す。

    ジョブのモードに応じて、スキップ、base再構築、フル保存を実行する。
    """
    if job.mode in {"skip", "error"}:
        return job.to_result()

    if job.mode == "rebuild_base":
        if job.basepath is None or job.df is None:
            return job.to_result()
        try:
            job.basepath.parent.mkdir(parents=True, exist_ok=True)
            base_df = compute_base_indicators(job.df)
        except Exception as exc:  # pragma: no cover - logging only
            logging.warning("%s: base計算に失敗 (%s)", job.symbol, exc)
            return job.to_result()
        if base_df is not None and not base_df.empty:
            base_reset = round_dataframe(
                base_df.reset_index(),
                CACHE_ROUND_DECIMALS,
            )
            base_reset.to_csv(job.basepath, index=False)
        return CacheResult(job.symbol, job.message, job.used_api, True)

    # mode == "save_full"
    df = job.df
    if df is None or df.empty:
        msg = f"{job.symbol}: 保存対象データが空でした"
        return CacheResult(job.symbol, msg, job.used_api, False)

    try:
        job.filepath.parent.mkdir(parents=True, exist_ok=True)
        try:
            full_df = add_indicators(df.copy())
        except Exception:
            full_df = add_indicators(df)
        df_reset = full_df.reset_index().rename(columns=str.lower)
        df_reset = round_dataframe(df_reset, CACHE_ROUND_DECIMALS)
        df_reset.to_csv(job.filepath, index=False)
    except Exception as exc:  # pragma: no cover - logging only
        logging.error("%s: データ保存中のエラー (%s)", job.symbol, exc)
        return CacheResult(
            job.symbol,
            f"{job.symbol}: 保存時にエラーが発生しました",
            job.used_api,
            False,
        )

    base_saved = False
    if job.basepath is not None:
        try:
            job.basepath.parent.mkdir(parents=True, exist_ok=True)
            base_df = compute_base_indicators(df)
        except Exception as exc:
            logging.warning("%s: base計算に失敗 (%s)", job.symbol, exc)
            base_df = None
        if base_df is not None and not base_df.empty:
            base_reset = round_dataframe(
                base_df.reset_index(),
                CACHE_ROUND_DECIMALS,
            )
            base_reset.to_csv(job.basepath, index=False)
            base_saved = True

    msg = job.message
    if base_saved and "base saved" not in msg:
        msg = f"{msg} (base saved)"
    return CacheResult(job.symbol, msg, job.used_api, True)


def cache_single(
    symbol: str,
    output_dir: Path,
    base_dir: Path | None = None,
    throttle_seconds: float | None = None,
) -> tuple[str, bool, bool]:
    """指定シンボルをキャッシュ。

    戻り値: (message, used_api, success)
    """
    _configure_api_throttle(1, throttle_seconds)
    job = _prepare_cache_job(symbol, output_dir, base_dir)
    result = _process_cache_job(job)
    return (result.message, result.used_api, result.success)


def cache_data(
    symbols: list[str],
    output_dir: Path | str = DATA_CACHE_DIR,
    base_dir: Path | None = BASE_CACHE_DIR,
    max_workers: int | None = None,
    fetch_workers: int | None = 1,
    save_workers: int | None = None,
    throttle_seconds: float | None = 0.0667,
    progress_interval: int = 600,
    heartbeat_seconds: int | None = 20,
) -> None:
    """指定シンボルリストのデータを並列でキャッシュする。

    API取得と保存/計算を別スレッドで実行し、効率を向上させる。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if base_dir is not None:
        base_dir = Path(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

    max_workers = int(max_workers or THREADS_DEFAULT)
    # API取得は常に順次実行（fetch_workers=1）に固定します。
    # 保存/指標計算のみを並列化して I/O を効率化します。
    fetch_workers = 1
    if save_workers is None:
        save_workers = max_workers

    fetch_workers = max(1, int(fetch_workers))
    save_workers = max(1, int(save_workers))

    effective_throttle = _configure_api_throttle(fetch_workers, throttle_seconds)
    configured_throttle = 0.0667 if throttle_seconds is None else float(throttle_seconds)
    if effective_throttle > 0:
        print(
            f"ℹ️ APIスロットリング: 設定値 {configured_throttle:.3f} 秒 → "
            f"実効 {effective_throttle:.3f} 秒/リクエスト (fetch workers={fetch_workers})",
            flush=True,
        )
    else:
        print(
            f"ℹ️ APIスロットリングなし (fetch workers={fetch_workers})",
            flush=True,
        )

    # 当月ブラックリストに該当する銘柄をスキップ
    monthly_blacklist = load_monthly_blacklist()
    symbols_to_fetch = [s for s in symbols if s.upper() not in monthly_blacklist]
    skipped_due_to_cooldown = len(symbols) - len(symbols_to_fetch)

    failed: list[str] = []
    succeeded: list[str] = []
    results_list: list[tuple[str, str, bool]] = []
    completed_count = 0
    pending_writers = 0

    def handle_result(result: CacheResult) -> None:
        """結果を処理し、統計を更新する。"""
        nonlocal completed_count
        index = completed_count
        completed_count += 1
        results_list.append((result.symbol, result.message, result.used_api))
        logging.info(result.message)
        print(f"[{index}] {result.message}")
        # 進捗表示
        if progress_interval > 0 and completed_count % progress_interval == 0:
            total = len(symbols_to_fetch)
            print(
                f"📊 進捗: {completed_count}/{total} 銘柄完了 "
                f"({completed_count / total * 100:.1f}%)",
                flush=True,
            )
        if not result.success:
            failed.append(result.symbol)
        else:
            succeeded.append(result.symbol)

    def drain_results(block: bool = False) -> None:
        """結果キューから結果を処理する。メモリ制限付き。"""
        nonlocal pending_writers
        if pending_writers <= 0:
            return
        timeout = 0.1 if block else 0
        while pending_writers > 0:
            try:
                result = results_queue.get(block=block, timeout=timeout)
            except queue.Empty:
                break
            pending_writers -= 1
            handle_result(result)

    # 保存・インジ計算ステージを別スレッドで実行して API 取得との重なりを確保する
    def writer_task(job: CacheJob) -> CacheResult:
        """保存タスクを実行する。キュー満杯時は待機。"""
        try:
            result = _process_cache_job(job)
        except Exception:  # pragma: no cover - logging only
            logging.exception("%s: 保存処理で予期せぬ例外", job.symbol)
            result = CacheResult(
                job.symbol,
                f"{job.symbol}: 保存処理で例外が発生しました",
                job.used_api,
                False,
            )
        try:
            results_queue.put(result, timeout=10)  # タイムアウト付きでキューに追加
        except queue.Full:
            logging.error("%s: 結果キューが満杯のため処理をスキップ", job.symbol)
            # 満杯時は直接処理
            handle_result(result)
            nonlocal pending_writers
            pending_writers -= 1
        return result

    results_queue: queue.Queue[CacheResult] = queue.Queue(maxsize=1000)  # メモリ制限: 最大1000件

    print(
        f"🚀 データキャッシュ処理を開始します: {len(symbols_to_fetch)} 銘柄 "
        f"(fetch_workers={fetch_workers} (sequential), save_workers={save_workers})",
        flush=True,
    )

    # 動作方針の明示: API取得は常に順次実行(fetch_workers=1)し、
    # CSV保存と指標計算のみを並列化して I/O を効率化します。
    print(
        "ℹ️ 動作方針: API取得は順次実行(fetch_workers=1)し、"
        "CSV保存と指標計算は並列化(save_workers)して効率化します。",
        flush=True,
    )

    # ハートビート監視スレッド: 一定秒ごとに進捗を出力します。
    stop_event = threading.Event()
    monitor_thread: threading.Thread | None = None
    try:
        hb = int(heartbeat_seconds) if heartbeat_seconds is not None else 0
    except Exception:
        hb = 0

    def _heartbeat_monitor() -> None:
        total = len(symbols_to_fetch)
        while not stop_event.wait(max(1, hb)):
            processed = completed_count
            pending = pending_writers
            pct = (processed / total * 100) if total else 0.0
            # ローカル時刻でミリ秒精度のタイムスタンプを付与し、
            # ログレベル風ラベルを角括弧で表示します。
            # 例: 2025-09-23T12:34:56.789+09:00 [HEARTBEAT] ⏱ 進捗: 12/100 (12.0%)
            now = datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")
            label = "[HEARTBEAT]"
            print(
                f"{now} {label} ⏱ 進捗: {processed}/{total} 銘柄完了 "
                f"({pct:.1f}%) - pending_writers={pending}",
                flush=True,
            )

    if hb and hb > 0:
        monitor_thread = threading.Thread(target=_heartbeat_monitor, daemon=True)
        monitor_thread.start()

    # 起動時にハートビート設定を明示
    if hb and hb > 0:
        print(f"ℹ️ ハートビート設定: {hb} 秒ごとに進捗を出力します（0で無効化）", flush=True)
    else:
        print(
            "ℹ️ ハートビートは無効化されています。表示は進捗コールバック依存になります。",
            flush=True,
        )

    with ThreadPoolExecutor(max_workers=save_workers) as writer_executor:
        with ThreadPoolExecutor(max_workers=fetch_workers) as fetch_executor:
            future_to_symbol = {
                fetch_executor.submit(
                    _prepare_cache_job,
                    symbol,
                    output_dir,
                    base_dir,
                ): symbol
                for symbol in symbols_to_fetch
            }
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    job = future.result()
                except Exception:  # pragma: no cover - logging only
                    logging.exception("%s: 取得処理で予期せぬ例外", symbol)
                    handle_result(
                        CacheResult(
                            symbol,
                            f"{symbol}: 取得処理で例外が発生しました",
                            True,
                            False,
                        )
                    )
                    continue
                if job.mode in {"save_full", "rebuild_base"}:
                    pending_writers += 1
                    writer_executor.submit(writer_task, job)
                else:
                    handle_result(job.to_result())
                drain_results()
        writer_executor.shutdown(wait=True)
        while pending_writers > 0:
            drain_results(block=True)
        drain_results()

    # ブラックリスト更新/回復削除
    if failed:
        update_failed_symbols(failed)
    if succeeded:
        remove_recovered_symbols(succeeded)

    # 統計の出力
    cached_count = sum(1 for _, _, used_api in results_list if not used_api)
    api_count = sum(1 for _, _, used_api in results_list if used_api)
    print(
        f"✅ キャッシュ済み: {cached_count}件, API使用: {api_count}件, "
        f"失敗: {len(failed)}件, クールダウン除外: {skipped_due_to_cooldown}件"
    )
    # 監視スレッドへ終了を通知して安全に停止を待つ（デーモンであるため必須ではないが明示）
    try:
        stop_event.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=1)
    except Exception:
        pass


def _cli_main() -> None:
    parser = argparse.ArgumentParser(description="EODHD デイリーデータのキャッシュを作成する")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="指定した場合、銘柄リストをこのサイズで分割して対象チャンクのみを取得する",
    )
    parser.add_argument(
        "--chunk-index",
        type=int,
        default=1,
        help="chunk-size と併用。1 始まりで何番目のチャンクを処理するかを指定する",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="ThreadPoolExecutor のワーカー数を上書きする",
    )
    parser.add_argument(
        "--fetch-workers",
        type=int,
        default=1,
        help="API取得ステージの並列度を指定する (既定: 1、順次実行でレート制限遵守)",
    )
    parser.add_argument(
        "--save-workers",
        type=int,
        default=None,
        help="保存/インジ計算ステージの並列度を指定する (既定: max_workers)",
    )
    parser.add_argument(
        "--throttle-seconds",
        type=float,
        default=0.0667,
        help="API 呼び出し間隔を秒単位で上書きする (既定: 0.0667秒、約15req/sec)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="強制的に full から再取得 (bulk をスキップしない)",
    )
    parser.add_argument(
        "--bulk-today",
        action="store_true",
        help="本日の Bulk 更新を明示的に実行する（従来のデフォルト動作に相当）",
    )
    parser.add_argument(
        "--skip-bulk",
        action="store_true",
        help="bulk 更新をスキップして API から取得する",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=300,
        help="進捗表示の間隔を件数で指定する (既定: 300件)",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=20,
        help="ハートビート監視の間隔を秒単位で指定する (既定: 20秒)。0で無効化",
    )
    # --parallel-fetch を廃止: API取得は常に順次(fetch_workers=1)
    args = parser.parse_args()

    # 変更: 引数が何も指定されなかった場合はデフォルトでフル取得する。
    # ただし `--bulk-today` を明示的に指定した場合のみ Bulk を実行する。
    if not args.full and not args.skip_bulk and not args.bulk_today:
        args.full = True

    # symbols = get_all_symbols()[:3]  # 簡易テスト用
    symbols = get_all_symbols()
    if not symbols:
        print("⚠️ 対象銘柄が検出できなかったため処理を終了します。", flush=True)
        return

    safe_symbols = [safe_filename(s) for s in symbols]
    try:
        save_symbol_manifest(safe_symbols, DATA_CACHE_DIR)
    except Exception as exc:  # pragma: no cover - logging only
        logging.warning("シンボルマニフェストの保存に失敗: %s", exc)

    # 全体件数をキャッシュしておく（チャンク処理や表示で使用）
    total_symbols = len(symbols)

    fallback_to_full = bool(args.full)
    if not args.full and not args.skip_bulk:
        stats = None
        try:
            stats = _attempt_bulk_refresh(symbols, progress_interval=args.progress_interval)
        except BaseException as exc:  # noqa: BLE001 - 中断検知のため
            if isinstance(exc, KeyboardInterrupt) or (
                CacheUpdateInterrupted is not None and isinstance(exc, CacheUpdateInterrupted)
            ):
                _report_bulk_interrupt(exc, total_symbols)
                return
            raise
        if stats is None:
            print(
                "⚠️ Bulk 更新が実行できなかったため API 再取得にフォールバックします。",
                flush=True,
            )
            fallback_to_full = True
        elif not stats.has_payload:
            print(
                "ℹ️ Bulk API の応答が空だったため追加更新はありませんでした。",
                flush=True,
            )
            return
        elif stats.filtered_rows == 0:
            print(
                "⚠️ Bulk データに処理対象銘柄が存在しなかったため "
                "API 再取得にフォールバックします。",
                flush=True,
            )
            fallback_to_full = True
        else:
            print(
                (
                    f"✅ Bulk更新完了: 対象={stats.processed_symbols} 銘柄 / "
                    f"更新={stats.updated_symbols} 銘柄 (フィルタ後 {stats.filtered_rows} 行)"
                ),
                flush=True,
            )
            if stats.universe_error:
                msg = stats.universe_error_message or "理由不明"
                print(
                    "⚠️ 銘柄ユニバース取得に問題があった可能性があります:",
                    msg,
                    flush=True,
                )
            if stats.updated_symbols == 0:
                print("ℹ️ キャッシュは既に最新のため追加取得は不要です。", flush=True)
            return

    if fallback_to_full or args.full or args.skip_bulk:
        if args.skip_bulk and not args.full:
            print("ℹ️ --skip-bulk 指定のため API からの再取得を実行します。", flush=True)

        # chunk_size適用
        if args.chunk_size:
            chunk_size = max(1, args.chunk_size)
            chunk_index = max(1, args.chunk_index)
            start = chunk_size * (chunk_index - 1)
            if start >= total_symbols:
                print(
                    f"⚠️ チャンク開始位置 {start + 1} が銘柄数 {total_symbols} を超えています。"
                    "処理をスキップします。"
                )
                return
            end = min(total_symbols, start + chunk_size)
            symbols = symbols[start:end]
            print(
                f"{total_symbols}銘柄中 {start + 1}〜{end} 件目 (計 {len(symbols)} 銘柄) を"
                f"取得します（チャンク {chunk_index}、サイズ {chunk_size}）。"
            )
        else:
            print(f"{len(symbols)}銘柄を取得します（クールダウン月次ブラックリスト適用後に除外）")

        cache_data(
            symbols,
            output_dir=DATA_CACHE_DIR,
            base_dir=BASE_CACHE_DIR,
            max_workers=args.max_workers,
            fetch_workers=args.fetch_workers,
            save_workers=args.save_workers,
            throttle_seconds=args.throttle_seconds,
            progress_interval=args.progress_interval,
            heartbeat_seconds=args.heartbeat_seconds,
        )
        print("データのキャッシュが完了しました。", flush=True)

    # chunk_sizeブロックを削除（上記に統合）
    # if args.chunk_size: ...


if __name__ == "__main__":
    _cli_main()
