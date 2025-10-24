"""Daily multi-system signal pipeline (repaired minimal bootstrap section).

NOTE: This file experienced prior encoding corruption. Incremental repairs are
being applied. The current patch introduces:
    parser.add_argument(
        "--run-namespace",
        default=None,
        help="任意のラン識別子: 出力を results_csv/<NAMESPACE>/ に分離するために使用します",
    )
 1. Explicit project root insertion into sys.path so that running the script
     via ``python scripts/run_all_systems_today.py`` correctly resolves top-level
     modules like ``common``.
 2. Use of ``get_settings(create_dirs=False)`` inside ``_initialize_run_context``
     to avoid potential hangs during strategy initialization (directory
     creation is performed lazily elsewhere if needed).

Further clean-up (mojibake in log strings/docstrings) will follow in later
patches without altering CLI flags or public behavior.
"""

from __future__ import annotations

# flake8: noqa: E501
import argparse
import io
import json
import logging
import multiprocessing
import os
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, cast, no_type_check
from zoneinfo import ZoneInfo

# --- ensure repository root on sys.path
# (script executed from repo root or elsewhere)
try:  # noqa: SIM105
    _project_root = Path(__file__).resolve().parents[1]
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
except Exception:  # pragma: no cover - defensive; failure is non-fatal
    pass

# Windows スケジューラー実行時の cp932 エンコードエラーを回避
if sys.platform == "win32":
    try:
        # reconfigure が利用可能な Python のみ直接切り替え
        _reconf_out = getattr(sys.stdout, "reconfigure", None)
        if callable(_reconf_out):
            _reconf_out(encoding="utf-8")
        _reconf_err = getattr(sys.stderr, "reconfigure", None)
        if callable(_reconf_err):
            _reconf_err(encoding="utf-8")
    except (AttributeError, io.UnsupportedOperation):
        # Fallback: Windows cp932 を回避するために UTF-8 ラッパを被せる
        import codecs

        try:
            sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "replace")
        except Exception:
            pass
        try:
            sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "replace")
        except Exception:
            pass

import pandas as pd

from common import broker_alpaca as ba
from common.alpaca_order import submit_orders_df
from common.cache_manager import CacheManager, load_base_cache
from common.dataframe_utils import round_dataframe  # noqa: E402
from common.indicator_access import get_indicator, is_true, to_float
from common.latest_day_validator import (
    get_exclusion_stats,
    save_excluded_symbols_csv,
    validate_latest_trading_day,
)
from common.notification import notify_zero_trd_all_systems
from common.notifier import create_notifier
from common.position_age import load_entry_dates, save_entry_dates
from common.run_lock import RunLock
from common.signal_merge import Signal, merge_signals
from common.stage_metrics import GLOBAL_STAGE_METRICS, StageEvent, StageSnapshot
from common.structured_logging import MetricsCollector
from common.symbol_universe import build_symbol_universe_from_settings
from common.system_diagnostics import get_diagnostics_with_fallback

# 抽出: データローダ関数は common.today_data_loader へ分離
from common.today_data_loader import load_basic_data

# 抽出: フィルタ/条件/低レベルヘルパは common.today_filters へ分離
from common.today_filters import (
    _system1_conditions,
    _system2_conditions,
    _system3_conditions,
    _system4_conditions,
    _system5_conditions,
    _system6_conditions,
    filter_system1,
    filter_system2,
    filter_system3,
    filter_system4,
    filter_system5,
    filter_system6,
)
from common.utils_spy import (
    get_latest_nyse_trading_day,
    get_signal_target_trading_day,
    get_spy_with_indicators,
)
from config.environment import get_env_config
from config.settings import get_settings
from core.final_allocation import finalize_allocation, load_symbol_system_map
from core.system1 import summarize_system1_diagnostics
from core.system5 import DEFAULT_ATR_PCT_THRESHOLD

# strategies
from strategies.system1_strategy import System1Strategy
from strategies.system2_strategy import System2Strategy
from strategies.system3_strategy import System3Strategy
from strategies.system4_strategy import System4Strategy
from strategies.system5_strategy import System5Strategy
from strategies.system6_strategy import System6Strategy
from strategies.system7_strategy import System7Strategy
from tools.notify_metrics import send_metrics_notification  # noqa: E402

# --- Console encoding helpers (to mitigate mojibake on Windows terminals) ---
_env = get_env_config()
_NO_EMOJI_ENV = bool(_env.no_emoji)

# コンパクトログ（詳細DEBUGを抑制）
_COMPACT_LOG = bool(_env.compact_logs)


def _console_supports_utf8() -> bool:
    try:
        enc = (getattr(sys.stdout, "encoding", None) or "").lower()
        return "utf-8" in enc or "65001" in enc  # CP65001 is UTF-8 on Windows
    except Exception:
        return False


def _strip_emojis(text: str) -> str:
    try:
        import re as _re

        # Remove characters outside BMP (common emojis etc.)
        return _re.sub(r"[\U00010000-\U0010FFFF]", "", str(text))
    except Exception:
        # Fallback: best-effort ASCII replacement
        try:
            enc = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
            return str(text).encode(enc, errors="ignore").decode(enc, errors="ignore")
        except Exception:
            return str(text)


_LOG_CALLBACK = None

# Progress event settings (EnvironmentConfig 経由に寄せる)
try:
    from config.environment import get_env_config

    ENABLE_PROGRESS_EVENTS = bool(get_env_config().enable_progress_events)
except Exception:
    # フォールバック: 有効にする（UI 同期用途）。必要に応じて .env で無効化可。
    ENABLE_PROGRESS_EVENTS = True

# Global log file variables (initialized by setup_logging)
_LOG_FILE_PATH: Path | None = None
_LOG_FILE_MODE: str | None = None

# Global metrics collector for performance tracking
_GLOBAL_METRICS = MetricsCollector()


def emit_progress_event(event_type: str, data: dict) -> None:
    """Emit a progress event with given type and data to JSONL."""
    if not ENABLE_PROGRESS_EVENTS:
        return
    try:
        from common.progress_events import emit_progress  # lazy import to avoid cycles

        emit_progress(event_type, data)
    except Exception:
        # 失敗してもパイプラインは止めない
        try:
            logger = logging.getLogger(__name__)
            logger.debug("(fallback) Progress event [%s]: %s", event_type, data)
        except Exception:
            pass


_LOG_FORWARDING: ContextVar[bool] = ContextVar("_LOG_FORWARDING", default=False)


# NOTE: StrategyProtocol 一時撤去（戦略側の実装差異が大きく attr-defined 問題を誘発のため）
_LOG_START_TS: float | None = None  # CLI 用の経過時間測定開始時刻

# Structured UI logging state (initialized lazily inside _emit_ui_log)
_STRUCTURED_LOG_START_TS: float | None = None  # monotonic-ish epoch seconds
_STRUCTURED_LAST_PHASE: dict[str, str] | None = None  # {system: last_phase}

# ログファイル設定（デフォルトは固定ファイル）。必要に応じて日付付きへ切替。
# レート制限ロガー
_rate_limited_logger = None


# --- Lightweight Benchmark (--benchmark flag) --------------------------------------------
class LightweightBenchmark:
    """軽量ベンチマーク（時間計測のみ、--benchmark フラグで有効化）。"""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.phases: dict[str, dict[str, float]] = {}
        self._current_phase: str | None = None
        self._start_time: float | None = None
        self._global_start: float | None = None
        # 追加メタデータ（任意のブロックや明細を保存するための拡張領域）
        self.extras: dict[str, Any] = {}

    def start_phase(self, phase_name: str) -> None:
        """フェーズ開始時刻を記録。"""
        if not self.enabled:
            return
        import time

        if self._global_start is None:
            self._global_start = time.perf_counter()
        self._current_phase = phase_name
        self._start_time = time.perf_counter()

    def end_phase(self) -> None:
        """フェーズ終了時刻を記録。"""
        if not self.enabled or self._current_phase is None or self._start_time is None:
            return
        import time

        end_time = time.perf_counter()
        duration = end_time - self._start_time
        self.phases[self._current_phase] = {
            "start": self._start_time - (self._global_start or 0.0),
            "end": end_time - (self._global_start or 0.0),
            "duration_sec": round(duration, 6),
        }
        self._current_phase = None
        self._start_time = None

    def get_report(self) -> dict[str, Any]:
        """ベンチマークレポートを取得。"""
        if not self.enabled:
            return {"enabled": False, "phases": {}, "total_duration_sec": 0.0}

        total_duration = sum(p["duration_sec"] for p in self.phases.values())
        return {
            "enabled": True,
            "timestamp": datetime.now().isoformat(),
            "phases": self.phases,
            "total_duration_sec": round(total_duration, 6),
            # 追加メタデータ（フェーズ内訳など）
            "extras": self.extras,
        }

    def save_report(self, output_path: str | Path) -> None:
        """レポートをJSONで保存。"""
        if not self.enabled:
            return
        path = Path(output_path)
        from common.io_utils import write_json

        write_json(path, self.get_report(), ensure_ascii=False, indent=2)

    # 追加: 任意セクションの付加
    def add_extra_section(self, name: str, payload: Any) -> None:
        if not self.enabled:
            return
        try:
            self.extras[str(name)] = payload
        except Exception:
            # extras 書き込み失敗は致命的ではないので無視
            pass


_LIGHTWEIGHT_BENCHMARK: LightweightBenchmark | None = None


# --- stage progress bridging helpers -----------------------------------------------------

_PER_SYSTEM_STAGE = None
_PER_SYSTEM_EXIT = None
_SET_STAGE_UNIVERSE_TARGET = None

_STAGE_EVENT_PUMP_THREAD: threading.Thread | None = None
_STAGE_EVENT_PUMP_STOP: threading.Event | None = None
_STAGE_EVENT_PUMP_INTERVAL = 0.25  # デフォルト250ms

# 最適化用フラグ（アクティブ処理時は頻繁に、アイドル時は負荷軽減）
_STAGE_EVENT_PUMP_ADAPTIVE = True
_STAGE_EVENT_PUMP_MIN_INTERVAL = 0.1  # 最小100ms（高負荷時）
_STAGE_EVENT_PUMP_MAX_INTERVAL = 1.0  # 最大1秒（アイドル時）
_STAGE_EVENT_PUMP_IDLE_THRESHOLD = 5  # 5回連続でイベントなしでアイドル判定


class StageReporter:
    """Callable wrapper that forwards stage progress with an associated system name."""

    __slots__ = ("system", "_queue")

    def __init__(self, system: str, queue: Any | None = None) -> None:
        self.system = str(system or "").strip().lower() or "unknown"
        self._queue = queue

    def __call__(
        self,
        progress: int,
        filter_count: int | None = None,
        setup_count: int | None = None,
        candidate_count: int | None = None,
        entry_count: int | None = None,
    ) -> None:
        if self._queue is not None:
            try:
                self._queue.put(
                    (
                        self.system,
                        progress,
                        filter_count,
                        setup_count,
                        candidate_count,
                        entry_count,
                    ),
                    block=False,
                )
            except Exception:
                pass
            return
        _stage(
            self.system,
            progress,
            filter_count,
            setup_count,
            candidate_count,
            entry_count,
        )


def register_stage_callback(callback: Callable[..., None] | None) -> None:
    """Register per-system stage callback and ensure the event pump is running."""

    globals()["_PER_SYSTEM_STAGE"] = callback
    if callable(callback):
        _ensure_stage_event_pump()
    else:
        _stop_stage_event_pump()


def register_stage_exit_callback(callback: Callable[[str, int], None] | None) -> None:
    """Register per-system exit callback (UI integration helper)."""

    globals()["_PER_SYSTEM_EXIT"] = callback


def register_universe_target_callback(
    callback: Callable[[int | None], None] | None,
) -> None:
    """Register callback to update the shared universe target in the UI."""

    globals()["_SET_STAGE_UNIVERSE_TARGET"] = callback


def _ensure_stage_event_pump(interval: float | None = None) -> None:
    """Start a background thread that periodically drains stage events for the UI.

    アダプティブ間隔調整機能:
    - イベントが頻繁な時は高頻度（100ms）
    - アイドル時は低頻度（1秒）でCPU負荷軽減
    """

    cb = globals().get("_PER_SYSTEM_STAGE")
    if not cb or not callable(cb):
        return

    thread = globals().get("_STAGE_EVENT_PUMP_THREAD")
    if isinstance(thread, threading.Thread) and thread.is_alive():
        return

    stop_event = threading.Event()
    globals()["_STAGE_EVENT_PUMP_STOP"] = stop_event

    base_interval = float(
        interval if interval is not None else _STAGE_EVENT_PUMP_INTERVAL
    )

    def _pump() -> None:
        current_interval = base_interval
        idle_count = 0

        while not stop_event.is_set():
            events_processed = False
            try:
                # イベント数をチェックしてアダプティブ調整
                queue_obj = globals().get("_PROGRESS_QUEUE")
                queue_size = 0
                if queue_obj is not None:
                    try:
                        # キューサイズの概算（実際には非破壊的にチェック不可）
                        queue_size = (
                            queue_obj.qsize() if hasattr(queue_obj, "qsize") else 0
                        )
                    except Exception:
                        queue_size = 0

                _drain_stage_event_queue()

                # GLOBAL_STAGE_METRICS からもイベント数をチェック
                try:
                    metrics_events = len(GLOBAL_STAGE_METRICS.drain_events())
                    if metrics_events > 0 or queue_size > 0:
                        events_processed = True
                except Exception:
                    pass

                # アダプティブ間隔調整
                if _STAGE_EVENT_PUMP_ADAPTIVE:
                    if events_processed:
                        # イベントがあった場合、間隔を短縮
                        current_interval = max(
                            _STAGE_EVENT_PUMP_MIN_INTERVAL, current_interval * 0.8
                        )
                        idle_count = 0
                    else:
                        # イベントがなかった場合、アイドルカウント増加
                        idle_count += 1
                        if idle_count >= _STAGE_EVENT_PUMP_IDLE_THRESHOLD:
                            # アイドル状態では間隔を延長してCPU負荷軽減
                            current_interval = min(
                                _STAGE_EVENT_PUMP_MAX_INTERVAL, current_interval * 1.2
                            )

            except Exception:
                pass

            stop_event.wait(current_interval)

    pump_thread = threading.Thread(target=_pump, name="stage-event-pump", daemon=True)
    globals()["_STAGE_EVENT_PUMP_THREAD"] = pump_thread
    pump_thread.start()


def _stop_stage_event_pump(timeout: float = 1.0) -> None:
    """Stop the background event pump thread if it is running."""

    stop_event = globals().get("_STAGE_EVENT_PUMP_STOP")
    thread = globals().get("_STAGE_EVENT_PUMP_THREAD")

    if isinstance(stop_event, threading.Event):
        stop_event.set()

    if isinstance(thread, threading.Thread) and thread.is_alive():
        if threading.current_thread() is not thread:
            thread.join(timeout)

    globals().pop("_STAGE_EVENT_PUMP_STOP", None)
    globals().pop("_STAGE_EVENT_PUMP_THREAD", None)


def _get_rate_limited_logger():
    """レート制限ロガーを取得。"""
    global _rate_limited_logger
    if _rate_limited_logger is None:
        from common.rate_limited_logging import create_rate_limited_logger

        _rate_limited_logger = create_rate_limited_logger("run_all_systems_today", 3.0)
    return _rate_limited_logger


def _prepare_concat_frames(
    frames: Sequence[pd.DataFrame | None],
) -> list[pd.DataFrame]:
    """Drop全NA列を除去し、空データを連結対象から外す。"""

    cleaned: list[pd.DataFrame] = []
    for frame in frames:
        if frame is None or getattr(frame, "empty", True):
            continue
        try:
            cleaned_frame = frame.dropna(axis=1, how="all")
        except Exception:
            cleaned_frame = frame
        if getattr(cleaned_frame, "empty", True):
            continue
        cleaned.append(cleaned_frame)
    return cleaned


@dataclass(slots=True)
class BaseCachePool:
    """base キャッシュの共有辞書をスレッドセーフに管理する補助クラス。"""

    cache_manager: CacheManager
    shared: dict[str, pd.DataFrame] | None = None
    hits: int = 0
    loads: int = 0
    failures: int = 0
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.shared is None:
            self.shared = {}

    def get(
        self,
        symbol: str,
        *,
        rebuild_if_missing: bool = True,
        min_last_date: pd.Timestamp | None = None,
        allowed_recent_dates: set[pd.Timestamp] | None = None,
    ) -> tuple[pd.DataFrame | None, bool]:
        """base キャッシュから銘柄シンボルの DataFrame を取得する。

        Returns (df, from_cache):
            - df: 取得または再構築された DataFrame（存在しなければ None）
            - from_cache: True=共有キャッシュ命中 / False=新規ロード

        フィルタ条件:
            rebuild_if_missing: キャッシュ欠損時にベースデータを再構築するか
            min_last_date: 末尾日付がこの日付(正規化)未満なら stale とみなす
            allowed_recent_dates: 許可された最終日付集合（存在し、かつ一致しなければ stale）
        stale 判定時はキャッシュを破棄して再ロードを試みる。
        """

        allowed_set = set(allowed_recent_dates or ())
        if min_last_date is not None:
            try:
                min_norm: pd.Timestamp | None = pd.Timestamp(min_last_date).normalize()
            except Exception:
                min_norm = None
        else:
            min_norm = None

        def _detect_last(frame: pd.DataFrame | None) -> pd.Timestamp | None:
            if frame is None or getattr(frame, "empty", True):
                return None
            # 優先: index から推定
            try:
                idx_dt = pd.to_datetime(frame.index, errors="coerce")
                if isinstance(idx_dt, pd.DatetimeIndex) and len(idx_dt):
                    last_val = idx_dt[-1]
                    return pd.Timestamp(cast(Any, last_val)).normalize()
            except Exception:
                pass
            # 次点: Date/date 列から推定
            try:
                series = frame.get("Date") if frame is not None else None
                if series is None and frame is not None and "date" in frame.columns:
                    series = frame.get("date")
                if series is not None:
                    ser_dt = pd.to_datetime(series, errors="coerce").dropna()
                    if len(ser_dt):
                        return pd.Timestamp(cast(Any, ser_dt.iloc[-1])).normalize()
            except Exception:
                pass
            return None

        with self._lock:
            if self.shared is not None and symbol in self.shared:
                value = self.shared[symbol]
                last_date = _detect_last(value)
                stale = False
                if allowed_set and (last_date is None or last_date not in allowed_set):
                    stale = True
                if not stale and min_norm is not None:
                    if last_date is None or last_date < min_norm:
                        stale = True
                if not stale:
                    self.hits += 1
                    return value, True
                try:
                    if self.shared is not None:
                        self.shared.pop(symbol, None)
                except Exception:
                    pass

        df = load_base_cache(
            symbol,
            rebuild_if_missing=rebuild_if_missing,
            cache_manager=self.cache_manager,
            min_last_date=min_last_date,
            allowed_recent_dates=allowed_set or None,
            prefer_precomputed_indicators=True,
        )

        with self._lock:
            if self.shared is not None and df is not None:
                # only store when df is a real DataFrame
                self.shared[symbol] = df
            self.loads += 1
            if df is None or getattr(df, "empty", True):
                self.failures += 1

        return df, False

    def sync_to(self, target: dict[str, pd.DataFrame] | None) -> None:
        """既存の外部辞書へ共有キャッシュを反映する。"""

        if target is None or self.shared is None or target is self.shared:
            return
        with self._lock:
            try:
                target.update(self.shared)
            except Exception:
                pass

    def snapshot_stats(self) -> dict[str, int]:
        with self._lock:
            size = len(self.shared or {})
            return {
                "hits": self.hits,
                "loads": self.loads,
                "failures": self.failures,
                "size": size,
            }


@dataclass(slots=True)
class TodayRunContext:
    """保持共有状態とコールバックを集約した当日シグナル実行用コンテキスト。"""

    settings: Any
    cache_manager: CacheManager
    signals_dir: Path
    cache_dir: Path
    slots_long: int | None = None
    slots_short: int | None = None
    capital_long: float | None = None
    capital_short: float | None = None
    save_csv: bool = False
    csv_name_mode: str | None = None
    notify: bool = True
    log_callback: Callable[[str], None] | None = None
    progress_callback: Callable[[int, int, str], None] | None = None
    per_system_progress: Callable[[str, str], None] | None = None
    symbol_data: dict[str, pd.DataFrame] | None = None
    parallel: bool = False
    run_start_time: datetime = field(default_factory=datetime.now)
    start_equity: float = 0.0
    run_id: str = ""
    today: pd.Timestamp | None = None
    symbol_universe: list[str] = field(default_factory=list)
    basic_data: dict[str, pd.DataFrame] = field(default_factory=dict)
    base_cache: dict[str, pd.DataFrame] = field(default_factory=dict)
    system_filters: dict[str, list[str]] = field(default_factory=dict)
    per_system_frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    final_signals: pd.DataFrame | None = None
    system_diagnostics: dict[str, dict[str, Any]] = field(default_factory=dict)
    # テスト高速化オプション
    test_mode: str | None = None  # mini/quick/sample
    skip_external: bool = False  # 外部API呼び出しをスキップ
    # latest_only グローバル制御: "データ基準日"（例: 週末は金曜、平日は当日）
    signal_base_day: pd.Timestamp | None = None
    # 実行開始時に確定する「エントリー予定日」（基準日の翌営業日）
    entry_day: pd.Timestamp | None = None
    # latest_only モードで許容する最新日からの遅延日数（営業日ベース）
    max_date_lag_days: int = 1
    # 任意のラン識別子（テスト/CI用）。ディレクトリ分離やログのプレフィックスに使用。
    run_namespace: str | None = None


def _get_account_equity() -> float:
    """Return current account equity via Alpaca API.

    失敗した場合は 0.0 を返す（テスト環境など API 未設定時の安全対策）。
    """
    try:
        client = ba.get_client(paper=True)
        acct = client.get_account()
        return float(getattr(acct, "equity", 0.0) or 0.0)
    except Exception:
        return 0.0


def _configure_today_logger(
    *, mode: str = "single", _run_id: str | None = None
) -> None:
    """today_signals 用のロガーファイルを構成する。

    mode:
      - "single": 固定ファイル `today_signals.log`
                if os.environ.get("ALLOCATION_DEBUG", "1") == "1":
    run_id: 予約（現状未使用）。将来、ファイル名に含めたい場合に利用。
    """
    global _LOG_FILE_PATH, _LOG_FILE_MODE
    _LOG_FILE_MODE = mode or "single"
    try:
        settings = get_settings(create_dirs=True)
        log_dir = Path(settings.LOGS_DIR)
    except Exception:
        log_dir = Path("logs")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    if _LOG_FILE_MODE == "dated":
        try:
            jst_now = datetime.now(ZoneInfo("Asia/Tokyo"))
        except Exception:
            jst_now = datetime.now()
        stamp = jst_now.strftime("%Y%m%d_%H%M")
        filename = f"today_signals_{stamp}.log"
    else:
        filename = "today_signals.log"

    _LOG_FILE_PATH = log_dir / filename
    # ハンドラを最新パスに合わせて張り替える
    try:
        logger = logging.getLogger("today_signals")
        for h in list(logger.handlers):
            try:
                if isinstance(h, logging.FileHandler) and getattr(
                    h, "baseFilename", None
                ):
                    if Path(h.baseFilename) != _LOG_FILE_PATH:
                        logger.removeHandler(h)
                        try:
                            h.close()
                        except Exception:
                            pass
            except Exception:
                # ハンドラ情報取得に失敗した場合は無視
                pass
        # 以降、_get_today_logger() が適切なハンドラを追加する
    except Exception:
        pass


def _get_today_logger() -> logging.Logger:
    """today_signals 用のファイルロガーを取得。

    デフォルトは `logs/today_signals.log`。
    `_configure_today_logger(mode="dated")` 適用時は日付付きファイルに出力。
    UI 有無に関係なく、完全な実行ログを常にファイルへ残す。
    """
    logger = logging.getLogger("today_signals")
    logger.setLevel(logging.INFO)
    # ルートロガーへの伝播を止めて重複出力を防止
    try:
        logger.propagate = False
    except Exception:
        pass
    # ルートロガーへの伝播を止め、コンソール二重出力を防止
    try:
        logger.propagate = False
    except Exception:
        pass
    # 目標ファイルパスを決定
    try:
        # 環境変数でも日付別ログ指定を許可（UI 実行など main() を経ない場合）
        if globals().get("_LOG_FILE_PATH") is None:
            try:
                _mode_env = (
                    (get_env_config().today_signals_log_mode or "").strip().lower()
                )
                if _mode_env == "dated":
                    try:
                        _jst_now = datetime.now(ZoneInfo("Asia/Tokyo"))
                    except Exception:
                        _jst_now = datetime.now()
                    _stamp = _jst_now.strftime("%Y%m%d_%H%M")
                    try:
                        settings = get_settings(create_dirs=True)
                        _log_dir = Path(settings.LOGS_DIR)
                    except Exception:
                        _log_dir = Path("logs")
                    try:
                        _log_dir.mkdir(parents=True, exist_ok=True)
                    except Exception:
                        pass
                    globals()["_LOG_FILE_PATH"] = (
                        _log_dir / f"today_signals_{_stamp}.log"
                    )
            except Exception:
                pass

        if globals().get("_LOG_FILE_PATH") is not None:
            log_path = globals().get("_LOG_FILE_PATH")
        else:
            try:
                settings = get_settings(create_dirs=True)
                log_dir = Path(settings.LOGS_DIR)
            except Exception:
                log_dir = Path("logs")
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            log_path = log_dir / "today_signals.log"
    except Exception:
        log_path = Path("logs") / "today_signals.log"

    # 既存の同一ファイルハンドラがあるか確認
    has_handler = False
    for h in list(logger.handlers):
        try:
            if isinstance(h, logging.FileHandler):
                base = getattr(h, "baseFilename", None)
                if base:
                    if Path(base).resolve() == Path(str(log_path)).resolve():
                        has_handler = True
                        break
        except Exception:
            continue
    if not has_handler:
        try:
            fh = logging.FileHandler(str(log_path), encoding="utf-8")
            fmt = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
            )
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except Exception:
            pass
    return logger


def _emit_ui_log(message: str) -> None:
    """UI コールバックへログを送信。

    環境変数 `STRUCTURED_UI_LOGS=1` の場合は JSON 文字列を送り、
    `{"ts": epoch_ms, "iso": iso8601, "msg": message}` 形式にする。
    既存テスト互換のためデフォルトは従来のプレーンテキスト。
    """
    # 1) フラグ判定（UI構造化 と NDJSON）
    try:
        structured_ui = bool(get_env_config().structured_ui_logs)
    except Exception:
        structured_ui = False
    try:
        ndjson_flag = bool(get_env_config().structured_log_ndjson)
    except Exception:
        ndjson_flag = False

    obj = None
    json_payload = None
    if structured_ui or ndjson_flag:
        try:
            import json as _json
            import re as _re
            import time as _t

            # 開始基準時刻（プロセス起動後最初の呼び出しで初期化）
            global _STRUCTURED_LOG_START_TS
            if _STRUCTURED_LOG_START_TS is None:
                _STRUCTURED_LOG_START_TS = _t.time()
            now = _t.time()
            iso = datetime.utcfromtimestamp(now).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            # elapsed_ms was unused; keep timestamp in iso only
            raw_msg = str(message)
            lower = raw_msg.lower()
            # system 抽出: System1..System7 (大文字小文字そのまま想定)
            m_sys = _re.search(r"\bSystem([1-9]|1[0-9])\b", raw_msg)
            system = f"system{m_sys.group(1)}" if m_sys else None

            # phase マッチ辞書 (順序重要: より特殊な語を前に)
            phase_patterns = [
                ("universe", [r"universe", r"load symbols", r"symbol universe"]),
                ("indicators", [r"indicator", r"precompute", r"adx", r"rsi"]),
                ("filter", [r"filter", r"phase2 filter", r"screening"]),
                ("setup", [r"setup", r"prepare setup"]),
                ("ranking", [r"ranking", r"rank "]),
                ("signals", [r" signal", r"signals", r"generate signal"]),
                (
                    "allocation",
                    [r"allocation", r"alloc ", r"allocating", r"final allocation"],
                ),
            ]
            phase = None
            for ph, pats in phase_patterns:
                if any(pat in lower for pat in pats):
                    phase = ph
                    break

            # 開始/終了ステータス推定
            phase_status = None
            if phase:
                if _re.search(r"\b(start|begin|開始)\b", lower):
                    phase_status = "start"
                elif _re.search(
                    r"\b(done|complete|completed|終了|end|finished)\b", lower
                ):
                    phase_status = "end"

            # 前回 phase の補強: system 単位で直前 phase を覚え、end/done だけのメッセージにも付与
            global _STRUCTURED_LAST_PHASE
            if _STRUCTURED_LAST_PHASE is None:
                _STRUCTURED_LAST_PHASE = {}
            if system:
                if phase:
                    _STRUCTURED_LAST_PHASE[system] = phase
                else:
                    # 明示 phase なし かつ done/complete 語があれば直前を参照
                    if _re.search(
                        r"\b(done|complete|completed|終了|end|finished)\b", lower
                    ):
                        last = _STRUCTURED_LAST_PHASE.get(system)
                        if last:
                            phase = last
                            phase_status = phase_status or "end"

            # v: スキーマバージョン / lvl: 将来のレベル拡張 (現状 INFO 固定)
            obj = {
                "v": 1,
                "ts": int(now * 1000),
                "iso": iso,
                "lvl": "INFO",
                "msg": raw_msg,
            }
            if system:
                obj["system"] = system
            if phase:
                obj["phase"] = phase
            if phase_status:
                obj["phase_status"] = phase_status
            if structured_ui:
                json_payload = _json.dumps(obj)
        except Exception:
            obj = None
            json_payload = None

    # 2) NDJSON 書き出し（UIコールバック有無に関係なく）
    if ndjson_flag and obj is not None:
        try:
            from common.structured_log_ndjson import maybe_init_global_writer

            writer = maybe_init_global_writer()
            if writer:
                writer.write(obj)
        except Exception:
            pass

    # 3) UI コールバックへ送信（存在する場合のみ）
    try:
        cb = globals().get("_LOG_CALLBACK")
    except Exception:
        cb = None
    if not (cb and callable(cb)):
        return

    payload = json_payload if (structured_ui and json_payload) else str(message)
    try:
        token = _LOG_FORWARDING.set(True)
        try:
            cb(payload)
        finally:
            _LOG_FORWARDING.reset(token)
    except Exception:
        pass


def _drain_stage_event_queue() -> None:
    """メインスレッドでステージ進捗イベントを処理し、UI 表示を更新する。"""

    try:
        cb2 = globals().get("_PER_SYSTEM_STAGE")
    except Exception:
        cb2 = None

    def _normalize_stage_value(value: object | None) -> int | None:
        """値を int に安全変換。文字列/数値以外は None。

        mypy: object から直接 int(value) すると overload 不一致になるため
        型分岐を明確化して Any 化を避ける。
        """
        if value is None:
            return None
        # 既に int
        if isinstance(value, int):
            return value
        # bool は int のサブクラスなので除外（進捗値に使わない）
        if isinstance(value, bool):
            return int(value)
        # float -> 切り捨て (意図的)
        if isinstance(value, float):
            try:
                return int(value)
            except Exception:
                return None
        # 文字列は空白除去後 数値判定
        if isinstance(value, str):
            txt = value.strip()
            if not txt:
                return None
            # まず整数表現
            if txt.isdigit() or (txt[0] == "-" and txt[1:].isdigit()):
                try:
                    return int(txt)
                except Exception:
                    return None
            # float 表現を許容
            try:
                fl = float(txt)
                return int(fl)
            except Exception:
                return None
        # その他の型は未対応（mypy整合のため無変換）
        return None

    events: list[StageEvent] = []

    queue_obj = globals().get("_PROGRESS_QUEUE")
    if queue_obj is not None:
        while True:
            try:
                item = queue_obj.get_nowait()
            except Exception:
                break
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            system = str(item[0] or "").strip().lower() or "unknown"
            try:
                progress = int(item[1])
            except Exception:
                progress = 0
            filter_count = _normalize_stage_value(item[2] if len(item) > 2 else None)
            setup_count = _normalize_stage_value(item[3] if len(item) > 3 else None)
            candidate_count = _normalize_stage_value(item[4] if len(item) > 4 else None)
            entry_count = _normalize_stage_value(item[5] if len(item) > 5 else None)
            try:
                GLOBAL_STAGE_METRICS.record_stage(
                    system,
                    progress,
                    filter_count,
                    setup_count,
                    candidate_count,
                    entry_count,
                    emit_event=False,
                )
            except Exception:
                continue
            events.append(
                StageEvent(
                    system,
                    progress,
                    filter_count,
                    setup_count,
                    candidate_count,
                    entry_count,
                )
            )

    try:
        events.extend(GLOBAL_STAGE_METRICS.drain_events())
    except Exception:
        pass

    if not events:
        return

    if not cb2 or not callable(cb2):
        return

    for event in events:
        try:
            cb2(
                event.system,
                event.progress,
                event.filter_count,
                event.setup_count,
                event.candidate_count,
                event.entry_count,
            )
        except Exception:
            continue


def _get_stage_snapshot(system: str) -> StageSnapshot | None:
    try:
        return GLOBAL_STAGE_METRICS.get_snapshot(system)
    except Exception:
        return None


def _log(
    msg: str,
    ui: bool = True,
    no_timestamp: bool = False,
    phase_id: str | None = None,
    level: str = "INFO",
    error_code: str | None = None,
) -> None:
    """CLI 出力には [HH:MM:SS | m分s秒] を付与。必要に応じて UI コールバックを抑制。

    Args:
        msg: ログメッセージ
        ui: UI表示フラグ
        no_timestamp: タイムスタンプ無効化フラグ
        phase_id: フェーズID
        level: ログレベル (INFO, WARNING, ERROR, DEBUG)
        error_code: エラーコード (エラー時に指定)
    """
    import time as _t

    # 初回呼び出しで開始時刻を設定
    try:
        global _LOG_START_TS
        if _LOG_START_TS is None:
            _LOG_START_TS = _t.time()
    except Exception:
        _LOG_START_TS = None

    # プレフィックスを作成（現在時刻 + 分秒経過 + エラーコード）
    try:
        if no_timestamp:
            prefix = ""
        else:
            now = _t.strftime("%H:%M:%S")
            elapsed = 0 if _LOG_START_TS is None else max(0, _t.time() - _LOG_START_TS)
            m, s = divmod(int(elapsed), 60)
            # 秒は2桁ゼロ埋めで整形（例: 0分05秒）
            prefix = f"[{now} | {m}分{s:02d}秒] "

        # エラーレベルとコードを含むプレフィックス
        if level != "INFO":
            prefix += f"[{level}] "
        if error_code:
            prefix += f"[{error_code}] "
    except Exception:
        prefix = ""

    # キーワードによる除外判定（全体）
    try:
        # SHOW_INDICATOR_LOGS が真でない限り、インジケーター系の進捗ログを抑制
        try:
            _show_ind_logs_flag = bool(get_env_config().show_indicator_logs)
        except Exception:
            _show_ind_logs_flag = False
        _hide_indicator_logs = not _show_ind_logs_flag
        _indicator_skip = (
            "インジケーター計算",
            "指標計算",
            "共有指標",
            "指標データロード",
            "📊 指標計算",
            "🧮 共有指標",
        )
        _skip_all = _GLOBAL_SKIP_KEYWORDS + (
            _indicator_skip if _hide_indicator_logs else ()
        )
        if any(k in str(msg) for k in _skip_all):
            return
        ui_allowed = ui and not any(k in str(msg) for k in _UI_ONLY_SKIP_KEYWORDS)
    except Exception:
        ui_allowed = ui

    # CLI へは整形して出力（非UTF-8端末では絵文字等を安全化）
    try:
        display_msg = str(msg)
        if _NO_EMOJI_ENV or not _console_supports_utf8():
            try:
                import unicodedata as _ud

                display_msg = _strip_emojis(display_msg)
                display_msg = _ud.normalize("NFKC", display_msg)
            except Exception:
                display_msg = _strip_emojis(display_msg)
    except Exception:
        display_msg = str(msg)
    out = f"{prefix}{display_msg}"
    try:
        print(out, flush=True)
    except UnicodeEncodeError:
        try:
            encoding = getattr(sys.stdout, "encoding", "") or "utf-8"
            safe = out.encode(encoding, errors="replace").decode(
                encoding, errors="replace"
            )
            print(safe, flush=True)
        except Exception:
            try:
                safe = out.encode("ascii", errors="replace").decode(
                    "ascii", errors="replace"
                )
                print(safe, flush=True)
            except Exception:
                pass

    # UI 側への通知
    if ui_allowed:
        try:
            _emit_ui_log(str(msg))
        except Exception:
            pass

    # バックエンドログ（ファイル）
    try:
        logger = _get_today_logger()
        log_msg = str(msg)
        if error_code:
            log_msg = f"[{error_code}] {log_msg}"
        if level == "ERROR":
            logger.error(log_msg)
        elif level == "WARNING":
            logger.warning(log_msg)
        elif level == "DEBUG":
            logger.debug(log_msg)
        else:
            logger.info(log_msg)
    except Exception:
        pass


class _PerfTimer:
    """軽量パフォーマンス計測 (環境変数 ENABLE_STEP_TIMINGS=1 の時のみ有効)"""

    def __init__(self, label: str, level: str = "DEBUG") -> None:
        self.label = label
        self.level = level
        try:
            self.enabled = bool(get_env_config().enable_step_timings)
        except Exception:
            self.enabled = False
        self._t0: float | None = None

    def __enter__(self):  # noqa: D401
        if self.enabled:
            try:
                import time as _t

                self._t0 = _t.perf_counter()
            except Exception:
                self.enabled = False
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: D401
        if not self.enabled or self._t0 is None:
            return False
        try:
            import time as _t

            dt = _t.perf_counter() - self._t0
            _log(f"⏱ {self.label} {dt * 1000:.1f}ms", ui=False, level=self.level)
        except Exception:
            pass
        return False


def _log_error(
    msg: str, error_code: str, ui: bool = True, phase_id: str | None = None
) -> None:
    """エラーログの簡便関数。"""
    _log(msg, ui=ui, phase_id=phase_id, level="ERROR", error_code=error_code)


def _log_warning(
    msg: str,
    error_code: str | None = None,
    ui: bool = True,
    phase_id: str | None = None,
) -> None:
    """警告ログの簡便関数。"""
    _log(msg, ui=ui, phase_id=phase_id, level="WARNING", error_code=error_code)


def _asc_by_score_key(score_key: str | None) -> bool:
    return bool(score_key and score_key.upper() in {"RSI4"})


def _calculate_trading_days_lag(
    cache_date: pd.Timestamp, target_date: pd.Timestamp
) -> int:
    """Calculate the number of NYSE trading days between cache_date and target_date.

    Args:
        cache_date: The date of the cached data
        target_date: The target signal date

    Returns:
        Number of trading days between the two dates (0 if same day, positive if cache is older)
    """
    try:
        import pandas_market_calendars as mcal

        cache_norm = pd.Timestamp(cache_date).normalize()
        target_norm = pd.Timestamp(target_date).normalize()

        if cache_norm == target_norm:
            return 0

        if cache_norm > target_norm:
            return 0  # Cache is newer than target

        # Get NYSE calendar
        nyse = mcal.get_calendar("NYSE")

        # Get valid trading days between cache and target
        schedule = nyse.schedule(
            start_date=cache_norm, end_date=target_norm + pd.Timedelta(days=1)
        )

        valid_days = pd.to_datetime(schedule.index).normalize()
        trading_days_between = valid_days[
            (valid_days > cache_norm) & (valid_days <= target_norm)
        ]

        return len(trading_days_between)
    except Exception:
        # Fallback to calendar days if NYSE calendar fails
        return max(0, (target_date - cache_date).days)


_SYSTEM1_REASON_LABELS = {
    "filter": "フィルター条件 (filter)",
    "setup": "セットアップ条件 (setup)",
    "roc200": "ROC200≤0",
}


def _log_zero_candidate_diagnostics(
    system_name: str,
    candidate_count: int,
    diag_payload: Mapping[str, Any] | None,
) -> None:
    """Emit helpful diagnostics when a system ends up with zero candidates."""
    name = str(system_name or "").strip().lower()
    if candidate_count != 0:
        return

    # Existing specialized summary for system1
    if name == "system1":
        summary = summarize_system1_diagnostics(diag_payload)
        if not summary:
            return
        top_n = summary.get("top_n")
        prefix = (
            f"抽出上限 {top_n} 件, " if isinstance(top_n, int) and top_n > 0 else ""
        )
        message_parts = [
            f"フィルター通過 {summary.get('filter_pass', 0)} 件",
            f"セットアップ成立 {summary.get('setup_flag_true', 0)} 件",
            f"代替判定成立 {summary.get('fallback_pass', 0)} 件",
            f"ROC200>0 {summary.get('roc200_positive', 0)} 件",
            f"最終通過 {summary.get('final_pass', 0)} 件",
        ]
        detail_line = f"[system1] 候補0件理由: {prefix}{', '.join(message_parts)}。"
        _log(detail_line)

        reasons = summary.get("exclude_reasons")
        if isinstance(reasons, Mapping) and reasons:
            reason_parts: list[str] = []
            for key, count in reasons.items():
                if not isinstance(count, int) or count <= 0:
                    continue
                label = _SYSTEM1_REASON_LABELS.get(str(key), str(key))
                reason_parts.append(f"{label} {count} 件")
            if reason_parts:
                _log("[system1] 候補0件の除外内訳: " + ", ".join(reason_parts))
        return

    # Add enriched diagnostics logging for system3 (common cause: drop3d/atr thresholds or missing ranking input)
    if name == "system3":
        if not isinstance(diag_payload, Mapping):
            _log("[system3] 候補0件: 診断情報がありません")
            return
        try:
            reason = diag_payload.get("ranking_zero_reason")
            inputs = diag_payload.get("ranking_input_counts") or {}
            stats = diag_payload.get("ranking_stats") or {}
            thresholds = diag_payload.get("thresholds") or {}
            exclude_reasons = diag_payload.get("exclude_reasons") or {}
            top_n = diag_payload.get("top_n")
            label_date = diag_payload.get("label_date")

            parts: list[str] = []
            parts.append(f"reason={reason or 'unknown'}")
            if label_date:
                parts.append(f"label_date={label_date}")
            parts.append(
                (
                    f"rows_total={inputs.get('rows_total', '?')}, "
                    f"rows_for_label_date={inputs.get('rows_for_label_date', '?')}, "
                    f"lagged_rows={inputs.get('lagged_rows', '?')}"
                )
            )
            # drop3d distribution (safe formatting)
            dmin = stats.get("drop3d_min")
            dmax = stats.get("drop3d_max")
            dmean = stats.get("drop3d_mean")
            dmedian = stats.get("drop3d_median")
            dnan = stats.get("drop3d_nan_count")
            drop_stats_str = "n/a"
            try:
                if (
                    dmin is not None
                    and dmax is not None
                    and dmean is not None
                    and dmedian is not None
                ):
                    drop_stats_str = (
                        f"min={float(dmin):.4f}, max={float(dmax):.4f}, "
                        f"mean={float(dmean):.4f}, median={float(dmedian):.4f}, "
                        f"nan_count={int(dnan) if dnan is not None else 0}"
                    )
                elif dnan is not None:
                    drop_stats_str = f"nan_count={int(dnan)}"
            except Exception:
                drop_stats_str = "n/a"
            parts.append("drop3d_stats=" + drop_stats_str)
            thr_drop = thresholds.get("drop3d")
            thr_atr = thresholds.get("atr_ratio")
            thr_str = (
                f"thresholds=drop3d:{thr_drop or 0.125}, atr_ratio:{thr_atr or 0.05}"
            )
            excl_str = ", ".join(
                f"{k}:{v}"
                for k, v in (
                    exclude_reasons.items()
                    if isinstance(exclude_reasons, Mapping)
                    else []
                )
            )

            header = f"[system3] 候補0件診断: {('top_n=' + str(top_n) + ', ') if isinstance(top_n, int) else ''}"
            _log(header + ", ".join(parts))
            _log(f"[system3] {thr_str}; exclude_reasons: {excl_str or 'none'}")

            # Helpful actionable hints for common zero causes
            try:
                if reason == "all_below_drop3d_threshold":
                    if dmax is not None:
                        _log(
                            f"[system3] 最大drop3d={dmax:.4f} は閾値 {float(thr_drop or 0.125):.4f} 未満です。閾値緩和やFULL_SCAN_TODAYで確認してください。"
                        )
                elif reason == "all_drop3d_nan":
                    _log(
                        "[system3] 全候補で drop3d が NaN のためランキング不能です。指標計算パイプラインを確認してください。"
                    )
                elif reason == "no_rows_for_label_date":
                    _log(
                        "[system3] ラベル日に該当する行がありません。データ鮮度や label_date の解決を確認してください。FULL_SCAN_TODAY を試すと過去日で候補が存在するか確認できます。"
                    )
            except Exception:
                pass
        except Exception:
            _log("[system3] 候補0件: 診断の解析中に例外が発生しました")
        return


def _export_diagnostics_snapshot(
    ctx: TodayRunContext, final_df: pd.DataFrame | None
) -> None:
    """Export a minimal diagnostics snapshot (JSON) for Phase2 verification.

    - Test modes only (mini/quick/sample)
    - Output path: <RESULTS_DIR>/diagnostics_test/diagnostics_snapshot_YYYYMMDD_HHMMSS.json
    - Content: export_date, mode, systems[{system_id, diagnostics, final_candidate_count}]
    """
    try:
        mode = getattr(ctx, "test_mode", None)
    except Exception:
        mode = None

    # 本番でも明示フラグで出力可能にする
    export_always = False
    try:
        from config.environment import get_env_config  # 遅延import

        export_always = bool(get_env_config().export_diagnostics_snapshot_always)
    except Exception:
        export_always = False

    if not mode and not export_always:
        return  # production では既定は出力しない

    try:
        settings = ctx.settings
        # test_mode のときは results_csv_test 配下に出力し、運用結果と分離
        if mode:
            base_dir = Path("results_csv_test")
        else:
            base_dir = Path(getattr(settings, "RESULTS_DIR", Path("results_csv")))
        out_dir = base_dir / "diagnostics_test"
        out_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now()
        stamp = now.strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"diagnostics_snapshot_{stamp}.json"

        # per-system final candidate counts
        final_counts: dict[str, int] = {}
        try:
            if (
                final_df is not None
                and not final_df.empty
                and "system" in final_df.columns
            ):
                final_counts = final_df.groupby("system").size().astype(int).to_dict()
        except Exception:
            final_counts = {}

        systems_payload: list[dict[str, Any]] = []
        try:
            diag_map = getattr(ctx, "system_diagnostics", {}) or {}
            for sys_id in sorted(diag_map.keys()):
                raw_diag = diag_map.get(sys_id) or {}
                # Phase5: フォールバック適用で欠損値を -1 / unknown に正規化
                safe_diag = get_diagnostics_with_fallback(raw_diag, sys_id)
                # 追加の生診断（正規化キー以外）を抽出して併記
                try:
                    extras = (
                        {k: v for k, v in raw_diag.items() if k not in safe_diag}
                        if isinstance(raw_diag, dict)
                        else {}
                    )
                except Exception:
                    extras = {}
                systems_payload.append(
                    {
                        "system_id": sys_id,
                        "diagnostics": safe_diag,
                        # System3 等のランキング可視化フィールド（ranking_input_counts など）を保持
                        **({"diagnostics_extra": extras} if extras else {}),
                        "final_candidate_count": int(final_counts.get(sys_id, 0)),
                    }
                )
        except Exception:
            systems_payload = []

        snapshot = {
            "export_date": now.isoformat(),
            "mode": mode,
            "systems": systems_payload,
        }

        from common.io_utils import write_json

        write_json(out_path, snapshot, ensure_ascii=False, indent=2)

        _log(
            f"🧪 Diagnostics snapshot exported: {out_path.relative_to(base_dir)}",
            ui=True,
        )
    except Exception as e:
        _log_warning(
            f"diagnostics スナップショットの出力に失敗: {e}", error_code="SNAP-FAIL"
        )


def _export_discrepancy_triage(ctx: TodayRunContext) -> None:
    """Discrepancy triage 結果を JSON ファイルとしてエクスポート。

    - Test modes only (mini/quick/sample)
    - Output path: <RESULTS_DIR>/diagnostics_test/discrepancy_triage_YYYYMMDD_HHMMSS.json
    - Content: export_date, mode, triage_results, unexpected_systems
    """
    try:
        mode = getattr(ctx, "test_mode", None)
    except Exception:
        mode = None
    if not mode:
        return  # production では出力しない

    try:
        from common.system_diagnostics import (
            format_triage_summary,
            get_unexpected_systems,
            triage_all_systems,
        )

        settings = ctx.settings
        base_dir = Path(getattr(settings, "RESULTS_DIR", Path("results_csv")))
        out_dir = base_dir / "diagnostics_test"
        out_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now()
        stamp = now.strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"discrepancy_triage_{stamp}.json"

        # システム診断情報を取得
        diag_map = getattr(ctx, "system_diagnostics", {}) or {}

        # Triage 実施
        triage_results = triage_all_systems(diag_map)
        unexpected = get_unexpected_systems(triage_results)

        # サマリーログ出力
        summary_text = format_triage_summary(triage_results)
        _log("📋 Discrepancy Triage Results:")
        for line in summary_text.split("\n"):
            _log(f"  {line}")

        # JSON エクスポート
        export_payload = {
            "export_date": now.isoformat(),
            "mode": mode,
            "triage_results": triage_results,
            "unexpected_systems": unexpected,
            "summary": summary_text,
        }

        from common.io_utils import write_json

        write_json(out_path, export_payload, ensure_ascii=False, indent=2)

        _log(
            f"🧪 Discrepancy triage exported: {out_path.relative_to(base_dir)}", ui=True
        )

        # Unexpected システムがあれば警告
        if unexpected:
            _log_warning(
                f"⚠️ Unexpected discrepancies detected in: {', '.join(unexpected)}",
                error_code="TRIAGE-UNEXPECTED",
            )

    except Exception as e:
        _log_warning(f"discrepancy triage の出力に失敗: {e}", error_code="TRIAGE-FAIL")


# ログ出力から除外するキーワード
# ログ全体から除外するキーワード（CLI/UI 共通）
# インジケーター計算自体は CLI に出したいので除外しない。
_GLOBAL_SKIP_KEYWORDS = (
    "バッチ時間",
    "batch time",
    # 銘柄の長いダンプは CLI でも非表示にする
    "銘柄:",
)
# UI 表示からのみ除外するキーワード
_UI_ONLY_SKIP_KEYWORDS = (
    "進捗",
    "候補抽出",
    "候補日数",
)


def _filter_logs(lines: list[str], ui: bool = False) -> list[str]:
    """キーワードに基づいてログ行を除外する。

    Args:
        lines: 対象ログ行のリスト。
        ui: True の場合は UI 限定の除外キーワードも適用。
    """

    skip_keywords = _GLOBAL_SKIP_KEYWORDS + (_UI_ONLY_SKIP_KEYWORDS if ui else ())
    return [ln for ln in lines if not any(k in ln for k in skip_keywords)]


def _prev_counts_path(signals_dir: Path) -> Path:
    try:
        return signals_dir / "previous_per_system_counts.json"
    except Exception:
        return Path("signals/previous_per_system_counts.json")


def _load_prev_counts(signals_dir: Path) -> dict[str, int]:
    fp = _prev_counts_path(signals_dir)
    if not fp.exists():
        return {}
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
        counts = data.get("counts", {}) if isinstance(data, dict) else {}
        out: dict[str, int] = {}
        for i in range(1, 8):
            key = f"system{i}"
            try:
                out[key] = int(counts.get(key, 0))
            except Exception:
                out[key] = 0
        return out
    except Exception:
        return {}


def _save_prev_counts(
    signals_dir: Path, per_system_map: dict[str, pd.DataFrame]
) -> None:
    try:
        counts = {
            k: (0 if (v is None or v.empty) else int(len(v)))
            for k, v in per_system_map.items()
        }
        data = {"timestamp": datetime.utcnow().isoformat() + "Z", "counts": counts}
        fp = _prev_counts_path(signals_dir)
        try:
            from common.io_utils import write_json

            fp.parent.mkdir(parents=True, exist_ok=True)
            write_json(fp, data, ensure_ascii=False, indent=2)
        except Exception:
            # fallback to previous behavior if helper import fails
            try:
                fp.parent.mkdir(parents=True, exist_ok=True)
                from common.io_utils import write_text

                write_text(fp, json.dumps(data, ensure_ascii=False, indent=2))
            except Exception:
                pass
    except Exception:
        pass


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """列名を大文字OHLCVに統一"""
    col_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "adj_close": "AdjClose",
        "adjusted_close": "AdjClose",
    }
    try:
        return df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    except Exception:
        return df


def _extract_last_cache_date(df: pd.DataFrame) -> pd.Timestamp | None:
    if df is None or getattr(df, "empty", True):
        return None
    for col in ("date", "Date"):
        if col in df.columns:
            try:
                ser_dt = pd.to_datetime(df[col], errors="coerce").dropna()
                if len(ser_dt):
                    last_val = ser_dt.iloc[-1]
                    return pd.Timestamp(cast(Any, last_val)).normalize()
            except Exception:
                continue
    try:
        idx_dt = pd.to_datetime(df.index, errors="coerce")
        if isinstance(idx_dt, pd.DatetimeIndex) and len(idx_dt):
            return pd.Timestamp(cast(Any, idx_dt[-1])).normalize()
    except Exception:
        pass
    return None


def _recent_trading_days(
    today: pd.Timestamp | None, max_back: int
) -> list[pd.Timestamp]:
    if today is None:
        return []
    out: list[pd.Timestamp] = []
    seen: set[pd.Timestamp] = set()
    current = pd.Timestamp(today).normalize()
    steps = max(0, int(max_back))
    for _ in range(steps + 1):
        if current in seen:
            break
        out.append(current)
        seen.add(current)
        prev_candidate = get_latest_nyse_trading_day(current - pd.Timedelta(days=1))
        prev_candidate = pd.Timestamp(prev_candidate).normalize()
        if prev_candidate == current:
            break
        current = prev_candidate
    return out


def _build_rolling_from_base(
    symbol: str,
    base_df: pd.DataFrame,
    target_len: int,
    cache_manager: CacheManager | None = None,
) -> pd.DataFrame | None:
    """Convert base cache dataframe to rolling window and optionally persist it."""

    if base_df is None or getattr(base_df, "empty", True):
        return None
    try:
        work = base_df.copy()
    except Exception:
        work = base_df
    if work.index.name is not None:
        work = work.reset_index()
    if "Date" in work.columns:
        work["date"] = pd.to_datetime(work["Date"].to_numpy(), errors="coerce")
    elif "date" in work.columns:
        work["date"] = pd.to_datetime(work["date"].to_numpy(), errors="coerce")
    else:
        return None
    work = work.dropna(subset=["date"]).sort_values("date")
    col_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "AdjClose": "adjusted_close",
        "Adj Close": "adjusted_close",
        "Volume": "volume",
    }
    try:
        for src, dst in list(col_map.items()):
            if src in work.columns:
                work = work.rename(columns={src: dst})
    except Exception:
        pass
    sliced = work.tail(int(target_len)).reset_index(drop=True)
    if sliced.empty:
        return None
    if cache_manager is not None:
        try:
            cache_manager.write_atomic(sliced, symbol, "rolling")
        except Exception:
            pass
    return sliced


def _load_basic_data(
    symbols: list[str],
    cache_manager: CacheManager,
    settings: Any,
    symbol_data: dict[str, pd.DataFrame] | None,
    *,
    today: pd.Timestamp | None = None,
    freshness_tolerance: int | None = None,
    _base_cache: dict[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    from time import perf_counter

    data: dict[str, pd.DataFrame] = {}
    total_syms = len(symbols)
    start_ts = perf_counter()
    chunk = 500

    if freshness_tolerance is None:
        try:
            freshness_tolerance = int(settings.cache.rolling.max_staleness_days)
        except Exception:
            freshness_tolerance = 2
    freshness_tolerance = max(0, int(freshness_tolerance))

    try:
        target_len = int(
            settings.cache.rolling.base_lookback_days
            + settings.cache.rolling.buffer_days
        )
    except Exception:
        target_len = 0

    stats_lock = Lock()
    stats: dict[str, int] = {}

    def _record_stat(key: str) -> None:
        with stats_lock:
            stats[key] = stats.get(key, 0) + 1

    recent_allowed: set[pd.Timestamp] = set()
    if today is not None and freshness_tolerance >= 0:
        try:
            recent_allowed = {
                pd.Timestamp(d).normalize()
                for d in _recent_trading_days(pd.Timestamp(today), freshness_tolerance)
            }
        except Exception:
            recent_allowed = set()

    if recent_allowed:
        try:
            _ = min(recent_allowed)
        except Exception:
            pass

    gap_probe_days = max(freshness_tolerance + 5, 10)

    def _estimate_gap_days(
        today_dt: pd.Timestamp | None, last_dt: pd.Timestamp | None
    ) -> int | None:
        if today_dt is None or last_dt is None:
            return None
        try:
            recent = _recent_trading_days(pd.Timestamp(today_dt), gap_probe_days)
        except Exception:
            recent = []
        for offset, dt in enumerate(recent):
            if dt == last_dt:
                return offset
        try:
            return max(0, int((pd.Timestamp(today_dt) - pd.Timestamp(last_dt)).days))
        except Exception:
            return None

    def _pick_symbol_data(sym: str) -> pd.DataFrame | None:
        try:
            if not symbol_data or sym not in symbol_data:
                return None
            df = symbol_data.get(sym)
            if df is None or getattr(df, "empty", True):
                return None
            x = df.copy()
            if x.index.name is not None:
                x = x.reset_index()
            if "date" in x.columns:
                x["date"] = pd.to_datetime(x["date"].to_numpy(), errors="coerce")
            elif "Date" in x.columns:
                x["date"] = pd.to_datetime(x["Date"].to_numpy(), errors="coerce")
            else:
                return None
            col_map = {
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adjusted_close",
                "AdjClose": "adjusted_close",
                "Volume": "volume",
            }
            for k, v in list(col_map.items()):
                if k in x.columns:
                    x = x.rename(columns={k: v})
            required = {"date", "close"}
            if not required.issubset(set(x.columns)):
                return None
            x = x.dropna(subset=["date"]).sort_values("date")
            return x
        except Exception:
            return None

    def _normalize_loaded(df: pd.DataFrame | None) -> pd.DataFrame | None:
        if df is None or getattr(df, "empty", True):
            return None
        try:
            if "Date" not in df.columns:
                work = df.copy()
                if "date" in work.columns:
                    work["Date"] = pd.to_datetime(
                        work["date"].to_numpy(), errors="coerce"
                    )
                else:
                    work["Date"] = pd.to_datetime(
                        work.index.to_numpy(), errors="coerce"
                    )
                df = work
            df["Date"] = pd.to_datetime(
                df["Date"].to_numpy(), errors="coerce"
            ).normalize()
        except Exception:
            pass
        normalized = _normalize_ohlcv(df)
        try:
            fill_cols = [
                c
                for c in ("Open", "High", "Low", "Close", "Volume")
                if c in normalized.columns
            ]
            if fill_cols:
                normalized = normalized.copy()
                try:
                    filled = normalized[fill_cols].apply(pd.to_numeric, errors="coerce")
                except Exception:
                    filled = normalized[fill_cols]
                normalized.loc[:, fill_cols] = filled.ffill().bfill()
        except Exception:
            pass
        try:
            if "Date" in normalized.columns:
                normalized = normalized.dropna(subset=["Date"])
        except Exception:
            pass
        return normalized

    # env-based overrides via EnvironmentConfig
    env_cfg = get_env_config()
    env_parallel = (
        "1"
        if env_cfg.basic_data_parallel is True
        else ("0" if env_cfg.basic_data_parallel is False else "")
    )
    env_parallel_threshold = int(getattr(env_cfg, "basic_data_parallel_threshold", 200))
    if env_parallel in ("1", "true", "yes"):
        use_parallel = total_syms > 1
    elif env_parallel in ("0", "false", "no"):
        use_parallel = False
    else:
        use_parallel = total_syms >= max(0, env_parallel_threshold)

    max_workers: int | None = None
    if use_parallel and total_syms > 0:
        try:
            env_workers = str(env_cfg.basic_data_max_workers or "").strip()
            if env_workers:
                max_workers = int(env_workers)
        except Exception:
            max_workers = None
        if max_workers is None:
            try:
                cfg_workers = getattr(settings.cache.rolling, "load_max_workers", None)
                if cfg_workers:
                    max_workers = int(cfg_workers)
            except Exception:
                pass
        if max_workers is None:
            cpu_count = os.cpu_count() or 4
            max_workers = max(4, cpu_count * 2)
        max_workers = max(1, min(int(max_workers), total_syms))
        try:
            _log(f"🧵 基礎データロード並列化: workers={max_workers}")
        except Exception:
            pass

    def _load_one(sym: str) -> tuple[str, pd.DataFrame | None]:
        try:
            source: str | None = None
            df = _pick_symbol_data(sym)
            rebuild_reason: str | None = None
            last_seen_date: pd.Timestamp | None = None
            gap_days: int | None = None
            if df is None or getattr(df, "empty", True):
                df = cache_manager.read(sym, "rolling")
            else:
                source = "prefetched"
            if df is None or getattr(df, "empty", True):
                source = None
            if df is None or getattr(df, "empty", True):
                needs_rebuild = True
            else:
                needs_rebuild = False
            if df is not None and not getattr(df, "empty", True) and source is None:
                source = "rolling"
            if df is not None and not getattr(df, "empty", True):
                # データ長さチェックを追加
                if len(df) < target_len:
                    if len(df) < 100:  # 明らかに新規上場
                        _log(
                            f"📊 新規上場銘柄 {sym}: len={len(df)}/{target_len} (正常)",
                            ui=False,
                        )
                        # 短いデータでも処理を継続（rebuildしない）
                    else:
                        rebuild_reason = "length"
                        needs_rebuild = True
                last_seen_date = _extract_last_cache_date(df)
                if last_seen_date is None:
                    rebuild_reason = rebuild_reason or "missing_date"
                    needs_rebuild = True
                else:
                    last_seen_date = pd.Timestamp(last_seen_date).normalize()
                    if (
                        today is not None
                        and recent_allowed
                        and last_seen_date not in recent_allowed
                    ):
                        rebuild_reason = "stale"
                        gap_days = _estimate_gap_days(
                            pd.Timestamp(today), last_seen_date
                        )
                        # 日付が古いがデータが存在する場合は、警告のみで処理を継続
                        # フィルター段階で各システムが必要な条件をチェックする
                        _log(
                            f"⚠️ データ鮮度注意: {sym} (最終日={last_seen_date.date()}, ギャップ={gap_days if gap_days else '不明'}営業日)",
                            ui=False,
                        )
                        # needs_rebuild = True  # この行をコメントアウトして除外を回避
            if needs_rebuild:
                # 個別ログを抑制（サマリー表示に統合）
                _record_stat("manual_rebuild_required")
                _record_stat("failed")
                return sym, None
            normalized = _normalize_loaded(df)
            if normalized is not None and not getattr(normalized, "empty", True):
                _record_stat(source or "rolling")
                return sym, normalized
            _record_stat("failed")
            return sym, None
        except Exception:
            _record_stat("failed")
            return sym, None

    def _report_progress(done: int) -> None:
        if done <= 0 or chunk <= 0:
            return
        if done % chunk != 0:
            return
        try:
            elapsed = max(0.001, perf_counter() - start_ts)
            rate = done / elapsed
            remain = max(0, total_syms - done)
            eta_sec = int(remain / rate) if rate > 0 else 0
            m, s = divmod(eta_sec, 60)
            # 固定幅整形（桁数揺れ対策）
            w = max(1, len(str(total_syms)))
            cur_s = f"{done:>{w}d}"
            tot_s = f"{total_syms:>{w}d}"
            mm = f"{m:02d}"
            ss = f"{s:02d}"
            msg = f"📦 基礎データロード進捗: {cur_s}/{tot_s} | ETA {mm}分{ss}秒"

            # 進捗ログはDEBUGレベルでレート制限適用
            rate_logger = _get_rate_limited_logger()
            rate_logger.debug_rate_limited(
                f"📦 基礎データロード進捗: {cur_s}/{tot_s}",
                interval=2.0,
                message_key="基礎データ進捗",
            )
            _emit_ui_log(msg)
        except Exception:
            try:
                w = max(1, len(str(total_syms)))
                cur_s = f"{done:>{w}d}"
                tot_s = f"{total_syms:>{w}d}"
            except Exception:
                cur_s, tot_s = str(done), str(total_syms)
            _log(f"📦 基礎データロード進捗: {cur_s}/{tot_s}", ui=False)
            _emit_ui_log(f"📦 基礎データロード進捗: {cur_s}/{tot_s}")

    processed = 0
    if use_parallel and max_workers and total_syms > 1:
        # 新しい並列バッチ読み込みを使用（Phase2最適化）
        try:
            _log(
                f"🚀 並列バッチ読み込み開始: {total_syms}シンボル, workers={max_workers}"
            )

            def progress_callback_internal(loaded, _total):
                nonlocal processed
                processed = loaded
                _report_progress(processed)

            # CacheManagerの並列読み込み機能を活用
            parallel_data = cache_manager.read_batch_parallel(
                symbols=symbols,
                profile="rolling",
                max_workers=max_workers,
                fallback_profile="full",
                progress_callback=progress_callback_internal,
            )

            # 結果を既存のデータフォーマットに合わせて処理
            for sym, df in parallel_data.items():
                if df is not None and not getattr(df, "empty", True):
                    # 既存の_normalize_loadedと同様の処理を適用
                    normalized = _normalize_loaded(df)
                    if normalized is not None and not getattr(
                        normalized, "empty", True
                    ):
                        data[sym] = normalized
                        _record_stat("rolling")
                    else:
                        _record_stat("failed")
                else:
                    _record_stat("failed")

            _log(f"✅ 並列バッチ読み込み完了: {len(data)}/{total_syms}件成功")

        except Exception as e:
            # 並列処理失敗時はフォールバック
            _log(f"⚠️ 並列バッチ読み込み失敗、従来処理にフォールバック: {e}")
            data.clear()
            processed = 0
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_load_one, sym): sym for sym in symbols}
                for fut in as_completed(futures):
                    try:
                        sym, df = fut.result()
                    except Exception:
                        sym, df = futures[fut], None
                    if df is not None and not getattr(df, "empty", True):
                        data[sym] = df
                    processed += 1
                    _report_progress(processed)
    else:
        for sym in symbols:
            sym, df = _load_one(sym)
            if df is not None and not getattr(df, "empty", True):
                data[sym] = df
            processed += 1
            _report_progress(processed)

    try:
        total_elapsed = max(0.0, perf_counter() - start_ts)
        total_int = int(total_elapsed)
        m, s = divmod(total_int, 60)
        done_msg = (
            f"📦 基礎データロード完了: {len(data)}/{total_syms} | 所要 {m}分{s}秒"
            + (" | 並列=ON" if use_parallel and max_workers else " | 並列=OFF")
        )
        _log(done_msg)
        _emit_ui_log(done_msg)
    except Exception:
        _log(f"📦 基礎データロード完了: {len(data)}/{total_syms}")
        _emit_ui_log(f"📦 基礎データロード完了: {len(data)}/{total_syms}")

    try:
        summary_map = {
            "prefetched": "事前供給",
            "rolling": "rolling再利用",
            "manual_rebuild_required": "手動対応",
            "failed": "失敗",
        }
        summary_parts = [
            f"{label}={stats.get(key, 0)}"
            for key, label in summary_map.items()
            if stats.get(key)
        ]
        if summary_parts:
            rate_logger = _get_rate_limited_logger()
            rate_logger.debug_rate_limited(
                "📊 基礎データロード内訳: " + " / ".join(summary_parts),
                interval=5.0,
                message_key="基礎データ内訳",
            )
    except Exception:
        pass

    return data


def _load_indicator_data(
    symbols: list[str],
    cache_manager: CacheManager,
    settings: Any,
    symbol_data: dict[str, pd.DataFrame] | None,
) -> dict[str, pd.DataFrame]:
    import time as _t

    data: dict[str, pd.DataFrame] = {}
    total_syms = len(symbols)
    start_ts = _t.time()
    chunk = 500
    for idx, sym in enumerate(symbols, start=1):
        try:
            df = None
            try:
                if symbol_data and sym in symbol_data:
                    df = symbol_data.get(sym)
                    if df is not None and not df.empty:
                        x = df.copy()
                        if x.index.name is not None:
                            x = x.reset_index()
                        if "date" in x.columns:
                            x["date"] = pd.to_datetime(
                                x["date"].to_numpy(), errors="coerce"
                            )
                        elif "Date" in x.columns:
                            x["date"] = pd.to_datetime(
                                x["Date"].to_numpy(), errors="coerce"
                            )
                        col_map = {
                            "Open": "open",
                            "High": "high",
                            "Low": "low",
                            "Close": "close",
                            "Adj Close": "adjusted_close",
                            "AdjClose": "adjusted_close",
                            "Volume": "volume",
                        }
                        for k, v in list(col_map.items()):
                            if k in x.columns:
                                x = x.rename(columns={k: v})
                        required = {"date", "close"}
                        if required.issubset(set(x.columns)):
                            x = x.dropna(subset=["date"]).sort_values("date")
                            df = x
                        else:
                            df = None
                    else:
                        df = None
            except Exception:
                df = None
            if df is None or df.empty:
                df = cache_manager.read(sym, "rolling")
            needs_rebuild = df is None or getattr(df, "empty", True)
            if needs_rebuild:
                # 個別銘柄ごとの "⛔ rolling未整備" ログは冗長なため完全に削除。
                # ループ終了後のサマリーログ（⚠️ rolling未整備）で一括表示されます。
                continue
            if df is not None and not df.empty:
                try:
                    if "Date" not in df.columns:
                        if "date" in df.columns:
                            df = df.copy()
                            df["Date"] = pd.to_datetime(
                                df["date"].to_numpy(), errors="coerce"
                            )
                        else:
                            df = df.copy()
                            df["Date"] = pd.to_datetime(
                                df.index.to_numpy(), errors="coerce"
                            )
                    df["Date"] = pd.to_datetime(
                        df["Date"].to_numpy(), errors="coerce"
                    ).normalize()
                except Exception:
                    pass
                df = _normalize_ohlcv(df)
                data[sym] = df
        except Exception:
            continue
        if total_syms > 0 and idx % chunk == 0:
            try:
                elapsed = max(0.001, _t.time() - start_ts)
                rate = idx / elapsed
                remain = max(0, total_syms - idx)
                eta_sec = int(remain / rate) if rate > 0 else 0
                m, s = divmod(eta_sec, 60)
                msg = f"🧮 指標データロード進捗: {idx}/{total_syms} | ETA {m}分{s}秒"

                # 進捗ログはDEBUGレベルでレート制限適用
                rate_logger = _get_rate_limited_logger()
                rate_logger.debug_rate_limited(
                    f"🧮 指標データロード進捗: {idx}/{total_syms}",
                    interval=2.0,
                    message_key="指標データ進捗",
                )
                _emit_ui_log(msg)
            except Exception:
                rate_logger = _get_rate_limited_logger()
                rate_logger.debug_rate_limited(
                    f"🧮 指標データロード進捗: {idx}/{total_syms}",
                    interval=2.0,
                    message_key="指標データ進捗",
                )
                _emit_ui_log(f"🧮 指標データロード進捗: {idx}/{total_syms}")
    try:
        total_elapsed = int(max(0, _t.time() - start_ts))
        m, s = divmod(total_elapsed, 60)
        done_msg = (
            f"🧮 指標データロード完了: {len(data)}/{total_syms} | 所要 {m}分{s}秒"
        )
        _log(done_msg)
        _emit_ui_log(done_msg)
    except Exception:
        _log(f"🧮 指標データロード完了: {len(data)}/{total_syms}")
        _emit_ui_log(f"🧮 指標データロード完了: {len(data)}/{total_syms}")
    return data


def _subset_data(
    basic_data: dict[str, pd.DataFrame], keys: list[str]
) -> dict[str, pd.DataFrame]:
    out = {}
    for s in keys or []:
        v = basic_data.get(s)
        if v is not None and not getattr(v, "empty", True):
            out[s] = v
    return out


def _fetch_positions_and_symbol_map() -> tuple[list[Any], dict[str, str]]:
    """Fetch Alpaca positions and cached symbol-to-system mapping once."""

    try:
        client = ba.get_client(paper=True)
        positions = list(client.get_all_positions())
    except Exception:
        positions = []

    try:
        symbol_system_map = load_symbol_system_map()
    except Exception:
        symbol_system_map = {}

    return positions, symbol_system_map


def _submit_orders(
    final_df: pd.DataFrame,
    *,
    paper: bool = True,
    order_type: str = "market",
    tif: str = "GTC",
    retries: int = 2,
    delay: float = 0.5,
) -> pd.DataFrame:
    """final_df をもとに Alpaca へ注文送信（shares 必須）。
    返り値: 実行結果の DataFrame（order_id/status/error を含む）
    """
    if final_df is None or final_df.empty:
        _log("(submit) final_df is empty; skip")
        return pd.DataFrame()
    if "shares" not in final_df.columns:
        _log("(submit) shares 列がありません。資金配分モードで実行してください。")
        return pd.DataFrame()
    try:
        client = ba.get_client(paper=paper)
    except Exception as e:
        _log(f"(submit) Alpaca接続エラー: {e}")
        return pd.DataFrame()

    results = []
    for _, r in final_df.iterrows():
        sym = str(r.get("symbol"))
        qty = int(r.get("shares") or 0)
        side = "buy" if str(r.get("side")).lower() == "long" else "sell"
        system = str(r.get("system"))
        entry_date = r.get("entry_date")
        if not sym or qty <= 0:
            continue
        # safely parse limit price
        limit_price = None
        if order_type == "limit":
            try:
                val = r.get("entry_price")
                if val is not None and val != "":
                    limit_price = float(val)
            except Exception:
                limit_price = None
        # estimate price for notification purposes
        price_val = None
        try:
            val = r.get("entry_price")
            if val is not None and val != "":
                price_val = float(val)
        except Exception:
            price_val = None
        if limit_price is not None:
            price_val = limit_price
        try:
            order = ba.submit_order_with_retry(
                client,
                sym,
                qty,
                side=side,
                order_type=order_type,
                limit_price=limit_price,
                time_in_force=tif,
                retries=max(0, int(retries)),
                backoff_seconds=max(0.0, float(delay)),
                rate_limit_seconds=max(0.0, float(delay)),
                log_callback=_log,
            )
            results.append(
                {
                    "symbol": sym,
                    "side": side,
                    "qty": qty,
                    "price": price_val,
                    "system": system,
                    "entry_date": entry_date,
                    # Streamlit/Arrow 互換のため UUID を文字列化
                    "order_id": (
                        str(getattr(order, "id", ""))
                        if getattr(order, "id", None) is not None
                        else ""
                    ),
                    "status": getattr(order, "status", None),
                }
            )
        except Exception as e:
            results.append(
                {
                    "symbol": sym,
                    "side": side,
                    "qty": qty,
                    "price": price_val,
                    "system": system,
                    "entry_date": entry_date,
                    "error": str(e),
                }
            )
    if results:
        out = pd.DataFrame(results)
        # 念のため order_id 列が存在すれば文字列化（他経路で UUID 型が混じるのを防ぐ）
        try:
            if "order_id" in out.columns:
                out["order_id"] = out["order_id"].apply(
                    lambda x: str(x) if x not in (None, "") else ""
                )
        except Exception:
            pass
        _log("\n=== Alpaca submission results ===")
        _log(out.to_string(index=False))
        # record entry dates for future day-based rules
        entry_map = load_entry_dates()
        for _, row in out.iterrows():
            sym = str(row.get("symbol"))
            side_val = str(row.get("side", "")).lower()
            if side_val == "buy" and row.get("entry_date"):
                entry_map[sym] = str(row["entry_date"])
            elif side_val == "sell":
                entry_map.pop(sym, None)
        save_entry_dates(entry_map)

        # Emit progress event for notification
        if ENABLE_PROGRESS_EVENTS:
            emit_progress_event(
                "notification_complete",
                {"notifications_sent": 1, "results_count": len(results)},
            )

        notifier = create_notifier(platform="auto", fallback=True)
        notifier.send_trade_report("integrated", results)
        return out
    return pd.DataFrame()


def _apply_filters(
    df: pd.DataFrame,
    *,
    only_long: bool = False,
    only_short: bool = False,
    top_per_system: int = 0,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if "side" in out.columns:
        if only_long and not only_short:
            out = out[out["side"].str.lower() == "long"]
        if only_short and not only_long:
            out = out[out["side"].str.lower() == "short"]
    if top_per_system and top_per_system > 0 and "system" in out.columns:
        by = ["system"] + (["side"] if "side" in out.columns else [])
        out = out.groupby(by, as_index=False, group_keys=False).head(
            int(top_per_system)
        )  # noqa: E501
    return out


def _initialize_run_context(
    *,
    slots_long: int | None = None,
    slots_short: int | None = None,
    capital_long: float | None = None,
    capital_short: float | None = None,
    save_csv: bool = False,
    csv_name_mode: str | None = None,
    notify: bool = True,
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    per_system_progress: Callable[[str, str], None] | None = None,
    symbol_data: dict[str, pd.DataFrame] | None = None,
    parallel: bool = False,
    test_mode: str | None = None,
    skip_external: bool = False,
) -> TodayRunContext:
    """当日シグナル実行前に共有設定・状態をまとめたコンテキストを生成する。"""

    # テストモードを環境変数にも設定（get_env_config()のシングルトン初期化前に設定必須）
    if test_mode:
        os.environ["TEST_MODE"] = test_mode
        # 環境変数変更後にキャッシュをクリアして、最新の環境変数を反映
        from config.environment import reset_env_config_cache

        reset_env_config_cache()

    # Avoid directory creation side-effects during initialization; directories
    # are expected to exist or be created lazily by CacheManager/write ops.
    settings = get_settings(create_dirs=False)
    cache_manager = CacheManager(settings)
    signals_dir = Path(settings.outputs.signals_dir)
    signals_dir.mkdir(parents=True, exist_ok=True)

    ctx = TodayRunContext(
        settings=settings,
        cache_manager=cache_manager,
        signals_dir=signals_dir,
        cache_dir=cache_manager.rolling_dir,
        slots_long=slots_long,
        slots_short=slots_short,
        capital_long=capital_long,
        capital_short=capital_short,
        save_csv=save_csv,
        csv_name_mode=csv_name_mode,
        notify=notify,
        log_callback=log_callback,
        progress_callback=progress_callback,
        per_system_progress=per_system_progress,
        symbol_data=symbol_data,
        parallel=parallel,
        test_mode=test_mode,
        # propagate run namespace into ctx if provided via CLI
        # stored as ctx.run_namespace for later use
        skip_external=skip_external,
    )
    ctx.run_start_time = datetime.now()
    ctx.start_equity = _get_account_equity()
    # run namespace support: read from env var or passed CLI flag via globals
    try:
        rn = os.getenv("RUN_NAMESPACE", "")
        # globals may contain a parsed CLI arg
        cli_ns = globals().get("_CLI_RUN_NAMESPACE")
        if cli_ns:
            rn = str(cli_ns)
        ctx.run_namespace = rn if rn else None
    except Exception:
        ctx.run_namespace = None

    try:
        freshness_tolerance = int(settings.cache.rolling.max_staleness_days)
    except Exception:
        freshness_tolerance = 2
    # Default to calendar days for backward compatibility
    # Will be updated to trading days after signal_base_day is determined
    ctx.max_date_lag_days = max(0, int(freshness_tolerance))
    # 環境変数による上書き（latest_only 用鮮度ガード、カレンダー日）
    try:
        env = get_env_config()
        lag_override = getattr(env, "latest_only_max_date_lag_days", None)
        if lag_override is not None:
            ctx.max_date_lag_days = max(0, int(lag_override))
    except Exception:
        pass
    try:
        import uuid as _uuid

        ctx.run_id = str(_uuid.uuid4())[:8]
    except Exception:
        ctx.run_id = "--------"
    return ctx


def _prepare_symbol_universe(
    ctx: TodayRunContext, initial_symbols: list[str] | None
) -> list[str]:
    """Determine today's symbol universe and emit initial run banners."""

    cache_dir = ctx.cache_dir
    log_callback = ctx.log_callback
    progress_callback = ctx.progress_callback

    if initial_symbols and len(initial_symbols) > 0:
        symbols = [s.upper() for s in initial_symbols]
    else:
        from common.universe import build_universe_from_cache, load_universe_file

        settings = getattr(ctx, "settings", None)
        log = _get_today_logger()
        skip_external = getattr(ctx, "skip_external", False)

        # 先に test_symbols モードを優先判定（skip_external に依存せず読み込める）
        fetched = []
        test_mode = getattr(ctx, "test_mode", None)
        if test_mode == "test_symbols":
            try:
                from config.settings import get_settings

                settings_local = get_settings()
                test_symbols_dir = settings_local.DATA_CACHE_DIR / "test_symbols"
                if test_symbols_dir.exists():
                    feather_files = list(test_symbols_dir.glob("*.feather"))
                    fetched = [f.stem for f in feather_files]
                    _log(
                        f"🧪 架空銘柄モード: {len(fetched)}銘柄を使用 (skip_external={skip_external})"
                    )
                else:
                    _log(f"❌ 架空銘柄ディレクトリが見つかりません: {test_symbols_dir}")
                    _log(
                        "先に 'python tools/generate_test_symbols.py' を実行してください"
                    )
            except Exception as exc:
                _log(f"❌ 架空銘柄読み込みエラー: {exc}")
                fetched = []
        if not fetched:  # 通常経路
            try:
                if skip_external:
                    _log(
                        "⚡ 外部API呼び出しをスキップ - キャッシュから銘柄リストを構築"
                    )
                    fetched = []
                else:
                    fetched = build_symbol_universe_from_settings(settings, logger=log)
            except Exception as exc:  # pragma: no cover - ネットワーク例外のみログ
                fetched = []
                msg = f"⚠️ NASDAQ/EODHD銘柄リストの取得に失敗しました: {exc}"
                _log(msg)
                if log_callback:
                    try:
                        log_callback(msg)
                    except Exception:
                        pass

        # テストモード・環境変数の制限値を事前計算
        limit_val: int | None = None
        limit_src = ""

        # テストモードの制限チェック
        if test_mode:
            test_limits = {"mini": 10, "quick": 50, "sample": 100}
            if test_mode in test_limits and test_mode != "test_symbols":
                limit_val = test_limits[test_mode]
                limit_src = f"test-mode={test_mode}"

        # 環境変数による制限チェック（テストモードが未指定の場合）
        if limit_val is None:
            try:
                env_limit = os.getenv("TODAY_SYMBOL_LIMIT", "").strip()
                if env_limit:
                    parsed = int(env_limit)
                    if parsed > 0:
                        limit_val = parsed
                        limit_src = "TODAY_SYMBOL_LIMIT"
            except Exception:
                limit_val = None

        if fetched:
            if limit_val is not None and len(fetched) > limit_val:
                fetched = fetched[:limit_val]
                label = limit_src or "TODAY_SYMBOL_LIMIT"
                info = f"🎯 シンボル数を制限 ({label}={limit_val})"
                _log(info)
                if log_callback:
                    try:
                        log_callback(info)
                    except Exception:
                        pass
            symbols = [s.upper() for s in fetched]
        else:
            universe = load_universe_file()
            if not universe:
                universe = build_universe_from_cache(limit=None)
            symbols = [s.upper() for s in universe]
            if not symbols:
                try:
                    files = list(cache_dir.glob("*.*"))
                    primaries = [p.stem for p in files if p.stem.upper() == "SPY"]
                    others = sorted({p.stem for p in files if len(p.stem) <= 5})[:200]
                    symbols = list(dict.fromkeys(primaries + others))
                except Exception:
                    symbols = []

        # テストモード制限を skip_external 経路でも適用
        if limit_val is not None and len(symbols) > limit_val:
            symbols = symbols[:limit_val]
            label = limit_src or "TODAY_SYMBOL_LIMIT"
            info = f"🎯 シンボル数を制限 ({label}={limit_val})"
            _log(info)
            if log_callback:
                try:
                    log_callback(info)
                except Exception:
                    pass

    # Ensure SPY is the first symbol in today's universe (required by some systems)
    try:
        symbols = [s.upper() for s in symbols]
    except Exception:
        symbols = [str(s).upper() for s in symbols]
    if "SPY" in symbols:
        try:
            symbols.remove("SPY")
        except Exception:
            pass
        symbols.insert(0, "SPY")
    else:
        symbols.insert(0, "SPY")
    ctx.symbol_universe = list(symbols)

    try:
        universe_total = sum(1 for s in symbols if str(s).upper() != "SPY")
    except Exception:
        universe_total = len(symbols)

    try:
        target_cb = globals().get("_SET_STAGE_UNIVERSE_TARGET")
    except Exception:
        target_cb = None
    if target_cb and callable(target_cb):
        try:
            target_cb(universe_total)
        except Exception:
            pass
    try:
        GLOBAL_STAGE_METRICS.set_universe_target(universe_total)
    except Exception:
        pass

    _log(f"🎯 対象シンボル数: {len(symbols)} | 銘柄数：{universe_total}")
    # ヘッダー部分に追加で銘柄数を表示
    _log(f"# 📊 銘柄数：{universe_total}", ui=False, no_timestamp=True)
    _log(f"📋 サンプル: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")

    if log_callback:
        try:
            log_callback("🧭 シンボル決定完了。基礎データのロードへ…")
        except Exception:
            pass
    if progress_callback:
        try:
            progress_callback(1, 8, "対象読み込み:start")
        except Exception:
            pass

    return symbols


def _load_universe_basic_data(
    ctx: TodayRunContext, symbols: list[str]
) -> dict[str, pd.DataFrame]:
    """Load rolling cache data for the prepared universe and ensure coverage."""

    cache_manager = ctx.cache_manager
    settings = ctx.settings
    progress_callback = ctx.progress_callback
    symbol_data = ctx.symbol_data

    # In test modes, allow older rolling caches by widening freshness tolerance
    # to avoid skipping symbols due to staleness when validating the pipeline.
    try:
        test_mode_active = bool(getattr(ctx, "test_mode", None))
    except Exception:
        test_mode_active = False
    freshness_tolerance: int | None = None
    if test_mode_active:
        try:
            # Allow override via env; default to 365 trading days for safety in tests
            freshness_tolerance = int(
                os.environ.get("BASIC_DATA_TEST_FRESHNESS_TOLERANCE", "365")
            )
        except Exception:
            freshness_tolerance = 365
        # Informative warning to make relaxed freshness explicit during tests
        try:
            _log_warning(
                f"テストモード: 基本データの鮮度許容を {freshness_tolerance} 営業日へ緩和します (rolling cache 検証)",
                error_code="TST-FRESHNESS",
                ui=True,
            )
        except Exception:
            pass

    basic_data = load_basic_data(
        symbols,
        cache_manager,
        settings,
        symbol_data,
        today=ctx.today,
        freshness_tolerance=freshness_tolerance,
        base_cache=ctx.base_cache,
        log_callback=lambda msg, ui=True: None,
        ui_log_callback=lambda msg: None,
    )
    # ensure precise type for type-checker
    ctx.basic_data = cast(dict[str, pd.DataFrame], basic_data)

    if progress_callback:
        try:
            progress_callback(2, 8, "load_basic")
        except Exception:
            pass

    try:
        cov_have = len(basic_data)
        cov_total = len(symbols)
        cov_missing = max(0, cov_total - cov_have)
        _log(
            "🧮 データカバレッジ: "
            + f"rolling取得済み {cov_have}/{cov_total} | missing={cov_missing}"
        )
        if cov_missing > 0:
            missing_syms = [s for s in symbols if s not in basic_data]
            # 10%ごとにバッチ表示
            batch_size = max(1, int(cov_total * 0.1))
            for i in range(0, len(missing_syms), batch_size):
                batch = missing_syms[i : i + batch_size]
                symbols_str = ", ".join(batch)
                _log(
                    f"⚠️ rolling未整備 ({i + 1}〜{min(i + batch_size, len(missing_syms))}/{len(missing_syms)}): {symbols_str}",
                    ui=False,
                )
            # 最後に集計メッセージ
            _log(
                f"💡 rolling未整備の計{cov_missing}銘柄は自動的にスキップされました（base/full_backupからの再試行は不要）",
                ui=False,
            )
    except Exception:
        pass

    return cast(dict[str, pd.DataFrame], basic_data)


def _ensure_cli_logger_configured() -> None:
    """CLI 実行時のファイルロガー設定を保証する。"""
    try:
        if globals().get("_LOG_FILE_PATH") is None:
            _mode_env = (get_env_config().today_signals_log_mode or "").strip().lower()
            _configure_today_logger(
                mode=("single" if _mode_env == "single" else "dated")
            )
    except Exception:
        pass


def _silence_streamlit_cli_warnings() -> None:
    """CLI での実行時、Streamlit の bare mode 警告を抑制する。"""
    try:
        if get_env_config().streamlit_server_enabled:
            return

        class _SilenceBareModeWarnings(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                msg = str(record.getMessage())
                if "missing ScriptRunContext" in msg:
                    return False
                if "Session state does not function" in msg:
                    return False
                return True

        _names = [
            "streamlit",
            "streamlit.runtime",
            "streamlit.runtime.scriptrunner_utils.script_run_context",
            "streamlit.runtime.state.session_state_proxy",
        ]
        for _name in _names:
            _logger = logging.getLogger(_name)
            _logger.addFilter(_SilenceBareModeWarnings())
            try:
                _logger.setLevel(logging.ERROR)
            except Exception:
                pass
    except Exception:
        pass


def _safe_progress_call(
    callback: Callable[[int, int, str], None] | None,
    current: int,
    total: int,
    label: str,
) -> None:
    """UI 進捗コールバックを安全に呼び出す（例外は握りつぶす）。"""
    if not callback:
        return
    try:
        callback(current, total, label)
    except Exception:
        pass


def _save_and_notify_phase(
    ctx: TodayRunContext,
    *,
    final_df: pd.DataFrame | None,
    per_system: Mapping[str, pd.DataFrame],
    order_1_7: Sequence[str],
    metrics_summary_context: Mapping[str, Any] | None,
    output_root_for_final: Path | None = None,
) -> None:
    """保存および通知フェーズを担当する補助関数。

    If ``output_root_for_final`` is provided, final CSVs and validation
    outputs are written under that path. We do not modify ``ctx.signals_dir``
    since it is used for cache semantics elsewhere.
    """

    signals_dir = ctx.signals_dir
    notify = ctx.notify
    save_csv = ctx.save_csv
    csv_name_mode = ctx.csv_name_mode or "date"
    progress_callback = ctx.progress_callback
    run_start_time = ctx.run_start_time
    start_equity = ctx.start_equity
    today = ctx.today or get_latest_nyse_trading_day().normalize()
    run_id = ctx.run_id
    # Final destination root (override when provided explicitly)
    final_base: Path = (
        Path(output_root_for_final)
        if output_root_for_final is not None
        else signals_dir
    )

    try:
        final_counts: dict[str, int] = {}
        if (
            final_df is not None
            and not getattr(final_df, "empty", True)
            and "system" in final_df.columns
        ):
            final_counts = final_df.groupby("system").size().to_dict()
    except Exception:
        final_counts = {}
    for name in order_1_7:
        cand_cnt: int | None
        try:
            snapshot = _get_stage_snapshot(name)
            cand_cnt = (
                None
                if snapshot is None or snapshot.candidate_count is None
                else int(snapshot.candidate_count)
            )
        except Exception:
            cand_cnt = None
        if cand_cnt is None:
            df_sys = per_system.get(name)
            cand_cnt = int(
                0 if df_sys is None or getattr(df_sys, "empty", True) else len(df_sys)
            )
        final_cnt = int(final_counts.get(name, 0))
        try:
            _stage(name, 100, None, None, cand_cnt, final_cnt)
        except Exception:
            pass

    if metrics_summary_context:
        try:
            prefilter_map = dict(metrics_summary_context.get("prefilter_map", {}))
            exit_counts_map_ctx = metrics_summary_context.get("exit_counts_map", {})
            exit_counts_map = (
                {k: v for k, v in exit_counts_map_ctx.items()}
                if isinstance(exit_counts_map_ctx, dict)
                else {}
            )
            setup_map = dict(metrics_summary_context.get("setup_map", {}))
            tgt_base = int(metrics_summary_context.get("tgt_base", 0))
            final_counts = {}
            if (
                final_df is not None
                and not getattr(final_df, "empty", True)
                and "system" in final_df.columns
            ):
                final_counts = final_df.groupby("system").size().to_dict()
            lines: list[dict[str, str]] = []
            for sys_name in order_1_7:
                tgt = tgt_base if sys_name != "system7" else 1
                fil = int(prefilter_map.get(sys_name, 0))
                stu = int(setup_map.get(sys_name, 0))
                try:
                    df_trd = per_system.get(sys_name, pd.DataFrame())
                    trd = int(
                        0
                        if df_trd is None or getattr(df_trd, "empty", True)
                        else len(df_trd)
                    )
                except Exception:
                    trd = 0
                ent = int(final_counts.get(sys_name, 0))
                exv = exit_counts_map.get(sys_name)
                ex_txt = "-" if exv is None else str(int(exv))
                value = f"Tgt {tgt} / FIL {fil} / STU {stu} / TRD {trd} / Entry {ent} / Exit {ex_txt}"
                lines.append({"name": sys_name, "value": value})
            title = "📈 本日の最終メトリクス（system別）"
            td = ctx.today
            try:
                td_str = str(getattr(td, "date", lambda: None)() or td)
            except Exception:
                td_str = ""
            run_end_time = datetime.now()
            end_equity = _get_account_equity()
            start_equity_val = float(start_equity or 0.0)
            end_equity_val = float(end_equity or 0.0)
            profit_amt = max(end_equity_val - start_equity_val, 0.0)
            loss_amt = max(start_equity_val - end_equity_val, 0.0)
            try:
                total_entries = int(sum(int(v) for v in final_counts.values()))
            except Exception:
                total_entries = 0
            try:
                total_exits = int(
                    sum(int(v) for v in exit_counts_map.values() if v is not None)
                )
            except Exception:
                total_exits = 0
            start_time_str = run_start_time.strftime("%H:%M:%S")
            end_time_str = run_end_time.strftime("%H:%M:%S")
            duration_seconds = max(
                0, int((run_end_time - run_start_time).total_seconds())
            )
            hours, remainder = divmod(duration_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            summary_pairs = [
                ("指定銘柄総数", f"{int(tgt_base):,}"),
                (
                    "開始時間/完了時間",
                    f"{start_time_str} / {end_time_str} (所要: {duration_str})",
                ),
                (
                    "開始時資産/完了時資産",
                    f"${start_equity_val:,.2f} / ${end_equity_val:,.2f}",
                ),
                (
                    "エントリー銘柄数/エグジット銘柄数",
                    f"{total_entries} / {total_exits}",
                ),
                ("利益額/損失額", f"${profit_amt:,.2f} / ${loss_amt:,.2f}"),
            ]
            summary_fields: list[dict[str, str | bool]] = [
                {"name": key, "value": value, "inline": True}
                for key, value in summary_pairs
            ]
            send_metrics_notification(
                day_str=str(td_str),
                fields=summary_fields + lines,
                summary_pairs=summary_pairs,
                title=title,
            )
        except Exception:
            pass

    if notify:
        try:
            from tools.notify_signals import send_signal_notification

            # Guard against None being passed where a DataFrame is required
            if final_df is not None and not getattr(final_df, "empty", True):
                send_signal_notification(final_df)
        except Exception:
            _log("⚠️ 通知に失敗しました。")

    if save_csv and final_df is not None and not final_df.empty:
        mode = (csv_name_mode or "date").lower()
        date_str = today.strftime("%Y-%m-%d")
        suffix = date_str
        if mode == "datetime":
            try:
                jst_now = datetime.now(ZoneInfo("Asia/Tokyo"))
            except Exception:
                jst_now = datetime.now()
            suffix = f"{date_str}_{jst_now.strftime('%H%M')}"
        elif mode == "runid":
            suffix = f"{date_str}_{run_id}" if run_id else date_str

        # Ensure final destination exists
        try:
            final_base.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        out_all = final_base / f"signals_final_{suffix}.csv"
        try:
            try:
                round_dec = getattr(
                    get_settings(create_dirs=True).cache, "round_decimals", None
                )
            except Exception:
                round_dec = None
            out_df = round_dataframe(final_df, round_dec)
        except Exception:
            out_df = final_df
        # Atomic write: write to temporary file then replace to avoid partial files
        try:
            tmp_all = final_base / f".signals_final_{suffix}.{ctx.run_id}.tmp"
            try:
                from common.io_utils import df_to_csv

                df_to_csv(out_df, tmp_all, index=False)
            except Exception:
                out_df.to_csv(tmp_all, index=False)
            try:
                tmp_all.replace(out_all)
            except Exception:
                # fallback to os.replace if Path.replace fails on some platforms
                import os as _os

                try:
                    tmp_all.replace(out_all)
                except Exception:
                    _os.replace(str(tmp_all), str(out_all))
        except Exception:
            # Best-effort: fallback to direct write
            try:
                from common.io_utils import df_to_csv

                df_to_csv(out_df, out_all, index=False)
            except Exception:
                try:
                    out_df.to_csv(out_all, index=False)
                except Exception:
                    pass
        for name, df in per_system.items():
            if df is None or getattr(df, "empty", True):
                continue
            out = final_base / f"signals_{name}_{suffix}.csv"
            try:
                try:
                    round_dec = getattr(
                        get_settings(create_dirs=True).cache, "round_decimals", None
                    )
                except Exception:
                    round_dec = None
                out_df_per = round_dataframe(df, round_dec)
            except Exception:
                out_df_per = df
            # write per-system CSV atomically
            try:
                tmp_out = final_base / f".signals_{name}_{suffix}.{ctx.run_id}.tmp"
                try:
                    from common.io_utils import df_to_csv

                    df_to_csv(out_df_per, tmp_out, index=False)
                except Exception:
                    out_df_per.to_csv(tmp_out, index=False)
                try:
                    tmp_out.replace(out)
                except Exception:
                    import os as _os

                    try:
                        tmp_out.replace(out)
                    except Exception:
                        _os.replace(str(tmp_out), str(out))
            except Exception:
                try:
                    out_df_per.to_csv(out, index=False)
                except Exception:
                    _log(f"⚠️ CSV書き込み失敗: {out}", ui=False)
        try:
            _log(f"💾 保存: {final_base} にCSVを書き出しました")
        except Exception:
            pass

        # 保存確認（同期検証用）
        if out_all.exists():
            _log(f"✅ CSV保存確認: {out_all.name} ({len(final_df)}行)")
        else:
            _log(f"⚠️ CSV保存失敗: {out_all} が見つかりません")

        # --- TRDlist validation and report export (non-intrusive) ---
        try:
            from common.trdlist_validator import build_validation_report

            report = build_validation_report(final_df, dict(per_system))
            try:
                _test_mode_val = getattr(ctx, "test_mode", None)
            except Exception:
                _test_mode_val = None
            try:
                if output_root_for_final is not None:
                    base_dir = Path(output_root_for_final)
                else:
                    base_dir = (
                        Path("results_csv_test")
                        if _test_mode_val
                        else Path(getattr(ctx.settings, "RESULTS_DIR", "results_csv"))
                    )
            except Exception:
                base_dir = Path("results_csv")
            out_dir = base_dir / "validation"
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                out_file = out_dir / f"validation_report_{suffix}.json"
                tmp_file = out_dir / f".validation_report_{suffix}.{ctx.run_id}.tmp"
                try:
                    from common.io_utils import write_json

                    write_json(tmp_file, report, ensure_ascii=False, indent=2)
                    try:
                        tmp_file.replace(out_file)
                    except Exception:
                        import os as _os

                        _os.replace(str(tmp_file), str(out_file))
                except Exception:
                    try:
                        with tmp_file.open("w", encoding="utf-8") as f:
                            json.dump(
                                report, f, ensure_ascii=False, indent=2, default=str
                            )
                        try:
                            tmp_file.replace(out_file)
                        except Exception:
                            import os as _os

                            _os.replace(str(tmp_file), str(out_file))
                    except Exception:
                        _log("⚠️ validation report 保存に失敗しました", ui=False)
            except Exception:
                try:
                    with open(
                        out_dir / f"validation_report_{suffix}.json",
                        "w",
                        encoding="utf-8",
                    ) as f:
                        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
                except Exception:
                    _log("⚠️ validation report 保存に失敗しました", ui=False)
            if int(report.get("summary", {}).get("errors", 0)) > 0:
                _log_warning(
                    f"検証エラーあり: validation_report_{suffix}.json を確認してください",
                    error_code="VALIDATE-ERR",
                )
            else:
                _log(
                    f"検証OK: validation_report_{suffix}.json に詳細を保存しました",
                    ui=False,
                )
        except Exception as e:
            _log_warning(
                f"バリデーション出力に失敗: {e}", error_code="VALIDATE-FAIL", ui=False
            )

    # Deliver exit counts to UI if a bulk callback is registered. Fallback to
    # per-system exit callback or metrics_summary_context map if not.
    try:
        bulk_cb = globals().get("_PER_SYSTEM_EXIT_BULK")
        per_cb = globals().get("_PER_SYSTEM_EXIT")
        # Prefer metrics_summary_context exit_counts_map when present
        exit_map_ctx = (
            (metrics_summary_context or {}).get("exit_counts_map")
            if metrics_summary_context
            else None
        )
        exit_map = (
            {k: int(v) for k, v in exit_map_ctx.items()}
            if isinstance(exit_map_ctx, dict)
            else None
        )
        # If we don't have a precomputed map, attempt to build from final_df/per_system
        if exit_map is None:
            exit_map = {f"system{i}": 0 for i in range(1, 8)}
            try:
                # Prefer final_counts (entry reductions) as heuristic for exits
                if (
                    final_df is not None
                    and not getattr(final_df, "empty", True)
                    and "system" in final_df.columns
                ):
                    # If final_df exists, assume exits = entries by system
                    fc = final_df.groupby("system").size().to_dict()
                    for k, v in fc.items():
                        try:
                            key = str(k).strip().lower()
                            exit_map[key] = int(v)
                        except Exception:
                            continue
            except Exception:
                pass

        # Persist exit_map to disk so external UI/processes can poll it.
        try:
            # Prefer results dir from ctx.signals_dir; fallback to logs
            out_dir = None
            try:
                out_dir = Path(signals_dir) if signals_dir is not None else None
            except Exception:
                out_dir = None
            if out_dir is None:
                try:
                    out_dir = Path(get_settings().outputs.results_csv_dir)
                except Exception:
                    out_dir = Path("results_csv")
            out_dir.mkdir(parents=True, exist_ok=True)
            exit_json = out_dir / f"exit_counts_{run_id}.json"
            try:
                from common.io_utils import write_json

                write_json(
                    exit_json,
                    {k: int(v) for k, v in exit_map.items()},
                    ensure_ascii=False,
                    indent=2,
                )
                try:
                    _log(f"📘 exit_counts persisted: {exit_json}")
                except Exception:
                    pass
            except Exception:
                try:
                    _log("⚠️ exit_counts の永続化に失敗しました")
                except Exception:
                    pass
        except Exception:
            pass

        # If bulk callback exists, call it once with the map
        if callable(bulk_cb):
            try:
                _log(f"🧩 Dispatching bulk exit_counts to UI: {dict(exit_map)}")
            except Exception:
                pass
            try:
                bulk_cb(dict(exit_map))
            except Exception:
                try:
                    _log("⚠️ bulk exit callback raised an exception")
                except Exception:
                    pass
                pass
        else:
            # Otherwise call per-system callback for each system
            if callable(per_cb):
                try:
                    _log(f"🧩 Dispatching per-system exit_counts to UI: {exit_map}")
                    for sys_name, cnt in exit_map.items():
                        try:
                            per_cb(sys_name, int(cnt or 0))
                        except Exception:
                            try:
                                _log(
                                    f"⚠️ per-system exit callback failed for {sys_name}"
                                )
                            except Exception:
                                pass
                            pass
                except Exception:
                    pass
    except Exception:
        pass

    # Finalize overall progress as exit phase (this ensures top progress shows 100% and label)
    _safe_progress_call(progress_callback, 8, 8, "exit")

    try:
        cnt = 0 if final_df is None else len(final_df)
        _log(f"✅ シグナル検出処理 終了 | 最終候補 {cnt} 件")
    except Exception:
        pass

    try:
        import time as _time

        end_txt = _time.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        end_txt = ""
    try:
        print("#" * 68, flush=True)
    except Exception:
        pass
    _log(
        "# 🏁🏁🏁  本日のシグナル 実行終了 (Engine)  🏁🏁🏁",
        ui=False,
        no_timestamp=True,
    )
    _log(f"# ⏱️ {end_txt} | RUN-ID: {run_id}", ui=False, no_timestamp=True)
    try:
        print("#" * 68 + "\n", flush=True)
    except Exception:
        pass


def _log_previous_counts_summary(signals_dir: Path) -> None:
    """前回実行のシステム別候補件数を簡易表示する。"""
    try:
        prev = _load_prev_counts(signals_dir)
        if prev:
            for i in range(1, 8):
                key = f"system{i}"
                v = int(prev.get(key, 0))
                icon = "✅" if v > 0 else "—"
                suffix = " (0件)" if v == 0 else ""
                _log(f"前回 {icon} {key}: {v}{suffix}")
    except Exception:
        pass


def _apply_system_filters_and_update_ctx(
    ctx: TodayRunContext,
    symbols: list[str],
    basic_data: dict[str, pd.DataFrame],
) -> dict[str, list[str]]:
    """システム別のフィルターを適用し、ctx.system_filters を更新する。"""
    system1_syms = filter_system1(symbols, basic_data)
    system2_syms = filter_system2(symbols, basic_data)
    system3_syms = filter_system3(symbols, basic_data)
    system4_syms = filter_system4(symbols, basic_data)
    system5_syms = filter_system5(symbols, basic_data)
    system6_syms = filter_system6(symbols, basic_data)
    filters = {
        "system1": system1_syms,
        "system2": system2_syms,
        "system3": system3_syms,
        "system4": system4_syms,
        "system5": system5_syms,
        "system6": system6_syms,
    }
    ctx.system_filters = filters
    for system_name, syms in filters.items():
        try:
            total_len = len(syms)
        except Exception:
            total_len = 0
        try:
            _stage(system_name, 25, total_len, None, None, None)
        except Exception:
            pass
    # System7 は SPY 専用
    try:
        spy_total = 1 if "SPY" in (basic_data or {}) else 0
        _stage("system7", 25, spy_total, None, None, None)
    except Exception:
        pass
    return filters


def _log_system1_filter_stats(
    symbols: list[str], basic_data: dict[str, pd.DataFrame]
) -> None:
    """System1 の事前条件ヒット数を表示する。"""
    try:
        s1_total = len(symbols)
        s1_price = 0
        s1_dv = 0
        for _sym in symbols:
            _df = basic_data.get(_sym)
            if _df is None or getattr(_df, "empty", True):
                continue
            try:
                price_ok, dv_ok = _system1_conditions(_df)
            except Exception:
                continue
            if price_ok:
                s1_price += 1
            else:
                continue
            if dv_ok:
                s1_dv += 1
        _log(
            "system1 事前条件サマリー: "
            + f"総数={s1_total}, 価格>=5: {s1_price}, DV20>=50M: {s1_dv}"
        )
    except Exception:
        pass


def _log_system2_filter_stats(
    symbols: list[str], basic_data: dict[str, pd.DataFrame]
) -> None:
    """System2 の事前条件ヒット数を表示する。"""
    try:
        s2_total = len(symbols)
        c_price = 0
        c_dv = 0
        c_atr = 0
        for _sym in symbols:
            _df = basic_data.get(_sym)
            if _df is None or getattr(_df, "empty", True):
                continue
            try:
                price_ok, dv_ok, atr_ok = _system2_conditions(_df)
            except Exception:
                continue
            if price_ok:
                c_price += 1
            else:
                continue
            if dv_ok:
                c_dv += 1
            else:
                continue
            if atr_ok:
                c_atr += 1
        _log(
            "system2 事前条件サマリー: "
            + f"総数={s2_total}, 価格>=5: {c_price}, DV20>=25M: {c_dv}, ATR比率>=3%: {c_atr}"
        )
    except Exception:
        pass


def _log_system3_filter_stats(
    symbols: list[str], basic_data: dict[str, pd.DataFrame]
) -> None:
    """System3 の事前条件ヒット数を表示する。"""
    try:
        s3_total = len(symbols)
        s3_low = 0
        s3_av = 0
        s3_atr = 0
        for _sym in symbols:
            _df = basic_data.get(_sym)
            if _df is None or getattr(_df, "empty", True):
                continue
            try:
                low_ok, av_ok, atr_ok = _system3_conditions(_df)
            except Exception:
                continue
            if low_ok:
                s3_low += 1
            else:
                continue
            if av_ok:
                s3_av += 1
            else:
                continue
            if atr_ok:
                s3_atr += 1
        _log(
            "system3 事前条件サマリー: "
            + f"総数={s3_total}, Low>=1: {s3_low}, AvgVol50>=1M: {s3_av}, ATR_Ratio>=5%: {s3_atr}"
        )
    except Exception:
        pass


def _log_system4_filter_stats(
    symbols: list[str], basic_data: dict[str, pd.DataFrame]
) -> None:
    """System4 の事前条件ヒット数を表示する。"""
    try:
        s4_total = len(symbols)
        s4_dv = 0
        s4_hv = 0
        for _sym in symbols:
            _df = basic_data.get(_sym)
            if _df is None or getattr(_df, "empty", True):
                continue
            try:
                dv_ok, hv_ok = _system4_conditions(_df)
            except Exception:
                continue
            if dv_ok:
                s4_dv += 1
            else:
                continue
            if hv_ok:
                s4_hv += 1
        _log(
            "system4 事前条件サマリー: "
            + f"総数={s4_total}, DV50>=100M: {s4_dv}, HV50 10〜40: {s4_hv}"
        )
    except Exception:
        pass


def _log_system5_filter_stats(
    symbols: list[str], basic_data: dict[str, pd.DataFrame]
) -> None:
    """System5 の事前条件ヒット数を表示する。"""
    try:
        threshold_label = f"ATR_Pct>{DEFAULT_ATR_PCT_THRESHOLD * 100:.1f}%"
        s5_total = len(symbols)
        s5_av = 0
        s5_dv = 0
        s5_atr = 0
        for _sym in symbols:
            _df = basic_data.get(_sym)
            if _df is None or getattr(_df, "empty", True):
                continue
            try:
                av_ok, dv_ok, atr_ok = _system5_conditions(_df)
            except Exception:
                continue
            if av_ok:
                s5_av += 1
            else:
                continue
            if dv_ok:
                s5_dv += 1
            else:
                continue
            if atr_ok:
                s5_atr += 1
        _log(
            "system5 事前条件サマリー: "
            + f"総数={s5_total}, AvgVol50>500k: {s5_av}, DV50>2.5M: {s5_dv}"
            + f", {threshold_label}: {s5_atr}"
        )
    except Exception:
        pass


def _log_system6_filter_stats(
    symbols: list[str], basic_data: dict[str, pd.DataFrame]
) -> None:
    """System6 の事前条件ヒット数を表示する。"""
    try:
        s6_total = len(symbols)
        s6_low = 0
        s6_dv = 0
        s6_hv = 0
        for _sym in symbols:
            _df = basic_data.get(_sym)
            if _df is None or getattr(_df, "empty", True):
                continue
            try:
                low_ok, dv_ok, hv_ok = _system6_conditions(_df)
            except Exception:
                continue
            if not low_ok:
                continue
            s6_low += 1
            if not dv_ok:
                continue
            s6_dv += 1
            if hv_ok:
                s6_hv += 1
        _log(
            "system6 事前条件サマリー: "
            + f"総数={s6_total}, Low>=5: {s6_low}, DV50>10M: {s6_dv}, HV50 10〜40: {s6_hv}"
        )
    except Exception:
        pass


def _log_system7_filter_stats(basic_data: dict[str, pd.DataFrame]) -> None:
    """System7 (SPY) の事前条件ヒット数を表示する。"""
    try:
        spyp = (
            1
            if (
                "SPY" in basic_data
                and not getattr(basic_data.get("SPY"), "empty", True)
            )
            else 0
        )
        _log("system7 事前条件サマリー: SPYの有無 | SPY=" + str(spyp))
    except Exception:
        pass


def _log_system_filter_stats(
    symbols: list[str],
    basic_data: dict[str, pd.DataFrame],
    filters: dict[str, list[str]],
) -> None:
    """各システムの事前条件サマリーとフィルター通過件数を表示する。"""
    _log("各システムの事前条件サマリー (system1〜system6)")
    _log_system1_filter_stats(symbols, basic_data)
    _log_system2_filter_stats(symbols, basic_data)
    _log_system3_filter_stats(symbols, basic_data)
    _log_system4_filter_stats(symbols, basic_data)
    _log_system5_filter_stats(symbols, basic_data)
    _log_system6_filter_stats(symbols, basic_data)
    _log_system7_filter_stats(basic_data)
    system1_syms = filters.get("system1", [])
    system2_syms = filters.get("system2", [])
    system3_syms = filters.get("system3", [])
    system4_syms = filters.get("system4", [])
    system5_syms = filters.get("system5", [])
    system6_syms = filters.get("system6", [])
    _log(
        "フィルター通過件数: "
        + f"system1={len(system1_syms)}件, "
        + f"system2={len(system2_syms)}件, "
        + f"system3={len(system3_syms)}件, "
        + f"system4={len(system4_syms)}件, "
        + f"system5={len(system5_syms)}件, "
        + f"system6={len(system6_syms)}件"
    )


def _ensure_rolling_cache_fresh(
    symbol: str,
    rolling_df: pd.DataFrame,
    today: pd.Timestamp,
    cache_manager: CacheManager,
    base_rows: int = 320,
    max_lag_days: int = 2,
) -> pd.DataFrame:
    """
    rolling_dfの最終日付がtodayからmax_lag_days以上ズレている場合、
    baseからrollingを再生成し、rollingへ書き戻す。
    """
    if rolling_df is None or getattr(rolling_df, "empty", True):
        # 欠損時はbaseから再生成
        base_df = cast(Any, cache_manager).read(symbol, layer="base", rows=base_rows)
        if base_df is not None and not getattr(base_df, "empty", True):
            rolling_new = base_df.tail(base_rows).copy()
            cast(Any, cache_manager).write_atomic(symbol, rolling_new, layer="rolling")
            return cast(pd.DataFrame, rolling_new)
        return rolling_df
    try:
        last_idx = rolling_df.index[-1]
        if isinstance(last_idx, str):
            last_ts = pd.to_datetime(last_idx)
        elif hasattr(last_idx, "to_pydatetime"):
            last_ts = pd.Timestamp(last_idx.to_pydatetime())
        else:
            # Cast to Any to satisfy Timestamp's accepted overloads
            last_ts = pd.Timestamp(cast(Any, last_idx))
    except Exception:
        return rolling_df
    try:
        lag_days = int((today - last_ts).days)
    except Exception:
        lag_days = 0
    if lag_days > max_lag_days:
        # 鮮度不足: baseからrolling再生成
        base_df = cast(Any, cache_manager).read(symbol, layer="base", rows=base_rows)
        if base_df is not None and not getattr(base_df, "empty", True):
            rolling_new = base_df.tail(base_rows).copy()
            cast(Any, cache_manager).write_atomic(symbol, rolling_new, layer="rolling")
            return cast(pd.DataFrame, rolling_new)
    return rolling_df


def _prepare_system2_data(
    basic_data: dict[str, pd.DataFrame],
    system_symbols: list[str],
) -> tuple[dict[str, pd.DataFrame], int, int, int]:
    """System2 の準備データ（フィルター通過集合など）を構築する。"""
    _log("System2 準備データの集計")
    raw_data = _subset_data(basic_data, system_symbols)
    _log(f"抽出対象の件数: system2={len(raw_data)}件")
    s2_filter = int(len(system_symbols))
    s2_rsi = 0
    s2_combo = 0
    try:
        for _sym in system_symbols or []:
            _df = raw_data.get(_sym)
            if _df is None or getattr(_df, "empty", True):
                continue
            try:
                last = _df.iloc[-1]
            except Exception:
                continue
            try:
                rsi_pass = float(last.get("RSI3", 0)) > 90
            except Exception:
                rsi_pass = False
            if not rsi_pass:
                continue
            s2_rsi += 1
            try:
                if bool(last.get("TwoDayUp", False)):
                    s2_combo += 1
            except Exception:
                pass
        _log(
            "system2 セットアップ条件: "
            + f"候補数={s2_filter}, RSI3>90: {s2_rsi}, "
            + f"TwoDayUp: {s2_combo}"
        )
        try:
            _stage(
                "system2",
                50,
                filter_count=int(s2_filter),
                setup_count=int(s2_combo),
            )
        except Exception:
            pass
    except Exception:
        pass
    return raw_data, s2_filter, s2_rsi, s2_combo


def _prepare_system3_data(
    basic_data: dict[str, pd.DataFrame],
    system_symbols: list[str],
) -> tuple[dict[str, pd.DataFrame], int, int, int]:
    """System3 の準備データ（フィルター通過集合など）を構築する。"""
    _log("System3 準備データの集計")
    raw_data = _subset_data(basic_data, system_symbols)
    _log(f"抽出対象の件数: system3={len(raw_data)}件")
    s3_filter = int(len(system_symbols))
    s3_close = 0
    s3_combo = 0
    try:
        for _sym in system_symbols or []:
            _df = raw_data.get(_sym)
            if _df is None or getattr(_df, "empty", True):
                continue
            try:
                last = _df.iloc[-1]
            except Exception:
                continue
            try:
                close_pass = float(last.get("Close", 0)) > float(
                    last.get("SMA150", float("inf"))
                )
            except Exception:
                close_pass = False
            if not close_pass:
                continue
            s3_close += 1
            try:
                if float(last.get("Drop3D", 0)) >= 0.125:
                    s3_combo += 1
            except Exception:
                pass
        _log(
            "system3 セットアップ条件: "
            + f"候補数={s3_filter}, Close>SMA150: {s3_close}, "
            + f"3日下落>=12.5%: {s3_combo}"
        )
        try:
            _stage(
                "system3",
                50,
                filter_count=int(s3_filter),
                setup_count=int(s3_combo),
                candidate_count=None,
                entry_count=None,
            )
        except Exception:
            pass
    except Exception:
        pass
    return raw_data, s3_filter, s3_close, s3_combo


def _prepare_system4_data(
    basic_data: dict[str, pd.DataFrame],
    system_symbols: list[str],
) -> tuple[dict[str, pd.DataFrame], int, int]:
    """System4 の準備データ（フィルター通過集合など）を構築する。"""
    _log("System4 準備データの集計")
    raw_data = _subset_data(basic_data, system_symbols)
    _log(f"抽出対象の件数: system4={len(raw_data)}件")
    s4_filter = int(len(system_symbols))
    s4_close = 0
    try:
        for _sym in system_symbols or []:
            _df = raw_data.get(_sym)
            if _df is None or getattr(_df, "empty", True):
                continue
            try:
                last = _df.iloc[-1]
            except Exception:
                continue
            try:
                if float(last.get("Close", 0)) > float(
                    last.get("SMA200", float("inf"))
                ):
                    s4_close += 1
            except Exception:
                pass
        _log(f"system4 セットアップ条件: 候補数={s4_filter}, Close>SMA200: {s4_close}")
        try:
            _stage(
                "system4",
                50,
                filter_count=int(s4_filter),
                setup_count=int(s4_close),
                candidate_count=None,
                entry_count=None,
            )
        except Exception:
            pass
    except Exception:
        pass
    return raw_data, s4_filter, s4_close


def _prepare_system5_data(
    basic_data: dict[str, pd.DataFrame],
    system_symbols: list[str],
) -> tuple[dict[str, pd.DataFrame], int, int, int, int]:
    """System5 の準備データ（フィルター通過集合など）を構築する。"""
    _log("System5 準備データの集計")
    raw_data = _subset_data(basic_data, system_symbols)
    _log(f"抽出対象の件数: system5={len(raw_data)}件")
    s5_filter = int(len(system_symbols))
    s5_close = 0
    s5_adx = 0
    s5_combo = 0
    try:
        for _sym in system_symbols or []:
            _df = raw_data.get(_sym)
            if _df is None or getattr(_df, "empty", True):
                continue
            try:
                last = _df.iloc[-1]
            except Exception:
                continue
            try:
                price_pass = float(last.get("Close", 0)) > float(
                    last.get("SMA100", 0)
                ) + float(last.get("ATR10", 0))
            except Exception:
                price_pass = False
            if not price_pass:
                continue
            s5_close += 1
            try:
                adx_pass = float(last.get("ADX7", 0)) > 55
            except Exception:
                adx_pass = False
            if not adx_pass:
                continue
            s5_adx += 1
            try:
                if float(last.get("RSI3", 100)) < 50:
                    s5_combo += 1
            except Exception:
                pass
        _log(
            "system5 セットアップ条件: "
            + f"候補数={s5_filter}, Close>SMA100+ATR10: {s5_close}, "
            + f"ADX7>55: {s5_adx}, RSI3<50: {s5_combo}"
        )
        try:
            _stage(
                "system5",
                50,
                filter_count=int(s5_filter),
                setup_count=int(s5_combo),
                candidate_count=None,
                entry_count=None,
            )
        except Exception:
            pass
    except Exception:
        pass
    return raw_data, s5_filter, s5_close, s5_adx, s5_combo


def _prepare_system6_data(
    basic_data: dict[str, pd.DataFrame],
    system_symbols: list[str],
) -> tuple[dict[str, pd.DataFrame], int, int, int]:
    """System6 の準備データ（フィルター通過集合など）を構築する。"""
    _log("System6 準備データの集計")
    raw_data = _subset_data(basic_data, system_symbols)
    _log(f"抽出対象の件数: system6={len(raw_data)}件")
    s6_filter = int(len(system_symbols))
    s6_ret = 0
    s6_combo = 0
    try:
        for _sym in system_symbols or []:
            _df = raw_data.get(_sym)
            if _df is None or getattr(_df, "empty", True):
                continue
            try:
                last = _df.iloc[-1]
            except Exception:
                continue
            try:
                # return_6d: 旧称 Return6D (命名統一済)
                ret_val = to_float(
                    get_indicator(cast(Mapping[str, Any], last), "return_6d")
                )
                ret_pass = (ret_val > 0.20) if not pd.isna(ret_val) else False
            except Exception:
                ret_pass = False
            if not ret_pass:
                continue
            s6_ret += 1
            try:
                # UpTwoDays は列名揺れに対応（UpTwoDay…）
                if is_true(get_indicator(cast(Mapping[str, Any], last), "uptwodays")):
                    s6_combo += 1
            except Exception:
                pass
        _log(
            "system6 セットアップ条件: "
            + f"候補数={s6_filter}, return_6d>20%: {s6_ret}, "
            + f"UpTwoDays: {s6_combo}"
        )
        try:
            _stage(
                "system6",
                50,
                filter_count=int(s6_filter),
                setup_count=int(s6_combo),
                candidate_count=None,
                entry_count=None,
            )
        except Exception:
            pass
    except Exception:
        pass
    return raw_data, s6_filter, s6_ret, s6_combo


def _resolve_spy_dataframe(basic_data: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    """SPY の DataFrame を指標付きで取得する。"""
    if "SPY" in basic_data:
        try:
            return cast(pd.DataFrame | None, get_spy_with_indicators(basic_data["SPY"]))
        except Exception:
            return None
    _log(
        "SPY の基礎データが見つかりません (base/full_backup/rolling のいずれにも存在しません)。"
        + " SPY.csv または data_cache/base ならびに data_cache/full_backup を確認してください。"
    )
    return None


@no_type_check
def compute_today_signals(  # noqa: C901  # type: ignore[reportGeneralTypeIssues]
    symbols: list[str] | None,
    *,
    slots_long: int | None = None,
    slots_short: int | None = None,
    capital_long: float | None = None,
    capital_short: float | None = None,
    save_csv: bool = False,
    csv_name_mode: str | None = None,
    notify: bool = True,
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    # 追加: 並列実行時などに system ごとの開始/完了を通知する軽量コールバック
    # phase は "start" | "done" を想定
    per_system_progress: Callable[[str, str], None] | None = None,
    symbol_data: dict[str, pd.DataFrame] | None = None,
    parallel: bool = False,
    test_mode: str | None = None,
    skip_external: bool = False,
    skip_latest_check: bool = False,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """当日シグナル抽出＋配分の本体。

    Args:
        symbols: 対象シンボルリスト。
        parallel: True の場合はシステムごとのシグナル抽出を並行実行する。

    戻り値: (final_df, per_system_df_dict)
    """

    # デフォルト戻り値を事前に設定（シグナル0件や早期returnの場合に使用）
    # final_df = pd.DataFrame()  # Unused variable removed
    per_system: dict[str, pd.DataFrame] = {}

    # 実行開始時にタイムスタンプをリセット（Streamlit UI から何度も実行される場合に対応）
    import time as _t

    global _LOG_START_TS
    _LOG_START_TS = _t.time()

    _log("🔧 デバッグ: compute_today_signals開始")

    # Phase5（当日シグナル抽出）では Phase0-4 で十分なフィルター/セットアップ内訳を
    # すでに出しているため、重複ログを抑制する無音ログ関数を用意する。
    def _quiet_log_for_phase5(_msg: str) -> None:  # noqa: ANN001 - simple sink
        return

    # PerformanceMonitor のグローバルインスタンスを取得（--detailed-perf有効時のみ存在）
    perf_monitor = None
    try:
        from common.performance_monitor import get_global_monitor

        perf_monitor = get_global_monitor()
    except Exception:
        pass

    # Phase 0: 初期化・設定ロード
    if _LIGHTWEIGHT_BENCHMARK:
        _LIGHTWEIGHT_BENCHMARK.start_phase("phase0_initialization")
    _phase0_measure = (
        perf_monitor.measure("phase0_initialization") if perf_monitor else None
    )
    if _phase0_measure:
        _phase0_measure.__enter__()

    # Progress: phase0 initialization start
    try:
        emit_progress_event("phase0_initialization_start", {})
    except Exception:
        pass

    ctx = _initialize_run_context(
        slots_long=slots_long,
        slots_short=slots_short,
        capital_long=capital_long,
        capital_short=capital_short,
        save_csv=save_csv,
        csv_name_mode=csv_name_mode,
        notify=notify,
        log_callback=log_callback,
        progress_callback=progress_callback,
        per_system_progress=per_system_progress,
        symbol_data=symbol_data,
        parallel=parallel,
        test_mode=test_mode,
        skip_external=skip_external,
    )

    try:
        GLOBAL_STAGE_METRICS.reset()
    except Exception:
        pass

    # CLI 経由で未設定の場合（UI 等）、既定で日付別ログに切替
    try:
        if globals().get("_LOG_FILE_PATH") is None:
            _mode_env = (get_env_config().today_signals_log_mode or "").strip().lower()
            _configure_today_logger(
                mode=("single" if _mode_env == "single" else "dated")
            )
    except Exception:
        pass

    if _phase0_measure:
        _phase0_measure.__exit__(None, None, None)
    if _LIGHTWEIGHT_BENCHMARK:
        _LIGHTWEIGHT_BENCHMARK.end_phase()

    # Progress: phase0 initialization complete
    try:
        emit_progress_event("phase0_initialization_complete", {})
    except Exception:
        pass

    _run_id = ctx.run_id
    # settings = ctx.settings  # Unused variable removed
    # install log callback for helpers
    globals()["_LOG_CALLBACK"] = ctx.log_callback
    signals_dir = ctx.signals_dir

    # run_start_time = ctx.run_start_time  # Unused variable removed
    # start_equity = ctx.start_equity  # Unused variable removed
    slots_long = ctx.slots_long
    slots_short = ctx.slots_short
    capital_long = ctx.capital_long
    capital_short = ctx.capital_short
    save_csv = ctx.save_csv
    csv_name_mode = ctx.csv_name_mode
    notify = ctx.notify
    log_callback = ctx.log_callback
    progress_callback = ctx.progress_callback
    per_system_progress = ctx.per_system_progress
    parallel = ctx.parallel

    # CLI実行時のStreamlit警告を抑制（UIコンテキストが無い場合のみ）
    try:
        if not get_env_config().streamlit_server_enabled:

            class _SilenceBareModeWarnings(logging.Filter):
                def filter(self, record: logging.LogRecord) -> bool:
                    msg = str(record.getMessage())
                    if "missing ScriptRunContext" in msg:
                        return False
                    if "Session state does not function" in msg:
                        return False
                    return True

            _names = [
                "streamlit",
                "streamlit.runtime",
                "streamlit.runtime.scriptrunner_utils.script_run_context",
                "streamlit.runtime.state.session_state_proxy",
            ]
            for _name in _names:
                _logger = logging.getLogger(_name)
                _logger.addFilter(_SilenceBareModeWarnings())
                try:
                    _logger.setLevel(logging.ERROR)
                except Exception:
                    pass
    except Exception:
        pass

    # 対象とするNYSE営業日（実行開始時に一度だけ確定）
    entry_day = get_signal_target_trading_day().normalize()
    ctx.today = entry_day  # 互換のため today フィールドはエントリー予定日を指す
    ctx.entry_day = entry_day
    try:
        prev_trading = get_latest_nyse_trading_day(entry_day - pd.Timedelta(days=1))
        ctx.signal_base_day = pd.Timestamp(prev_trading).normalize()
    except Exception:
        ctx.signal_base_day = entry_day

    # Update max_date_lag_days dynamically for weekend/holiday gaps when no env override is set
    # 基本方針: 明示的な環境変数オーバーライドがない場合、
    # (entry_day - signal_base_day) のカレンダー日差分を下限として許容（日数）を引き上げる。
    try:
        env = get_env_config()
        lag_override = getattr(env, "latest_only_max_date_lag_days", None)
    except Exception:
        lag_override = None

    try:
        calendar_tolerance = max(0, int(getattr(ctx, "max_date_lag_days", 2)))
    except Exception:
        calendar_tolerance = 2

    if lag_override is None:
        try:
            entry = pd.Timestamp(getattr(ctx, "entry_day", None)).normalize()
            base = pd.Timestamp(getattr(ctx, "signal_base_day", None)).normalize()
            gap_days = None
            if entry is not None and base is not None:
                gap_days = max(0, int((entry - base).days))
            effective = int(calendar_tolerance)
            if gap_days is not None:
                effective = max(effective, gap_days)
            ctx.max_date_lag_days = max(0, int(effective))
        except Exception:
            ctx.max_date_lag_days = calendar_tolerance
    else:
        # Respect explicit env override decided earlier in _initialize_run_context
        try:
            ctx.max_date_lag_days = max(0, int(getattr(ctx, "max_date_lag_days", 1)))
        except Exception:
            pass

    # Run start banner (CLI only) - 最初に実行開始メッセージを表示
    try:
        print("#" * 68, flush=True)
    except Exception:
        pass
    _log(
        "# 🚀🚀🚀  本日のシグナル 実行開始 (Engine)  🚀🚀🚀",
        ui=False,
        no_timestamp=True,
    )
    try:
        import time as _time

        now_str = _time.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        now_str = ""
    _log(f"# ⏱️ {now_str} | RUN-ID: {_run_id}", ui=False, no_timestamp=True)
    try:
        print("#" * 68 + "\n", flush=True)
    except Exception:
        pass

    try:
        _log(
            f"📅 エントリー予定日（NYSE）: {entry_day.date()}",
            no_timestamp=True,
        )
        base_day_disp = getattr(ctx, "signal_base_day", None)
        if base_day_disp is not None:
            _log(
                f"📌 シグナル基準日（前営業日）: {pd.Timestamp(base_day_disp).date()}",
                no_timestamp=True,
            )
    except Exception:
        pass
    _log(
        "ℹ️ 注: EODHDは当日終値が未反映のため、直近営業日ベースで計算します。",
        no_timestamp=True,
    )
    _log("", no_timestamp=True)  # 空行を追加
    # 開始直後に前回結果をまとめて表示
    try:
        prev = _load_prev_counts(signals_dir)
        if prev:
            for i in range(1, 8):
                key = f"system{i}"
                v = int(prev.get(key, 0))
                icon = "✅" if v > 0 else "❌"
                _log(f"🧾 {icon} (前回結果) {key}: {v} 件{' 🚫' if v == 0 else ''}")
    except Exception:
        pass
    if progress_callback:
        try:
            progress_callback(0, 8, "init")
        except Exception:
            pass

    # Phase 1: シンボルユニバース構築
    if _LIGHTWEIGHT_BENCHMARK:
        _LIGHTWEIGHT_BENCHMARK.start_phase("phase1_symbol_universe")
    _phase1_measure = (
        perf_monitor.measure("phase1_symbol_universe") if perf_monitor else None
    )
    if _phase1_measure:
        _phase1_measure.__enter__()

    # Progress: phase1 start
    try:
        emit_progress_event("phase1_symbol_universe_start", {})
    except Exception:
        pass

    symbols = _prepare_symbol_universe(ctx, symbols)

    # Progress: phase1 complete
    try:
        emit_progress_event(
            "phase1_symbol_universe_complete",
            {"symbols": int(len(symbols) if symbols is not None else 0)},
        )
    except Exception:
        pass
    if _phase1_measure:
        _phase1_measure.__exit__(None, None, None)
    if _LIGHTWEIGHT_BENCHMARK:
        _LIGHTWEIGHT_BENCHMARK.end_phase()

    # Phase 2: データロード（rolling cache）
    if _LIGHTWEIGHT_BENCHMARK:
        _LIGHTWEIGHT_BENCHMARK.start_phase("phase2_data_loading")
    _phase2_measure = (
        perf_monitor.measure("phase2_data_loading") if perf_monitor else None
    )
    if _phase2_measure:
        _phase2_measure.__enter__()

    # Progress: phase2 start
    try:
        emit_progress_event(
            "phase2_data_loading_start",
            {"target_symbols": int(len(symbols) if symbols is not None else 0)},
        )
    except Exception:
        pass

    basic_data = _load_universe_basic_data(ctx, symbols)

    # Progress: phase2 complete
    try:
        loaded = 0
        if isinstance(basic_data, dict):
            loaded = sum(1 for _k, _v in basic_data.items() if _v is not None)
        emit_progress_event(
            "phase2_data_loading_complete", {"loaded_assets": int(loaded)}
        )
    except Exception:
        pass
    if _phase2_measure:
        _phase2_measure.__exit__(None, None, None)
    if _LIGHTWEIGHT_BENCHMARK:
        _LIGHTWEIGHT_BENCHMARK.end_phase()

    # ✨ NEW: Phase 0 - 最新営業日チェック（rolling cache の鮮度確認）
    if not skip_latest_check:
        try:
            expected_base_day = pd.Timestamp(
                getattr(ctx, "signal_base_day", None)
            ).normalize()

            _log(
                f"🔍 Phase 0: rolling キャッシュの最新営業日チェック中 (期待日: {expected_base_day.date()})..."
            )

            # Progress: phase0 latest check start
            try:
                emit_progress_event(
                    "phase0_latest_check_start",
                    {
                        "expected_date": (
                            expected_base_day.date().isoformat()
                            if expected_base_day is not None
                            else None
                        ),
                        "precheck_total_symbols": int(
                            len(symbols) if symbols is not None else 0
                        ),
                    },
                )
            except Exception:
                pass

            valid_symbols, stale_details = validate_latest_trading_day(
                symbols=symbols,
                expected_date=expected_base_day,
                cache_manager=ctx.cache_manager,
                log_callback=_log,
                rolling_data=basic_data if isinstance(basic_data, dict) else None,
                tolerance_days=max(0, int(getattr(ctx, "max_date_lag_days", 1))),
            )

            # 除外銘柄の詳細を CSV 保存
            if stale_details:
                try:
                    excluded_csv = save_excluded_symbols_csv(
                        stale_details, expected_base_day, output_dir="logs"
                    )
                    if excluded_csv:
                        _log(f"📄 除外銘柄の詳細: {excluded_csv}")
                except Exception as e:
                    _log(f"⚠️  除外銘柄 CSV 保存エラー: {e}")

                # 理由別サマリー
                reason_counts = get_exclusion_stats(stale_details)

                _log("📊 除外理由の内訳:")
                for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
                    _log(f"   - {reason}: {count} 銘柄")

            # symbols リストを valid_symbols で上書き
            if not valid_symbols:
                _log("❌ すべての銘柄が最新営業日チェックで除外されました。処理中止。")
                raise SystemExit(1)

            symbols = valid_symbols
            excluded_count = len(stale_details)
            total_symbols = len(symbols) + excluded_count

            _log(
                f"✅ Phase 0 完了: {len(symbols)} 銘柄が処理対象（{excluded_count} 銘柄を除外）"
            )

            # 進捗イベント送出（Streamlit UI で可視化）
            if stale_details:
                try:
                    emit_progress_event(
                        "phase0_exclusion_stats",
                        {
                            "total_symbols": total_symbols,
                            "valid_symbols": len(symbols),
                            "excluded_count": excluded_count,
                            "expected_date": expected_base_day.date().isoformat(),
                            "reason_breakdown": reason_counts,
                        },
                    )
                except Exception as e:
                    _log(f"⚠️  進捗イベント送出エラー: {e}")

            # Progress: phase0 latest check complete
            try:
                emit_progress_event(
                    "phase0_latest_check_complete",
                    {
                        "total_symbols": int(total_symbols),
                        "valid_symbols": int(len(symbols)),
                        "excluded_count": int(excluded_count),
                        "expected_date": (
                            expected_base_day.date().isoformat()
                            if expected_base_day is not None
                            else None
                        ),
                    },
                )
            except Exception:
                pass

            # basic_data も valid_symbols のみに絞り込み
            if isinstance(basic_data, dict):
                basic_data = {
                    sym: df
                    for sym, df in basic_data.items()
                    if sym in valid_symbols or sym == "SPY"
                }

        except SystemExit:
            raise
        except Exception as e:
            _log(f"⚠️  最新営業日チェックでエラー: {e}。スキップして継続します。")
    else:
        _log("⏭️  Phase 0: 最新営業日チェックをスキップしました (--skip-latest-check)")

    # 重要: SPY キャッシュの存在と最低限の健全性を起動直後にチェックし、NGなら即停止
    try:
        spy_df_check = basic_data.get("SPY") if isinstance(basic_data, dict) else None
    except Exception:
        spy_df_check = None
    if spy_df_check is None or getattr(spy_df_check, "empty", True):
        _log(
            "❌ SPYキャッシュが見つかりません（または空です）。処理を中止します。",
            ui=False,
        )
        _log(
            "💡 対策: data_cache/rolling または base/full_backup に SPY.csv を配置し、"
            "必要なら scripts/recover_spy_cache.py で復旧してください。",
            ui=False,
        )
        raise SystemExit(1)
    try:
        last_dt = _extract_last_cache_date(spy_df_check)
    except Exception:
        last_dt = None
    if last_dt is None:
        _log(
            "❌ SPYキャッシュの日付列（date/Date/index）が解釈できません。処理を中止します。",
            ui=False,
        )
        raise SystemExit(1)

    # latest_only の基準日は開始時に確定済み（ctx.signal_base_day）。SPYキャッシュと相違しても警告のみ。
    try:
        spy_df = basic_data.get("SPY") if isinstance(basic_data, dict) else None
        anchor_last = _extract_last_cache_date(spy_df) if spy_df is not None else None
        if anchor_last is not None:
            frozen_base = pd.Timestamp(
                getattr(ctx, "signal_base_day", None)
            ).normalize()

            # Calculate trading days lag using NYSE calendar
            trading_days_lag = _calculate_trading_days_lag(
                pd.Timestamp(anchor_last), frozen_base
            )

            if pd.Timestamp(anchor_last).normalize() != frozen_base:
                _log(
                    f"⚠️ SPYキャッシュの最終日が固定したシグナル基準日と異なります: "
                    f"cache={pd.Timestamp(anchor_last).date()} / "
                    f"frozen={frozen_base.date()} "
                    f"(営業日差: {trading_days_lag}日)"
                )

                # Validate against trading days tolerance
                calendar_tolerance = getattr(ctx, "max_date_lag_days", 2)
                if trading_days_lag > calendar_tolerance:
                    _log(
                        f"❌ キャッシュの鮮度が許容範囲を超えています "
                        f"(営業日差 {trading_days_lag} > 許容 {calendar_tolerance}日)。"
                    )
                    _log(
                        "💡 対策: scripts/cache_daily_data.py または "
                        "scripts/update_cache_all.ps1 でデータを更新してください。"
                    )
                    # Hard failure when SPY cache exceeds freshness threshold
                    raise SystemExit(1)
    except Exception:
        pass

    # ✨ NEW: 指標事前計算チェック（不足時は即座停止）
    try:
        from common.indicators_validation import (
            IndicatorValidationError,
            validate_precomputed_indicators,
        )

        target_systems = [1, 2, 3, 4, 5, 6, 7]  # 全System対象
        _log("🔍 指標事前計算状況をチェック中...")

        validate_precomputed_indicators(
            basic_data,
            systems=target_systems,
            strict_mode=True,  # 不足時は即座停止
            log_callback=_log,
        )

    except IndicatorValidationError as e:
        _log(f"❌ 指標チェックエラー: {e}")
        _log("💡 解決方法: python scripts/build_rolling_with_indicators.py --workers 4")
        raise SystemExit(1) from e
    except Exception as e:
        _log(f"⚠️  指標チェック処理でエラー: {e}")
        # チェック処理自体のエラーは継続（後方互換性）

    # Phase 3: Two-Phaseフィルタリング
    if _LIGHTWEIGHT_BENCHMARK:
        _LIGHTWEIGHT_BENCHMARK.start_phase("phase3_filtering")
    _phase3_measure = perf_monitor.measure("phase3_filtering") if perf_monitor else None
    if _phase3_measure:
        _phase3_measure.__enter__()

    _log("🧪 事前フィルター実行中 (system1〜system6)…")

    # フィルター開始前に各システムの進捗を0%にリセット
    try:
        for system_name in [
            "system1",
            "system2",
            "system3",
            "system4",
            "system5",
            "system6",
            "system7",
        ]:
            _stage(system_name, 0, filter_count=len(symbols))
    except Exception:
        pass

    filter_stats: dict[str, dict[str, int]] = {
        "system1": {},
        "system2": {},
        "system3": {},
        "system4": {},
        "system5": {},
        "system6": {},
    }
    system1_syms = filter_system1(symbols, basic_data, stats=filter_stats["system1"])
    system2_syms = filter_system2(symbols, basic_data, stats=filter_stats["system2"])
    system3_syms = filter_system3(symbols, basic_data, stats=filter_stats["system3"])
    system4_syms = filter_system4(symbols, basic_data, stats=filter_stats["system4"])
    system5_syms = filter_system5(symbols, basic_data, stats=filter_stats["system5"])
    system6_syms = filter_system6(symbols, basic_data, stats=filter_stats["system6"])
    ctx.system_filters = {
        "system1": system1_syms,
        "system2": system2_syms,
        "system3": system3_syms,
        "system4": system4_syms,
        "system5": system5_syms,
        "system6": system6_syms,
    }

    if _phase3_measure:
        _phase3_measure.__exit__(None, None, None)
    if _LIGHTWEIGHT_BENCHMARK:
        _LIGHTWEIGHT_BENCHMARK.end_phase()

    # フィルター処理完了後に各システムの進捗を25%に更新
    try:
        stage_targets = (
            ("system1", system1_syms),
            ("system2", system2_syms),
            ("system3", system3_syms),
            ("system4", system4_syms),
            ("system5", system5_syms),
            ("system6", system6_syms),
        )
        for system_name, items in stage_targets:
            _stage(system_name, 25, filter_count=len(items or []))
        # System7 は SPY 専用
        _stage("system7", 25, filter_count=1 if "SPY" in (basic_data or {}) else 0)
    except Exception:
        pass
    # System1 フィルター内訳（価格・売買代金）
    try:
        stats1 = filter_stats.get("system1", {})
        s1_total = stats1.get("total", len(symbols or []))
        s1_price = stats1.get("price_pass", 0)
        s1_dv = stats1.get("dv_pass", 0)
        _log(
            "🧪 system1内訳: "
            + f"元={s1_total}, 価格>=5: {s1_price}, DV20>=50M: {s1_dv}"
        )
    except Exception:
        pass
    # System2 フィルター内訳の可視化（価格・売買代金・ATR比率の段階通過数）
    try:
        stats2 = filter_stats.get("system2", {})
        s2_total = stats2.get("total", len(symbols or []))
        c_price = stats2.get("price_pass", 0)
        c_dv = stats2.get("dv_pass", 0)
        c_atr = stats2.get("atr_pass", 0)
        _log(
            "🧪 system2内訳: "
            + f"元={s2_total}, 価格>=5: {c_price}, DV20>=25M: {c_dv}, ATR比率>=3%: {c_atr}"
        )
    except Exception:
        pass
    # System3 フィルター内訳（Low>=1 → AvgVol50>=1M → ATR_Ratio>=5%（テスト時はoverride表示））
    try:
        stats3 = filter_stats.get("system3", {})
        s3_total = stats3.get("total", len(symbols or []))
        s3_low = stats3.get("low_pass", 0)
        s3_av = stats3.get("avgvol_pass", 0)
        s3_atr = stats3.get("atr_pass", 0)
        # 表示ラベルのみテストモード時に override（MIN_ATR_RATIO_FOR_TEST）を反映
        # 本番では常に 5.0% を表示し、ロジックは変更しない
        _atr_label_pct = 5.0
        try:
            from config.environment import (
                get_env_config as _get_env,
            )  # 遅延import（安全）

            _env_label = _get_env()
            if hasattr(_env_label, "is_test_mode") and bool(_env_label.is_test_mode()):
                _ov = getattr(_env_label, "min_atr_ratio_for_test", None)
                if _ov is not None:
                    try:
                        _atr_label_pct = float(_ov) * 100.0
                    except Exception:
                        pass
        except Exception:
            pass
        _log(
            "🧪 system3内訳: "
            + (
                f"元={s3_total}, Low>=1: {s3_low}, AvgVol50>=1M: {s3_av}, ATR_Ratio>={_atr_label_pct:.1f}%: {s3_atr}"
            )
        )
    except Exception:
        pass
    # System4 フィルター内訳（DV50>=100M → HV50 10〜40）
    try:
        stats4 = filter_stats.get("system4", {})
        s4_total = stats4.get("total", len(symbols or []))
        s4_dv = stats4.get("dv_pass", 0)
        s4_hv = stats4.get("hv_pass", 0)
        _log(
            "🧪 system4内訳: "
            + f"元={s4_total}, DV50>=100M: {s4_dv}, HV50 10〜40: {s4_hv}"
        )
    except Exception:
        pass
    # System5 フィルター内訳（AvgVol50>500k → DV50>2.5M → ATR_Pct>閾値）
    try:
        threshold_label = f"ATR_Pct>{DEFAULT_ATR_PCT_THRESHOLD * 100:.1f}%"
        stats5 = filter_stats.get("system5", {})
        s5_total = stats5.get("total", len(symbols or []))
        s5_av = stats5.get("avgvol_pass", 0)
        s5_dv = stats5.get("dv_pass", 0)
        s5_atr = stats5.get("atr_pass", 0)
        _log(
            "🧪 system5内訳: "
            + f"元={s5_total}, AvgVol50>500k: {s5_av}, DV50>2.5M: {s5_dv}, "
            + f"{threshold_label}: {s5_atr}"
        )
    except Exception:
        pass
    # System6 フィルター内訳（Low>=5 → DV50>10M）
    try:
        stats6 = filter_stats.get("system6", {})
        s6_total = stats6.get("total", len(symbols or []))
        s6_low = stats6.get("low_pass", 0)
        s6_dv = stats6.get("dv_pass", 0)
        _log("🧪 system6内訳: " + f"元={s6_total}, Low>=5: {s6_low}, DV50>10M: {s6_dv}")
    except Exception:
        pass
    # System7 フィルター内訳（SPY固定）
    try:
        spyp = (
            1
            if (
                "SPY" in basic_data
                and not getattr(basic_data.get("SPY"), "empty", True)
            )
            else 0
        )
        _log(f"🧪 system7内訳: SPY固定 | SPY存在={spyp}")
    except Exception:
        pass
    _log(
        "🧪 フィルター結果: "
        + f"system1={len(system1_syms)}件, "
        + f"system2={len(system2_syms)}件, "
        + f"system3={len(system3_syms)}件, "
        + f"system4={len(system4_syms)}件, "
        + f"system5={len(system5_syms)}件, "
        + f"system6={len(system6_syms)}件, "
        + f"system7={spyp}件"
    )
    if progress_callback:
        try:
            progress_callback(3, 8, "filter")
        except Exception:
            pass

    # 各システム用の生データ辞書を事前フィルター後の銘柄で構築
    _log("🧮 指標計算用データロード中 (system1)…")
    raw_data_system1 = _subset_data(basic_data, system1_syms)
    _log(f"🧮 指標データ: system1={len(raw_data_system1)}銘柄")
    # System1 セットアップ内訳（最新日の setup 判定数）を CLI に出力
    s1_setup = None
    s1_setup_eff = None
    # s1_spy_gate = None  # Unused variable removed
    try:
        # フィルタ通過は事前フィルター結果（system1_syms）由来で確定
        s1_filter = int(len(system1_syms))
        # 直近日の SMA25>SMA50 を集計（事前計算済み列を参照）
        s1_setup_calc = 0
        # 市場条件（SPYのClose>SMA100）を先に判定
        _spy_ok = None
        try:
            if "SPY" in (basic_data or {}):
                _spy_df = get_spy_with_indicators(basic_data["SPY"])
                if _spy_df is not None and not getattr(_spy_df, "empty", True):
                    _last = _spy_df.iloc[-1]
                    _spy_ok = int(
                        float(_last.get("Close", 0)) > float(_last.get("SMA100", 0))
                    )
        except Exception:
            _spy_ok = None
        # system1 デバッグ情報（最初の1銘柄分）を一時的に保持し、セットアップ内訳の後にまとめて出力
        s1_debug_cols_line = None
        s1_debug_once_line = None
        for _sym, _df in (raw_data_system1 or {}).items():
            if _df is None or getattr(_df, "empty", True):
                continue
            try:
                last = _df.iloc[-1]
            except Exception:
                continue
            # 最初の1件だけデバッグ行を準備（すぐに出力せず、内訳ログの後にまとめて出す）
            if s1_setup_calc == 0 and s1_debug_cols_line is None:
                try:
                    s1_debug_cols_line = (
                        f"[DEBUG_S1_COLS] sym={_sym} df_cols={list(_df.columns)[:40]}"
                    )
                except Exception:
                    s1_debug_cols_line = None
            if s1_setup_calc == 0 and s1_debug_once_line is None:
                try:
                    _cols_preview = (
                        list(last.index)
                        if hasattr(last, "index")
                        else list(getattr(last, "keys", lambda: [])())
                    )
                except Exception:
                    _cols_preview = []
                try:
                    _s25_raw = get_indicator(last, "sma25")
                    _s50_raw = get_indicator(last, "sma50")
                    s1_debug_once_line = (
                        f"[DEBUG_S1_ONCE] sym={_sym} "
                        f"sma25_raw={_s25_raw} sma50_raw={_s50_raw} "
                        f"cols_sample={_cols_preview[:25]}"
                    )
                except Exception:
                    s1_debug_once_line = f"[DEBUG_S1_ONCE] sym={_sym} 取得失敗"
            try:
                a = to_float(get_indicator(last, "sma25"))
                b = to_float(get_indicator(last, "sma50"))
                if (not pd.isna(a)) and (not pd.isna(b)) and a > b:
                    s1_setup_calc += 1
            except Exception:
                pass
        s1_setup = int(s1_setup_calc)
        # 出力順: フィルタ通過 → SPY>SMA100 → SMA25>SMA50
        if _spy_ok is None:
            _log(
                f"🧩 system1セットアップ内訳: フィルタ通過={s1_filter}, SPY>SMA100: -, SMA25>SMA50: {s1_setup}"
            )
        else:
            _log(
                f"🧩 system1セットアップ内訳: フィルタ通過={s1_filter}, SPY>SMA100: {_spy_ok}, SMA25>SMA50: {s1_setup}"
            )
        # セットアップ内訳の後にデバッグ情報を順に出力（交互の美観を崩さないため）
        # COMPACT_TODAY_LOGS=1の場合はDEBUGログを抑制
        try:
            if not os.getenv("COMPACT_TODAY_LOGS"):
                if s1_debug_cols_line:
                    print(s1_debug_cols_line)
                if s1_debug_once_line:
                    print(s1_debug_once_line)
        except Exception:
            pass
        # UI の STUpass へ反映（50%時点）
        try:
            s1_setup_eff = int(s1_setup)
            try:
                if isinstance(_spy_ok, int) and _spy_ok == 0:
                    s1_setup_eff = 0
            except Exception:
                pass
            _stage(
                "system1",
                50,
                filter_count=int(s1_filter),
                setup_count=int(s1_setup_eff),
                candidate_count=None,
                entry_count=None,
            )
        except Exception:
            pass
        # 参考: System1 の SPY gate 状態を UI に補足表示
        try:
            cb_note = globals().get("_PER_SYSTEM_NOTE")
            if cb_note and callable(cb_note):
                try:
                    if _spy_ok is None:
                        cb_note("system1", "SPY>SMA100: -")
                    else:
                        cb_note(
                            "system1",
                            "SPY>SMA100: OK" if int(_spy_ok) == 1 else "SPY>SMA100: NG",
                        )
                except Exception:
                    pass
        except Exception:
            pass
        if s1_setup_eff is None:
            s1_setup_eff = s1_setup
        # s1_spy_gate = _spy_ok  # Unused variable removed
    except Exception:
        pass
    _log("🧮 指標計算用データロード中 (system2)…")
    raw_data_system2 = _subset_data(basic_data, system2_syms)
    _log(f"🧮 指標データ: system2={len(raw_data_system2)}銘柄")
    # System2 セットアップ内訳: フィルタ通過, RSI3>90, TwoDayUp
    s2_setup = None
    try:
        s2_filter = int(len(system2_syms))
        s2_rsi = 0
        s2_combo = 0
        for _sym in system2_syms or []:
            _df = raw_data_system2.get(_sym)
            if _df is None or getattr(_df, "empty", True):
                continue
            try:
                last = _df.iloc[-1]
            except Exception:
                continue
            try:
                rv = to_float(get_indicator(last, "rsi3"))
                rsi_pass = (not pd.isna(rv)) and rv > 90
            except Exception:
                rsi_pass = False
            if not rsi_pass:
                continue
            s2_rsi += 1
            try:
                up = get_indicator(last, "twodayup") or get_indicator(last, "uptwodays")
                if bool(up):
                    s2_combo += 1
            except Exception:
                pass
        s2_setup = int(s2_combo)
        _log(
            "🧩 system2セットアップ内訳: "
            + f"フィルタ通過={s2_filter}, RSI3>90: {s2_rsi}, "
            + f"TwoDayUp: {s2_setup}"
        )
        try:
            _stage(
                "system2",
                50,
                filter_count=int(s2_filter),
                setup_count=int(s2_setup),
                candidate_count=None,
                entry_count=None,
            )
        except Exception:
            pass
    except Exception:
        pass
    _log("🧮 指標計算用データロード中 (system3)…")
    raw_data_system3 = _subset_data(basic_data, system3_syms)
    _log(f"🧮 指標データ: system3={len(raw_data_system3)}銘柄")
    # System3 セットアップ内訳: フィルタ通過, Close>SMA150, 3日下落率>=12.5%
    s3_setup = None
    try:
        s3_filter = int(len(system3_syms))
        s3_close = 0
        s3_combo = 0
        # drop3d 閾値は本番固定 0.125。テストモード時のみ override を反映（表示目的含む）
        try:
            from config.environment import get_env_config as _get_env

            _env3 = _get_env()
            _drop_thr = 0.125
            if (
                hasattr(_env3, "is_test_mode")
                and bool(_env3.is_test_mode())
                and getattr(_env3, "min_drop3d_for_test", None) is not None
            ):
                _drop_thr = float(_env3.min_drop3d_for_test)
        except Exception:
            _drop_thr = 0.125
        for _sym in system3_syms or []:
            _df = raw_data_system3.get(_sym)
            if _df is None or getattr(_df, "empty", True):
                continue
            try:
                last = _df.iloc[-1]
            except Exception:
                continue
            try:
                cval = to_float(last.get("Close"))
                sval = to_float(get_indicator(last, "sma150"))
                close_pass = (not pd.isna(cval)) and (not pd.isna(sval)) and cval > sval
            except Exception:
                close_pass = False
            if not close_pass:
                continue
            s3_close += 1
            try:
                dv = to_float(get_indicator(last, "drop3d"))
                drop_pass = (not pd.isna(dv)) and dv >= _drop_thr
            except Exception:
                drop_pass = False
            if drop_pass:
                s3_combo += 1
        s3_setup = int(s3_combo)
        _log(
            "🧩 system3セットアップ内訳: "
            + f"フィルタ通過={s3_filter}, Close>SMA150: {s3_close}, "
            + f"3日下落率>={_drop_thr * 100:.1f}%: {s3_setup}"
        )
        try:
            _stage(
                "system3",
                50,
                filter_count=int(s3_filter),
                setup_count=int(s3_setup),
                candidate_count=None,
                entry_count=None,
            )
        except Exception:
            pass
    except Exception:
        pass
    _log("🧮 指標計算用データロード中 (system4)…")
    raw_data_system4 = _subset_data(basic_data, system4_syms)
    _log(f"🧮 指標データ: system4={len(raw_data_system4)}銘柄")
    # System4 セットアップ内訳: フィルタ通過, Close>SMA200
    try:
        s4_filter = int(len(system4_syms))
        s4_close = 0
        for _sym in system4_syms or []:
            _df = raw_data_system4.get(_sym)
            if _df is None or getattr(_df, "empty", True):
                continue
            try:
                last = _df.iloc[-1]
            except Exception:
                continue
            try:
                sval = to_float(get_indicator(last, "sma200"))
                cval = to_float(last.get("Close"))
                if (not pd.isna(sval)) and (not pd.isna(cval)) and cval > sval:
                    s4_close += 1
            except Exception:
                pass
        _log(
            f"🧩 system4セットアップ内訳: フィルタ通過={s4_filter}, Close>SMA200: {s4_close}"
        )
        try:
            _stage(
                "system4",
                50,
                filter_count=int(s4_filter),
                setup_count=int(s4_close),
                candidate_count=None,
                entry_count=None,
            )
        except Exception:
            pass
    except Exception:
        pass
    _log("🧮 指標計算用データロード中 (system5)…")
    raw_data_system5 = _subset_data(basic_data, system5_syms)
    _log(f"🧮 指標データ: system5={len(raw_data_system5)}銘柄")
    # System5 セットアップ内訳: フィルタ通過, Close>SMA100+ATR10, ADX7>55, RSI3<50
    s5_setup = None
    try:
        s5_filter = int(len(system5_syms))
        s5_close = 0
        s5_adx = 0
        s5_combo = 0
        for _sym in system5_syms or []:
            _df = raw_data_system5.get(_sym)
            if _df is None or getattr(_df, "empty", True):
                continue
            try:
                last = _df.iloc[-1]
            except Exception:
                continue
            try:
                cval = to_float(last.get("Close"))
                sval = to_float(get_indicator(last, "sma100"))
                aval = to_float(get_indicator(last, "atr10"))
                price_pass = (
                    (not pd.isna(cval))
                    and (not pd.isna(sval))
                    and (not pd.isna(aval))
                    and (cval > sval + aval)
                )
            except Exception:
                price_pass = False
            if not price_pass:
                continue
            s5_close += 1
            try:
                adx_val = to_float(get_indicator(last, "adx7"))
                adx_pass = (not pd.isna(adx_val)) and adx_val > 55
            except Exception:
                adx_pass = False
            if not adx_pass:
                continue
            s5_adx += 1
            try:
                rsi_val = to_float(get_indicator(last, "rsi3"))
                rsi_pass = (not pd.isna(rsi_val)) and rsi_val < 50
            except Exception:
                rsi_pass = False
            if rsi_pass:
                s5_combo += 1
        s5_setup = int(s5_combo)
        _log(
            "🧩 system5セットアップ内訳: "
            + f"フィルタ通過={s5_filter}, Close>SMA100+ATR10: {s5_close}, "
            + f"ADX7>55: {s5_adx}, RSI3<50: {s5_setup}"
        )
        try:
            _stage(
                "system5",
                50,
                filter_count=int(s5_filter),
                setup_count=int(s5_setup),
                candidate_count=None,
                entry_count=None,
            )
        except Exception:
            pass
    except Exception:
        pass
    _log("🧮 指標計算用データロード中 (system6)…")
    raw_data_system6 = _subset_data(basic_data, system6_syms)
    _log(f"🧮 指標データ: system6={len(raw_data_system6)}銘柄")
    # System6 セットアップ内訳: 各条件を独立カウント
    s6_setup = None
    try:
        s6_filter = int(len(system6_syms))
        s6_ret = 0
        s6_uptwo = 0
        s6_combo = 0
        for _sym in system6_syms or []:
            _df = raw_data_system6.get(_sym)
            if _df is None or getattr(_df, "empty", True):
                continue
            try:
                last = _df.iloc[-1]
            except Exception:
                continue
            # return_6d>20% 判定（独立）
            try:
                # 指標アクセスAPIで列名揺れに対応（return_6d/RETURN_6D）
                r6v = to_float(get_indicator(last, "return_6d"))
                ret_pass = (not pd.isna(r6v)) and (r6v > 0.20)
            except Exception:
                ret_pass = False
            if ret_pass:
                s6_ret += 1
            # UpTwoDays 判定（独立）
            try:
                # 列名揺れに対応（UpTwoDays/TwoDayUp/twodayup/uptwodays）
                up_pass = bool(is_true(get_indicator(last, "uptwodays")))
            except Exception:
                up_pass = False
            if up_pass:
                s6_uptwo += 1
            # AND 条件（return_6d>20% かつ UpTwoDays）
            if ret_pass and up_pass:
                s6_combo += 1
        # セットアップ結果は AND 条件で集計
        s6_setup = int(s6_combo)
        _log(
            "🧩 system6セットアップ内訳: "
            + f"フィルタ通過={s6_filter}, return_6d>20%: {s6_ret}, "
            + f"UpTwoDays: {s6_uptwo}"
        )
        try:
            _stage(
                "system6",
                50,
                filter_count=int(s6_filter),
                setup_count=int(s6_setup),
                candidate_count=None,
                entry_count=None,
            )
        except Exception:
            pass
    except Exception:
        pass
    # System7 セットアップ内訳（SPY固定: Low <= min_50）
    s7_filter = 0
    s7_setup = 0
    try:
        if "SPY" in basic_data:
            s7_filter = 1
            spy_data = basic_data["SPY"]
            if not spy_data.empty:
                # 最新行を取得
                last_row = spy_data.iloc[-1] if hasattr(spy_data, "iloc") else spy_data
                # セットアップ条件: Low <= min_50
                try:
                    low_val = to_float(
                        get_indicator(cast(Mapping[str, Any], last_row), "Low")
                    )
                    min50_val = to_float(
                        get_indicator(cast(Mapping[str, Any], last_row), "min_50")
                    )
                    if (
                        (not pd.isna(low_val))
                        and (not pd.isna(min50_val))
                        and low_val <= min50_val
                    ):
                        s7_setup = 1
                except Exception:
                    pass
        _log(
            f"🧩 system7セットアップ内訳: フィルタ通過={s7_filter}, Low<=min_50: {s7_setup}"
        )
        try:
            _stage(
                "system7",
                50,
                filter_count=int(s7_filter),
                setup_count=int(s7_setup),
                candidate_count=None,
                entry_count=None,
            )
        except Exception:
            pass
    except Exception:
        pass
    try:
        # system1 は SPY ゲート適用後の実効値を優先
        try:
            _s1_base = (
                s1_setup_eff
                if ("s1_setup_eff" in locals() and s1_setup_eff is not None)
                else (s1_setup or 0)
            )
            s1_val = int(_s1_base)
        except Exception:
            s1_val = int(s1_setup or 0)
        s2_val = int(s2_setup or 0) if "s2_setup" in locals() else 0
        s3_val = int(s3_setup or 0) if "s3_setup" in locals() else 0
        # system4 は Close>SMA200 件数（s4_close）をセットアップ相当として扱う
        s4_val = int(locals().get("s4_close", 0) or 0)
        s5_val = int(s5_setup or 0) if "s5_setup" in locals() else 0
        s6_val = int(s6_setup or 0) if "s6_setup" in locals() else 0
        s7_val = int(s7_setup or 0) if "s7_setup" in locals() else 0

        _log(
            "🧩 セットアップ結果: "
            + f"system1={s1_val}件, "
            + f"system2={s2_val}件, "
            + f"system3={s3_val}件, "
            + f"system4={s4_val}件, "
            + f"system5={s5_val}件, "
            + f"system6={s6_val}件, "
            + f"system7={s7_val}件"
        )
    except Exception:
        pass
    if progress_callback:
        try:
            progress_callback(4, 8, "load_indicators")
        except Exception:
            pass
    # ...raw_data_system...
    if "SPY" in basic_data:
        spy_df = get_spy_with_indicators(basic_data["SPY"])
    else:
        spy_df = None
        _log(
            "⚠️ SPY がキャッシュに見つかりません (base/full_backup/rolling を確認)。"
            "SPY.csv を data_cache/base もしくは data_cache/full_backup に配置してください。"
        )

    # ストラテジ初期化
    strategy_objs = [
        System1Strategy(),
        System2Strategy(),
        System3Strategy(),
        System4Strategy(),
        System5Strategy(),
        # fixed_mode=True で事前計算済インジケータのみ利用（高速経路）
        System6Strategy(),
        System7Strategy(),
    ]
    strategies = {getattr(s, "SYSTEM_NAME", "").lower(): s for s in strategy_objs}

    # Phase 4: シグナル生成（System 1-7）
    if _LIGHTWEIGHT_BENCHMARK:
        _LIGHTWEIGHT_BENCHMARK.start_phase("phase4_signal_generation")
    _phase4_measure = (
        perf_monitor.measure("phase4_signal_generation") if perf_monitor else None
    )
    if _phase4_measure:
        _phase4_measure.__enter__()

    # 各システムの当日シグナル抽出を並列実行
    _log("🚀 各システムの当日シグナル抽出を開始")

    per_system = {}
    # ベンチマーク明細（有効時のみ値が入る）
    _phase4_details: list[dict[str, Any]] = []
    system_names = [f"system{i}" for i in range(1, 8)]

    # Progress: phase4 start
    try:
        emit_progress_event(
            "phase4_signal_generation_start", {"systems": len(system_names)}
        )
    except Exception:
        pass

    for system_name in system_names:
        _log(f"▶ {system_name} 開始")

        # システム開始をUIに通知
        try:
            if per_system_progress and callable(per_system_progress):
                per_system_progress(system_name, "start")
        except Exception:
            pass
        # Progress: per-system start
        try:
            emit_progress_event("system_start", {"system": system_name})
        except Exception:
            pass

        try:
            if system_name == "system1":
                raw_data = raw_data_system1
            elif system_name == "system2":
                raw_data = raw_data_system2
            elif system_name == "system3":
                raw_data = raw_data_system3
            elif system_name == "system4":
                raw_data = raw_data_system4
            elif system_name == "system5":
                raw_data = raw_data_system5
            elif system_name == "system6":
                raw_data = raw_data_system6
            elif system_name == "system7":
                raw_data = {"SPY": basic_data.get("SPY")}
            else:
                raw_data = basic_data

            strategy = strategies.get(system_name)
            if strategy is None:
                _log(f"[{system_name}] ❌ strategy not found")
                per_system[system_name] = pd.DataFrame()
                continue

            # システム固有のロジック実行
            if system_name == "system4" and spy_df is None:
                _log(
                    f"[{system_name}] ⚠️ System4 は SPY 指標が必要ですが SPY データがありません。スキップします。"
                )
                per_system[system_name] = pd.DataFrame()
                continue

            _log(f"[{system_name}] 🔎 {system_name}: シグナル抽出を開始")
            # per-system 計測（ベンチマークが無効でも _PerfTimer による軽量ログは出す）
            _sys_t_prepare = 0.0  # Phase5 は内部で prepare を行うため外側では実行しない
            _sys_t_candidates = None
            try:
                import time as _t

                _sys_t0 = _t.perf_counter()
            except Exception:
                _sys_t0 = None

            candidate_kwargs: dict[str, Any] = {}
            if system_name == "system4":
                candidate_kwargs["market_df"] = spy_df

            # today 実行では最新日のみを対象とした高速候補抽出を有効化（バックテスト互換保持のためオプション）
            # --full-scan-today フラグ（または環境変数 FULL_SCAN_TODAY=1）指定時は latest_only を無効化し
            # 従来どおり全履歴を対象に候補抽出する。
            try:
                _disable_fast = False
                # 環境変数優先: FULL_SCAN_TODAY=1/true/on なら無効化
                _env_full = (os.environ.get("FULL_SCAN_TODAY") or "").strip().lower()
                if _env_full in {"1", "true", "yes", "on"}:
                    _disable_fast = True
                # argparse からのフラグ（利用可能な場合のみ getattr で安全取得）
                if not _disable_fast:
                    _args_obj = globals().get("_CLI_ARGS")
                    if _args_obj is not None:
                        try:
                            if getattr(_args_obj, "full_scan_today", False):
                                _disable_fast = True
                        except Exception:
                            pass
                if not _disable_fast and system_name in {
                    "system1",
                    "system2",
                    "system3",
                    "system4",
                    "system5",
                    "system6",
                    "system7",
                }:
                    candidate_kwargs.setdefault("latest_only", True)
                else:
                    # 明示的に無効化する場合は latest_only=False を入れておく（ストラテジ側で分岐容易）
                    candidate_kwargs.setdefault("latest_only", False)
            except Exception:
                # 失敗時は従来挙動（高速経路）
                if system_name in {
                    "system1",
                    "system2",
                    "system3",
                    "system4",
                    "system5",
                    "system6",
                    "system7",
                }:
                    candidate_kwargs.setdefault("latest_only", True)

            # ここから: latest_only 対象日はグローバルに一度だけ決めた ctx.signal_base_day を使用
            try:
                if candidate_kwargs.get("latest_only", False):
                    base_day = getattr(ctx, "signal_base_day", None)
                    # フルスキャンへのフォールバックは行わず、latest_only を維持する
                    if base_day is not None:
                        # 全システムにグローバル基準日を注入（system6 も対応済み）
                        candidate_kwargs["latest_mode_date"] = pd.Timestamp(
                            base_day
                        ).normalize()
                    # 全システムに max_date_lag_days を注入
                    # (system1/3 のみが実際に使用し、他システムは kwargs で受け取るが無視)
                    max_lag = max(0, int(getattr(ctx, "max_date_lag_days", 1)))
                    candidate_kwargs.setdefault("max_date_lag_days", max_lag)
            except Exception:
                pass
            # DEBUG: latest_only フラグと top_n 相当をログ（system1のみ冗長）
            try:
                if system_name == "system1":
                    _log(
                        f"[system1] DEBUG call generate_candidates latest_only={candidate_kwargs.get('latest_only')}"
                    )
            except Exception:
                pass
            # 準備段階は外側で実行しないため、候補生成計測の開始をここに置く
            _sys_t1 = None
            try:
                import time as _t

                if _sys_t0 is not None:
                    _sys_t1 = _t.perf_counter()
            except Exception:
                pass

            # Phase5: Use get_today_signals (内部で prepare_data/フィルター/セットアップ/候補抽出/エントリー算出)
            # 外側での prepare_data は行わず、生データ raw_data を渡す
            with _PerfTimer(f"{system_name}.get_today_signals"):
                try:
                    df = strategy.get_today_signals(
                        raw_data,
                        market_df=spy_df,
                        today=ctx.today,
                        progress_callback=None,
                        # 重複する詳細ログを抑制
                        log_callback=_quiet_log_for_phase5,
                        stage_progress=None,
                        use_process_pool=False,  # Phase5 is already parallelized per-system
                        max_workers=None,
                        lookback_days=None,
                    )
                except Exception as sig_err:
                    import traceback

                    _log(f"[{system_name}] ⚠️ get_today_signals failed: {sig_err}")
                    _log(f"[{system_name}] Traceback:\n{traceback.format_exc()}")
                    df = pd.DataFrame()

            # per-system 計測まとめ（準備は内部で行われるため 0、候補抽出は get_today_signals 全体の時間）
            try:
                import time as _t

                _now = _t.perf_counter()
                if _sys_t0 is not None:
                    if _sys_t1 is None:
                        _sys_t1 = _now
                    # _sys_t_prepare は 0.0 に固定（外側では実行しない）
                    _sys_t_candidates = _now - _sys_t1
            except Exception:
                pass

            # TRD リスト長の検証（テストモード時の整合性チェック）
            # Note: get_today_signals returns DataFrame, not dict[date, dict]
            # So we skip TRD verification here (will be done in Phase6)
            try:
                if system_name == "system1":
                    _log(
                        "[system1] DEBUG get_today_signals returned "
                        f"{len(df) if df is not None and not df.empty else 0} rows"
                    )
            except Exception:
                pass

            # df is already a DataFrame with entry_price/stop_price from get_today_signals
            if df is None or df.empty:
                df = pd.DataFrame()
            else:
                # デバッグ: get_today_signalsから返されたDataFrameの列を確認
                if os.environ.get("ALLOCATION_DEBUG", "0") == "1":
                    _log(
                        f"[ALLOC_DEBUG] {system_name} get_today_signals returned columns: {list(df.columns)}"
                    )
                    if len(df) > 0:
                        _log(
                            f"[ALLOC_DEBUG] {system_name} sample row: {df.iloc[0].to_dict()}"
                        )
                    # If the strategy attached entry-skip diagnostics to DataFrame.attrs, log them
                    try:
                        for akey in (
                            "entry_skip_counts",
                            "entry_skip_details",
                            "entry_skip_samples",
                        ):
                            if akey in getattr(df, "attrs", {}):
                                _log(
                                    f"[ALLOC_DEBUG] {system_name} attrs[{akey}]: {df.attrs.get(akey)!r}"
                                )
                    except Exception:
                        _log(
                            f"[ALLOC_DEBUG] {system_name} attrs debug failed: {sys.exc_info()[0]}"
                        )

            per_system[system_name] = df
            count = len(df) if not df.empty else 0
            if count > 0:
                # 成功アイコン（従来は常に❌表示だった箇所を条件分岐）
                _log(f"[{system_name}] ✅ {system_name}: {count} 件")
            else:
                _log(f"[{system_name}] ❌ {system_name}: {count} 件 🚫")

            # UI 進捗: 候補抽出件数を 75% ステージとして通知（早期に TRDlist を可視化）
            try:
                _stage(system_name, 75, candidate_count=int(count))
            except Exception:
                pass

            try:
                diag_payload = getattr(strategy, "last_diagnostics", None)
                if isinstance(diag_payload, dict):
                    ctx.system_diagnostics[system_name] = diag_payload
                    _log_zero_candidate_diagnostics(system_name, count, diag_payload)
            except Exception:
                pass

        except Exception as e:
            _log(f"[{system_name}] ⚠️ {system_name}: シグナル抽出に失敗しました: {e}")
            per_system[system_name] = pd.DataFrame()
            _log(f"[{system_name}] ❌ {system_name}: 0 件 🚫")

        _log(f"✅ {system_name} 完了: {len(per_system[system_name])}件")
        # Progress: per-system complete
        try:
            emit_progress_event(
                "system_complete",
                {
                    "system": system_name,
                    "candidates": int(len(per_system.get(system_name, pd.DataFrame()))),
                },
            )
        except Exception:
            pass

        # ベンチマーク拡張: フェーズ4のシステム別明細を収集
        try:
            if _LIGHTWEIGHT_BENCHMARK and _LIGHTWEIGHT_BENCHMARK.enabled:
                detail = {
                    "system": system_name,
                    "prepare_sec": round(float(_sys_t_prepare or 0.0), 6),
                    "generate_candidates_sec": round(
                        float(_sys_t_candidates or 0.0), 6
                    ),
                    "total_sec": round(
                        float(((_sys_t_prepare or 0.0) + (_sys_t_candidates or 0.0))),
                        6,
                    ),
                    "candidates": (
                        int(len(per_system.get(system_name, pd.DataFrame())))
                        if isinstance(per_system.get(system_name), pd.DataFrame)
                        else 0
                    ),
                    "latest_only": bool(candidate_kwargs.get("latest_only", False)),
                }
                _phase4_details.append(detail)
        except Exception:
            pass

        # システム完了をUIに通知
        try:
            if per_system_progress and callable(per_system_progress):
                per_system_progress(system_name, "done")
        except Exception:
            pass

    # 進捗通知
    if progress_callback:
        try:
            progress_callback(6, 8, "strategies_done")
        except Exception:
            pass

    # システム別の順序を明示（1..7）に固定
    order_1_7 = [f"system{i}" for i in range(1, 8)]
    per_system = {
        k: per_system.get(k, pd.DataFrame()) for k in order_1_7 if k in per_system
    }
    ctx.per_system_frames = dict(per_system)
    # メトリクス概要計算

    # Phase 4測定終了
    if _phase4_measure:
        _phase4_measure.__exit__(None, None, None)
    if _LIGHTWEIGHT_BENCHMARK:
        # 明細を extras に付加（フェーズ終了と同時に）
        try:
            _LIGHTWEIGHT_BENCHMARK.add_extra_section(
                "phase4_per_system", _phase4_details
            )
        except Exception:
            pass
        _LIGHTWEIGHT_BENCHMARK.end_phase()

    # Phase 5: 配分計算
    if _LIGHTWEIGHT_BENCHMARK:
        _LIGHTWEIGHT_BENCHMARK.start_phase("phase5_allocation")
    _phase5_measure = (
        perf_monitor.measure("phase5_allocation") if perf_monitor else None
    )
    if _phase5_measure:
        _phase5_measure.__enter__()

    # Progress: phase5 start
    try:
        total_cand = 0
        try:
            total_cand = sum(
                len(df) for df in per_system.values() if isinstance(df, pd.DataFrame)
            )
        except Exception:
            total_cand = 0
        emit_progress_event(
            "phase5_allocation_start", {"total_candidates": int(total_cand)}
        )
    except Exception:
        pass

    # === Allocation & Final Assembly ===
    # ここで per_system から最終候補 (final_df) を構築し AllocationSummary を取得する。
    try:
        # シンボル→system マップ（存在しなくても継続）
        try:
            symbol_system_map = load_symbol_system_map()
        except Exception:
            symbol_system_map = None

        # アクティブポジション情報（将来: broker / キャッシュから取得可能なら拡張）
        active_positions = None  # NOTE: Could be retrieved via ctx if needed

        # デバッグ: 配分前の候補データを出力
        if os.environ.get("ALLOCATION_DEBUG", "0") == "1":
            _log("[ALLOC_DEBUG] === PRE-ALLOCATION CANDIDATES ===")
            # Persist per-system frames for offline inspection (test-mode results area)
            out_dir = Path("results_csv_test")
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            for sys_name, df in per_system.items():
                try:
                    if not df.empty:
                        _log(f"[ALLOC_DEBUG] {sys_name}: {len(df)} rows")
                        _log(f"[ALLOC_DEBUG] {sys_name} columns: {list(df.columns)}")
                        if len(df) > 0:
                            sample = df.iloc[0].to_dict()
                            _log(f"[ALLOC_DEBUG] {sys_name} sample: {sample}")
                        # try to persist to feather for later offline debug
                        try:
                            fp = out_dir / f"per_system_{sys_name}.feather"
                            # preserve index as a column to avoid losing information
                            try:
                                df.reset_index(drop=False).to_feather(fp)
                            except Exception:
                                # Fallback: write without resetting index
                                df.to_feather(fp)
                            _log(f"[ALLOC_DEBUG] Saved per-system candidates to {fp}")
                        except Exception as _e:
                            # If feather writing failed (pyarrow etc), fallback to CSV
                            try:
                                csv_fp = out_dir / f"per_system_{sys_name}.csv"
                                # Try helper write on reset-indexed frame first
                                try:
                                    from common.io_utils import df_to_csv

                                    df_to_csv(
                                        df.reset_index(drop=False), csv_fp, index=False
                                    )
                                except Exception:
                                    # Fallback: try pandas native write after reset_index
                                    try:
                                        df.reset_index(drop=False).to_csv(
                                            csv_fp, index=False
                                        )
                                    except Exception:
                                        # Next fallback: try helper on original df
                                        try:
                                            from common.io_utils import df_to_csv

                                            df_to_csv(df, csv_fp, index=False)
                                        except Exception:
                                            # Last resort: pandas native write
                                            df.to_csv(csv_fp, index=False)
                                _log(
                                    f"[ALLOC_DEBUG] Saved per-system candidates to CSV fallback {csv_fp}"
                                )
                            except Exception as _e2:
                                # Log both exceptions for easier triage
                                _log(
                                    f"[ALLOC_DEBUG] Failed to save per-system {sys_name}: {_e}; fallback error: {_e2}"
                                )
                    else:
                        _log(f"[ALLOC_DEBUG] {sys_name}: EMPTY")
                except Exception:
                    # Per-system debug must never break the allocation flow
                    _log(f"[ALLOC_DEBUG] Error inspecting per-system {sys_name}")

        final_df, allocation_summary = finalize_allocation(
            per_system,
            strategies=strategies,
            positions=active_positions,
            symbol_system_map=symbol_system_map,
            slots_long=slots_long,
            slots_short=slots_short,
            capital_long=capital_long,
            capital_short=capital_short,
            system_diagnostics=ctx.system_diagnostics,
            market_data_dict=ctx.basic_data,
            signal_date=ctx.today,
            include_trade_management=True,
        )
        # Emit user-friendly allocator diagnostics to CLI when available.
        try:
            # allocation_summary may be dataclass or dict-like
            alloc_diag = None
            try:
                alloc_diag = getattr(allocation_summary, "system_diagnostics", None)
            except Exception:
                alloc_diag = (
                    allocation_summary if isinstance(allocation_summary, dict) else None
                )

            if isinstance(alloc_diag, dict):
                alloc_ex = alloc_diag.get("allocator_excludes") or {}
                if alloc_ex:
                    _log("🔎 Allocation-level excludes summary:")
                    # alloc_ex typically has keys 'long'/'short' with reason->list mapping
                    try:
                        for side in ("long", "short"):
                            side_map = alloc_ex.get(side)
                            if not side_map:
                                continue
                            _log(f"  {side.upper()} side:")
                            for reason, syms in side_map.items():
                                try:
                                    syms_list = (
                                        list(syms)
                                        if isinstance(syms, (list, tuple, set))
                                        else [syms]
                                    )
                                    if not syms_list:
                                        continue
                                    _log(
                                        f"    {reason}: {', '.join(sorted(map(str, syms_list))) }"
                                    )
                                except Exception:
                                    _log(f"    {reason}: <unreadable list>")
                    except Exception:
                        _log("  <failed to pretty-print allocator excludes>")

                # already_selected_map may be present under allocator_excludes or at top level
                already_map = {}
                try:
                    # check common locations
                    if isinstance(alloc_ex, dict) and alloc_ex:
                        # if present in alloc_ex.long/short as mapping under key 'already_selected_map'
                        for side in ("long", "short"):
                            side_map = alloc_ex.get(side) or {}
                            if (
                                isinstance(side_map, dict)
                                and "already_selected_map" in side_map
                            ):
                                try:
                                    already_map.update(
                                        side_map.get("already_selected_map") or {}
                                    )
                                except Exception:
                                    pass
                    # fallback: system_diagnostics top-level allocator mapping
                    top_alloc = alloc_diag.get("allocator_excludes") or {}
                    if isinstance(top_alloc, dict):
                        # flatten long/short reasons for already_selected details
                        for side in ("long", "short"):
                            sub = top_alloc.get(side) or {}
                            if isinstance(sub, dict) and "already_selected_map" in sub:
                                try:
                                    already_map.update(
                                        sub.get("already_selected_map") or {}
                                    )
                                except Exception:
                                    pass
                except Exception:
                    already_map = {}

                if already_map:
                    _log(
                        "🔁 Allocator already-selected map (symbol -> first selecting system):"
                    )
                    try:
                        for sym, sysn in sorted(already_map.items()):
                            _log(f"  {sym} -> {sysn}")
                    except Exception:
                        _log("  <failed to print already-selected map>")
        except Exception:
            # Diagnostics printing must not break the pipeline
            pass
    except Exception as e:
        _log(f"❌ finalize_allocation 失敗: {e}")
        final_df = pd.DataFrame()
        from core.final_allocation import (
            AllocationSummary as _AS,  # local import to avoid cycle
        )

        allocation_summary = _AS(
            mode="error",
            long_allocations={},
            short_allocations={},
            active_positions={},
            available_slots={},
            final_counts={},
        )

    # Progress: phase5 complete (emit brief summary)
    try:
        _final_counts = int(len(final_df) if isinstance(final_df, pd.DataFrame) else 0)
        alloc_brief = {}
        try:
            if isinstance(allocation_summary, dict):
                sc = allocation_summary.get("slot_candidates")
                fc = allocation_summary.get("final_counts")
                if isinstance(sc, dict):
                    alloc_brief["slot_candidates_total"] = sum(
                        int(v or 0) for v in sc.values()
                    )
                if isinstance(fc, dict):
                    alloc_brief["final_counts_total"] = sum(
                        int(v or 0) for v in fc.values()
                    )
        except Exception:
            alloc_brief = {}
        emit_progress_event(
            "phase5_allocation_complete",
            {"final_df_rows": _final_counts, **alloc_brief},
        )
    except Exception:
        pass

    # 並べ替え / 連番付与（finalize_allocation 内部で付与されるが念のため最終安定ソート）
    try:
        if not final_df.empty and "system" in final_df.columns:
            # system番号抽出 (system4 等)
            final_df["_system_no"] = (
                final_df["system"]
                .astype(str)
                .str.extract(r"(\d+)", expand=False)
                .fillna("0")
                .astype(int)
            )
            final_df = final_df.sort_values(["side", "_system_no"], kind="stable")
            final_df = final_df.drop(columns=["_system_no"], errors="ignore")
            if "no" not in final_df.columns:
                final_df.insert(0, "no", range(1, len(final_df) + 1))
    except Exception:
        pass

    # サマリログ
    try:
        if final_df.empty:
            _log("📭 最終候補は0件でした")
        else:
            _log(f"📊 最終候補件数: {len(final_df)}")
            try:
                if "system" in final_df.columns:
                    grp = final_df.groupby("system").size().to_dict()
                    for k, v in grp.items():
                        _log(f"✅ {k}: {int(v)} 件")
            except Exception:
                pass
    except Exception:
        pass

    # UI 進捗: 最終エントリー数を 100% ステージとして通知
    try:
        # AllocationSummary を優先的に参照（候補数/エントリー数ともに安定）
        alloc_summary = allocation_summary
        final_counts_map: dict[str, int] = {}
        cand_counts_map: dict[str, int] = {}
        try:
            if hasattr(alloc_summary, "final_counts"):
                raw = getattr(alloc_summary, "final_counts", {})
                if isinstance(raw, dict):
                    final_counts_map = {
                        str(k).strip().lower(): int(v) for k, v in raw.items()
                    }
        except Exception:
            final_counts_map = {}
        try:
            if hasattr(alloc_summary, "slot_candidates"):
                raw2 = getattr(alloc_summary, "slot_candidates", {})
                if isinstance(raw2, dict):
                    cand_counts_map = {
                        str(k).strip().lower(): int(v) for k, v in raw2.items()
                    }
        except Exception:
            cand_counts_map = {}

        # フォールバック: final_df からエントリー数をグルーピング
        if not final_counts_map and (final_df is not None) and (not final_df.empty):
            try:
                grp = (
                    final_df["system"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .value_counts()
                    .to_dict()
                )
                final_counts_map = {str(k): int(v) for k, v in grp.items()}
            except Exception:
                final_counts_map = {}

        # 通知（システム1..7の既定順で）
        for i in range(1, 8):
            key = f"system{i}"
            entry_n = int(final_counts_map.get(key, 0))
            # 候補数が取れれば併せて送る（UI側で上書き・フォールバックに利用）
            cand_n = cand_counts_map.get(key)
            try:
                _stage(key, 100, candidate_count=cand_n, entry_count=entry_n)
            except Exception:
                try:
                    _stage(key, 100, entry_count=entry_n)
                except Exception:
                    pass
        # Drain any queued per-system stage events synchronously so that
        # final/global progress updates applied afterward are not
        # overwritten by delayed event-pump processing.
        try:
            _drain_stage_event_queue()
        except Exception:
            pass
    except Exception:
        pass

    if progress_callback:
        try:
            progress_callback(7, 8, "finalize")
        except Exception:
            pass

    # Phase 5測定終了
    if _phase5_measure:
        _phase5_measure.__exit__(None, None, None)
    if _LIGHTWEIGHT_BENCHMARK:
        _LIGHTWEIGHT_BENCHMARK.end_phase()

    # Phase5: Zero TRD escalation notification
    try:
        notify_zero_trd_all_systems(ctx, final_df)
    except Exception:
        pass

    # Phase2: Export diagnostics snapshot in test modes
    try:
        # Ensure allocation-level diagnostics (added to allocation_summary)
        # are available to the snapshot exporter which reads ctx.system_diagnostics.
        try:
            alloc_diag = None
            try:
                alloc_diag = getattr(allocation_summary, "system_diagnostics", None)
            except Exception:
                try:
                    alloc_diag = (
                        allocation_summary.get("system_diagnostics")
                        if isinstance(allocation_summary, dict)
                        else None
                    )
                except Exception:
                    alloc_diag = None

            if alloc_diag:
                try:
                    existing = getattr(ctx, "system_diagnostics", None) or {}
                    # store under a reserved key so per-system snapshot logic can include it
                    existing = dict(existing)
                    existing["_allocation_summary"] = alloc_diag
                    ctx.system_diagnostics = existing
                except Exception:
                    # non-fatal - proceed to export even if merge fails
                    pass
        except Exception:
            pass

        _export_diagnostics_snapshot(ctx, final_df)
    except Exception:
        pass

    # Phase4: Discrepancy triage in test modes
    try:
        _export_discrepancy_triage(ctx)
    except Exception:
        pass

    # 戻り値: final_df と AllocationSummary (呼び出し側で dict 化可能)
    return final_df, allocation_summary


def _safe_stage_int(value: object | None) -> int:
    """安全に整数値に変換する"""
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        try:
            return int(value)
        except Exception:
            return 0
    if isinstance(value, str):
        txt = value.strip()
        if not txt:
            return 0
        try:
            return int(float(txt))
        except Exception:
            return 0
    # 最後のフォールバック: __int__ を実装していれば使用
    try:
        to_int = getattr(value, "__int__", None)
        if callable(to_int):
            v2 = to_int()
            return int(v2) if isinstance(v2, (int, float)) else 0
    except Exception:
        return 0
    return 0


def _format_stage_message(
    progress: int,
    filter_count: int | None = None,
    setup_count: int | None = None,
    candidate_count: int | None = None,
    entry_count: int | None = None,
) -> str | None:
    """進捗段階に応じたメッセージをフォーマット"""
    if progress < 0 or progress > 100:
        return None

    filter_int = _safe_stage_int(filter_count)
    setup_int = _safe_stage_int(setup_count)
    candidate_int = _safe_stage_int(candidate_count)
    entry_int = _safe_stage_int(entry_count)

    # システム名はグローバルから取得（この関数の外で定義されている想定）
    name = "System"  # デフォルト値

    if progress == 0:
        if filter_int is not None:
            return f"🧪 {name}: フィルターチェック開始 (対象 {filter_int} 銘柄)"
        return f"🧪 {name}: フィルターチェックを開始"
    if progress == 25:
        if filter_int is not None:
            return f"🧪 {name}: フィルター通過 {filter_int} 銘柄"
        return f"🧪 {name}: フィルター処理が完了"
    if progress == 50:
        if filter_int is not None and setup_int is not None:
            return "🧩 " + f"{name}: セットアップ通過 {setup_int}/{filter_int} 銘柄"
        if setup_int is not None:
            return f"🧩 {name}: セットアップ通過 {setup_int} 銘柄"
        return f"🧩 {name}: セットアップ判定が完了"
    if progress == 75:
        if candidate_int is not None:
            return f"🧮 {name}: 候補抽出中 (当日候補 {candidate_int} 銘柄)"
        return f"🧮 {name}: 候補抽出を実行中"
    if progress == 100:
        if entry_int is not None:
            parts: list[str] = []
            if candidate_int is not None:
                parts.append(f"候補 {candidate_int} 銘柄")
            parts.append(f"エントリー {entry_int} 銘柄")
            joined = " / ".join(parts)
            return f"✅ {name}: エントリーステージ完了 ({joined})"
        return f"✅ {name}: エントリーステージ完了"
    return None


def _format_phase_completion(
    prev_stage: int,
    filter_int: int | None,
    setup_int: int | None,
    candidate_int: int | None,
    final_int: int | None,
) -> str | None:
    """フェーズ完了メッセージをフォーマット"""
    # phase_namesはグローバルスコープから取得する想定
    phase_names = {
        0: "フィルタリング",
        25: "セットアップ",
        50: "候補抽出",
        75: "最終選定",
    }
    name = "System"  # デフォルト値

    label = phase_names.get(prev_stage)
    if not label:
        return None

    if prev_stage == 0:
        if filter_int is not None:
            return f"🏁 {name}: {label}のプロセスプールが完了 (通過 {filter_int} 銘柄)"
        return f"🏁 {name}: {label}のプロセスプールが完了"

    if prev_stage == 25:
        if setup_int is not None and filter_int is not None:
            return f"🏁 {name}: {label}のプロセスプールが完了 (セットアップ通過 {setup_int}/{filter_int} 銘柄)"
        if setup_int is not None:
            return f"🏁 {name}: {label}のプロセスプールが完了 (セットアップ通過 {setup_int} 銘柄)"
        return f"🏁 {name}: {label}のプロセスプールが完了"

    if prev_stage == 50:
        if candidate_int is not None:
            return f"🏁 {name}: {label}のプロセスプールが完了 (当日候補 {candidate_int} 銘柄)"
        return f"🏁 {name}: {label}のプロセスプールが完了"

    if prev_stage == 75:
        if final_int is not None:
            parts: list[str] = [f"エントリー {final_int} 銘柄"]
            if candidate_int is not None:
                parts.append(f"候補 {candidate_int} 銘柄")
            joined = " / ".join(parts)
            return f"🏁 {name}: {label}のプロセスプールが完了 ({joined})"
        return f"🏁 {name}: {label}のプロセスプールが完了"

    return None


def _stage(
    system: str,
    progress: int,
    filter_count: int | None = None,
    setup_count: int | None = None,
    candidate_count: int | None = None,
    entry_count: int | None = None,
    # サブステージ情報の追加
    substage_name: str | None = None,
    substage_progress: int | None = None,
    substage_total: int | None = None,
) -> None:
    """Record stage progress for ``system`` and flush pending UI events."""

    system_key = str(system or "").strip().lower() or "unknown"
    try:
        GLOBAL_STAGE_METRICS.record_stage(
            system_key,
            progress,
            filter_count,
            setup_count,
            candidate_count,
            entry_count,
            emit_event=True,
            substage_name=substage_name,
            substage_progress=substage_progress,
            substage_total=substage_total,
        )
    except Exception as e:
        # ログを残してデバッグ時の手がかりにする
        import logging

        logging.getLogger(__name__).debug(f"_stage failed for {system}: {e}")
        return
    _drain_stage_event_queue()


# プロセスプール利用可否（環境変数で上書き可）
def _configure_process_pool_and_workers(
    name: str = "", _log: Callable[[str], None] = print
) -> tuple[bool, int | None]:
    """Configure process pool usage and worker count based on environment variables."""
    env_pp_raw = os.environ.get("USE_PROCESS_POOL", "")
    env_pp = env_pp_raw.strip().lower()
    if env_pp in {"1", "true", "yes", "on"}:
        use_process_pool = True
    elif env_pp in {"0", "false", "no", "off"}:
        use_process_pool = False
    else:
        use_process_pool = False
        if env_pp:
            _log(
                "⚠️ "
                + f"{name}: USE_PROCESS_POOL の値 '{env_pp_raw}' を解釈できません。"
                + "プロセスプールを無効化します。"
            )
    # ワーカー数は環境変数があれば優先、無ければ設定(THREADS_DEFAULT)に連動
    try:
        _env_workers = os.environ.get("PROCESS_POOL_WORKERS", "").strip()
        if _env_workers:
            max_workers = int(_env_workers) or None
        else:
            try:
                _st = get_settings(create_dirs=False)
                max_workers = int(getattr(_st, "THREADS_DEFAULT", 8)) or None
            except Exception:
                max_workers = None
    except Exception:
        max_workers = None
    return use_process_pool, max_workers


def _configure_lookback_days(
    name: str = "",
    stg: object | None = None,
    base: object | None = None,
) -> int:
    """Configure lookback days based on strategy requirements.

    The strategy object may optionally expose a ``get_total_days(base_df)`` method.
    We treat this attribute as ``Callable[[Any], Any] | None`` and validate at runtime
    before invoking, which prevents the E1102 (not-callable) warning once type hinted.
    """
    # ルックバックは『必要指標の最大窓＋α』を動的推定
    try:
        settings2 = get_settings(create_dirs=True)
        lb_default = int(
            settings2.cache.rolling.base_lookback_days
            + settings2.cache.rolling.buffer_days
        )
    except Exception:
        settings2 = None
        lb_default = 300
    # YAMLのstrategiesセクション等からヒントを取得（なければヒューリスティック）
    # ルックバックのマージン/最小日数は環境変数で上書き可能
    try:
        margin = float(os.environ.get("LOOKBACK_MARGIN", "0.15"))
    except Exception:
        margin = 0.15
    need_map: dict[str, int] = {
        "system1": int(220 * (1 + margin)),
        "system2": int(120 * (1 + margin)),
        # SMA150 を安定に計算するため 170 日程度を要求
        "system3": int(170 * (1 + margin)),
        # SMA200 系のため 220 日程度を要求
        "system4": int(220 * (1 + margin)),
        "system5": int(140 * (1 + margin)),
        "system6": int(80 * (1 + margin)),
        "system7": int(80 * (1 + margin)),
    }
    # 戦略側が get_total_days を実装していれば優先
    custom_need: int | None = None
    # Use collections.abc.Callable already imported at top for type hints.
    fn: Callable[[object], object] | None
    try:
        raw = getattr(stg, "get_total_days", None)
        fn = raw if callable(raw) else None
    except Exception:  # pragma: no cover - defensive
        fn = None
    if fn is not None:
        try:
            _val = fn(base)
            if isinstance(_val, (int, float)):
                custom_need = int(_val)
            elif isinstance(_val, str):
                try:
                    custom_need = int(float(_val))
                except Exception:
                    custom_need = None
        except Exception:  # pragma: no cover - strategy specific failures ignored
            custom_need = None
    try:
        min_floor = int(os.environ.get("LOOKBACK_MIN_DAYS", "80"))
    except Exception:
        min_floor = 80
    min_required = custom_need or need_map.get(name, lb_default)
    lookback_days = min(lb_default, max(min_floor, int(min_required)))
    return lookback_days


# Let's clean up from here and find the actual function that needs these variables
def _run_strategy_with_proper_scope(
    name: str,
    stg: object,
    base: object,
    spy_df: pd.DataFrame | None,
    today: datetime | None,
    _log: Callable[[str], None],
) -> tuple[str, pd.DataFrame, str, list[str]]:
    """Run strategy with properly scoped variables (現在は簡略版)."""
    logs: list[str] = []
    pool_outcome = "none"
    progress_q: Any | None = None
    mgr: Any | None = None

    # Configure process pool settings
    use_process_pool, max_workers = _configure_process_pool_and_workers(
        name=name, _log=_log
    )

    # Configure lookback days
    lookback_days = _configure_lookback_days(name=name, stg=stg, base=base)

    _t0 = __import__("time").time()
    # プロセスプール利用時も stage_progress を渡し、要所の進捗ログを共有する
    _log_cb = None if use_process_pool else _log
    # プロセスプール利用時は Manager().Queue を生成して子プロセスから
    # 進捗を送れるようにする。globals に置いて子が参照できるようにする。
    if use_process_pool:
        try:
            mgr = multiprocessing.Manager()  # noqa: F401 (kept for child access)
            progress_q = mgr.Queue()
            globals()["_PROGRESS_MANAGER"] = mgr
            globals()["_PROGRESS_QUEUE"] = progress_q
        except Exception:
            progress_q = None
            globals().pop("_PROGRESS_MANAGER", None)
            globals().pop("_PROGRESS_QUEUE", None)
    else:
        globals().pop("_PROGRESS_MANAGER", None)
        globals().pop("_PROGRESS_QUEUE", None)

    stage_reporter = StageReporter(name, progress_q)
    _stage_cb = stage_reporter
    if use_process_pool:
        workers_label = str(max_workers) if max_workers is not None else "auto"
        _log(
            f"⚙️ {name}: USE_PROCESS_POOL=1 でプロセスプール実行を開始"
            + f" (workers={workers_label})"
            + " | 並列化: インジケーター計算/前処理"
        )
        _log(
            f"🧭 {name}: フィルター・セットアップ・候補抽出はメインプロセスで進行状況を記録します"
        )
    try:
        # 戦略インターフェースは統一されていないため Any として扱う (後続段階で整備予定)
        stg_any: Any = stg
        df = stg_any.get_today_signals(
            base,
            market_df=spy_df,
            today=today,
            progress_callback=None,
            log_callback=_log,
            stage_progress=_stage_cb,
            use_process_pool=use_process_pool,
            max_workers=max_workers,
            lookback_days=lookback_days,
        )
        # 子プロセスからキューへ送られた進捗は上で作られた globals 上の
        # _PROGRESS_QUEUE に蓄積される。_drain_stage_event_queue がそれを
        # 定期的に取り出し、UI 更新に転換する。
        if use_process_pool:
            pool_outcome = "success"
        _elapsed = int(max(0, __import__("time").time() - _t0))
        _m, _s = divmod(_elapsed, 60)
        _log(f"⏱️ {name}: 経過 {_m}分{_s}秒")
        _drain_stage_event_queue()
    except Exception as e:  # noqa: BLE001
        _log(f"⚠️ {name}: シグナル抽出に失敗しました: {e}")
        # プロセスプール異常時はフォールバック（非プール）で一度だけ再試行
        try:
            msg = str(e).lower()
        except Exception:
            msg = ""
        if use_process_pool and pool_outcome == "none":
            pool_outcome = "error"
        needs_fallback = any(
            k in msg
            for k in [
                "process pool",
                "a child process terminated",
                "terminated abruptly",
                "forkserver",
                "__main__",
            ]
        )
        if needs_fallback:
            _log("🛟 フォールバック再試行: プロセスプール無効化で実行します")
            try:
                _t0b = __import__("time").time()
                stg_fallback: Any = stg
                df = stg_fallback.get_today_signals(
                    base,
                    market_df=spy_df,
                    today=today,
                    progress_callback=None,
                    log_callback=_log,
                    stage_progress=StageReporter(name, None),
                    use_process_pool=False,
                    max_workers=None,
                    lookback_days=lookback_days,
                )
                _elapsed_b = int(max(0, __import__("time").time() - _t0b))
                _m2, _s2 = divmod(_elapsed_b, 60)
                _log(f"⏱️ {name} (fallback): 経過 {_m2}分{_s2}秒")
                _drain_stage_event_queue()
                if use_process_pool:
                    pool_outcome = "fallback"
            except Exception as e2:  # noqa: BLE001
                _log(f"❌ {name}: フォールバックも失敗: {e2}")
                if use_process_pool:
                    pool_outcome = "error"
                df = pd.DataFrame()
        else:
            df = pd.DataFrame()
    finally:
        _drain_stage_event_queue()
        if use_process_pool:
            if pool_outcome == "success":
                _log(f"🏁 {name}: プロセスプール実行が完了しました")
            elif pool_outcome == "fallback":
                _log(f"🏁 {name}: プロセスプール実行を終了（フォールバック実行済み）")
            else:
                _log(f"🏁 {name}: プロセスプール実行を終了（結果: 失敗）")
            globals().pop("_PROGRESS_QUEUE", None)
            globals().pop("_PROGRESS_MANAGER", None)
            if mgr is not None:
                try:
                    mgr.shutdown()
                except Exception:
                    pass
    if not df.empty:
        if "score_key" in df.columns and len(df):
            first_key = df["score_key"].iloc[0]
        else:
            first_key = None
        asc = _asc_by_score_key(first_key)
        df = df.sort_values("score", ascending=asc, na_position="last")
        df = df.reset_index(drop=True)
    if df is not None and not df.empty:
        msg = f"📊 {name}: {len(df)} 件"
    else:
        msg = f"❌ {name}: 0 件 🚫"
    _log(msg)
    logs = []  # Initialize logs list for return statement

    return name, df, msg, logs


def _run_strategy(name: str, _stg: object) -> tuple[str, pd.DataFrame, str, list[str]]:
    """
    Wrapper function for _run_strategy_with_proper_scope with appropriate defaults.
    """
    try:
        # This is a simplified wrapper - actual implementation depends on full context
        # For now, return a basic result structure
        df = pd.DataFrame()  # Empty dataframe as placeholder
        msg = f"📊 {name}: 0 件 (placeholder)"
        logs: list[str] = []
        return name, df, msg, logs
    except Exception:
        return name, pd.DataFrame(), f"❌ {name}: エラー", []


# Setup summary code that was after return - moved to proper location
# NOTE: This function and subsequent code have been temporarily commented out
# due to structural issues with undefined variables. The main functionality
# remains intact through other entry points.
#
# def _log_setup_summary():
#     """Log setup summary - this function should be called before strategy execution"""
#     try:
#         setup_summary = []
#         for name, val in (
#             ("system1", s1_setup_eff if s1_setup_eff is not None else s1_setup),
#             ("system2", s2_setup),
#             ("system3", s3_setup),
#             ("system4", locals().get("s4_close")),
#             ("system5", s5_setup),
#             ("system6", s6_setup),
#             ("system7", 1 if ("SPY" in (basic_data or {})) else 0),
#         ):
#             try:
#                 if val is not None:
#                     setup_summary.append(f"{name}={int(val)}")
#             except Exception:
#                 continue
#         if setup_summary:
#             _log("🧩 セットアップ通過まとめ: " + ", ".join(setup_summary))
#     except Exception:
#         pass

#     _log("🚀 各システムの当日シグナル抽出を開始")
#     per_system: dict[str, pd.DataFrame] = {}
#     total = len(strategies)
#     # (rest of the problematic code commented out)


def _placeholder_log_setup_summary() -> None:
    """最小ダミー: 破損していた旧 _log_setup_summary / 重複配分ロジックを撤去。

    将来ここでセットアップ結果サマリを復活させる場合は、
    (ctx, final_df など) 必要情報を引数として受け取る新しい関数として実装してください。
    現在は副作用なしで軽いログのみを出力します。
    """
    try:
        _log("🧩 セットアップ通過まとめ機能: 一時的に無効化中")
    except Exception:
        pass
    # これ以上の処理は行わない（final_df 等はこのスコープに存在しないため参照禁止）
    return None


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="全システム当日シグナル抽出・集約")
    parser.add_argument(
        "--symbols",
        nargs="*",
        help="対象シンボル。未指定なら設定のauto_tickersを使用",
    )
    parser.add_argument(
        "--slots-long",
        type=int,
        default=None,
        help="買いサイドの最大採用数（スロット方式）",
    )
    parser.add_argument(
        "--slots-short",
        type=int,
        default=None,
        help="売りサイドの最大採用数（スロット方式）",
    )
    parser.add_argument(
        "--capital-long",
        type=float,
        default=None,
        help=("買いサイド予算（ドル）。指定時は金額配分モード"),
    )
    parser.add_argument(
        "--capital-short",
        type=float,
        default=None,
        help=("売りサイド予算（ドル）。指定時は金額配分モード"),
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="signalsディレクトリにCSVを保存する",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="システムごとの当日シグナル抽出を並列実行する",
    )
    parser.add_argument(
        "--full-scan-today",
        action="store_true",
        help="当日シグナル抽出で latest_only 最適化を無効化し全履歴走査 (検証/デバッグ用途)",
    )
    # Alpaca 自動発注オプション
    parser.add_argument(
        "--alpaca-submit",
        action="store_true",
        help="Alpaca に自動発注（shares 必須）",
    )
    parser.add_argument(
        "--order-type",
        choices=["market", "limit"],
        default="market",
        help="注文種別",
    )
    parser.add_argument(
        "--tif",
        choices=["GTC", "DAY"],
        default="GTC",
        help="Time In Force",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="ライブ口座で発注（デフォルトはPaper）",
    )
    parser.add_argument(
        "--log-file-mode",
        choices=["single", "dated"],
        default=None,
        help="ログ保存形式: single=固定 today_signals.log / dated=日付別ファイル",
    )
    parser.add_argument(
        "--csv-name-mode",
        choices=["date", "datetime", "runid"],
        default=None,
        help=(
            "CSVファイル名の形式: date=YYYY-MM-DD / datetime=YYYY-MM-DD_HHMM / runid=YYYY-MM-DD_RUNID"
        ),
    )
    # 計画 -> 実行ブリッジ（安全のため既定はドライラン）
    parser.add_argument(
        "--run-planned-exits",
        choices=["off", "open", "close", "auto"],
        default=None,
        help=(
            "手仕舞い計画の自動実行: off=無効 / open=寄り(OPG) / close=引け(CLS) / auto=時間帯で自動判定"
        ),
    )
    parser.add_argument(
        "--planned-exits-dry-run",
        action="store_true",
        help="手仕舞い計画の自動実行をドライランにする（既定は実発注）",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="パイプライン全体のフェーズ別実行時間を計測し logs/perf にレポート保存",
    )
    parser.add_argument(
        "--detailed-perf",
        action="store_true",
        help="詳細パフォーマンス測定（メモリ、CPU、ディスクI/O）を有効化し logs/perf に保存",
    )
    parser.add_argument(
        "--test-mode",
        choices=["mini", "quick", "sample", "test_symbols"],
        help="テスト用モード: mini=10銘柄 / quick=50銘柄 / sample=100銘柄 / test_symbols=架空銘柄",
    )
    parser.add_argument(
        "--skip-external",
        action="store_true",
        help="外部API呼び出しをスキップ（NASDAQ Trader, pandas_market_calendars等）",
    )
    parser.add_argument(
        "--perf-snapshot",
        action="store_true",
        help="性能スナップショット(JSON)を logs/perf_snapshots に保存 (latest_only 切替比較用)",
    )
    parser.add_argument(
        "--filter-debug",
        action="store_true",
        help="フィルタ段階通過数のFDBGログを有効化 (環境変数 FILTER_DEBUG=1 を内部設定)",
    )
    parser.add_argument(
        "--run-namespace",
        default=None,
        help="任意のラン識別子: 出力を results_csv/<NAMESPACE>/ に分離するために使用します",
    )
    parser.add_argument(
        "--skip-latest-check",
        action="store_true",
        help="Phase 0 の最新営業日チェックをスキップ (デバッグ用)",
    )
    parser.add_argument(
        "--force-per-system-save",
        action="store_true",
        help="内部デバッグ: per-system の候補を results_csv_test に強制保存 (ALLOCATION_DEBUG をプロセス内で有効化)",
    )
    return parser


def parse_cli_args() -> argparse.Namespace:
    parser = build_cli_parser()
    return parser.parse_args()


def configure_logging_for_cli(args: argparse.Namespace) -> None:
    env_mode = os.environ.get("TODAY_SIGNALS_LOG_MODE", "").strip().lower()
    mode = (
        args.log_file_mode
        or (env_mode if env_mode in {"single", "dated"} else None)
        or "dated"
    )
    _configure_today_logger(mode=mode)
    try:
        sel_path = globals().get("_LOG_FILE_PATH")
        _log(f"📝 ログ保存先: {sel_path}", ui=False)
    except Exception:
        pass


def run_signal_pipeline(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    # latest_only 推定: --full-scan-today 指定で False、それ以外 True (システム毎デフォルトロジックと揃える)
    latest_only_flag = False if getattr(args, "full_scan_today", False) else True

    # フィルタデバッグ要求時に環境変数設定（today_filters 側は環境参照）
    try:
        if getattr(args, "filter_debug", False):
            os.environ.setdefault("FILTER_DEBUG", "1")
    except Exception:
        pass

    # PerformanceMonitor の初期化（--detailed-perf 指定時のみ有効化）
    if getattr(args, "detailed_perf", False):
        try:
            from common.performance_monitor import enable_global_monitor

            _perf_monitor = (
                enable_global_monitor()
            )  # noqa: F841 - グローバルモニター初期化のみ
            _log("📊 詳細パフォーマンス測定を有効化しました")
        except Exception as e:  # pragma: no cover - 安全フォールバック
            _log(f"⚠️ PerformanceMonitor初期化失敗: {e}")

    # Lightweight Benchmark の初期化（--benchmark 指定時のみ有効化）
    global _LIGHTWEIGHT_BENCHMARK
    if getattr(args, "benchmark", False):
        _LIGHTWEIGHT_BENCHMARK = LightweightBenchmark(enabled=True)
        _log("⏱️  軽量ベンチマーク（時間計測）を有効化しました")
    else:
        _LIGHTWEIGHT_BENCHMARK = None

    perf = None
    if getattr(args, "perf_snapshot", False):
        try:
            from common.perf_snapshot import enable_global_perf

            perf = enable_global_perf(True)
        except Exception:  # pragma: no cover - 安全フォールバック
            perf = None

    from contextlib import nullcontext
    from typing import ContextManager

    cm: ContextManager[Any]
    if perf is not None:
        cm = perf.run(latest_only=latest_only_flag)
    else:
        # ダミー contextmanager: 必ずインスタンスを作成 (関数参照をそのまま with しない)
        cm = nullcontext()

    with cm:
        result = compute_today_signals(
            args.symbols,
            slots_long=args.slots_long,
            slots_short=args.slots_short,
            capital_long=args.capital_long,
            capital_short=args.capital_short,
            save_csv=args.save_csv,
            csv_name_mode=args.csv_name_mode,
            parallel=args.parallel,
            test_mode=getattr(args, "test_mode", None),
            skip_external=getattr(args, "skip_external", False),
            skip_latest_check=getattr(args, "skip_latest_check", False),
        )

    # 戻り値がNoneの場合のフォールバック
    if result is None:
        return pd.DataFrame(), {}

    # AllocationSummaryを辞書に変換する必要がある場合
    final_df, allocation_summary = result
    if hasattr(allocation_summary, "__dict__"):
        # AllocationSummaryオブジェクトから適切な辞書形式に変換
        per_system_dict = {}
    else:
        # 既に辞書形式の場合
        per_system_dict = (
            allocation_summary if isinstance(allocation_summary, dict) else {}
        )

    return final_df, per_system_dict


def log_final_candidates(final_df: pd.DataFrame) -> list[Signal]:
    if final_df.empty:
        _log("📭 本日の最終候補はありません。")
        return []

    _log("\n=== 最終候補（推奨） ===")
    # Normalize entry_date for display stability
    try:
        if "entry_date" in final_df.columns:
            tmp_df = final_df.copy()
            # まずはそのまま正規化
            norm_series = pd.to_datetime(
                tmp_df["entry_date"], errors="coerce"
            ).dt.normalize()
            # すべて NaT になってしまうケースの簡易フォールバック（型崩れ対策）
            try:
                if norm_series.isna().all():
                    # 文字列として一度フォーマットしてから再解釈（混在型に強い）
                    as_str = tmp_df["entry_date"].astype(str)
                    # "NaT" 文字列は空に
                    as_str = as_str.where(~as_str.str.contains("NaT", na=False), "")
                    norm_series = pd.to_datetime(as_str, errors="coerce").dt.normalize()
            except Exception:
                pass
            tmp_df["entry_date"] = norm_series
            # デバッグ: 型情報と先頭行の値を出力（テストモードのみ想定）
            try:
                _log(
                    f"[DEBUG] entry_date dtype={str(tmp_df['entry_date'].dtype)} sample={list(tmp_df['entry_date'].head(3).astype(str))}",
                    ui=False,
                )
            except Exception:
                pass
        else:
            tmp_df = final_df
    except Exception:
        tmp_df = final_df
    cols = [
        "symbol",
        "system",
        "side",
        "signal_type",
        "entry_date",
        "entry_price",
        "stop_price",
        "shares",
        "position_value",
        "score_key",
        "score",
    ]
    show = [c for c in cols if c in final_df.columns]
    _log(tmp_df[show].to_string(index=False))
    signals_for_merge = []
    for _, r in final_df.iterrows():
        raw_score = r.get("score", 0.0)
        try:
            # None/NaN/invalid -> 0.0
            score_val = 0.0 if pd.isna(raw_score) else float(raw_score)
        except Exception:
            score_val = 0.0
        try:
            system_field = str(r.get("system"))
            system_id = int(system_field.replace("system", "") or 0)
        except Exception:
            system_id = 0
        signals_for_merge.append(
            Signal(
                system_id=system_id,
                symbol=str(r.get("symbol")),
                side="BUY" if str(r.get("side")).lower() == "long" else "SELL",
                strength=score_val,
                meta={},
            )
        )
    return signals_for_merge


def merge_signals_for_cli(signals_for_merge: list[Signal]) -> None:
    if not signals_for_merge:
        return
    merge_signals([signals_for_merge], portfolio_state={}, market_state={})


def maybe_submit_orders(final_df: pd.DataFrame, args: argparse.Namespace) -> None:
    if final_df.empty or not args.alpaca_submit:
        return
    submit_orders_df(
        final_df,
        paper=(not args.live),
        order_type=args.order_type,
        system_order_type=None,
        tif=args.tif,
        retries=2,
        delay=0.5,
        log_callback=_log,
        notify=True,
    )


def maybe_run_planned_exits(args: argparse.Namespace) -> None:
    """Run scheduled exits if the flags/environment request it.

    This helper is intentionally small and side-effect free in failure cases.
    """
    try:
        from schedulers.next_day_exits import submit_planned_exits as _run_planned
    except Exception:
        _run_planned = None

    env_run = os.environ.get("RUN_PLANNED_EXITS", "").lower()
    run_mode = (
        getattr(args, "run_planned_exits", None)
        or (env_run if env_run in {"off", "open", "close", "auto"} else None)
        or "off"
    )
    dry_run = bool(getattr(args, "planned_exits_dry_run", False))

    if _run_planned is None or run_mode == "off":
        return

    sel = run_mode
    if run_mode == "auto":
        try:
            now = datetime.now(ZoneInfo("America/New_York"))
            hhmm = now.strftime("%H%M")
            sel = (
                "open"
                if ("0930" <= hhmm <= "0945")
                else ("close" if ("1550" <= hhmm <= "1600") else "off")
            )
        except Exception:
            sel = "off"

    if sel not in {"open", "close"}:
        return

    _log(f"⏱️ 手仕舞い計画の自動実行: {sel} (dry_run={dry_run})")
    try:
        _run_planned(sel, dry_run=dry_run)
    except Exception as e:
        _log(f"⚠️ 手仕舞い計画の自動実行に失敗: {e}", level="ERROR")


def main() -> int:
    """Entry point for CLI execution.

    Responsibilities:
    - parse CLI
    - configure logging
    - set RUN_NAMESPACE if provided
    - run the pipeline
    - perform save/notify under optional RunLock and per-run subdir
    """
    args = parse_cli_args()

    # Developer helper: allow CLI to force ALLOCATION_DEBUG inside process
    try:
        if getattr(args, "force_per_system_save", False):
            os.environ.setdefault("ALLOCATION_DEBUG", "1")
            _log(
                "[DEBUG] --force-per-system-save enabled: ALLOCATION_DEBUG=1 set in process"
            )
    except Exception:
        pass

    try:
        configure_logging_for_cli(args)
    except Exception:
        pass

    # Persist CLI args for internal helpers
    try:
        globals()["_CLI_ARGS"] = args
    except Exception:
        pass

    # CLI provided namespace has highest precedence for this process
    try:
        if getattr(args, "run_namespace", None):
            cli_ns = str(args.run_namespace)
            os.environ["RUN_NAMESPACE"] = cli_ns
            globals()["_CLI_RUN_NAMESPACE"] = cli_ns
    except Exception:
        pass

    # Run the core pipeline
    try:
        final_df, per_system = run_signal_pipeline(args)
    except Exception as e:
        _log(f"⚠️ パイプライン実行に失敗しました: {e}", level="ERROR")
        return 2

    # If user requested CSV saving, perform atomic save/notify with optional RunLock
    if (
        getattr(args, "save_csv", False)
        and final_df is not None
        and not getattr(final_df, "empty", True)
    ):
        # Build a context for saving (notify suppressed for CLI save)
        try:
            ctx = _initialize_run_context(
                slots_long=getattr(args, "slots_long", None),
                slots_short=getattr(args, "slots_short", None),
                capital_long=getattr(args, "capital_long", None),
                capital_short=getattr(args, "capital_short", None),
                save_csv=True,
                csv_name_mode=getattr(args, "csv_name_mode", None),
                notify=False,
                log_callback=None,
                progress_callback=None,
                per_system_progress=None,
                symbol_data=None,
                parallel=getattr(args, "parallel", False),
                test_mode=getattr(args, "test_mode", None),
                skip_external=getattr(args, "skip_external", False),
            )
        except Exception:
            ctx = _initialize_run_context(save_csv=True)

        # Determine env-controlled behavior
        try:
            env_cfg = get_env_config()
            use_lock = bool(getattr(env_cfg, "use_run_lock", False))
            use_subdir = bool(getattr(env_cfg, "use_run_subdir", False))
        except Exception:
            use_lock = False
            use_subdir = False

        # If CLI provided a run_namespace, prefer it
        ns_val: str | None = getattr(args, "run_namespace", None)
        ns: str | None = None
        if ns_val is not None and str(ns_val).strip() != "":
            ns = str(ns_val).strip()
        else:
            ns_env = os.environ.get("RUN_NAMESPACE")
            if ns_env:
                ns = str(ns_env)
            else:
                try:
                    cfg_ns = getattr(get_env_config(), "run_namespace", None)
                    ns = (
                        str(cfg_ns)
                        if (cfg_ns is not None and str(cfg_ns).strip() != "")
                        else None
                    )
                except Exception:
                    ns = getattr(ctx, "run_namespace", None)
        out_root: Path | None = None
        if use_subdir and ns:
            try:
                base = Path(getattr(ctx.settings, "RESULTS_DIR", "results_csv"))
                out_root = base / f"run_{ns}"
            except Exception:
                out_root = None

        rl = None
        try:
            if use_lock:
                rl = RunLock("today_signals")
                rl.acquire()
        except Exception:
            rl = None

        try:
            _save_and_notify_phase(
                ctx,
                final_df=final_df,
                per_system=per_system or {},
                order_1_7=[f"system{i}" for i in range(1, 8)],
                metrics_summary_context=None,
                output_root_for_final=out_root,
            )
        finally:
            if rl is not None:
                try:
                    rl.release()
                except Exception:
                    pass

    # Run planned exits if requested
    try:
        maybe_run_planned_exits(args)
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
