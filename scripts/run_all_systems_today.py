from __future__ import annotations

"""Daily multi-system signal pipeline (repaired minimal bootstrap section).

NOTE: This file experienced prior encoding corruption. Incremental repairs are
being applied. The current patch introduces:
 1. Explicit project root insertion into sys.path so that running the script
    via `python scripts/run_all_systems_today.py` correctly resolves top-level
    modules like `common`.
 2. Use of `get_settings(create_dirs=False)` inside `_initialize_run_context`
    to avoid potential hangs during strategy initialization (directory
    creation is performed lazily elsewhere if needed).

Further clean-up (mojibake in log strings/docstrings) will follow in later
patches without altering CLI flags or public behavior.
"""

import argparse
import sys
from pathlib import Path as _PathBootstrap

# --- ensure repository root on sys.path (script executed from repo root or elsewhere) ---
try:  # noqa: SIM105
    _project_root = _PathBootstrap(__file__).resolve().parents[1]
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
except Exception:  # pragma: no cover - defensive; failure is non-fatal
    pass
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
import multiprocessing
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import os
from pathlib import Path
from threading import Lock
import threading
from typing import Any, cast, no_type_check
from zoneinfo import ZoneInfo

import pandas as pd

from common import broker_alpaca as ba
from common.alpaca_order import submit_orders_df
from common.cache_manager import CacheManager, load_base_cache
from common.cache_manager import round_dataframe  # noqa: E402
from common.notifier import create_notifier
from common.position_age import load_entry_dates, save_entry_dates
from common.signal_merge import Signal, merge_signals
from common.stage_metrics import GLOBAL_STAGE_METRICS, StageEvent, StageSnapshot
from common.symbol_universe import build_symbol_universe_from_settings
from common.system_groups import format_group_counts, format_group_counts_and_values
from common.utils_spy import (
    get_latest_nyse_trading_day,
    get_signal_target_trading_day,
    get_spy_with_indicators,
)
from config.settings import get_settings
from core.final_allocation import AllocationSummary, finalize_allocation, load_symbol_system_map
from core.system5 import DEFAULT_ATR_PCT_THRESHOLD, format_atr_pct_threshold_label

# strategies
from strategies.system1_strategy import System1Strategy
from strategies.system2_strategy import System2Strategy
from strategies.system3_strategy import System3Strategy
from strategies.system4_strategy import System4Strategy
from strategies.system5_strategy import System5Strategy
from strategies.system6_strategy import System6Strategy
from strategies.system7_strategy import System7Strategy
from tools.notify_metrics import send_metrics_notification

_LOG_CALLBACK = None

# Progress event settings
ENABLE_PROGRESS_EVENTS = os.getenv("ENABLE_PROGRESS_EVENTS", "false").lower() == "true"


def emit_progress_event(event_type: str, data: dict) -> None:
    """Emit a progress event with given type and data."""
    if not ENABLE_PROGRESS_EVENTS:
        return

    try:
        logger = logging.getLogger(__name__)
        logger.debug(f"Progress event [{event_type}]: {data}")
    except Exception:
        pass


_LOG_FORWARDING = ContextVar("_LOG_FORWARDING", default=False)
_LOG_START_TS = None  # CLI 用の経過時間測定開始時刻

# ログファイル設定（デフォルトは固定ファイル）。必要に応じて日付付きへ切替。
# レート制限ロガー
_rate_limited_logger = None


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
            try:
                if isinstance(frame.index, pd.DatetimeIndex) and len(frame.index):
                    return pd.Timestamp(frame.index[-1]).normalize()
            except Exception:
                pass
            try:
                idx = pd.to_datetime(frame.index.to_numpy(), errors="coerce")
                idx = idx[~pd.isna(idx)]
                if len(idx):
                    return pd.Timestamp(idx[-1]).normalize()
            except Exception:
                pass
            try:
                series = frame.get("Date") if frame is not None else None
                if series is not None:
                    # convert to numpy array to match pandas.to_datetime overloads
                    series = pd.to_datetime(series.to_numpy(), errors="coerce")
                    if hasattr(series, "dropna"):
                        series = series.dropna()
                    if getattr(series, "size", 0):
                        # ensure we have an indexable array/series
                        try:
                            if isinstance(series, pd.DatetimeIndex):
                                return pd.Timestamp(series[-1]).normalize()
                            # pandas Series/Index support iloc; fallback to index access
                            try:
                                return pd.Timestamp(series.iloc[-1]).normalize()
                            except Exception:
                                return pd.Timestamp(series[-1]).normalize()
                        except Exception:
                            pass
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


def _configure_today_logger(*, mode: str = "single", run_id: str | None = None) -> None:
    """today_signals 用のロガーファイルを構成する。

    mode:
      - "single": 固定ファイル `today_signals.log`
      - "dated":  日付付き `today_signals_YYYYMMDD_HHMM.log`（JST）
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
                if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None):
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
                import os as _os

                _mode_env = (_os.environ.get("TODAY_SIGNALS_LOG_MODE") or "").strip().lower()
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
                    globals()["_LOG_FILE_PATH"] = _log_dir / f"today_signals_{_stamp}.log"
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
            fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except Exception:
            pass
    return logger


def _emit_ui_log(message: str) -> None:
    """UI 側のログコールバックが登録されていれば、そのまま文字列を送信する。"""
    try:
        cb = globals().get("_LOG_CALLBACK")
        if cb and callable(cb):
            token = _LOG_FORWARDING.set(True)
            try:
                cb(str(message))
            finally:
                _LOG_FORWARDING.reset(token)
    except Exception:
        # UI コールバック未設定や例外は黙って無視（CLI 実行時を考慮）
        pass


def _drain_stage_event_queue() -> None:
    """メインスレッドでステージ進捗イベントを処理し、UI 表示を更新する。"""

    try:
        cb2 = globals().get("_PER_SYSTEM_STAGE")
    except Exception:
        cb2 = None

    events: list[StageEvent] = GLOBAL_STAGE_METRICS.drain_events()

    # もしプロセスマネージャー経由の進捗キューが存在すればそこからも取り出す
    try:
        _mgr = globals().get("_PROGRESS_MANAGER")
    except Exception:
        _mgr = None
    if _mgr is not None:
        try:
            q = globals().get("_PROGRESS_QUEUE")
            if q is not None:
                while True:
                    try:
                        item = q.get_nowait()
                    except Exception:
                        break
                    try:
                        # item expected: (system, progress, filter, setup, cand, entry)
                        if isinstance(item, list | tuple) and len(item) >= 2:
                            GLOBAL_STAGE_METRICS.record_stage(
                                item[0],
                                int(item[1]),
                                item[2] if len(item) > 2 else None,
                                item[3] if len(item) > 3 else None,
                                item[4] if len(item) > 4 else None,
                                item[5] if len(item) > 5 else None,
                                emit_event=True,
                            )
                    except Exception:
                        continue
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


def _log(msg: str, ui: bool = True, no_timestamp: bool = False):
    """CLI 出力には [HH:MM:SS | m分s秒] を付与。必要に応じて UI コールバックを抑制。"""
    import time as _t

    # 初回呼び出しで開始時刻を設定
    try:
        global _LOG_START_TS
        if _LOG_START_TS is None:
            _LOG_START_TS = _t.time()
    except Exception:
        _LOG_START_TS = None

    # プレフィックスを作成（現在時刻 + 分秒経過）
    try:
        if no_timestamp:
            prefix = ""
        else:
            now = _t.strftime("%H:%M:%S")
            elapsed = 0 if _LOG_START_TS is None else max(0, _t.time() - _LOG_START_TS)
            m, s = divmod(int(elapsed), 60)
            prefix = f"[{now} | {m}分{s}秒] "
    except Exception:
        prefix = ""

    # キーワードによる除外判定（全体）
    try:
        import os as _os

        # SHOW_INDICATOR_LOGS が真でない限り、インジケーター系の進捗ログを抑制
        _show_ind_logs = (_os.environ.get("SHOW_INDICATOR_LOGS") or "").strip().lower()
        _hide_indicator_logs = _show_ind_logs not in {"1", "true", "yes", "on"}
        _indicator_skip = (
            "インジケーター計算",
            "指標計算",
            "共有指標",
            "指標データロード",
            "📊 指標計算",
            "🧮 共有指標",
        )
        _skip_all = _GLOBAL_SKIP_KEYWORDS + (_indicator_skip if _hide_indicator_logs else ())
        if any(k in str(msg) for k in _skip_all):
            return
        ui_allowed = ui and not any(k in str(msg) for k in _UI_ONLY_SKIP_KEYWORDS)
    except Exception:
        ui_allowed = ui

    # CLI へは整形して出力
    out = f"{prefix}{msg}"
    try:
        print(out, flush=True)
    except UnicodeEncodeError:
        try:
            import sys as _sys

            encoding = getattr(_sys.stdout, "encoding", "") or "utf-8"
            safe = out.encode(encoding, errors="replace").decode(encoding, errors="replace")
            print(safe, flush=True)
        except Exception:
            try:
                safe = out.encode("ascii", errors="replace").decode("ascii", errors="replace")
                print(safe, flush=True)
            except Exception:
                pass
    except Exception:
        pass

    # UI 側のコールバックにはフィルタ済みで通知（UI での重複プレフィックス回避）
    if ui_allowed:
        _emit_ui_log(str(msg))

    # 常にファイルへもINFOで出力（UI/CLI の別なく完全なログを保存）
    try:
        _get_today_logger().info(str(msg))
    except Exception:
        pass


def _asc_by_score_key(score_key: str | None) -> bool:
    return bool(score_key and score_key.upper() in {"RSI4"})


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


def _save_prev_counts(signals_dir: Path, per_system_map: dict[str, pd.DataFrame]) -> None:
    try:
        counts = {
            k: (0 if (v is None or v.empty) else int(len(v))) for k, v in per_system_map.items()
        }
        data = {"timestamp": datetime.utcnow().isoformat() + "Z", "counts": counts}
        fp = _prev_counts_path(signals_dir)
        try:
            fp.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        fp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
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


def _pick_series(df: pd.DataFrame, names: list[str]):
    try:
        for nm in names:
            if nm in df.columns:
                s = df[nm]
                if isinstance(s, pd.DataFrame):
                    try:
                        # only try 2D iloc if DataFrame-like
                        if getattr(s, "ndim", None) == 2 and hasattr(s, "iloc"):
                            s = cast(pd.Series, cast(Any, s).iloc[:, 0])
                        else:
                            # fallback: convert to first column via list of columns
                            cols = list(s.columns or [])
                            if cols:
                                s = s[cols[0]]
                            else:
                                continue
                    except Exception:
                        continue
                try:
                    s = pd.to_numeric(s, errors="coerce")
                except Exception:
                    pass
                return s
    except Exception:
        pass
    return None


def _last_scalar(series):
    try:
        if series is None:
            return None
        s2 = series.dropna()
        if s2.empty:
            return None
        return float(s2.iloc[-1])
    except Exception:
        return None


def _calc_dollar_volume_from_series(close_series, volume_series, window: int) -> float | None:
    if close_series is None or volume_series is None:
        return None
    try:
        product = close_series * volume_series
    except Exception:
        return None
    try:
        tail = product.tail(window) if hasattr(product, "tail") else product
    except Exception:
        tail = product
    try:
        tail = pd.to_numeric(tail, errors="coerce")
    except Exception:
        pass
    try:
        if hasattr(tail, "dropna"):
            tail = tail.dropna()
        if getattr(tail, "empty", False):
            return None
        return float(tail.mean())
    except Exception:
        return None


def _calc_average_volume_from_series(volume_series, window: int) -> float | None:
    if volume_series is None:
        return None
    try:
        tail = volume_series.tail(window) if hasattr(volume_series, "tail") else volume_series
    except Exception:
        tail = volume_series
    try:
        tail = pd.to_numeric(tail, errors="coerce")
    except Exception:
        pass
    try:
        if hasattr(tail, "dropna"):
            tail = tail.dropna()
        if getattr(tail, "empty", False):
            return None
        return float(tail.mean())
    except Exception:
        return None


def _resolve_atr_ratio(
    df: pd.DataFrame,
    close_series=None,
    last_close: float | None = None,
) -> float | None:
    ratio_series = _pick_series(df, ["ATR_Ratio", "ATR_Pct"])
    ratio_val = _last_scalar(ratio_series)
    if ratio_val is not None:
        return ratio_val

    atr_series = _pick_series(df, ["ATR10", "ATR20"])
    atr_val = _last_scalar(atr_series)
    if atr_val is None:
        high_series = _pick_series(df, ["High", "high"])
        low_series = _pick_series(df, ["Low", "low"])
        if high_series is not None and low_series is not None:
            try:
                tr = high_series - low_series
                if hasattr(tr, "dropna"):
                    tr = tr.dropna()
                tr_tail = tr.tail(10) if hasattr(tr, "tail") else tr
                if getattr(tr_tail, "empty", False):
                    atr_val = None
                else:
                    atr_val = float(tr_tail.mean())
            except Exception:
                atr_val = None
    if atr_val is None:
        return None

    if last_close is None:
        try:
            close_series = close_series or _pick_series(df, ["Close", "close"])
        except Exception:
            close_series = None
        last_close = _last_scalar(close_series)
    try:
        if last_close is None:
            return None
        close_val = float(last_close)
        if close_val == 0:
            return None
    except Exception:
        return None

    try:
        return float(atr_val) / close_val
    except Exception:
        return None


def _system1_conditions(df: pd.DataFrame) -> tuple[bool, bool]:
    close_series = _pick_series(df, ["Close", "close"])
    last_close = _last_scalar(close_series)
    price_ok = bool(last_close is not None and last_close >= 5)

    dv_series = _pick_series(df, ["DollarVolume20"])
    dv20 = _last_scalar(dv_series)
    if dv20 is None:
        volume_series = _pick_series(df, ["Volume", "volume"])
        dv20 = _calc_dollar_volume_from_series(close_series, volume_series, 20)
    dv_ok = bool(dv20 is not None and dv20 >= 50_000_000)

    return price_ok, dv_ok


def _system2_conditions(df: pd.DataFrame) -> tuple[bool, bool, bool]:
    close_series = _pick_series(df, ["Close", "close"])
    last_close = _last_scalar(close_series)
    price_ok = bool(last_close is not None and last_close >= 5)

    dv_series = _pick_series(df, ["DollarVolume20"])
    dv20 = _last_scalar(dv_series)
    if dv20 is None:
        volume_series = _pick_series(df, ["Volume", "volume"])
        dv20 = _calc_dollar_volume_from_series(close_series, volume_series, 20)
    dv_ok = bool(dv20 is not None and dv20 >= 25_000_000)

    atr_ratio = _resolve_atr_ratio(df, close_series, last_close)
    atr_ok = bool(atr_ratio is not None and atr_ratio >= 0.03)

    return price_ok, dv_ok, atr_ok


def _system3_conditions(df: pd.DataFrame) -> tuple[bool, bool, bool]:
    low_series = _pick_series(df, ["Low", "low"])
    low_val = _last_scalar(low_series)
    low_ok = bool(low_val is not None and low_val >= 1)

    av_series = _pick_series(df, ["AvgVolume50"])
    av_val = _last_scalar(av_series)
    if av_val is None:
        volume_series = _pick_series(df, ["Volume", "volume"])
        av_val = _calc_average_volume_from_series(volume_series, 50)
    av_ok = bool(av_val is not None and av_val >= 1_000_000)

    atr_ratio = _resolve_atr_ratio(df)
    atr_ok = bool(atr_ratio is not None and atr_ratio >= 0.05)

    return low_ok, av_ok, atr_ok


def _system4_conditions(df: pd.DataFrame) -> tuple[bool, bool]:
    close_series = _pick_series(df, ["Close", "close"])
    volume_series = _pick_series(df, ["Volume", "volume"])
    dv_series = _pick_series(df, ["DollarVolume50"])
    dv50 = _last_scalar(dv_series)
    if dv50 is None:
        dv50 = _calc_dollar_volume_from_series(close_series, volume_series, 50)
    dv_ok = bool(dv50 is not None and dv50 > 100_000_000)

    hv_series = _pick_series(df, ["HV50"])
    hv_val = _last_scalar(hv_series)
    hv_ok = bool(hv_val is not None and 10 <= hv_val <= 40)

    return dv_ok, hv_ok


def _system5_conditions(df: pd.DataFrame) -> tuple[bool, bool, bool]:
    volume_series = _pick_series(df, ["Volume", "volume"])
    av_series = _pick_series(df, ["AvgVolume50"])
    av_val = _last_scalar(av_series)
    if av_val is None:
        av_val = _calc_average_volume_from_series(volume_series, 50)
    av_ok = bool(av_val is not None and av_val > 500_000)

    close_series = _pick_series(df, ["Close", "close"])
    dv_series = _pick_series(df, ["DollarVolume50"])
    dv50 = _last_scalar(dv_series)
    if dv50 is None:
        dv50 = _calc_dollar_volume_from_series(close_series, volume_series, 50)
    dv_ok = bool(dv50 is not None and dv50 > 2_500_000)

    atr_series = _pick_series(df, ["ATR_Pct", "ATR_Ratio"])
    atr_val = _last_scalar(atr_series)
    if atr_val is None:
        atr_val = _resolve_atr_ratio(df, close_series)
    atr_ok = bool(atr_val is not None and atr_val > DEFAULT_ATR_PCT_THRESHOLD)

    return av_ok, dv_ok, atr_ok


def _system6_conditions(df: pd.DataFrame) -> tuple[bool, bool]:
    low_series = _pick_series(df, ["Low", "low"])
    low_val = _last_scalar(low_series)
    low_ok = bool(low_val is not None and low_val >= 5)

    close_series = _pick_series(df, ["Close", "close"])
    volume_series = _pick_series(df, ["Volume", "volume"])
    dv_series = _pick_series(df, ["DollarVolume50"])
    dv50 = _last_scalar(dv_series)
    if dv50 is None:
        dv50 = _calc_dollar_volume_from_series(close_series, volume_series, 50)
    dv_ok = bool(dv50 is not None and dv50 > 10_000_000)

    return low_ok, dv_ok


def filter_system1(symbols, data, stats: dict[str, int] | None = None):
    result = []
    total = len(symbols or [])
    price_pass = 0
    dv_pass = 0
    for sym in symbols or []:
        df = data.get(sym)
        if df is None or df.empty:
            continue
        price_ok, dv_ok = _system1_conditions(df)
        if not price_ok:
            continue
        price_pass += 1
        if not dv_ok:
            continue
        dv_pass += 1
        result.append(sym)
    if stats is not None:
        stats["total"] = total
        stats["price_pass"] = price_pass
        stats["dv_pass"] = dv_pass
    return result


def filter_system2(symbols, data, stats: dict[str, int] | None = None):
    result = []
    total = len(symbols or [])
    price_pass = 0
    dv_pass = 0
    atr_pass = 0
    for sym in symbols or []:
        df = data.get(sym)
        if df is None or df.empty:
            continue
        price_ok, dv_ok, atr_ok = _system2_conditions(df)
        if not price_ok:
            continue
        price_pass += 1
        if not dv_ok:
            continue
        dv_pass += 1
        if not atr_ok:
            continue
        atr_pass += 1
        result.append(sym)
    if stats is not None:
        stats["total"] = total
        stats["price_pass"] = price_pass
        stats["dv_pass"] = dv_pass
        stats["atr_pass"] = atr_pass
    return result


def filter_system3(symbols, data, stats: dict[str, int] | None = None):
    result = []
    total = len(symbols or [])
    low_pass = 0
    av_pass = 0
    atr_pass = 0
    for sym in symbols or []:
        df = data.get(sym)
        if df is None or df.empty:
            continue
        low_ok, av_ok, atr_ok = _system3_conditions(df)
        if not low_ok:
            continue
        low_pass += 1
        if not av_ok:
            continue
        av_pass += 1
        if not atr_ok:
            continue
        atr_pass += 1
        result.append(sym)
    if stats is not None:
        stats["total"] = total
        stats["low_pass"] = low_pass
        stats["avgvol_pass"] = av_pass
        stats["atr_pass"] = atr_pass
    return result


def filter_system4(symbols, data, stats: dict[str, int] | None = None):
    result = []
    total = len(symbols or [])
    dv_pass = 0
    hv_pass = 0
    for sym in symbols or []:
        df = data.get(sym)
        if df is None or df.empty:
            continue
        dv_ok, hv_ok = _system4_conditions(df)
        if not dv_ok:
            continue
        dv_pass += 1
        if not hv_ok:
            continue
        hv_pass += 1
        result.append(sym)
    if stats is not None:
        stats["total"] = total
        stats["dv_pass"] = dv_pass
        stats["hv_pass"] = hv_pass
    return result


def filter_system5(symbols, data, stats: dict[str, int] | None = None):
    result = []
    total = len(symbols or [])
    av_pass = 0
    dv_pass = 0
    atr_pass = 0
    for sym in symbols or []:
        df = data.get(sym)
        if df is None or df.empty:
            continue
        av_ok, dv_ok, atr_ok = _system5_conditions(df)
        if not av_ok:
            continue
        av_pass += 1
        if not dv_ok:
            continue
        dv_pass += 1
        if not atr_ok:
            continue
        atr_pass += 1
        result.append(sym)
    if stats is not None:
        stats["total"] = total
        stats["avgvol_pass"] = av_pass
        stats["dv_pass"] = dv_pass
        stats["atr_pass"] = atr_pass
    return result


def filter_system6(symbols, data, stats: dict[str, int] | None = None):
    result = []
    total = len(symbols or [])
    low_pass = 0
    dv_pass = 0
    for sym in symbols or []:
        df = data.get(sym)
        if df is None or df.empty:
            continue
        low_ok, dv_ok = _system6_conditions(df)
        if not low_ok:
            continue
        low_pass += 1
        if not dv_ok:
            continue
        dv_pass += 1
        result.append(sym)
    if stats is not None:
        stats["total"] = total
        stats["low_pass"] = low_pass
        stats["dv_pass"] = dv_pass
    return result


def _extract_last_cache_date(df: pd.DataFrame) -> pd.Timestamp | None:
    if df is None or getattr(df, "empty", True):
        return None
    for col in ("date", "Date"):
        if col in df.columns:
            try:
                values = pd.to_datetime(df[col].to_numpy(), errors="coerce")
                values = values.dropna()
                if not values.empty:
                    return pd.Timestamp(values[-1]).normalize()
            except Exception:
                continue
    try:
        idx = pd.to_datetime(df.index.to_numpy(), errors="coerce")
        mask = ~pd.isna(idx)
        if mask.any():
            return pd.Timestamp(idx[mask][-1]).normalize()
    except Exception:
        pass
    return None


def _recent_trading_days(today: pd.Timestamp | None, max_back: int) -> list[pd.Timestamp]:
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
    base_cache: dict[str, pd.DataFrame] | None = None,
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
            settings.cache.rolling.base_lookback_days + settings.cache.rolling.buffer_days
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
                    work["Date"] = pd.to_datetime(work["date"].to_numpy(), errors="coerce")
                else:
                    work["Date"] = pd.to_datetime(work.index.to_numpy(), errors="coerce")
                df = work
            df["Date"] = pd.to_datetime(df["Date"].to_numpy(), errors="coerce").normalize()
        except Exception:
            pass
        normalized = _normalize_ohlcv(df)
        try:
            fill_cols = [
                c for c in ("Open", "High", "Low", "Close", "Volume") if c in normalized.columns
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

    env_parallel = (os.environ.get("BASIC_DATA_PARALLEL", "") or "").strip().lower()
    try:
        env_parallel_threshold = int(os.environ.get("BASIC_DATA_PARALLEL_THRESHOLD", "200"))
    except Exception:
        env_parallel_threshold = 200
    if env_parallel in ("1", "true", "yes"):
        use_parallel = total_syms > 1
    elif env_parallel in ("0", "false", "no"):
        use_parallel = False
    else:
        use_parallel = total_syms >= max(0, env_parallel_threshold)

    max_workers: int | None = None
    if use_parallel and total_syms > 0:
        try:
            env_workers = (os.environ.get("BASIC_DATA_MAX_WORKERS", "") or "").strip()
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
                        gap_days = _estimate_gap_days(pd.Timestamp(today), last_seen_date)
                        needs_rebuild = True
            if needs_rebuild:
                reason_map = {
                    "stale": "鮮度不足",
                    "missing_date": "日付欠損",
                    "length": "行数不足",
                }
                reason_key = rebuild_reason or (
                    "missing" if df is None or getattr(df, "empty", True) else "unknown"
                )
                reason_label = reason_map.get(reason_key, "未整備")
                detail_parts: list[str] = []
                if rebuild_reason == "stale":
                    gap_label = f"約{gap_days}営業日" if gap_days is not None else "不明"
                    last_label = (
                        str(last_seen_date.date()) if last_seen_date is not None else "不明"
                    )
                    detail_parts.append(f"最終日={last_label}")
                    detail_parts.append(f"ギャップ={gap_label}")
                elif rebuild_reason == "length" and df is not None:
                    try:
                        detail_parts.append(f"len={len(df)}/{target_len}")
                    except Exception:
                        pass
                elif rebuild_reason == "missing_date":
                    detail_parts.append("date列欠損")
                if df is None or getattr(df, "empty", True):
                    detail_parts.append("rolling未生成")
                skip_msg = f"⛔ rolling未整備: {sym} ({reason_label})"
                if detail_parts:
                    skip_msg += " | " + ", ".join(detail_parts)
                skip_msg += " → 手動で rolling キャッシュを更新してください"
                _log(skip_msg, ui=False)
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
            msg = f"📦 基礎データロード進捗: {done}/{total_syms} | ETA {m}分{s}秒"

            # 進捗ログはDEBUGレベルでレート制限適用
            rate_logger = _get_rate_limited_logger()
            rate_logger.debug_rate_limited(
                f"📦 基礎データロード進捗: {done}/{total_syms}",
                interval=2.0,
                message_key="基礎データ進捗",
            )
            _emit_ui_log(msg)
        except Exception:
            _log(f"📦 基礎データロード進捗: {done}/{total_syms}", ui=False)
            _emit_ui_log(f"📦 基礎データロード進捗: {done}/{total_syms}")

    processed = 0
    if use_parallel and max_workers and total_syms > 1:
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
        done_msg = f"📦 基礎データロード完了: {len(data)}/{total_syms} | 所要 {m}分{s}秒" + (
            " | 並列=ON" if use_parallel and max_workers else " | 並列=OFF"
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
            f"{label}={stats.get(key, 0)}" for key, label in summary_map.items() if stats.get(key)
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
                            x["date"] = pd.to_datetime(x["date"].to_numpy(), errors="coerce")
                        elif "Date" in x.columns:
                            x["date"] = pd.to_datetime(x["Date"].to_numpy(), errors="coerce")
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
            target_len = int(
                settings.cache.rolling.base_lookback_days + settings.cache.rolling.buffer_days
            )
            needs_rebuild = df is None or getattr(df, "empty", True)
            if needs_rebuild:
                if df is None or getattr(df, "empty", True):
                    reason_desc = "rolling未生成"
                else:
                    try:
                        reason_desc = f"len={len(df)}/{target_len}"
                    except Exception:
                        reason_desc = "行数不足"
                _log(
                    f"⛔ rolling未整備: {sym} ({reason_desc}) → 手動更新を実行してください",
                    ui=False,
                )
                continue
            if df is not None and not df.empty:
                try:
                    if "Date" not in df.columns:
                        if "date" in df.columns:
                            df = df.copy()
                            df["Date"] = pd.to_datetime(df["date"].to_numpy(), errors="coerce")
                        else:
                            df = df.copy()
                            df["Date"] = pd.to_datetime(df.index.to_numpy(), errors="coerce")
                    df["Date"] = pd.to_datetime(df["Date"].to_numpy(), errors="coerce").normalize()
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
        done_msg = f"🧮 指標データロード完了: {len(data)}/{total_syms} | 所要 {m}分{s}秒"
        _log(done_msg)
        _emit_ui_log(done_msg)
    except Exception:
        _log(f"🧮 指標データロード完了: {len(data)}/{total_syms}")
        _emit_ui_log(f"🧮 指標データロード完了: {len(data)}/{total_syms}")
    return data


def _subset_data(basic_data: dict[str, pd.DataFrame], keys: list[str]) -> dict[str, pd.DataFrame]:
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
                "notification_complete", {"notifications_sent": 1, "results_count": len(results)}
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
) -> TodayRunContext:
    """当日シグナル実行前に共有設定・状態をまとめたコンテキストを生成する。"""

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
    )
    ctx.run_start_time = datetime.now()
    ctx.start_equity = _get_account_equity()
    try:
        import uuid as _uuid

        ctx.run_id = str(_uuid.uuid4())[:8]
    except Exception:
        ctx.run_id = "--------"
    return ctx


def _prepare_symbol_universe(ctx: TodayRunContext, initial_symbols: list[str] | None) -> list[str]:
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
        try:
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

        if fetched:
            limit_val: int | None = None
            limit_src = ""
            try:
                env_limit = os.getenv("TODAY_SYMBOL_LIMIT", "").strip()
                if env_limit:
                    parsed = int(env_limit)
                    if parsed > 0:
                        limit_val = parsed
                        limit_src = "TODAY_SYMBOL_LIMIT"
            except Exception:
                limit_val = None
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

    _log(f"🎯 対象シンボル数: {len(symbols)} | 銘柄数：{universe_total}")
    # ヘッダー部分に追加で銘柄数を表示
    _log(f"# 📊 銘柄数：{universe_total}", ui=False, no_timestamp=True)
    _log(f"📋 サンプル: {', '.join(symbols[:10])}" f"{'...' if len(symbols) > 10 else ''}")

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


def _load_universe_basic_data(ctx: TodayRunContext, symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Load rolling cache data for the prepared universe and ensure coverage."""

    cache_manager = ctx.cache_manager
    settings = ctx.settings
    progress_callback = ctx.progress_callback
    symbol_data = ctx.symbol_data

    basic_data = _load_basic_data(
        symbols,
        cache_manager,
        settings,
        symbol_data,
        today=ctx.today,
        base_cache=ctx.base_cache,
    )
    ctx.basic_data = basic_data

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
            preview = ", ".join(missing_syms[:10])
            if len(missing_syms) > 10:
                preview += " …"
            _log(
                "⚠️ rolling未整備: "
                + f"{cov_missing}銘柄 → 手動でキャッシュを更新してください"
                + (f" | 例: {preview}" if preview else ""),
                ui=False,
            )
    except Exception:
        pass

    return basic_data


def _precompute_shared_indicators_phase(
    ctx: TodayRunContext, basic_data: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """Optionally pre-compute shared indicators for the loaded dataset."""

    if not basic_data:
        return basic_data

    try:
        import os as _os

        from common.indicators_precompute import (
            PRECOMPUTED_INDICATORS,
            precompute_shared_indicators,
        )

        # Rolling データに既に指標が含まれているかチェック
        sample_symbols = list(basic_data.keys())[:5]  # サンプル数銘柄をチェック
        indicators_already_exist = True

        # 指標の存在をより柔軟にチェック（大文字・小文字両対応）
        from common.cache_manager import get_indicator_column_flexible

        for sym in sample_symbols:
            df = basic_data[sym]
            if df is None or df.empty:
                continue

            indicators_found = 0
            total_indicators = 0
            required_indicators = [
                "ATR10",
                "ATR20",
                "SMA25",
                "SMA50",
                "RSI4",
                "ROC200",
                "DollarVolume20",
            ]

            for indicator in required_indicators:
                total_indicators += 1
                if get_indicator_column_flexible(df, indicator) is not None:
                    indicators_found += 1
                # ATRの代替をチェック（ATR14など）
                elif indicator.startswith("ATR") and any(
                    col.upper().startswith("ATR") and any(c.isdigit() for c in col)
                    for col in df.columns
                ):
                    indicators_found += 1
                # RSIの代替をチェック（RSI3, RSI14など）
                elif indicator.startswith("RSI") and any(
                    col.upper().startswith("RSI") and any(c.isdigit() for c in col)
                    for col in df.columns
                ):
                    indicators_found += 1

            # 7つ中5つ以上の指標があれば最適化を適用
            if indicators_found < 5:
                indicators_already_exist = False
                break

        if indicators_already_exist:
            _log("🧮 共有指標の前計算: スキップ（rollingデータに既に指標が含まれています）")
            return basic_data

        try:
            thr_syms = int(_os.environ.get("PRECOMPUTE_SYMBOLS_THRESHOLD", "300"))
        except Exception:
            thr_syms = 300
        if len(basic_data) < max(0, thr_syms):
            _log(
                f"🧮 共有指標の前計算: スキップ（対象銘柄 {len(basic_data)} 件 < 閾値 {thr_syms}）"
            )
            return basic_data

        try:
            _log(
                "🧮 共有指標の前計算を開始: "
                + ", ".join(list(PRECOMPUTED_INDICATORS)[:8])
                + (" …" if len(PRECOMPUTED_INDICATORS) > 8 else "")
            )
        except Exception:
            _log("🧮 共有指標の前計算を開始 (ATR/SMA/ADX ほか)")

        force_parallel = _os.environ.get("PRECOMPUTE_PARALLEL", "").lower()
        try:
            thr_parallel = int(_os.environ.get("PRECOMPUTE_PARALLEL_THRESHOLD", "200"))
        except Exception:
            thr_parallel = 200
        if force_parallel in ("1", "true", "yes"):
            use_parallel = True
        elif force_parallel in ("0", "false", "no"):
            use_parallel = False
        else:
            use_parallel = len(basic_data) >= max(0, thr_parallel)

        try:
            st = get_settings(create_dirs=False)
            pre_workers = int(getattr(st, "THREADS_DEFAULT", 12))
        except Exception:
            pre_workers = 12
        if use_parallel:
            max_workers = max(1, min(int(pre_workers), len(basic_data)))
            try:
                _log(f"🧵 前計算 並列ワーカー: {max_workers}")
            except Exception:
                pass
        else:
            max_workers = None
        from time import perf_counter as _perf

        pre_start = _perf()
        basic_data = precompute_shared_indicators(
            basic_data,
            log=_log,
            parallel=use_parallel,
            max_workers=max_workers,
        )
        ctx.basic_data = basic_data
        elapsed = int(max(0, _perf() - pre_start))
        m, s = divmod(elapsed, 60)
        mode_label = "ON" if use_parallel else "OFF"
        _log(f"🧮 共有指標の前計算が完了 | 所要 {m}分{s}秒 | 並列={mode_label}")
    except Exception as e:
        _log(f"⚠️ 共有指標の前計算に失敗: {e}")
    return basic_data


def _ensure_cli_logger_configured() -> None:
    """CLI ???????????????????"""
    try:
        if globals().get("_LOG_FILE_PATH") is None:
            import os as _os

            _mode_env = (_os.environ.get("TODAY_SIGNALS_LOG_MODE") or "").strip().lower()
            _configure_today_logger(mode=("single" if _mode_env == "single" else "dated"))
    except Exception:
        pass


def _silence_streamlit_cli_warnings() -> None:
    """CLI ???? Streamlit ? bare mode ????????"""
    try:
        import logging as _lg
        import os as _os

        if _os.environ.get("STREAMLIT_SERVER_ENABLED"):
            return

        class _SilenceBareModeWarnings(_lg.Filter):
            def filter(self, record: _lg.LogRecord) -> bool:
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
            _logger = _lg.getLogger(_name)
            _logger.addFilter(_SilenceBareModeWarnings())
            try:
                _logger.setLevel(_lg.ERROR)
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
    """?????????????????"""
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
) -> None:
    """保存および通知フェーズを担当する補助関数。"""

    signals_dir = ctx.signals_dir
    notify = ctx.notify
    save_csv = ctx.save_csv
    csv_name_mode = ctx.csv_name_mode or "date"
    progress_callback = ctx.progress_callback
    run_start_time = ctx.run_start_time
    start_equity = ctx.start_equity
    today = ctx.today or get_latest_nyse_trading_day().normalize()
    run_id = ctx.run_id

    try:
        cb2 = globals().get("_PER_SYSTEM_STAGE")
    except Exception:
        cb2 = None
    if cb2 and callable(cb2):
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
                cb2(name, 100, None, None, cand_cnt, final_cnt)
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
            lines = []
            for sys_name in order_1_7:
                tgt = tgt_base if sys_name != "system7" else 1
                fil = int(prefilter_map.get(sys_name, 0))
                stu = int(setup_map.get(sys_name, 0))
                try:
                    df_trd = per_system.get(sys_name, pd.DataFrame())
                    trd = int(
                        0 if df_trd is None or getattr(df_trd, "empty", True) else len(df_trd)
                    )
                except Exception:
                    trd = 0
                ent = int(final_counts.get(sys_name, 0))
                exv = exit_counts_map.get(sys_name)
                ex_txt = "-" if exv is None else str(int(exv))
                value = (
                    f"Tgt {tgt} / FIL {fil} / STU {stu} / TRD {trd} / Entry {ent} / Exit {ex_txt}"
                )
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
                total_exits = int(sum(int(v) for v in exit_counts_map.values() if v is not None))
            except Exception:
                total_exits = 0
            start_time_str = run_start_time.strftime("%H:%M:%S")
            end_time_str = run_end_time.strftime("%H:%M:%S")
            duration_seconds = max(0, int((run_end_time - run_start_time).total_seconds()))
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
            summary_fields = [
                {"name": key, "value": value, "inline": True} for key, value in summary_pairs
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

        out_all = signals_dir / f"signals_final_{suffix}.csv"
        try:
            try:
                round_dec = getattr(get_settings(create_dirs=True).cache, "round_decimals", None)
            except Exception:
                round_dec = None
            out_df = round_dataframe(final_df, round_dec)
        except Exception:
            out_df = final_df
        out_df.to_csv(out_all, index=False)
        for name, df in per_system.items():
            if df is None or getattr(df, "empty", True):
                continue
            out = signals_dir / f"signals_{name}_{suffix}.csv"
            try:
                try:
                    round_dec = getattr(
                        get_settings(create_dirs=True).cache, "round_decimals", None
                    )
                except Exception:
                    round_dec = None
                out_df = round_dataframe(df, round_dec)
            except Exception:
                out_df = df
            out_df.to_csv(out, index=False)
        _log(f"💾 保存: {signals_dir} にCSVを書き出しました")

    _safe_progress_call(progress_callback, 8, 8, "done")

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
    _log("# 🏁🏁🏁  本日のシグナル 実行終了 (Engine)  🏁🏁🏁", ui=False, no_timestamp=True)
    _log(f"# ⏱️ {end_txt} | RUN-ID: {run_id}", ui=False, no_timestamp=True)
    try:
        print("#" * 68 + "\n", flush=True)
    except Exception:
        pass


def _log_previous_counts_summary(signals_dir: Path) -> None:
    """?????????????????"""
    try:
        prev = _load_prev_counts(signals_dir)
        if prev:
            for i in range(1, 8):
                key = f"system{i}"
                v = int(prev.get(key, 0))
                icon = "?" if v > 0 else "?"
                suffix = " ??" if v == 0 else ""
                _log(f"?? {icon} (????) {key}: {v} ?{suffix}")
    except Exception:
        pass


def _apply_system_filters_and_update_ctx(
    ctx: TodayRunContext,
    symbols: list[str],
    basic_data: dict[str, pd.DataFrame],
) -> dict[str, list[str]]:
    """????????????????????????"""
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
    try:
        cb2 = globals().get("_PER_SYSTEM_STAGE")
    except Exception:
        cb2 = None
    if cb2 and callable(cb2):
        try:
            cb2("system1", 25, len(system1_syms), None, None, None)
            cb2("system2", 25, len(system2_syms), None, None, None)
            cb2("system3", 25, len(system3_syms), None, None, None)
            cb2("system4", 25, len(system4_syms), None, None, None)
            cb2("system5", 25, len(system5_syms), None, None, None)
            cb2("system6", 25, len(system6_syms), None, None, None)
            cb2(
                "system7",
                25,
                1 if "SPY" in (basic_data or {}) else 0,
                None,
                None,
                None,
            )
        except Exception:
            pass
    return filters


def _log_system1_filter_stats(symbols: list[str], basic_data: dict[str, pd.DataFrame]) -> None:
    """System1 ???????????????????"""
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
        _log("?? system1???????: " + f"??={s1_total}, ??>=5: {s1_price}, DV20>=50M: {s1_dv}")
    except Exception:
        pass


def _log_system2_filter_stats(symbols: list[str], basic_data: dict[str, pd.DataFrame]) -> None:
    """System2 ???????????????????"""
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
            "?? system2???????: "
            + f"??={s2_total}, ??>=5: {c_price}, DV20>=25M: {c_dv}, ATR比率>=3%: {c_atr}"
        )
    except Exception:
        pass


def _log_system3_filter_stats(symbols: list[str], basic_data: dict[str, pd.DataFrame]) -> None:
    """System3 ???????????????????"""
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
            "?? system3???????: "
            + f"??={s3_total}, Low>=1: {s3_low}, AvgVol50>=1M: {s3_av}, ATR_Ratio>=5%: {s3_atr}"
        )
    except Exception:
        pass


def _log_system4_filter_stats(symbols: list[str], basic_data: dict[str, pd.DataFrame]) -> None:
    """System4 ???????????????????"""
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
        _log("?? system4???????: " + f"??={s4_total}, DV50>=100M: {s4_dv}, HV50 10?40: {s4_hv}")
    except Exception:
        pass


def _log_system5_filter_stats(symbols: list[str], basic_data: dict[str, pd.DataFrame]) -> None:
    """System5 ???????????????????"""
    try:
        threshold_label = format_atr_pct_threshold_label()
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
            "?? system5???????: "
            + f"??={s5_total}, AvgVol50>500k: {s5_av}, DV50>2.5M: {s5_dv}"
            + f", {threshold_label}: {s5_atr}"
        )
    except Exception:
        pass


def _log_system6_filter_stats(symbols: list[str], basic_data: dict[str, pd.DataFrame]) -> None:
    """System6 ???????????????????"""
    try:
        s6_total = len(symbols)
        s6_low = 0
        s6_dv = 0
        for _sym in symbols:
            _df = basic_data.get(_sym)
            if _df is None or getattr(_df, "empty", True):
                continue
            try:
                low_ok, dv_ok = _system6_conditions(_df)
            except Exception:
                continue
            if low_ok:
                s6_low += 1
            else:
                continue
            if dv_ok:
                s6_dv += 1
        _log("?? system6???????: " + f"??={s6_total}, Low>=5: {s6_low}, DV50>10M: {s6_dv}")
    except Exception:
        pass


def _log_system7_filter_stats(basic_data: dict[str, pd.DataFrame]) -> None:
    """System7 (SPY) ?????????????????"""
    try:
        spyp = (
            1 if ("SPY" in basic_data and not getattr(basic_data.get("SPY"), "empty", True)) else 0
        )
        _log("?? system7???????: SPY?? | SPY??=" + str(spyp))
    except Exception:
        pass


def _log_system_filter_stats(
    symbols: list[str],
    basic_data: dict[str, pd.DataFrame],
    filters: dict[str, list[str]],
) -> None:
    """????????????????????????"""
    _log("?? ?????????? (system1?system6)?")
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
        "?? ???????: "
        + f"system1={len(system1_syms)}?, "
        + f"system2={len(system2_syms)}?, "
        + f"system3={len(system3_syms)}?, "
        + f"system4={len(system4_syms)}?, "
        + f"system5={len(system5_syms)}?, "
        + f"system6={len(system6_syms)}?"
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
            return rolling_new
        return rolling_df
    last_date = None
    try:
        last_date = rolling_df.index[-1]
        if isinstance(last_date, str):
            # wrap scalar into list to satisfy type checker overloads
            last_date = pd.to_datetime([last_date])[0]
    except Exception:
        return rolling_df
    lag_days = (today - last_date).days
    if lag_days > max_lag_days:
        # 鮮度不足: baseからrolling再生成
        base_df = cast(Any, cache_manager).read(symbol, layer="base", rows=base_rows)
        if base_df is not None and not getattr(base_df, "empty", True):
            rolling_new = base_df.tail(base_rows).copy()
            cast(Any, cache_manager).write_atomic(symbol, rolling_new, layer="rolling")
            return rolling_new
    return rolling_df


def _prepare_system2_data(
    basic_data: dict[str, pd.DataFrame],
    system_symbols: list[str],
) -> tuple[dict[str, pd.DataFrame], int, int, int]:
    """System2 ???????????????????"""
    _log("?? ?????????????? (system2)?")
    raw_data = _subset_data(basic_data, system_symbols)
    _log(f"?? ???????: system2={len(raw_data)}??")
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
            "?? system2????????: "
            + f"??????={s2_filter}, RSI3>90: {s2_rsi}, "
            + f"TwoDayUp: {s2_combo}"
        )
        try:
            cb2 = globals().get("_PER_SYSTEM_STAGE")
        except Exception:
            cb2 = None
        if cb2 and callable(cb2):
            try:
                cb2("system2", 50, int(s2_filter), int(s2_combo), None, None)
            except Exception:
                pass
    except Exception:
        pass
    return raw_data, s2_filter, s2_rsi, s2_combo


def _prepare_system3_data(
    basic_data: dict[str, pd.DataFrame],
    system_symbols: list[str],
) -> tuple[dict[str, pd.DataFrame], int, int, int]:
    """System3 ???????????????????"""
    _log("?? ?????????????? (system3)?")
    raw_data = _subset_data(basic_data, system_symbols)
    _log(f"?? ???????: system3={len(raw_data)}??")
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
                close_pass = float(last.get("Close", 0)) > float(last.get("SMA150", float("inf")))
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
            "?? system3????????: "
            + f"??????={s3_filter}, Close>SMA150: {s3_close}, "
            + f"3????>=12.5%: {s3_combo}"
        )
        try:
            cb2 = globals().get("_PER_SYSTEM_STAGE")
        except Exception:
            cb2 = None
        if cb2 and callable(cb2):
            try:
                cb2("system3", 50, int(s3_filter), int(s3_combo), None, None)
            except Exception:
                pass
    except Exception:
        pass
    return raw_data, s3_filter, s3_close, s3_combo


def _prepare_system4_data(
    basic_data: dict[str, pd.DataFrame],
    system_symbols: list[str],
) -> tuple[dict[str, pd.DataFrame], int, int]:
    """System4 ???????????????????"""
    _log("?? ?????????????? (system4)?")
    raw_data = _subset_data(basic_data, system_symbols)
    _log(f"?? ???????: system4={len(raw_data)}??")
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
                if float(last.get("Close", 0)) > float(last.get("SMA200", float("inf"))):
                    s4_close += 1
            except Exception:
                pass
        _log(f"?? system4????????: ??????={s4_filter}, Close>SMA200: {s4_close}")
        try:
            cb2 = globals().get("_PER_SYSTEM_STAGE")
        except Exception:
            cb2 = None
        if cb2 and callable(cb2):
            try:
                cb2("system4", 50, int(s4_filter), int(s4_close), None, None)
            except Exception:
                pass
    except Exception:
        pass
    return raw_data, s4_filter, s4_close


def _prepare_system5_data(
    basic_data: dict[str, pd.DataFrame],
    system_symbols: list[str],
) -> tuple[dict[str, pd.DataFrame], int, int, int, int]:
    """System5 ???????????????????"""
    _log("?? ?????????????? (system5)?")
    raw_data = _subset_data(basic_data, system_symbols)
    _log(f"?? ???????: system5={len(raw_data)}??")
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
                price_pass = float(last.get("Close", 0)) > float(last.get("SMA100", 0)) + float(
                    last.get("ATR10", 0)
                )
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
            "?? system5????????: "
            + f"??????={s5_filter}, Close>SMA100+ATR10: {s5_close}, "
            + f"ADX7>55: {s5_adx}, RSI3<50: {s5_combo}"
        )
        try:
            cb2 = globals().get("_PER_SYSTEM_STAGE")
        except Exception:
            cb2 = None
        if cb2 and callable(cb2):
            try:
                cb2("system5", 50, int(s5_filter), int(s5_combo), None, None)
            except Exception:
                pass
    except Exception:
        pass
    return raw_data, s5_filter, s5_close, s5_adx, s5_combo


def _prepare_system6_data(
    basic_data: dict[str, pd.DataFrame],
    system_symbols: list[str],
) -> tuple[dict[str, pd.DataFrame], int, int, int]:
    """System6 ???????????????????"""
    _log("?? ?????????????? (system6)?")
    raw_data = _subset_data(basic_data, system_symbols)
    _log(f"?? ???????: system6={len(raw_data)}??")
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
                ret_pass = float(last.get("return_6d", 0)) > 0.20
            except Exception:
                ret_pass = False
            if not ret_pass:
                continue
            s6_ret += 1
            try:
                if bool(last.get("UpTwoDays", False)):
                    s6_combo += 1
            except Exception:
                pass
        _log(
            "?? system6????????: "
            + f"??????={s6_filter}, return_6d>20%: {s6_ret}, "
            + f"UpTwoDays: {s6_combo}"
        )
        try:
            cb2 = globals().get("_PER_SYSTEM_STAGE")
        except Exception:
            cb2 = None
        if cb2 and callable(cb2):
            try:
                cb2("system6", 50, int(s6_filter), int(s6_combo), None, None)
            except Exception:
                pass
    except Exception:
        pass
    return raw_data, s6_filter, s6_ret, s6_combo


def _resolve_spy_dataframe(basic_data: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    """SPY ??????????????????"""
    if "SPY" in basic_data:
        try:
            return get_spy_with_indicators(basic_data["SPY"])
        except Exception:
            return None
    _log(
        "?? SPY ?????????????? (base/full_backup/rolling ?????????)"
        + " SPY.csv ? data_cache/base ???? data_cache/full_backup ?????????"
    )
    return None


@no_type_check
def compute_today_signals(
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
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """当日シグナル抽出＋配分の本体。

    Args:
        symbols: 対象シンボルリスト。
        parallel: True の場合はシステムごとのシグナル抽出を並行実行する。

    戻り値: (final_df, per_system_df_dict)
    """
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
    )

    try:
        GLOBAL_STAGE_METRICS.reset()
    except Exception:
        pass

    # CLI 経由で未設定の場合（UI 等）、既定で日付別ログに切替
    try:
        if globals().get("_LOG_FILE_PATH") is None:
            import os as _os

            _mode_env = (_os.environ.get("TODAY_SIGNALS_LOG_MODE") or "").strip().lower()
            _configure_today_logger(mode=("single" if _mode_env == "single" else "dated"))
    except Exception:
        pass

    _run_id = ctx.run_id
    settings = ctx.settings
    # install log callback for helpers
    globals()["_LOG_CALLBACK"] = ctx.log_callback
    signals_dir = ctx.signals_dir

    run_start_time = ctx.run_start_time
    start_equity = ctx.start_equity
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
        import logging as _lg
        import os as _os

        if not _os.environ.get("STREAMLIT_SERVER_ENABLED"):

            class _SilenceBareModeWarnings(_lg.Filter):
                def filter(self, record: _lg.LogRecord) -> bool:
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
                _logger = _lg.getLogger(_name)
                _logger.addFilter(_SilenceBareModeWarnings())
                try:
                    _logger.setLevel(_lg.ERROR)
                except Exception:
                    pass
    except Exception:
        pass

    # 対象とするNYSE営業日
    today = get_signal_target_trading_day().normalize()
    ctx.today = today

    # Run start banner (CLI only) - 最初に実行開始メッセージを表示
    try:
        print("#" * 68, flush=True)
    except Exception:
        pass
    _log("# 🚀🚀🚀  本日のシグナル 実行開始 (Engine)  🚀🚀🚀", ui=False, no_timestamp=True)
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

    _log(f"📅 対象営業日（NYSE）: {today.date()}", no_timestamp=True)
    _log("ℹ️ 注: EODHDは当日終値が未反映のため、直近営業日ベースで計算します。", no_timestamp=True)
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

    symbols = _prepare_symbol_universe(ctx, symbols)
    basic_data = _load_universe_basic_data(ctx, symbols)

    basic_data = _precompute_shared_indicators_phase(ctx, basic_data)
    _log("🧪 事前フィルター実行中 (system1〜system6)…")
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
    # 各システムのフィルター通過件数をUIへ通知
    try:
        cb2 = globals().get("_PER_SYSTEM_STAGE")
    except Exception:
        cb2 = None
    if cb2 and callable(cb2):
        try:
            cb2("system1", 25, len(system1_syms), None, None, None)
            cb2("system2", 25, len(system2_syms), None, None, None)
            cb2("system3", 25, len(system3_syms), None, None, None)
            cb2("system4", 25, len(system4_syms), None, None, None)
            cb2("system5", 25, len(system5_syms), None, None, None)
            cb2("system6", 25, len(system6_syms), None, None, None)
            cb2(
                "system7",
                25,
                1 if "SPY" in (basic_data or {}) else 0,
                None,
                None,
                None,
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
    # System1 フィルター内訳（価格・売買代金）
    try:
        stats1 = filter_stats.get("system1", {})
        s1_total = stats1.get("total", len(symbols or []))
        s1_price = stats1.get("price_pass", 0)
        s1_dv = stats1.get("dv_pass", 0)
        rate_logger = _get_rate_limited_logger()
        rate_logger.debug_rate_limited(
            f"🧪 system1内訳: 元={s1_total}, 価格>=5: {s1_price}, DV20>=50M: {s1_dv}",
            interval=10.0,
            message_key="system1内訳",
        )
    except Exception:
        pass
    # System3 フィルター内訳（Low>=1 → AvgVol50>=1M → ATR_Ratio>=5%）
    try:
        stats3 = filter_stats.get("system3", {})
        s3_total = stats3.get("total", len(symbols or []))
        s3_low = stats3.get("low_pass", 0)
        s3_av = stats3.get("avgvol_pass", 0)
        s3_atr = stats3.get("atr_pass", 0)
        _log(
            "🧪 system3内訳: "
            + f"元={s3_total}, Low>=1: {s3_low}, AvgVol50>=1M: {s3_av}, ATR_Ratio>=5%: {s3_atr}"
        )
    except Exception:
        pass
    # System4 フィルター内訳（DV50>=100M → HV50 10〜40）
    try:
        stats4 = filter_stats.get("system4", {})
        s4_total = stats4.get("total", len(symbols or []))
        s4_dv = stats4.get("dv_pass", 0)
        s4_hv = stats4.get("hv_pass", 0)
        rate_limited_logger.debug_rate_limited(
            f"🧪 system4内訳: 元={s4_total}, DV50>=100M: {s4_dv}, HV50 10〜40: {s4_hv}",
            message_key="system4_detail",
            interval=10,
        )
    except Exception:
        pass
    # System5 フィルター内訳（AvgVol50>500k → DV50>2.5M → ATR_Pct>閾値）
    try:
        threshold_label = format_atr_pct_threshold_label()
        stats5 = filter_stats.get("system5", {})
        s5_total = stats5.get("total", len(symbols or []))
        s5_av = stats5.get("avgvol_pass", 0)
        s5_dv = stats5.get("dv_pass", 0)
        s5_atr = stats5.get("atr_pass", 0)
        rate_limited_logger.debug_rate_limited(
            f"🧪 system5内訳: 元={s5_total}, AvgVol50>500k: {s5_av}, DV50>2.5M: {s5_dv}, "
            f"{threshold_label}: {s5_atr}",
            message_key="system5_detail",
            interval=10,
        )
    except Exception:
        pass
    # System6 フィルター内訳（Low>=5 → DV50>10M）
    try:
        stats6 = filter_stats.get("system6", {})
        s6_total = stats6.get("total", len(symbols or []))
        s6_low = stats6.get("low_pass", 0)
        s6_dv = stats6.get("dv_pass", 0)
        rate_limited_logger.debug_rate_limited(
            f"🧪 system6内訳: 元={s6_total}, Low>=5: {s6_low}, DV50>10M: {s6_dv}",
            message_key="system6_detail",
            interval=10,
        )
    except Exception:
        pass
    # System7 は SPY 固定（参考情報のみ）
    try:
        spyp = (
            1 if ("SPY" in basic_data and not getattr(basic_data.get("SPY"), "empty", True)) else 0
        )
        rate_limited_logger.debug_rate_limited(
            f"🧪 system7内訳: SPY固定 | SPY存在={spyp}", message_key="system7_detail", interval=10
        )
    except Exception:
        pass
    _log(
        "🧪 フィルター結果: "
        + f"system1={len(system1_syms)}件, "
        + f"system2={len(system2_syms)}件, "
        + f"system3={len(system3_syms)}件, "
        + f"system4={len(system4_syms)}件, "
        + f"system5={len(system5_syms)}件, "
        + f"system6={len(system6_syms)}件"
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
    s1_spy_gate = None
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
                    _spy_ok = int(float(_last.get("Close", 0)) > float(_last.get("SMA100", 0)))
        except Exception:
            _spy_ok = None
        for _sym, _df in (raw_data_system1 or {}).items():
            if _df is None or getattr(_df, "empty", True):
                continue
            try:
                last = _df.iloc[-1]
            except Exception:
                continue
            try:
                sma_pass = float(last.get("SMA25", float("nan"))) > float(
                    last.get("SMA50", float("nan"))
                )
            except Exception:
                sma_pass = False
            if sma_pass:
                s1_setup_calc += 1
        s1_setup = int(s1_setup_calc)
        # 出力順: フィルタ通過 → SPY>SMA100 → SMA25>SMA50
        if _spy_ok is None:
            _log(
                f"🧩 system1セットアップ内訳: フィルタ通過={s1_filter}, SPY>SMA100: -, "
                f"SMA25>SMA50: {s1_setup}"
            )
        else:
            _log(
                f"🧩 system1セットアップ内訳: フィルタ通過={s1_filter}, SPY>SMA100: {_spy_ok}, "
                f"SMA25>SMA50: {s1_setup}"
            )
        # UI の STUpass へ反映（50%時点）
        try:
            cb2 = globals().get("_PER_SYSTEM_STAGE")
            if cb2 and callable(cb2):
                # SPY ゲート（Close>SMA100）が偽なら STUpass は 0 扱い
                s1_setup_eff = int(s1_setup)
                try:
                    if isinstance(_spy_ok, int) and _spy_ok == 0:
                        s1_setup_eff = 0
                except Exception:
                    pass
                cb2("system1", 50, int(s1_filter), int(s1_setup_eff), None, None)
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
        s1_spy_gate = _spy_ok
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
        s2_setup = int(s2_combo)
        _log(
            "🧩 system2セットアップ内訳: "
            + f"フィルタ通過={s2_filter}, RSI3>90: {s2_rsi}, "
            + f"TwoDayUp: {s2_setup}"
        )
        try:
            cb2 = globals().get("_PER_SYSTEM_STAGE")
            if cb2 and callable(cb2):
                cb2("system2", 50, int(s2_filter), int(s2_setup), None, None)
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
        for _sym in system3_syms or []:
            _df = raw_data_system3.get(_sym)
            if _df is None or getattr(_df, "empty", True):
                continue
            try:
                last = _df.iloc[-1]
            except Exception:
                continue
            try:
                close_pass = float(last.get("Close", 0)) > float(last.get("SMA150", float("inf")))
            except Exception:
                close_pass = False
            if not close_pass:
                continue
            s3_close += 1
            try:
                drop_pass = float(last.get("Drop3D", 0)) >= 0.125
            except Exception:
                drop_pass = False
            if drop_pass:
                s3_combo += 1
        s3_setup = int(s3_combo)
        _log(
            "🧩 system3セットアップ内訳: "
            + f"フィルタ通過={s3_filter}, Close>SMA150: {s3_close}, "
            + f"3日下落率>=12.5%: {s3_setup}"
        )
        try:
            cb2 = globals().get("_PER_SYSTEM_STAGE")
            if cb2 and callable(cb2):
                cb2("system3", 50, int(s3_filter), int(s3_setup), None, None)
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
                if float(last.get("Close", 0)) > float(last.get("SMA200", float("inf"))):
                    s4_close += 1
            except Exception:
                pass
        _log(f"🧩 system4セットアップ内訳: フィルタ通過={s4_filter}, Close>SMA200: {s4_close}")
        try:
            cb2 = globals().get("_PER_SYSTEM_STAGE")
            if cb2 and callable(cb2):
                cb2("system4", 50, int(s4_filter), int(s4_close), None, None)
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
                price_pass = float(last.get("Close", 0)) > float(last.get("SMA100", 0)) + float(
                    last.get("ATR10", 0)
                )
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
                rsi_pass = float(last.get("RSI3", 100)) < 50
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
            cb2 = globals().get("_PER_SYSTEM_STAGE")
            if cb2 and callable(cb2):
                cb2("system5", 50, int(s5_filter), int(s5_setup), None, None)
        except Exception:
            pass
    except Exception:
        pass
    _log("🧮 指標計算用データロード中 (system6)…")
    raw_data_system6 = _subset_data(basic_data, system6_syms)
    _log(f"🧮 指標データ: system6={len(raw_data_system6)}銘柄")
    # System6 セットアップ内訳: フィルタ通過, return_6d>20%, UpTwoDays
    s6_setup = None
    try:
        s6_filter = int(len(system6_syms))
        s6_ret = 0
        s6_combo = 0
        for _sym in system6_syms or []:
            _df = raw_data_system6.get(_sym)
            if _df is None or getattr(_df, "empty", True):
                continue
            try:
                last = _df.iloc[-1]
            except Exception:
                continue
            try:
                # return_6d: 旧称 Return6D (命名統一済)
                ret_pass = float(last.get("return_6d", 0)) > 0.20
            except Exception:
                ret_pass = False
            if not ret_pass:
                continue
            s6_ret += 1
            try:
                if bool(last.get("UpTwoDays", False)):
                    s6_combo += 1
            except Exception:
                pass
        s6_setup = int(s6_combo)
        _log(
            "🧩 system6セットアップ内訳: "
            + f"フィルタ通過={s6_filter}, return_6d>20%: {s6_ret}, "
            + f"UpTwoDays: {s6_setup}"
        )
        try:
            cb2 = globals().get("_PER_SYSTEM_STAGE")
            if cb2 and callable(cb2):
                cb2("system6", 50, int(s6_filter), int(s6_setup), None, None)
        except Exception:
            pass
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
        System6Strategy(fixed_mode=True),
        System7Strategy(),
    ]
    strategies = {getattr(s, "SYSTEM_NAME", "").lower(): s for s in strategy_objs}
    # エンジン層はUI依存を排除（UI表示はlog/progressコールバック側に任せる）

    def _run_strategy(name: str, stg) -> tuple[str, pd.DataFrame, str, list[str]]:
        logs: list[str] = []

        def _local_log(message: str) -> None:
            logs.append(str(message))
            # UI コールバックがあればフィルタ済みで送信、無ければ CLI に出力
            try:
                cb = globals().get("_LOG_CALLBACK")
            except Exception:
                cb = None
            # Only invoke UI callback from the main thread. Background
            # worker threads should not call Streamlit APIs directly; their
            # logs will be forwarded by the main thread after task
            # completion.
            try:
                is_main = threading.current_thread() is threading.main_thread()
            except Exception:
                is_main = False
            if cb and callable(cb) and is_main:
                _emit_ui_log(f"[{name}] {message}")
            else:
                try:
                    print(f"[{name}] {message}", flush=True)
                except Exception:
                    pass

        if name == "system1":
            base = raw_data_system1
        elif name == "system2":
            base = raw_data_system2
        elif name == "system3":
            base = raw_data_system3
        elif name == "system4":
            base = raw_data_system4
        elif name == "system5":
            base = raw_data_system5
        elif name == "system6":
            base = raw_data_system6
        elif name == "system7":
            base = {"SPY": basic_data.get("SPY")}
        else:
            base = basic_data
        if name == "system4" and spy_df is None:
            _local_log(
                "⚠️ System4 は SPY 指標が必要ですが "
                + "SPY データがありません。"
                + "スキップします。"
            )
            return name, pd.DataFrame(), f"❌ {name}: 0 件 🚫", logs
        _local_log(f"🔎 {name}: シグナル抽出を開始")
        pool_outcome: str | None = None
        df = pd.DataFrame()
        try:
            # 段階進捗: 0/25/50/75/100 を UI 側に橋渡し
            stage_state: dict[int, tuple[int | None, int | None, int | None, int | None]] = {}
            phase_names = {
                0: "フィルターフェーズ",
                25: "セットアップフェーズ",
                50: "トレード候補フェーズ",
                75: "エントリーフェーズ",
            }
            prev_phase_map = {25: 0, 50: 25, 75: 50, 100: 75}
            phase_started: set[int] = set()
            phase_completed: set[int] = set()

            def _safe_stage_int(value: int | float | None) -> int | None:
                try:
                    if value is None:
                        return None
                    return int(value)
                except Exception:
                    return None

            def _format_stage_message(
                progress: int,
                filter_count: int | None,
                setup_count: int | None,
                candidate_count: int | None,
                final_count: int | None,
            ) -> str | None:
                filter_int = _safe_stage_int(filter_count)
                setup_int = _safe_stage_int(setup_count)
                candidate_int = _safe_stage_int(candidate_count)
                final_int = _safe_stage_int(final_count)

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
                    if final_int is not None:
                        parts: list[str] = []
                        if candidate_int is not None:
                            parts.append(f"候補 {candidate_int} 銘柄")
                        parts.append(f"エントリー {final_int} 銘柄")
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
                label = phase_names.get(prev_stage)
                if not label:
                    return None
                if prev_stage == 0:
                    if filter_int is not None:
                        return f"🏁 {name}: {label}のプロセスプールが完了 (通過 {filter_int} 銘柄)"
                    return f"🏁 {name}: {label}のプロセスプールが完了"
                if prev_stage == 25:
                    if setup_int is not None and filter_int is not None:
                        return (
                            f"🏁 {name}: {label}のプロセスプールが完了 "
                            f"(セットアップ通過 {setup_int}/{filter_int} 銘柄)"
                        )
                    if setup_int is not None:
                        return (
                            f"🏁 {name}: {label}のプロセスプールが完了 "
                            f"(セットアップ通過 {setup_int} 銘柄)"
                        )
                    return f"🏁 {name}: {label}のプロセスプールが完了"
                if prev_stage == 50:
                    if candidate_int is not None:
                        return (
                            f"🏁 {name}: {label}のプロセスプールが完了 "
                            f"(当日候補 {candidate_int} 銘柄)"
                        )
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
                v: int,
                f: int | None = None,
                s: int | None = None,
                c: int | None = None,
                fin: int | None = None,
            ) -> None:
                progress_val = max(0, min(100, int(v)))
                f_int = _safe_stage_int(f)
                s_int = _safe_stage_int(s)
                c_int = _safe_stage_int(c)
                fin_int = _safe_stage_int(fin)
                try:
                    cb2 = globals().get("_PER_SYSTEM_STAGE")
                except Exception:
                    cb2 = None
                # Only call the per-system UI callback directly from the
                # main thread. When running in background threads (e.g.
                # via ThreadPoolExecutor) we must avoid invoking Streamlit
                # APIs from non-main threads — instead record the stage
                # into GLOBAL_STAGE_METRICS and let the main thread drain
                # and forward events.
                try:
                    is_main = threading.current_thread() is threading.main_thread()
                except Exception:
                    is_main = False
                if cb2 and callable(cb2) and is_main:
                    try:
                        cb2(name, progress_val, f_int, s_int, c_int, fin_int)
                    except Exception:
                        pass
                # TRDlist件数スナップショットを更新（後段のメインスレッド通知で使用）
                if use_process_pool:
                    try:
                        # 正規化したタプルで前回値と比較し、変化があれば必ずイベントを
                        # 登録する。None と 0 や空文字列のような微妙な差を吸収する
                        # ため、整数化した値で比較する。
                        key = (
                            _safe_stage_int(f_int),
                            _safe_stage_int(s_int),
                            _safe_stage_int(c_int),
                            _safe_stage_int(fin_int),
                        )
                        prev = stage_state.get(progress_val)
                        if prev != key:
                            stage_state[progress_val] = key
                            try:
                                # 常に emit_event=True でイベントを積む（UI 側で重複
                                # 表示抑制する責務を負わせることも可能だが、ここは
                                # イベントの喪失を避けるため明示的に通知する）
                                GLOBAL_STAGE_METRICS.record_stage(
                                    name,
                                    progress_val,
                                    f_int,
                                    s_int,
                                    c_int,
                                    fin_int,
                                    emit_event=True,
                                )
                            except Exception:
                                pass
                            prev_stage_val = prev_phase_map.get(progress_val)
                            if prev_stage_val is not None and prev_stage_val not in phase_completed:
                                completion_msg = _format_phase_completion(
                                    prev_stage_val, f_int, s_int, c_int, fin_int
                                )
                                if completion_msg:
                                    _local_log(completion_msg)
                                phase_completed.add(prev_stage_val)
                            msg = _format_stage_message(progress_val, f_int, s_int, c_int, fin_int)
                            if msg:
                                _local_log(msg)
                            if progress_val in phase_names and progress_val not in phase_started:
                                _local_log(
                                    f"⚙️ {name}: {phase_names[progress_val]}のプロセスプールを開始"
                                )
                                phase_started.add(progress_val)
                    except Exception:
                        pass
                else:
                    try:
                        GLOBAL_STAGE_METRICS.record_stage(
                            name,
                            progress_val,
                            f_int,
                            s_int,
                            c_int,
                            fin_int,
                        )
                    except Exception:
                        pass

            import os as _os

            # プロセスプール利用可否（環境変数で上書き可）
            env_pp_raw = _os.environ.get("USE_PROCESS_POOL", "")
            env_pp = env_pp_raw.strip().lower()
            if env_pp in {"1", "true", "yes", "on"}:
                use_process_pool = True
            elif env_pp in {"0", "false", "no", "off"}:
                use_process_pool = False
            else:
                use_process_pool = False
                if env_pp:
                    _local_log(
                        "⚠️ "
                        + f"{name}: USE_PROCESS_POOL の値 '{env_pp_raw}' を解釈できません。"
                        + "プロセスプールを無効化します。"
                    )
            # ワーカー数は環境変数があれば優先、無ければ設定(THREADS_DEFAULT)に連動
            try:
                _env_workers = _os.environ.get("PROCESS_POOL_WORKERS", "").strip()
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
            # ルックバックは『必要指標の最大窓＋α』を動的推定
            try:
                settings2 = get_settings(create_dirs=True)
                lb_default = int(
                    settings2.cache.rolling.base_lookback_days + settings2.cache.rolling.buffer_days
                )
            except Exception:
                settings2 = None
                lb_default = 300
            # YAMLのstrategiesセクション等からヒントを取得（なければヒューリスティック）
            # ルックバックのマージン/最小日数は環境変数で上書き可能
            try:
                margin = float(_os.environ.get("LOOKBACK_MARGIN", "0.15"))
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
            custom_need = None
            try:
                fn = getattr(stg, "get_total_days", None)
                if callable(fn):
                    _val = fn(base)
                    if isinstance(_val, int | float):
                        custom_need = int(_val)
                    elif isinstance(_val, str):
                        try:
                            custom_need = int(float(_val))
                        except Exception:
                            custom_need = None
                    else:
                        custom_need = None
            except Exception:
                custom_need = None
            try:
                min_floor = int(_os.environ.get("LOOKBACK_MIN_DAYS", "80"))
            except Exception:
                min_floor = 80
            min_required = custom_need or need_map.get(name, lb_default)
            lookback_days = min(lb_default, max(min_floor, int(min_required)))
            _t0 = __import__("time").time()
            # プロセスプール利用時も stage_progress を渡し、要所の進捗ログを共有する
            _stage_cb = _stage
            _log_cb = None if use_process_pool else _local_log
            # プロセスプール利用時は Manager().Queue を生成して子プロセスから
            # 進捗を送れるようにする。globals に置いて子が参照できるようにする。
            if use_process_pool:
                try:
                    mgr = multiprocessing.Manager()  # noqa: F401 (kept for child access)
                    progress_q = mgr.Queue()
                    globals()["_PROGRESS_MANAGER"] = mgr
                    globals()["_PROGRESS_QUEUE"] = progress_q
                except Exception:
                    globals().pop("_PROGRESS_MANAGER", None)
                    globals().pop("_PROGRESS_QUEUE", None)
            if use_process_pool:
                workers_label = str(max_workers) if max_workers is not None else "auto"
                _local_log(
                    f"⚙️ {name}: USE_PROCESS_POOL=1 でプロセスプール実行を開始"
                    + f" (workers={workers_label})"
                    + " | 並列化: インジケーター計算/前処理"
                )
                _local_log(
                    f"🧭 {name}: フィルター・セットアップ・候補抽出は"
                    "メインプロセスで進行状況を記録します"
                )
            df = stg.get_today_signals(
                base,
                market_df=spy_df,
                today=today,
                progress_callback=None,
                log_callback=_log_cb,
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
            _local_log(f"⏱️ {name}: 経過 {_m}分{_s}秒")
        except Exception as e:  # noqa: BLE001
            _local_log(f"⚠️ {name}: シグナル抽出に失敗しました: {e}")
            # プロセスプール異常時はフォールバック（非プール）で一度だけ再試行
            try:
                msg = str(e).lower()
            except Exception:
                msg = ""
            if use_process_pool and pool_outcome is None:
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
                _local_log("🛟 フォールバック再試行: プロセスプール無効化で実行します")
                try:
                    _t0b = __import__("time").time()
                    df = stg.get_today_signals(
                        base,
                        market_df=spy_df,
                        today=today,
                        progress_callback=None,
                        log_callback=_local_log,
                        stage_progress=None,
                        use_process_pool=False,
                        max_workers=None,
                        lookback_days=lookback_days,
                    )
                    _elapsed_b = int(max(0, __import__("time").time() - _t0b))
                    _m2, _s2 = divmod(_elapsed_b, 60)
                    _local_log(f"⏱️ {name} (fallback): 経過 {_m2}分{_s2}秒")
                    if use_process_pool:
                        pool_outcome = "fallback"
                except Exception as e2:  # noqa: BLE001
                    _local_log(f"❌ {name}: フォールバックも失敗: {e2}")
                    if use_process_pool:
                        pool_outcome = "error"
                    df = pd.DataFrame()
            else:
                df = pd.DataFrame()
        finally:
            if use_process_pool:
                if pool_outcome == "success":
                    _local_log(f"🏁 {name}: プロセスプール実行が完了しました")
                elif pool_outcome == "fallback":
                    _local_log(f"🏁 {name}: プロセスプール実行を終了（フォールバック実行済み）")
                else:
                    _local_log(f"🏁 {name}: プロセスプール実行を終了（結果: 失敗）")
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
        _local_log(msg)
        return name, df, msg, logs

    # 抽出開始前にセットアップ通過のまとめを出力
    try:
        setup_summary = []
        for name, val in (
            ("system1", s1_setup_eff if s1_setup_eff is not None else s1_setup),
            ("system2", s2_setup),
            ("system3", s3_setup),
            ("system4", locals().get("s4_close")),
            ("system5", s5_setup),
            ("system6", s6_setup),
            ("system7", 1 if ("SPY" in (basic_data or {})) else 0),
        ):
            try:
                if val is not None:
                    setup_summary.append(f"{name}={int(val)}")
            except Exception:
                continue
        if setup_summary:
            _log("🧩 セットアップ通過まとめ: " + ", ".join(setup_summary))
    except Exception:
        pass

    _log("🚀 各システムの当日シグナル抽出を開始")
    per_system: dict[str, pd.DataFrame] = {}
    total = len(strategies)
    # 事前に全システムへステージ0%（filter開始）を同時通知（UI同期表示用）
    try:
        cb2 = globals().get("_PER_SYSTEM_STAGE")
    except Exception:
        cb2 = None
    if cb2 and callable(cb2):
        # 0% ステージの「対象→」はユニバース総数ベース（SPYは除外）
        try:
            universe_total = sum(1 for s in (symbols or []) if str(s).upper() != "SPY")
        except Exception:
            universe_total = len(symbols) if symbols is not None else 0
            try:
                has_spy = 1 if "SPY" in (symbols or []) else 0
                universe_total = max(0, int(universe_total) - has_spy)
            except Exception:
                pass
        for name in strategies.keys():
            try:
                cb2(name, 0, int(universe_total), None, None, None)
            except Exception:
                pass
    if parallel:
        if progress_callback:
            try:
                progress_callback(5, 8, "run_strategies")
            except Exception:
                pass
        with ThreadPoolExecutor() as executor:
            futures: dict[Future, str] = {}
            for name, stg in strategies.items():
                # systemごとの開始を通知
                if per_system_progress:
                    try:
                        per_system_progress(name, "start")
                    except Exception:
                        pass
                # CLI専用: 各システム開始を即時表示（UIには出さない）
                try:
                    _log(f"▶ {name} 開始", ui=False)
                except Exception:
                    pass
                fut = executor.submit(_run_strategy, name, stg)
                futures[fut] = name
            pending: set[Future] = set(futures.keys())
            completed_count = 0
            while pending:
                done, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                _drain_stage_event_queue()
                if not done:
                    continue
                for fut in done:
                    name, df, msg, logs = fut.result()
                    per_system[name] = df
                    try:
                        cb2 = globals().get("_PER_SYSTEM_STAGE")
                    except Exception:
                        cb2 = None
                    if cb2 and callable(cb2):
                        try:
                            try:
                                _mx = int(get_settings(create_dirs=False).risk.max_positions)
                            except Exception:
                                _mx = 10
                            _cand_cnt: int | None
                            try:
                                snapshot = _get_stage_snapshot(name)
                                _cand_cnt = (
                                    None
                                    if snapshot is None or snapshot.candidate_count is None
                                    else int(snapshot.candidate_count)
                                )
                            except Exception:
                                _cand_cnt = None
                            if _cand_cnt is None:
                                _cand_cnt = (
                                    0
                                    if (df is None or getattr(df, "empty", True))
                                    else int(len(df))
                                )
                            if _mx > 0:
                                _cand_cnt = min(int(_cand_cnt), int(_mx))
                            _entry_cnt: int
                            try:
                                _entry_cnt = (
                                    0
                                    if (df is None or getattr(df, "empty", True))
                                    else int(len(df))
                                )
                            except Exception:
                                _entry_cnt = 0
                            if _mx > 0:
                                _entry_cnt = min(int(_entry_cnt), int(_mx))
                            try:
                                GLOBAL_STAGE_METRICS.record_stage(
                                    name,
                                    100,
                                    None,
                                    None,
                                    _cand_cnt,
                                    _entry_cnt,
                                    emit_event=False,
                                )
                            except Exception:
                                pass
                            cb2(name, 75, None, None, int(_cand_cnt), None)
                            cb2(name, 100, None, None, int(_cand_cnt), int(_entry_cnt))
                        except Exception:
                            pass
                    for line in _filter_logs(logs, ui=False):
                        _log(f"[{name}] {line}", ui=False)
                    if per_system_progress:
                        try:
                            per_system_progress(name, "done")
                        except Exception:
                            pass
                    try:
                        _cnt = 0 if (df is None or getattr(df, "empty", True)) else int(len(df))
                    except Exception:
                        _cnt = -1
                    try:
                        _log(f"✅ {name} 完了: {('?' if _cnt < 0 else _cnt)}件", ui=False)
                    except Exception:
                        pass
                    if progress_callback:
                        try:
                            progress_callback(5 + min(completed_count + 1, 1), 8, name)
                        except Exception:
                            pass
                    completed_count += 1
                    try:
                        del futures[fut]
                    except Exception:
                        pass
                    _drain_stage_event_queue()
            _drain_stage_event_queue()
        if progress_callback:
            try:
                progress_callback(6, 8, "strategies_done")
            except Exception:
                pass
    else:
        for _idx, (name, stg) in enumerate(strategies.items(), start=1):
            if progress_callback:
                try:
                    progress_callback(5, 8, name)
                except Exception:
                    pass
            # 順次実行時も開始を通知
            if per_system_progress:
                try:
                    per_system_progress(name, "start")
                except Exception:
                    pass
            # CLI専用: 各システム開始を即時表示（UIには出さない）
            try:
                _log(f"▶ {name} 開始", ui=False)
            except Exception:
                pass
            name, df, msg, logs = _run_strategy(name, stg)
            per_system[name] = df
            _drain_stage_event_queue()
            # CLI専用: ワーカー収集ログを常に出力（UIには送らない）
            for line in _filter_logs(logs, ui=False):
                _log(f"[{name}] {line}", ui=False)
            # 即時: TRDlist（候補件数）を75%段階として通知（上限はmax_positions）
            try:
                cb2 = globals().get("_PER_SYSTEM_STAGE")
            except Exception:
                cb2 = None
            if cb2 and callable(cb2):
                try:
                    try:
                        _mx = int(get_settings(create_dirs=False).risk.max_positions)
                    except Exception:
                        _mx = 10
                    _cand_cnt: int | None
                    try:
                        snapshot = _get_stage_snapshot(name)
                        _cand_cnt = (
                            None
                            if snapshot is None or snapshot.candidate_count is None
                            else int(snapshot.candidate_count)
                        )
                    except Exception:
                        _cand_cnt = None
                    if _cand_cnt is None:
                        _cand_cnt = (
                            0 if (df is None or getattr(df, "empty", True)) else int(len(df))
                        )
                    if _mx > 0:
                        _cand_cnt = min(int(_cand_cnt), int(_mx))
                    _entry_cnt: int
                    try:
                        _entry_cnt = (
                            0 if (df is None or getattr(df, "empty", True)) else int(len(df))
                        )
                    except Exception:
                        _entry_cnt = 0
                    if _mx > 0:
                        _entry_cnt = min(int(_entry_cnt), int(_mx))
                    try:
                        GLOBAL_STAGE_METRICS.record_stage(
                            name,
                            100,
                            None,
                            None,
                            _cand_cnt,
                            _entry_cnt,
                            emit_event=False,
                        )
                    except Exception:
                        pass
                    cb2(name, 75, None, None, int(_cand_cnt), None)
                    cb2(name, 100, None, None, int(_cand_cnt), int(_entry_cnt))
                except Exception:
                    pass
            if per_system_progress:
                try:
                    per_system_progress(name, "done")
                except Exception:
                    pass
            # CLI専用: 完了を簡潔表示（件数付き）
            try:
                _cnt = 0 if (df is None or getattr(df, "empty", True)) else int(len(df))
            except Exception:
                _cnt = -1
            try:
                _log(f"✅ {name} 完了: {('?' if _cnt < 0 else _cnt)}件", ui=False)
            except Exception:
                pass
        _drain_stage_event_queue()
        # 即時の75%再通知は行わない（メインスレッド側で一括通知）
        # 前回結果は開始時にまとめて出力するため、ここでは出さない
        if progress_callback:
            try:
                progress_callback(6, 8, "strategies_done")
            except Exception:
                pass

    # システム別の順序を明示（1..7）に固定
    order_1_7 = [f"system{i}" for i in range(1, 8)]
    per_system = {k: per_system.get(k, pd.DataFrame()) for k in order_1_7 if k in per_system}
    ctx.per_system_frames = dict(per_system)

    metrics_summary_context = None

    # 並列実行時はワーカースレッドからの UI 更新が抑制されるため、
    # メインスレッドで候補件数（TRDlist）を75%段階として通知する
    try:
        cb2 = globals().get("_PER_SYSTEM_STAGE")
    except Exception:
        cb2 = None
    if cb2 and callable(cb2):
        try:
            # UIのTRDlist表示は最大ポジション数を超えないよう丸める
            try:
                _mx = int(get_settings(create_dirs=False).risk.max_positions)
            except Exception:
                _mx = 10
            for _name in order_1_7:
                # ワーカーからのスナップショットがあれば優先（型ゆらぎ等を超えて信頼できる値）
                _cand_cnt = None
                try:
                    snapshot = _get_stage_snapshot(_name)
                    if snapshot is not None and snapshot.candidate_count is not None:
                        _cand_cnt = int(snapshot.candidate_count)
                except Exception:
                    _cand_cnt = None
                if _cand_cnt is None:
                    _df_sys = per_system.get(_name, pd.DataFrame())
                    _cand_cnt = int(
                        0 if _df_sys is None or getattr(_df_sys, "empty", True) else len(_df_sys)
                    )
                if _mx > 0:
                    _cand_cnt = min(int(_cand_cnt), int(_mx))
                cb2(_name, 75, None, None, int(_cand_cnt), None)
        except Exception:
            pass

    # メトリクス保存前に、当日のトレード候補Top10を簡易出力（デバッグ/可視化用）
    try:
        # 追加: 候補日キーの診断（today/prev日正規化の確認）
        try:
            from common.today_signals import get_latest_nyse_trading_day as _gln  # type: ignore
        except Exception:
            _gln = None
        all_rows: list[pd.DataFrame] = []
        for _sys_name, df in per_system.items():
            if df is None or df.empty:
                continue
            x = df.copy()
            if "score" in x.columns:
                try:
                    asc = False
                    if "score_key" in x.columns and len(x):
                        asc = _asc_by_score_key(str(x.iloc[0].get("score_key")))
                    x["_sort_val"] = x["score"].astype(float)
                    if not asc:
                        x["_sort_val"] = -x["_sort_val"]
                except Exception:
                    x["_sort_val"] = 0.0
            else:
                x["_sort_val"] = 0.0
            all_rows.append(x)
        if all_rows:
            concat_rows = _prepare_concat_frames(all_rows)
            _log("📝 事前トレードリスト(Top10, メトリクス保存前)")
            if concat_rows:
                merged = pd.concat(concat_rows, ignore_index=True)
                merged = merged.sort_values("_sort_val", kind="stable", na_position="last")
                top10 = merged.head(10).drop(columns=["_sort_val"], errors="ignore")
                cols = [
                    c
                    for c in [
                        "symbol",
                        "system",
                        "side",
                        "entry_date",
                        "entry_price",
                        "stop_price",
                        "score_key",
                        "score",
                    ]
                    if c in top10.columns
                ]
                if not top10.empty:
                    _log(top10[cols].to_string(index=False))
                else:
                    _log("(候補なし)")
            else:
                _log("(候補なし)")
        # 追加: システム別のTop10を個別に出力（system2〜system6）
        try:
            for _sys_name in [f"system{i}" for i in range(2, 7)]:
                _df = per_system.get(_sys_name, pd.DataFrame())
                _log(f"📝 事前トレードリスト({_sys_name} Top10, メトリクス保存前)")
                if _df is None or getattr(_df, "empty", True):
                    _log("(候補なし)")
                    continue
                x = _df.copy()
                if "score" in x.columns:
                    try:
                        asc = False
                        if "score_key" in x.columns and len(x):
                            asc = _asc_by_score_key(str(x.iloc[0].get("score_key")))
                        x["_sort_val"] = x["score"].astype(float)
                        if not asc:
                            x["_sort_val"] = -x["_sort_val"]
                    except Exception:
                        x["_sort_val"] = 0.0
                else:
                    x["_sort_val"] = 0.0
                x = x.sort_values("_sort_val", kind="stable", na_position="last")
                top10_s = x.head(10).drop(columns=["_sort_val"], errors="ignore")
                cols_s = [
                    c
                    for c in [
                        "symbol",
                        "system",
                        "side",
                        "entry_date",
                        "entry_price",
                        "stop_price",
                        "score_key",
                        "score",
                    ]
                    if c in top10_s.columns
                ]
                if not top10_s.empty:
                    _log(top10_s[cols_s].to_string(index=False))
                else:
                    _log("(候補なし)")
            # 追加: 各systemで entry_date のユニーク日付を出力（最大3件）
            try:
                if "entry_date" in _df.columns and not _df.empty:
                    uniq = sorted(
                        {
                            pd.to_datetime([v])[0].date()
                            for v in _df["entry_date"].tolist()
                            if v is not None
                        }
                    )
                    sample_dates = ", ".join([str(d) for d in uniq[:3]])
                    _log(
                        f"🗓️ {_sys_name} entry日ユニーク: {sample_dates}"
                        + (" ..." if len(uniq) > 3 else "")
                    )
            except Exception:
                pass
        except Exception:
            pass
    except Exception:
        pass

    positions_cache: list[Any] | None = None
    symbol_system_map_cache: dict[str, str] | None = None

    # --- 日次メトリクス（事前フィルタ通過数・候補数）の保存 ---
    try:
        metrics_rows = []
        # 事前フィルタ通過数（存在しないシステムは0扱い）
        prefilter_map = {
            "system1": len(locals().get("system1_syms", []) or []),
            "system2": len(locals().get("system2_syms", []) or []),
            "system3": len(locals().get("system3_syms", []) or []),
            "system4": len(locals().get("system4_syms", []) or []),
            "system5": len(locals().get("system5_syms", []) or []),
            "system6": len(locals().get("system6_syms", []) or []),
            "system7": 1 if ("SPY" in (locals().get("basic_data", {}) or {})) else 0,
        }
        # 候補数（per_systemの行数）
        for sys_name in order_1_7:
            df_sys = per_system.get(sys_name, pd.DataFrame())
            candidates = int(0 if df_sys is None or getattr(df_sys, "empty", True) else len(df_sys))
            pre_count = int(prefilter_map.get(sys_name, 0))
            metrics_rows.append(
                {
                    "date": locals().get("today"),
                    "system": sys_name,
                    "prefilter_pass": pre_count,
                    "candidates": candidates,
                }
            )
        if metrics_rows:
            metrics_df = pd.DataFrame(metrics_rows)
            try:
                settings_out = get_settings(create_dirs=True)
                out_dir = Path(settings_out.outputs.results_csv_dir)
            except Exception:
                out_dir = Path("results_csv")
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            out_fp = out_dir / "daily_metrics.csv"
            try:
                try:
                    round_dec = getattr(settings_out.cache, "round_decimals", None)
                except Exception:
                    round_dec = None
                try:
                    metrics_out = round_dataframe(metrics_df, round_dec)
                except Exception:
                    metrics_out = metrics_df
                if out_fp.exists():
                    metrics_out.to_csv(
                        out_fp, mode="a", header=False, index=False, encoding="utf-8"
                    )
                else:
                    metrics_out.to_csv(out_fp, index=False, encoding="utf-8")
                _log(f"📈 メトリクス保存: {out_fp} に {len(metrics_rows)} 行を追記")
            except Exception as e:
                _log(f"⚠️ メトリクス保存に失敗: {e}")
            # 通知: 最終ステージ形式（Tgt/FILpass/STUpass/TRDlist/Entry/Exit）で送信
            try:
                # 0%のTgtはユニバース総数（SPY除く）
                try:
                    tgt_base = sum(1 for s in (symbols or []) if str(s).upper() != "SPY")
                except Exception:
                    tgt_base = len(symbols) if symbols is not None else 0
                    try:
                        if "SPY" in (symbols or []):
                            tgt_base = max(0, int(tgt_base) - 1)
                    except Exception:
                        pass

                # Exit 件数を簡易推定（Alpaca の保有ポジションと各 Strategy の compute_exit を利用）
                if positions_cache is None or symbol_system_map_cache is None:
                    positions_cache, symbol_system_map_cache = _fetch_positions_and_symbol_map()

                def _estimate_exit_counts_today(
                    positions0: Sequence[object],
                    symbol_system_map0: Mapping[str, str],
                ) -> dict[str, int]:
                    counts: dict[str, int] = {}
                    try:
                        # 価格ロード関数は共通ローダーを利用
                        from common.data_loader import load_price as _load_price  # lazy import

                        # SPY から本日の基準日（最新営業日）を推定
                        latest_trading_day = None
                        try:
                            spy_df0 = _load_price("SPY", cache_profile="rolling")
                            if spy_df0 is not None and not spy_df0.empty:
                                latest_trading_day = pd.to_datetime([spy_df0.index[-1]])[
                                    0
                                ].normalize()
                        except Exception:
                            latest_trading_day = None

                        # エントリー日のローカル記録と system 推定マップ
                        entry_map0 = load_entry_dates()
                        symbol_map_local = symbol_system_map0 or {}

                        for pos in positions0 or []:
                            try:
                                sym = str(getattr(pos, "symbol", "")).upper()
                                if not sym:
                                    continue
                                qty = int(abs(float(getattr(pos, "qty", 0)) or 0))
                                if qty <= 0:
                                    continue
                                pos_side = str(getattr(pos, "side", "")).lower()
                                mapped = symbol_map_local.get(sym)
                                if mapped is None and sym.lower() in symbol_map_local:
                                    mapped = symbol_map_local.get(sym.lower())
                                system0 = str(mapped or "").lower()
                                if not system0:
                                    if sym == "SPY" and pos_side == "short":
                                        system0 = "system7"
                                    else:
                                        continue
                                if system0 == "system7":
                                    continue
                                entry_date_str0 = entry_map0.get(sym)
                                if not entry_date_str0:
                                    continue
                                # 価格データ読込（full）
                                dfp = _load_price(sym, cache_profile="full")
                                if dfp is None or dfp.empty:
                                    continue
                                try:
                                    dfp2 = dfp.copy(deep=False)
                                    if "Date" in dfp2.columns:
                                        dfp2.index = pd.Index(
                                            pd.to_datetime(dfp2["Date"].to_numpy()).normalize()
                                        )
                                    else:
                                        dfp2.index = pd.Index(
                                            pd.to_datetime(dfp2.index.to_numpy()).normalize()
                                        )
                                except Exception:
                                    continue
                                if latest_trading_day is None and len(dfp2.index) > 0:
                                    latest_trading_day = pd.to_datetime([dfp2.index[-1]])[
                                        0
                                    ].normalize()
                                # エントリー日のインデックス
                                try:
                                    idx = dfp2.index
                                    ent_dt = pd.to_datetime([entry_date_str0])[0].normalize()
                                    if ent_dt in idx:
                                        ent_arr = idx.get_indexer([ent_dt])
                                    else:
                                        ent_arr = idx.get_indexer([ent_dt], method="bfill")
                                    entry_idx0 = (
                                        int(ent_arr[0]) if len(ent_arr) and ent_arr[0] >= 0 else -1
                                    )
                                    if entry_idx0 < 0:
                                        continue
                                except Exception:
                                    continue

                                # Strategy毎の entry/stop を近似（UIと同等の簡易版）
                                entry_price0 = None
                                stop_price0 = None
                                try:
                                    prev_close0 = float(
                                        dfp2.iloc[int(max(0, entry_idx0 - 1))]["Close"]
                                    )
                                    if system0 == "system1":
                                        stg0 = System1Strategy()
                                        entry_price0 = float(dfp2.iloc[int(entry_idx0)]["Open"])
                                        atr20 = float(
                                            dfp2.iloc[int(max(0, entry_idx0 - 1))]["ATR20"]
                                        )
                                        stop_mult0 = float(
                                            stg0.config.get("stop_atr_multiple", 5.0)
                                        )
                                        stop_price0 = entry_price0 - stop_mult0 * atr20
                                    elif system0 == "system2":
                                        stg0 = System2Strategy()
                                        entry_price0 = float(dfp2.iloc[int(entry_idx0)]["Open"])
                                        atr = float(dfp2.iloc[int(max(0, entry_idx0 - 1))]["ATR10"])
                                        stop_mult0 = float(
                                            stg0.config.get("stop_atr_multiple", 3.0)
                                        )
                                        stop_price0 = entry_price0 + stop_mult0 * atr
                                    elif system0 == "system6":
                                        # fixed_mode=True を強制し再計算を避ける
                                        stg0 = System6Strategy(fixed_mode=True)
                                        ratio0 = float(
                                            stg0.config.get("entry_price_ratio_vs_prev_close", 1.05)
                                        )
                                        entry_price0 = round(prev_close0 * ratio0, 2)
                                        atr = float(dfp2.iloc[int(max(0, entry_idx0 - 1))]["ATR10"])
                                        stop_mult0 = float(
                                            stg0.config.get("stop_atr_multiple", 3.0)
                                        )
                                        stop_price0 = entry_price0 + stop_mult0 * atr
                                    elif system0 == "system3":
                                        stg0 = System3Strategy()
                                        ratio0 = float(
                                            stg0.config.get("entry_price_ratio_vs_prev_close", 0.93)
                                        )
                                        entry_price0 = round(prev_close0 * ratio0, 2)
                                        atr = float(dfp2.iloc[int(max(0, entry_idx0 - 1))]["ATR10"])
                                        stop_mult0 = float(
                                            stg0.config.get("stop_atr_multiple", 2.5)
                                        )
                                        stop_price0 = entry_price0 - stop_mult0 * atr
                                    elif system0 == "system4":
                                        stg0 = System4Strategy()
                                        entry_price0 = float(dfp2.iloc[int(entry_idx0)]["Open"])
                                        atr40 = float(
                                            dfp2.iloc[int(max(0, entry_idx0 - 1))]["ATR40"]
                                        )
                                        stop_mult0 = float(
                                            stg0.config.get("stop_atr_multiple", 1.5)
                                        )
                                        stop_price0 = entry_price0 - stop_mult0 * atr40
                                    elif system0 == "system5":
                                        stg0 = System5Strategy()
                                        ratio0 = float(
                                            stg0.config.get("entry_price_ratio_vs_prev_close", 0.97)
                                        )
                                        entry_price0 = round(prev_close0 * ratio0, 2)
                                        atr = float(dfp2.iloc[int(max(0, entry_idx0 - 1))]["ATR10"])
                                        stop_mult0 = float(
                                            stg0.config.get("stop_atr_multiple", 3.0)
                                        )
                                        stop_price0 = entry_price0 - stop_mult0 * atr
                                        try:
                                            stg0._last_entry_atr = atr  # type: ignore[attr-defined]
                                        except Exception:
                                            pass
                                    else:
                                        continue
                                except Exception:
                                    continue
                                if entry_price0 is None or stop_price0 is None:
                                    continue
                                try:
                                    exit_price0, exit_date0 = stg0.compute_exit(
                                        dfp2,
                                        int(entry_idx0),
                                        float(entry_price0),
                                        float(stop_price0),
                                    )
                                except Exception:
                                    continue
                                today_norm0 = pd.to_datetime([dfp2.index[-1]])[0].normalize()
                                if latest_trading_day is not None:
                                    today_norm0 = latest_trading_day
                                is_today_exit0 = (
                                    pd.to_datetime([exit_date0])[0].normalize() == today_norm0
                                )
                                if is_today_exit0:
                                    if system0 == "system5":
                                        # System5 は翌日寄り決済のためカウント対象外
                                        pass
                                    else:
                                        counts[system0] = counts.get(system0, 0) + 1
                            except Exception:
                                continue
                    except Exception:
                        return {}
                    return counts

                exit_counts_map = (
                    _estimate_exit_counts_today(
                        positions_cache or [], symbol_system_map_cache or {}
                    )
                    or {}
                )
                # UI へも Exit 件数を送る（早期に可視化）
                try:
                    cb_exit = globals().get("_PER_SYSTEM_EXIT")
                except Exception:
                    cb_exit = None
                if cb_exit and callable(cb_exit):
                    try:
                        for _nm, _cnt in (exit_counts_map or {}).items():
                            try:
                                GLOBAL_STAGE_METRICS.record_exit(_nm, _cnt, emit_event=False)
                            except Exception:
                                pass
                        for _nm, _cnt in (exit_counts_map or {}).items():
                            try:
                                cb_exit(_nm, int(_cnt))
                            except Exception:
                                pass
                    except Exception:
                        pass
                # エグジット件数を UI ログへも要約表示
                try:
                    exit_counts_norm = {
                        str(k).strip().lower(): int(v)
                        for k, v in (exit_counts_map or {}).items()
                        if k is not None
                    }
                except Exception:
                    exit_counts_norm = {}
                exit_logged = False
                for _sys_name in order_1_7:
                    try:
                        cnt_val = int(exit_counts_norm.get(_sys_name, 0))
                    except Exception:
                        cnt_val = 0
                    if cnt_val:
                        try:
                            GLOBAL_STAGE_METRICS.record_exit(_sys_name, cnt_val, emit_event=False)
                        except Exception:
                            pass
                    if cnt_val > 0:
                        _log(f"🚪 {_sys_name}: 本日エグジット予定 {cnt_val} 件")
                        exit_logged = True
                if not exit_logged:
                    _log("🚪 本日エグジット予定はありません")
                # 既に集計済みの値を再構成
                setup_map = {
                    "system1": int(
                        (s1_setup_eff if s1_setup_eff is not None else (s1_setup or 0)) or 0
                    ),
                    "system2": int(s2_setup or 0),
                    "system3": int(s3_setup or 0),
                    "system4": int(locals().get("s4_close") or 0),
                    "system5": int(s5_setup or 0),
                    "system6": int(s6_setup or 0),
                    "system7": 1 if ("SPY" in (locals().get("basic_data", {}) or {})) else 0,
                }
                if isinstance(s1_spy_gate, int) and s1_spy_gate == 0:
                    setup_map["system1"] = 0
                metrics_summary_context = {
                    "prefilter_map": dict(prefilter_map),
                    "exit_counts_map": dict(exit_counts_map),
                    "setup_map": dict(setup_map),
                    "tgt_base": int(tgt_base),
                }
                # UI が StageTracker を登録していれば、ユニバース総数を通知して表示を揃える
                try:
                    cb_stage_set = globals().get("_SET_STAGE_UNIVERSE_TARGET")
                except Exception:
                    cb_stage_set = None
                if cb_stage_set and callable(cb_stage_set):
                    try:
                        cb_stage_set(int(tgt_base))
                    except Exception:
                        pass
                try:
                    GLOBAL_STAGE_METRICS.set_universe_target(int(tgt_base))
                except Exception:
                    pass
            except Exception:
                pass
        # 簡易ログ
        try:
            summary = ", ".join(
                [
                    (f"{r['system']}: 対象→{r['prefilter_pass']}, trade候補数→{r['candidates']}")
                    for r in metrics_rows
                ]
            )
            if summary:
                _log(f"📊 メトリクス概要: {summary}")
        except Exception:
            pass
    except Exception:
        _log("⚠️ メトリクス集計で例外が発生しました（処理続行）")

    if positions_cache is None or symbol_system_map_cache is None:
        positions_cache, symbol_system_map_cache = _fetch_positions_and_symbol_map()

    # 1) 枠配分（スロット）モード or 2) 金額配分モード
    try:
        settings_alloc_long = getattr(settings.ui, "long_allocations", {}) or {}
        settings_alloc_short = getattr(settings.ui, "short_allocations", {}) or {}
    except Exception:
        settings_alloc_long, settings_alloc_short = {}, {}

    try:
        max_positions_default = int(getattr(settings.risk, "max_positions", 10))
    except Exception:
        max_positions_default = 10

    slots_long_total = slots_long if slots_long is not None else max_positions_default
    slots_short_total = slots_short if slots_short is not None else max_positions_default

    try:
        default_capital = float(getattr(settings.ui, "default_capital", 100000))
    except Exception:
        default_capital = 100000.0
    try:
        default_long_ratio = float(getattr(settings.ui, "default_long_ratio", 0.5))
    except Exception:
        default_long_ratio = 0.5

    # Emit progress event for allocation start
    if ENABLE_PROGRESS_EVENTS:
        from common.progress_events import emit_progress_event

        emit_progress_event(
            "allocation_start",
            {"total_candidates": len(per_system), "target_positions": max_positions_default},
        )

    _log("🧷 候補の配分（スロット方式 or 金額配分）を実行")
    allocation_summary: AllocationSummary
    final_df, allocation_summary = finalize_allocation(
        per_system,
        strategies=strategies,
        positions=positions_cache,
        symbol_system_map=symbol_system_map_cache,
        long_allocations=settings_alloc_long,
        short_allocations=settings_alloc_short,
        slots_long=slots_long_total,
        slots_short=slots_short_total,
        capital_long=capital_long,
        capital_short=capital_short,
        default_capital=default_capital,
        default_long_ratio=default_long_ratio,
        default_max_positions=max_positions_default,
    )

    # Emit progress event for allocation completion
    if ENABLE_PROGRESS_EVENTS:
        emit_progress_event(
            "allocation_complete",
            {
                "final_positions": len(final_df) if final_df is not None else 0,
                "active_positions_total": sum(allocation_summary.active_positions.values()),
            },
        )

    active_positions_map = dict(allocation_summary.active_positions)
    if active_positions_map:
        try:
            summary_line = ", ".join(
                f"{name}={int(count)}"
                for name, count in sorted(active_positions_map.items())
                if int(count) > 0
            )
            if summary_line:
                _log("📦 現在保有ポジション数: " + summary_line)
        except Exception:
            pass

    available_slots_map = dict(allocation_summary.available_slots)
    try:
        lines = []
        for name in sorted(available_slots_map.keys()):
            remain = int(available_slots_map.get(name, 0))
            limit = remain + int(active_positions_map.get(name, 0))
            if limit > 0 and remain < limit:
                lines.append(f"{name}={remain}/{limit}")
        if lines:
            _log("🪧 利用可能スロット (残/上限): " + ", ".join(lines))
    except Exception:
        pass

    long_alloc_norm = dict(allocation_summary.long_allocations)
    short_alloc_norm = dict(allocation_summary.short_allocations)
    slot_candidates = allocation_summary.slot_candidates or {}

    if allocation_summary.mode == "slot":

        def _fmt_slot(name: str) -> str:
            cand = int(slot_candidates.get(name, 0))
            avail = min(cand, int(available_slots_map.get(name, 0)))
            return f"{name}={avail}" if avail == cand else f"{name}={avail}/{cand}"

        long_msg = ", ".join(_fmt_slot(name) for name in long_alloc_norm)
        short_msg = ", ".join(_fmt_slot(name) for name in short_alloc_norm)
        _log(
            "🧮 枠配分（利用可能スロット/候補数）: "
            + (long_msg if long_msg else "-")
            + " | "
            + (short_msg if short_msg else "-")
        )
    else:
        cap_long = float(allocation_summary.capital_long or 0.0)
        cap_short = float(allocation_summary.capital_short or 0.0)
        _log(f"💰 金額配分: long=${cap_long:,.0f}, short=${cap_short:,.0f}")
        try:
            budgets = allocation_summary.budgets or {}
            long_lines = [f"{name}=${budgets.get(name, 0.0):,.0f}" for name in long_alloc_norm]
            short_lines = [f"{name}=${budgets.get(name, 0.0):,.0f}" for name in short_alloc_norm]
            if long_lines:
                _log("📊 long予算内訳: " + ", ".join(long_lines))
            if short_lines:
                _log("📊 short予算内訳: " + ", ".join(short_lines))
        except Exception:
            pass

    if not final_df.empty:
        # 並びは side → system番号 → 各systemのスコア方向（RSI系のみ昇順、それ以外は降順）
        tmp = final_df.copy()
        if "system" in tmp.columns:
            try:
                tmp["_system_no"] = (
                    tmp["system"].astype(str).str.extract(r"(\d+)").fillna(0).astype(int)
                )
            except Exception:
                tmp["_system_no"] = 0
        # 一旦 side, system 番号で安定ソート
        tmp = tmp.sort_values(
            [c for c in ["side", "_system_no"] if c in tmp.columns], kind="stable"
        )
        # system ごとに score を方向指定で並べ替え
        try:
            parts2: list[pd.DataFrame] = []
            for sys_name, g in tmp.groupby("system", sort=False):
                if "score" in g.columns:
                    asc = False
                    try:
                        # system4（RSI系）はスコア小さいほど良い
                        if isinstance(sys_name, str) and sys_name.lower() == "system4":
                            asc = True
                    except Exception:
                        asc = False
                    g = g.sort_values("score", ascending=asc, na_position="last", kind="stable")
                parts2.append(g)
            concat_parts2 = _prepare_concat_frames(parts2)
            if concat_parts2:
                tmp = pd.concat(concat_parts2, ignore_index=True)
        except Exception:
            pass
        tmp = tmp.drop(columns=["_system_no"], errors="ignore")
        final_df = tmp.reset_index(drop=True)
        # 先頭に連番（1始まり）を付与
        try:
            final_df.insert(0, "no", range(1, len(final_df) + 1))
        except Exception:
            pass
        # system別の件数/金額サマリを出力
        try:
            if "position_value" in final_df.columns:
                grp = (
                    final_df.groupby("system")["position_value"].agg(["count", "sum"]).reset_index()
                )
                counts_map = {
                    str(r["system"]).strip().lower(): int(r["count"])
                    for _, r in grp.iterrows()
                    if str(r["system"]).strip()
                }
                values_map = {
                    str(r["system"]).strip().lower(): float(r["sum"])
                    for _, r in grp.iterrows()
                    if str(r["system"]).strip()
                }
                summary_lines = format_group_counts_and_values(counts_map, values_map)
                if summary_lines:
                    _log("🧾 Long/Shortサマリ: " + ", ".join(summary_lines))
            else:
                grp = final_df.groupby("system").size().to_dict()
                counts_map = {
                    str(key).strip().lower(): int(value)
                    for key, value in grp.items()
                    if str(key).strip()
                }
                summary_lines = format_group_counts(counts_map)
                if summary_lines:
                    _log("🧾 Long/Shortサマリ: " + ", ".join(summary_lines))
            # system ごとの最終エントリー数を出力
            try:
                if isinstance(grp, dict):
                    for k, v in grp.items():
                        _log(f"✅ {k}: {int(v)} 件")
                else:
                    for _, r in grp.iterrows():
                        _log(f"✅ {r['system']}: {int(r['count'])} 件")
            except Exception:
                pass
            # 追加: エントリー銘柄の system ごとのまとめ
            try:
                lines = []
                for sys_name, g in final_df.groupby("system"):
                    syms = ", ".join(list(g["symbol"].astype(str))[:20])
                    lines.append(f"{sys_name}: {syms}")
                if lines:
                    _log("🧾 エントリー内訳:\n" + "\n".join(lines))
            except Exception:
                pass
        except Exception:
            pass
        _log(f"📊 最終候補件数: {len(final_df)}")
    else:
        _log("📭 最終候補は0件でした")
    if progress_callback:
        try:
            progress_callback(7, 8, "finalize")
        except Exception:
            pass

    _save_and_notify_phase(
        ctx,
        final_df=final_df,
        per_system=per_system,
        order_1_7=order_1_7,
        metrics_summary_context=metrics_summary_context,
    )

    # clear callback
    try:
        globals().pop("_LOG_CALLBACK", None)
    except Exception:
        pass

    ctx.final_signals = final_df
    return final_df, per_system


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
            "CSVファイル名の形式: date=YYYY-MM-DD / "
            "datetime=YYYY-MM-DD_HHMM / runid=YYYY-MM-DD_RUNID"
        ),
    )
    # 計画 -> 実行ブリッジ（安全のため既定はドライラン）
    parser.add_argument(
        "--run-planned-exits",
        choices=["off", "open", "close", "auto"],
        default=None,
        help=(
            "手仕舞い計画の自動実行: off=無効 / open=寄り(OPG) / "
            "close=引け(CLS) / auto=時間帯で自動判定"
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
    return parser


def parse_cli_args() -> argparse.Namespace:
    parser = build_cli_parser()
    return parser.parse_args()


def configure_logging_for_cli(args: argparse.Namespace) -> None:
    env_mode = os.environ.get("TODAY_SIGNALS_LOG_MODE", "").strip().lower()
    mode = args.log_file_mode or (env_mode if env_mode in {"single", "dated"} else None) or "dated"
    _configure_today_logger(mode=mode)
    try:
        sel_path = globals().get("_LOG_FILE_PATH")
        _log(f"📝 ログ保存先: {sel_path}", ui=False)
    except Exception:
        pass


def run_signal_pipeline(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    return compute_today_signals(
        args.symbols,
        slots_long=args.slots_long,
        slots_short=args.slots_short,
        capital_long=args.capital_long,
        capital_short=args.capital_short,
        save_csv=args.save_csv,
        csv_name_mode=args.csv_name_mode,
        parallel=args.parallel,
    )


def log_final_candidates(final_df: pd.DataFrame) -> list[Signal]:
    if final_df.empty:
        _log("📭 本日の最終候補はありません。")
        return []

    _log("\n=== 最終候補（推奨） ===")
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
    _log(final_df[show].to_string(index=False))
    signals_for_merge = [
        Signal(
            system_id=int(str(r.get("system")).replace("system", "") or 0),
            symbol=str(r.get("symbol")),
            side="BUY" if str(r.get("side")).lower() == "long" else "SELL",
            strength=float(r.get("score", 0.0)),
            meta={},
        )
        for _, r in final_df.iterrows()
    ]
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
    try:
        from schedulers.next_day_exits import submit_planned_exits as _run_planned
    except Exception:
        _run_planned = None
    env_run = os.environ.get("RUN_PLANNED_EXITS", "").lower()
    run_mode = (
        args.run_planned_exits
        or (env_run if env_run in {"off", "open", "close", "auto"} else None)
        or "off"
    )
    dry_run = True if args.planned_exits_dry_run else False
    if _run_planned and run_mode != "off":
        sel = run_mode
        if run_mode == "auto":
            now = datetime.now(ZoneInfo("America/New_York"))
            hhmm = now.strftime("%H%M")
            sel = (
                "open"
                if ("0930" <= hhmm <= "0945")
                else ("close" if ("1550" <= hhmm <= "1600") else "off")
            )
        if sel in {"open", "close"}:
            _log(f"⏱️ 手仕舞い計画の自動実行: {sel} (dry_run={dry_run})")
            try:
                df_exec = _run_planned(sel, dry_run=dry_run)
                if df_exec is not None and not df_exec.empty:
                    _log(df_exec.to_string(index=False), ui=False)
                else:
                    _log("対象の手仕舞い計画はありません。", ui=False)
            except Exception as e:
                _log(f"⚠️ 手仕舞い計画の自動実行に失敗: {e}")


def main():
    args = parse_cli_args()
    configure_logging_for_cli(args)
    final_df, _per_system = run_signal_pipeline(args)
    signals_for_merge = log_final_candidates(final_df)
    merge_signals_for_cli(signals_for_merge)
    maybe_submit_orders(final_df, args)
    maybe_run_planned_exits(args)


if __name__ == "__main__":
    main()
