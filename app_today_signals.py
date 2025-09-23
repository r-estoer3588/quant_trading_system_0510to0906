from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, TYPE_CHECKING

import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from common import broker_alpaca as ba
from common import universe as univ
from common.alpaca_order import submit_orders_df
from common.cache_manager import round_dataframe  # type: ignore
from common.data_loader import load_price
from common.exit_planner import decide_exit_schedule
from common.notifier import create_notifier
from common.position_age import fetch_entry_dates_from_alpaca, load_entry_dates, save_entry_dates
from common.profit_protection import evaluate_positions
from common.stage_metrics import (
    DEFAULT_SYSTEM_ORDER,
    GLOBAL_STAGE_METRICS,
    StageMetricsStore,
    StageSnapshot,
)
from common.symbol_universe import build_symbol_universe_from_settings
from common.system_groups import format_group_counts, format_group_counts_and_values
from common.today_signals import LONG_SYSTEMS, SHORT_SYSTEMS
from common.today_signals import run_all_systems_today as compute_today_signals
from common.utils_spy import get_latest_nyse_trading_day
from config.settings import get_settings
import scripts.run_all_systems_today as _run_today_mod

if TYPE_CHECKING:  # pragma: no cover - static typing only
    try:  # type: ignore - optional import for type checkers
        import alpaca.trading.requests as _alpaca_trading_requests  # type: ignore
    except Exception:  # pragma: no cover - runtime fallback
        _alpaca_trading_requests = Any  # type: ignore


def _import_alpaca_requests():
    """Runtime-safe importer for `alpaca.trading.requests`.

    Returns the module or None if not importable.
    """
    try:
        import importlib

        return importlib.import_module("alpaca.trading.requests")
    except Exception:
        return None


def _running_in_streamlit() -> bool:
    try:
        if get_script_run_ctx(suppress_warning=True) is not None:
            return True
    except Exception:
        pass
    try:
        flag = (os.environ.get("STREAMLIT_SERVER_ENABLED") or "").strip().lower()
        if flag in {"1", "true", "yes"}:
            return True
    except Exception:
        pass
    try:
        argv_text = " ".join(sys.argv).lower()
        if "streamlit" in argv_text:
            return True
    except Exception:
        pass
    return False


_IS_STREAMLIT_RUNTIME = _running_in_streamlit()

if not _IS_STREAMLIT_RUNTIME:
    if __name__ == "__main__":
        print("このスクリプトはStreamlitで実行してください: `streamlit run app_today_signals.py`")
    raise SystemExit

try:
    # Streamlit の実行コンテキスト有無を判定（スレッド外からの UI 呼び出しを防ぐ）
    from streamlit.runtime.scriptrunner import get_script_run_ctx as _st_get_ctx  # type: ignore

    def _has_st_ctx() -> bool:
        if not _IS_STREAMLIT_RUNTIME:
            return False
        try:
            return _st_get_ctx() is not None
        except Exception:
            return False

except Exception:

    def _has_st_ctx() -> bool:  # type: ignore
        return _IS_STREAMLIT_RUNTIME


# Streamlit checkbox の重複ID対策（key未指定時に自動で一意キーを付与）
try:
    if not hasattr(st, "_orig_checkbox"):
        st._orig_checkbox = st.checkbox  # type: ignore[attr-defined]

        def _unique_checkbox(label, *args, **kwargs):
            if "key" not in kwargs:
                base = f"auto_cb_{abs(hash(str(label))) % 10**8}"
                count_key = f"_{base}_cnt"
                try:
                    cnt = int(st.session_state.get(count_key, 0)) + 1
                except Exception:
                    cnt = 1
                st.session_state[count_key] = cnt
                kwargs["key"] = f"{base}_{cnt}"
            return st._orig_checkbox(  # type: ignore[attr-defined]
                label,
                *args,
                **kwargs,
            )

        st.checkbox = _unique_checkbox  # type: ignore[attr-defined]
except Exception:
    # 失敗しても従来動作のまま進める
    pass

st.set_page_config(page_title="本日のシグナル", layout="wide")
st.title("📈 本日のシグナル（全システム）")

settings = get_settings(create_dirs=True)
notifier = create_notifier(platform="slack", fallback=True)
# この実行ループで結果を表示したかのフラグ（保存ボタン等でのリラン対策）
st.session_state.setdefault("today_shown_this_run", False)


def _reset_shown_flag() -> None:
    """リラン後の前回結果再表示を有効にするフラグをリセットする。"""
    st.session_state["today_shown_this_run"] = False


def _build_position_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """side×system 別の保有件数サマリーを作成する。"""

    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    allowed_systems = {
        *(s.lower() for s in LONG_SYSTEMS),
        *(s.lower() for s in SHORT_SYSTEMS),
    }

    def _norm_side(value: Any) -> str | None:
        if isinstance(value, str):
            side = value.strip().lower()
            if side in {"long", "short"}:
                return side
        return None

    def _norm_system(value: Any) -> str | None:
        if isinstance(value, str):
            system = value.strip().lower()
            if system in allowed_systems:
                return system
        return None

    work["side_norm"] = work["side"].map(_norm_side)
    work["system_norm"] = work["system"].map(_norm_system)

    invalid_side_mask = work["side_norm"].isna()
    if invalid_side_mask.any():
        invalid_values = sorted(
            {str(v) for v in work.loc[invalid_side_mask, "side"].tolist()}
        )  # noqa: E501
        raise ValueError(f"未対応のsideが含まれています: {invalid_values}")

    invalid_system_mask = work["system_norm"].isna()
    if invalid_system_mask.any():
        invalid_values = sorted(
            {str(v) for v in work.loc[invalid_system_mask, "system"].tolist()}
        )  # noqa: E501
        raise ValueError(f"未対応のsystemが含まれています: {invalid_values}")

    long_conflict_mask = (work["side_norm"] == "long") & (
        ~work["system_norm"].isin(LONG_SYSTEMS)
    )  # noqa: E501
    if long_conflict_mask.any():
        conflict = sorted({str(v) for v in work.loc[long_conflict_mask, "system"].tolist()})
        raise ValueError(f"Longサイドに想定外のsystemが含まれています: {conflict}")

    short_conflict_mask = (work["side_norm"] == "short") & (
        ~work["system_norm"].isin(SHORT_SYSTEMS)
    )
    if short_conflict_mask.any():
        conflict = sorted({str(v) for v in work.loc[short_conflict_mask, "system"].tolist()})
        raise ValueError(f"Shortサイドに想定外のsystemが含まれています: {conflict}")

    def _sorted_systems(systems: set[str]) -> list[str]:
        def _key(name: str) -> tuple[int, int | str]:
            base = name.strip().lower()
            if base.startswith("system"):
                suffix = base[6:]
                if suffix.isdigit():
                    return (0, int(suffix))
            return (1, base)

        return sorted({s.strip().lower() for s in systems if s}, key=_key)

    long_order = _sorted_systems(LONG_SYSTEMS)
    short_order = _sorted_systems(SHORT_SYSTEMS)
    system_columns: list[str] = []
    for name in [*long_order, *short_order]:
        if name and name not in system_columns:
            system_columns.append(name)
    columns_all = [*system_columns, "合計"]

    def _format_system_label(name: str) -> str:
        base = name.strip().lower()
        if base.startswith("system"):
            suffix = base[6:]
            if suffix.isdigit():
                return f"System{int(suffix)}"
        return name

    def _build_row(side_key: str, allowed: list[str]) -> dict[str, int]:
        subset = work[work["side_norm"] == side_key]
        counts = subset["system_norm"].value_counts()
        row = {col: 0 for col in columns_all}
        for system_name in allowed:
            row[system_name] = int(counts.get(system_name, 0))
        row["合計"] = int(counts.sum())
        return row

    summary_rows: list[dict[str, int]] = []
    index_labels: list[str] = []

    summary_rows.append(_build_row("long", long_order))
    index_labels.append("Long")
    summary_rows.append(_build_row("short", short_order))
    index_labels.append("Short")

    summary = pd.DataFrame(summary_rows, index=index_labels)
    summary = summary.reindex(columns=columns_all, fill_value=0)

    rename_map = {name: _format_system_label(name) for name in system_columns}
    rename_map["合計"] = "合計"
    summary = summary.rename(columns=rename_map)

    summary.index.name = "side"
    summary.columns.name = None

    return summary.astype(int)


def _normalize_price_history(df: pd.DataFrame, rows: int) -> pd.DataFrame | None:
    """ロード済み株価データをUI用に正規化する。"""

    try:
        work = df.copy()
    except Exception:
        return None

    try:
        work.columns = [str(col) for col in work.columns]
    except Exception:
        work = pd.DataFrame(work)
        work.columns = [str(col) for col in work.columns]

    lower_map = {col.lower(): col for col in work.columns}

    # 日付列を決定（存在しない場合は index から生成）
    date_col = lower_map.get("date")
    if date_col is not None:
        work["date"] = pd.to_datetime(work[date_col], errors="coerce")
    else:
        try:
            idx = pd.to_datetime(work.index, errors="coerce")
            work = work.assign(date=idx)
        except Exception:
            return None

    work = work.dropna(subset=["date"]).sort_values("date")

    rename_src = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "adj close": "adjusted_close",
        "adjclose": "adjusted_close",
    }
    for key, target in rename_src.items():
        col = lower_map.get(key)
        if col is not None:
            work.rename(columns={col: target}, inplace=True)

    try:
        work.columns = [str(col).lower() for col in work.columns]
    except Exception:
        work.columns = [str(col) for col in work.columns]

    if "date" not in work.columns or "close" not in work.columns:
        return None

    # `date` を先頭に維持しつつ既知カラムを優先表示
    known_order = ["date", "open", "high", "low", "close", "volume", "adjusted_close"]
    ordered: list[str] = []
    for col in known_order:
        if col in work.columns:
            ordered.append(col)
    if hasattr(work, "columns") and isinstance(work.columns, pd.Index):
        for col in list(work.columns):
            if col not in ordered:
                ordered.append(col)
        work = work.loc[:, ordered]
    else:
        # work.columns が存在しない場合や反復できない場合は空DataFrameを返す
        return pd.DataFrame()

    if rows > 0:
        work = work.tail(rows)

    return work.reset_index(drop=True)


_ROLLING_REQUIRED_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "sma25",
    "sma50",
    "sma100",
    "sma150",
    "sma200",
    "atr20",
    "roc200",
]

_ROLLING_IMPORTANT_COLUMNS = [
    "ema20",
    "ema50",
    "atr10",
    "atr14",
    "atr40",
    "atr50",
    "adx7",
    "rsi3",
    "rsi14",
    "hv50",
    "return6d",
    "drop3d",
]

_ROLLING_NAN_THRESHOLD = 0.20
_ROLLING_RECENT_WINDOW = 120
_ROLLING_RECENT_STRICT_WINDOW = 30
_ROLLING_RECENT_STRICT_THRESHOLD = 0.0

# Per-column lookback (rows required before values become available).
# When computing NaN ratios we exclude the initial warm-up rows for indicators
# that naturally produce NaN for the first `lookback-1` rows (e.g. ROC200,
# SMA100). Keys are lower-cased to match `col_map` usage.
_ROLLING_COLUMN_LOOKBACK: dict[str, int] = {
    # price / basic
    "date": 0,
    "open": 0,
    "high": 0,
    "low": 0,
    "close": 0,
    "volume": 0,
    # SMAs
    "sma25": 25,
    "sma50": 50,
    "sma100": 100,
    "sma150": 150,
    "sma200": 200,
    # ATR / ROC
    "atr20": 20,
    "roc200": 200,
    # common optional indicators
    "ema20": 20,
    "ema50": 50,
    "atr10": 10,
    "atr14": 14,
    "atr40": 40,
    "atr50": 50,
    "adx7": 7,
    "rsi3": 3,
    "rsi14": 14,
    "hv50": 50,
    "return6d": 6,
    "drop3d": 3,
}

# When True, emit a single info log summarizing how many indicator columns
# were skipped because their series length was shorter than the configured
# lookback (useful for debugging false-positive NaN warnings).
_ROLLING_DEBUG_LOG_SKIPPED = False


def _has_recent_valid_window(numeric: pd.Series) -> bool:
    """Return True if recent rows provide enough non-NaN coverage."""

    if numeric.empty:
        return False

    recent_len = int(min(len(numeric), _ROLLING_RECENT_WINDOW))
    if recent_len <= 0:
        return False
    recent = numeric.iloc[-recent_len:]
    try:
        recent_ratio = float(recent.isna().mean())
    except Exception:
        recent_ratio = 1.0
    if recent_ratio <= _ROLLING_NAN_THRESHOLD:
        return True

    strict_len = int(min(len(numeric), _ROLLING_RECENT_STRICT_WINDOW))
    if strict_len <= 0:
        return False
    strict_recent = recent.iloc[-strict_len:]
    try:
        strict_ratio = float(strict_recent.isna().mean())
    except Exception:
        strict_ratio = 1.0
    return strict_ratio <= _ROLLING_RECENT_STRICT_THRESHOLD


def _analyze_rolling_cache(df: pd.DataFrame | None) -> tuple[bool, dict[str, Any]]:
    if df is None or df.empty:
        return False, {"status": "rolling_missing"}
    try:
        columns = list(df.columns)
    except Exception:
        columns = []
    col_map = {str(col).lower(): col for col in columns}
    missing_required = [col for col in _ROLLING_REQUIRED_COLUMNS if col not in col_map]
    missing_optional = [col for col in _ROLLING_IMPORTANT_COLUMNS if col not in col_map]
    nan_required: list[tuple[str, float]] = []
    nan_optional: list[tuple[str, float]] = []
    skipped_lookback_count = 0
    for name in {*_ROLLING_REQUIRED_COLUMNS, *_ROLLING_IMPORTANT_COLUMNS}:
        actual = col_map.get(name)
        if actual is None:
            continue
        try:
            numeric = pd.to_numeric(df[actual], errors="coerce")
        except Exception:
            continue
        # Exclude initial warm-up rows for indicators that naturally produce NaNs
        # by using a per-column lookback. If the series is shorter than lookback,
        # the indicator cannot be computed yet — treat it as "not applicable"
        # for NaN-warning purposes (do not flag as NaN過多).
        lookback = _ROLLING_COLUMN_LOOKBACK.get(name, 0)
        try:
            # If a lookback is defined but the series is too short, skip this
            # column entirely (it's not a problem — the indicator simply
            # couldn't have been computed yet).
            if lookback and len(numeric) <= lookback:
                skipped_lookback_count += 1
                # mark as not-applicable by continuing to next column
                continue
            if lookback and len(numeric) > lookback:
                # exclude the first (lookback - 1) rows from the recent window
                # so only rows where the indicator could exist are counted.
                effective_start = max(
                    0,
                    len(numeric) - _ROLLING_RECENT_WINDOW,
                    lookback - 1 - (len(numeric) - _ROLLING_RECENT_WINDOW),
                )
                eval_series = numeric.iloc[effective_start:]
                # If eval_series is empty fallback to full-series ratio
                if len(eval_series) > 0:
                    ratio = float(eval_series.isna().mean())
                else:
                    ratio = float(numeric.isna().mean())
            else:
                ratio = float(numeric.isna().mean())
        except Exception:
            ratio = 1.0
        if ratio > _ROLLING_NAN_THRESHOLD:
            if name in _ROLLING_REQUIRED_COLUMNS:
                nan_required.append((name, ratio))
            else:
                nan_optional.append((name, ratio))
    issues: dict[str, Any] = {}
    fatal = False
    if missing_required:
        issues["missing_required"] = missing_required
        fatal = True
    if nan_required or nan_optional:
        issues["nan_columns"] = [*nan_required, *nan_optional]
    if nan_required:
        fatal = True
    if missing_optional:
        issues["missing_optional"] = missing_optional
    if fatal:
        issues.setdefault(
            "status",
            "missing_required" if missing_required else "nan_columns",
        )
        return False, issues
    if missing_optional:
        issues.setdefault("status", "missing_optional")
        return True, issues
    if nan_optional:
        issues.setdefault("status", "nan_optional")
        return True, issues
    # Optionally log a debug summary about skipped lookback-short columns
    if _ROLLING_DEBUG_LOG_SKIPPED and skipped_lookback_count:
        try:
            import logging

            logger = logging.getLogger("today_signals")
            logger.info(f"lookback未満でスキップされた列は{skipped_lookback_count}件でした")
        except Exception:
            pass

    return True, {}


def _format_nan_columns(values: list[tuple[str, float]]) -> str:
    if not values:
        return ""
    return ", ".join(f"{name}:{ratio:.1%}" for name, ratio in values)


def _issues_to_note(issues: dict[str, Any]) -> str:
    if not issues:
        return ""
    parts: list[str] = []
    missing_required = issues.get("missing_required") or []
    if missing_required:
        parts.append("required=" + ", ".join(str(x) for x in missing_required))
    missing_optional = issues.get("missing_optional") or []
    if missing_optional:
        parts.append("optional=" + ", ".join(str(x) for x in missing_optional))
    nan_columns = issues.get("nan_columns") or []
    if nan_columns:
        parts.append("nan=" + _format_nan_columns(list(nan_columns)))
    return "; ".join(parts)


def _merge_note(base: str, addition: str) -> str:
    parts = [part for part in [base, addition] if part]
    return " / ".join(parts)


def _build_missing_detail(
    symbol: str,
    issues: dict[str, Any],
    rows_before: int,
) -> dict[str, Any]:
    missing_required = issues.get("missing_required") or []
    missing_optional = issues.get("missing_optional") or []
    nan_columns = issues.get("nan_columns") or []
    return {
        "symbol": symbol,
        "status": issues.get("status", "missing"),
        "missing_required": ", ".join(str(x) for x in missing_required),
        "missing_optional": ", ".join(str(x) for x in missing_optional),
        "nan_columns": _format_nan_columns(list(nan_columns)),
        "rows_before": int(rows_before),
        "rows_after": 0,
        "action": "",
        "resolved": False,
        "note": "",
    }


def _log_manual_rebuild_notice(
    symbol: str,
    detail: dict[str, Any],
    log_fn: Callable[[str], None] | None = None,
) -> None:
    if log_fn is None:
        return
    status = str(detail.get("status") or "rolling_missing")
    reason_map = {
        "rolling_missing": "rolling未生成",
        "missing_required": "必須列不足",
        "missing_optional": "任意列不足",
        "nan_columns": "NaN過多",
    }
    reason_label = reason_map.get(status, status)
    parts: list[str] = []
    rows_before = detail.get("rows_before")
    try:
        rows_val = int(rows_before) if rows_before is not None else None
    except Exception:
        rows_val = None
    if rows_val:
        parts.append(f"rows={rows_val}")
    missing_required = str(detail.get("missing_required") or "").strip()
    if missing_required:
        parts.append(f"必須: {missing_required}")
    missing_optional = str(detail.get("missing_optional") or "").strip()
    if missing_optional:
        parts.append(f"任意: {missing_optional}")
    nan_columns = str(detail.get("nan_columns") or "").strip()
    if nan_columns:
        parts.append(f"NaN: {nan_columns}")
    message = f"⛔ rolling未整備: {symbol} ({reason_label})"
    if parts:
        message += " | " + ", ".join(parts)
    message += " → 手動で rolling キャッシュを更新してください"
    try:
        log_fn(message)
    except Exception:
        pass


def _collect_symbol_data(
    symbols: list[str],
    *,
    rows: int,
    log_fn: Callable[[str], None] | None = None,
    debug_scan: bool = False,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    """指定シンボルの株価履歴をまとめて取得し、欠損も記録する。"""

    start_ts = time.time()
    total = len(symbols)
    if total == 0:
        return {}, []

    step = max(1, total // 20)
    fetched: dict[str, pd.DataFrame] = {}
    malformed: list[str] = []
    missing_details: list[dict[str, Any]] = []
    for idx, sym in enumerate(symbols, start=1):
        try:
            df = load_price(sym, cache_profile="rolling")
        except Exception:
            df = None
        rows_before = 0 if df is None else int(len(df))
        ok, issues = _analyze_rolling_cache(df)
        detail: dict[str, Any] | None = None
        if not ok:
            detail = _build_missing_detail(sym, issues, rows_before)
            if debug_scan:
                detail["action"] = "debug_scan"
                detail["note"] = _issues_to_note(issues)
                missing_details.append(detail)
                df = None
            else:
                detail["action"] = "manual_rebuild_required"
                manual_note = _merge_note(
                    _issues_to_note(issues),
                    "手動で rolling キャッシュを更新してください",
                )
                detail["note"] = manual_note
                missing_details.append(detail)
                _log_manual_rebuild_notice(sym, detail, log_fn)
                df = None
            continue

        # df may be None if earlier checks failed; guard before normalization
        if df is None:
            malformed.append(sym)
            continue
        norm = _normalize_price_history(df, rows)
        if norm is not None and not norm.empty:
            fetched[sym] = norm
        else:
            malformed.append(sym)

        if log_fn and (idx % step == 0 or idx == total):
            try:
                log_fn(f"📦 基礎データロード進捗: {idx}/{total}")
            except Exception:
                pass

    if log_fn:
        try:
            elapsed = int(max(0, time.time() - start_ts))
            minutes, seconds = divmod(elapsed, 60)
            log_fn(f"📦 基礎データロード完了: {len(fetched)}/{total} | 所要 {minutes}分{seconds}秒")
        except Exception:
            pass
        manual_symbols = [
            detail["symbol"]
            for detail in missing_details
            if detail.get("action") == "manual_rebuild_required"
        ]
        if manual_symbols:
            sample = ", ".join(manual_symbols[:5])
            if len(manual_symbols) > 5:
                sample += f" ほか{len(manual_symbols) - 5}件"
            try:
                log_fn(
                    "⚠️ rolling未整備: " + sample + " → 手動で rolling キャッシュを更新してください"
                )
            except Exception:
                pass
        if malformed:
            sample = ", ".join(malformed[:5])
            if len(malformed) > 5:
                sample += f" ほか{len(malformed) - 5}件"
            try:
                log_fn(f"⚠️ データ整形不可: {sample}")
            except Exception:
                pass
        if debug_scan:
            try:
                if missing_details:
                    log_fn(f"🧪 欠損洗い出し検出: {len(missing_details)}件")
                else:
                    log_fn("🧪 欠損洗い出し: 問題は検出されませんでした")
            except Exception:
                pass

    return fetched, missing_details


def _get_today_logger() -> logging.Logger:
    """本日のシグナル実行用ロガー。

    - orchestrator(`scripts.run_all_systems_today`)が設定したログパスがあればそれに合わせる
    - 無い場合は `TODAY_SIGNALS_LOG_MODE`（single|dated）を参照
    - 既定は dated（JST: today_signals_YYYYMMDD_HHMM.log）
    """
    logger = logging.getLogger("today_signals")
    logger.setLevel(logging.INFO)
    try:
        logger.propagate = False
    except Exception:
        pass

    # ログディレクトリ
    try:
        log_dir = Path(settings.LOGS_DIR)
    except Exception:
        log_dir = Path("logs")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # orchestrator 側の設定を最優先
    log_path: Path | None = None
    try:
        sel = getattr(_run_today_mod, "_LOG_FILE_PATH", None)
        if isinstance(sel, Path):
            log_path = sel
    except Exception:
        log_path = None

    # 無ければ環境変数を見て決定
    if log_path is None:
        try:
            mode_env = (os.environ.get("TODAY_SIGNALS_LOG_MODE") or "").strip().lower()
        except Exception:
            mode_env = ""
        if mode_env == "single":
            log_path = log_dir / "today_signals.log"
        else:
            try:
                from datetime import datetime
                from zoneinfo import ZoneInfo

                jst_now = datetime.now(ZoneInfo("Asia/Tokyo"))
            except Exception:
                from datetime import datetime

                jst_now = datetime.now()
            stamp = jst_now.strftime("%Y%m%d_%H%M")
            log_path = log_dir / f"today_signals_{stamp}.log"

    # 既存のハンドラを整理（異なるファイルへのハンドラは除去）
    try:
        for h in list(logger.handlers):
            try:
                if isinstance(h, logging.FileHandler):
                    base = getattr(h, "baseFilename", None)
                    if base and Path(base) != log_path:
                        logger.removeHandler(h)
                        try:
                            h.close()
                        except Exception:
                            pass
            except Exception:
                pass
    except Exception:
        pass

    # 同一ファイル向けが未追加なら追加
    has_handler = False
    for h in list(logger.handlers):
        try:
            if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None):
                if Path(h.baseFilename) == log_path:
                    has_handler = True
                    break
        except Exception:
            continue
    if not has_handler:
        try:
            fh = logging.FileHandler(str(log_path), encoding="utf-8")
            fmt = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
            )  # noqa: E501
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except Exception:
            pass
    return logger


@dataclass
class RunConfig:
    symbols: list[str]
    capital_long: float
    capital_short: float
    save_csv: bool
    csv_name_mode: str
    notify: bool
    run_parallel: bool
    scan_missing_only: bool = False


@dataclass
class TradeOptions:
    paper_mode: bool
    retries: int
    delay: float
    poll_status: bool
    do_trade: bool
    update_bp_after: bool


class ProgressUI:
    """全体進捗とログ表示を管理するヘルパー。"""

    def __init__(self, ui_vis: dict[str, Any]):
        self.show_overall = bool(ui_vis.get("overall_progress", True))
        self.show_data_load = bool(ui_vis.get("data_load_progress_lines", True))
        self.phase_title_area = st.empty()
        self.progress_area = st.empty()
        self.progress_bar = st.progress(0) if self.show_overall else None
        self.progress_text = st.empty() if self.show_overall else None
        self.phase_state: dict[str, Any] = {"percent": 0, "label": "対象読み込み"}
        self._render_title()

    def set_label(self, label: str) -> None:
        if not self.show_overall:
            return
        self.phase_state["label"] = label
        self._render_title()

    def update(self, done: int, total: int, tag: str) -> None:
        if not self.show_overall or self.progress_bar is None:
            return
        total = max(1, int(total))
        ratio = min(max(int(done), 0), total) / total
        percent = int(ratio * 100)
        self.phase_state["percent"] = percent
        mapped = self._map_phase(tag)
        if mapped:
            self.phase_state["label"] = mapped
        try:
            self.progress_bar.progress(percent)
        except Exception:
            pass
        if self.progress_text is not None:
            try:
                self.progress_text.text(
                    f"進捗 {percent}%: {self.phase_state.get('label', '対象読み込み')}"
                )
            except Exception:
                pass
        self._render_title()

    def update_label_for_stage(self, stage_value: int) -> None:
        if not self.show_overall:
            return
        if stage_value <= 0:
            label = "対象準備"
        elif stage_value < 10:
            label = "対象読み込み"
        elif stage_value < 30:
            label = "フィルター"
        elif stage_value < 60:
            label = "セットアップ"
        elif stage_value < 90:
            label = "トレード候補選定"
        else:
            label = "エントリー"
        self.set_label(label)

    def _render_title(self) -> None:
        if not self.show_overall:
            return
        try:
            percent = int(self.phase_state.get("percent", 0))
            label = str(self.phase_state.get("label", "対象読み込み"))
            self.phase_title_area.markdown(f"## 進捗 {percent}%: {label}フェーズ")
        except Exception:
            pass

    @staticmethod
    def _map_phase(tag: str) -> str:
        try:
            t = (tag or "").lower()
        except Exception:
            t = ""
        if t in {
            "init",
            "対象読み込み:start",
            "load_basic:start",
            "load_basic",
            "load_indicators",
            "spx",
            "spy",
        }:
            return "対象読み込み"
        if t in {"filter", "フィルター"}:
            return "フィルター"
        if t in {"run_strategies", "setup"} or t.startswith("system"):
            return "セットアップ"
        if t in {"strategies_done", "trade候補", "トレード候補選定"}:
            return "トレード候補選定"
        if t in {"finalize", "done", "エントリー"}:
            return "エントリー"
        return "対象読み込み"


class StageTracker:
    """システム別の進捗と件数メトリクスを管理する。"""

    def __init__(self, ui_vis: dict[str, Any], progress_ui: ProgressUI):
        self.progress_ui = progress_ui
        self.show_ui = bool(ui_vis.get("per_system_progress", True)) and _has_st_ctx()
        self.bars: dict[str, Any] = {}
        self.stage_txt: dict[str, Any] = {}
        self.metrics_txt: dict[str, Any] = {}
        self.states: dict[str, int] = {}
        self.metrics_store = StageMetricsStore(DEFAULT_SYSTEM_ORDER)
        self.stage_counts = self.metrics_store.stage_counts
        self.universe_total: int | None = None
        self.universe_target: int | None = None
        if self.show_ui:
            sys_cols = st.columns(7)
            sys_labels = [f"System{i}" for i in range(1, 8)]
            for i, col in enumerate(sys_cols, start=1):
                key = f"system{i}"
                try:
                    col.caption(sys_labels[i - 1])
                    self.bars[key] = col.progress(0)
                    self.stage_txt[key] = col.empty()
                    self.metrics_txt[key] = col.empty()
                    self._render_metrics(key)
                except Exception:
                    self.show_ui = False
                    break
        self._initialize_from_store()

    def update_progress(self, name: str, phase: str) -> None:
        if not self.show_ui:
            return
        key = str(name).lower()
        bar = self.bars.get(key)
        if bar is None:
            return
        value = 50 if phase == "start" else 100 if phase == "done" else None
        if value is None:
            return
        self.states[key] = value
        try:
            bar.progress(value)
            self.stage_txt[key].text("run 50%" if value == 50 else "done 100%")
        except Exception:
            pass

    def _initialize_from_store(self) -> None:
        try:
            stored_target = GLOBAL_STAGE_METRICS.get_universe_target()
            if stored_target is not None:
                self.universe_target = int(stored_target)
        except Exception:
            pass
        try:
            snapshots = GLOBAL_STAGE_METRICS.all_snapshots()
        except Exception:
            snapshots = {}
        for sys_name, snapshot in snapshots.items():
            try:
                self._apply_snapshot(sys_name, snapshot)
            except Exception:
                continue

    def _apply_snapshot(self, name: str, snapshot: StageSnapshot) -> None:
        key = str(name).lower()
        counts = self._ensure_counts(key)
        if snapshot.target is not None:
            try:
                counts["target"] = int(snapshot.target)
                self.universe_total = int(snapshot.target)
            except Exception:
                pass
        if snapshot.filter_pass is not None:
            try:
                counts["filter"] = int(snapshot.filter_pass)
            except Exception:
                pass
        if snapshot.setup_pass is not None:
            try:
                counts["setup"] = int(snapshot.setup_pass)
            except Exception:
                pass
        if snapshot.candidate_count is not None:
            try:
                counts["cand"] = self._clamp_trdlist(snapshot.candidate_count)
            except Exception:
                pass
        if snapshot.entry_count is not None:
            try:
                counts["entry"] = int(snapshot.entry_count)
            except Exception:
                pass
        if snapshot.exit_count is not None:
            try:
                counts["exit"] = int(snapshot.exit_count)
            except Exception:
                pass
        self._update_bar(key, snapshot.progress)
        self.progress_ui.update_label_for_stage(snapshot.progress)
        self._render_metrics(key)

    def update_stage(
        self,
        name: str,
        value: int,
        filter_cnt: int | None = None,
        setup_cnt: int | None = None,
        cand_cnt: int | None = None,
        final_cnt: int | None = None,
    ) -> None:
        key = str(name).lower()
        snapshot: StageSnapshot | None
        try:
            snapshot = GLOBAL_STAGE_METRICS.record_stage(
                key,
                value,
                filter_cnt,
                setup_cnt,
                cand_cnt,
                final_cnt,
                emit_event=False,
            )
        except Exception:
            snapshot = None
        if snapshot is not None:
            self._apply_snapshot(key, snapshot)
            return
        counts = self._ensure_counts(key)
        if filter_cnt is not None:
            try:
                if value == 0:
                    counts["target"] = int(filter_cnt)
                    self.universe_total = int(filter_cnt)
                else:
                    counts["filter"] = int(filter_cnt)
            except Exception:
                counts["filter"] = counts.get("filter")
        if setup_cnt is not None:
            counts["setup"] = int(setup_cnt)
        if cand_cnt is not None:
            counts["cand"] = self._clamp_trdlist(cand_cnt)
        if final_cnt is not None:
            counts["entry"] = int(final_cnt)
        self._update_bar(key, value)
        self.progress_ui.update_label_for_stage(value)
        self._render_metrics(key)

    def set_universe_target(self, tgt: int | None) -> None:
        """全体ユニバース（Tgt）を設定。UI に即時反映する。

        - 引数が None の場合は既定動作（各 system の target/filter を表示）に戻る。
        - 整数が与えられた場合、各 system の表示上の `Tgt` はこの値を表示する。
        """
        try:
            if tgt is None:
                self.universe_target = None
            else:
                self.universe_target = int(tgt)
            GLOBAL_STAGE_METRICS.set_universe_target(self.universe_target)
        except Exception:
            self.universe_target = None
            try:
                GLOBAL_STAGE_METRICS.set_universe_target(None)
            except Exception:
                pass
        # 全 system の表示を更新
        self.refresh_all()

    def update_exit(self, name: str, count: int) -> None:
        key = str(name).lower()
        snapshot: StageSnapshot | None
        try:
            snapshot = GLOBAL_STAGE_METRICS.record_exit(key, count, emit_event=False)
        except Exception:
            snapshot = None
        if snapshot is not None:
            self._apply_snapshot(key, snapshot)
            return
        counts = self._ensure_counts(key)
        counts["exit"] = int(count)
        self._render_metrics(key)

    def finalize_counts(
        self, final_df: pd.DataFrame, per_system: dict[str, pd.DataFrame]
    ) -> None:  # noqa: E501
        for name, counts in self.stage_counts.items():
            snapshot: StageSnapshot | None
            try:
                snapshot = GLOBAL_STAGE_METRICS.get_snapshot(name)
            except Exception:
                snapshot = None
            if snapshot is not None:
                if counts.get("target") is None and snapshot.target is not None:
                    try:
                        counts["target"] = int(snapshot.target)
                        if self.universe_total is None:
                            self.universe_total = int(snapshot.target)
                    except Exception:
                        pass
                if counts.get("filter") is None and snapshot.filter_pass is not None:
                    try:
                        counts["filter"] = int(snapshot.filter_pass)
                    except Exception:
                        pass
                if counts.get("setup") is None and snapshot.setup_pass is not None:
                    try:
                        counts["setup"] = int(snapshot.setup_pass)
                    except Exception:
                        pass
                if counts.get("cand") is None and snapshot.candidate_count is not None:
                    try:
                        counts["cand"] = self._clamp_trdlist(snapshot.candidate_count)
                    except Exception:
                        pass
                if counts.get("entry") is None and snapshot.entry_count is not None:
                    try:
                        counts["entry"] = int(snapshot.entry_count)
                    except Exception:
                        pass
                if counts.get("exit") is None and snapshot.exit_count is not None:
                    try:
                        counts["exit"] = int(snapshot.exit_count)
                    except Exception:
                        pass
        try:
            system_series = (
                final_df["system"].astype(str).str.strip().str.lower()
                if "system" in final_df.columns
                else pd.Series(dtype=str)
            )
        except Exception:
            system_series = pd.Series(dtype=str)
        for name, counts in self.stage_counts.items():
            if counts.get("cand") is None:
                df_sys = per_system.get(name)
                if df_sys is None or df_sys.empty:
                    counts["cand"] = 0
                else:
                    counts["cand"] = self._clamp_trdlist(len(df_sys))
            if counts.get("entry") is None and not system_series.empty:
                try:
                    counts["entry"] = int((system_series == name).sum())
                except Exception:
                    counts["entry"] = counts.get("entry")
            if counts.get("target") is None:
                if self.universe_total is not None:
                    counts["target"] = self.universe_total
                elif counts.get("filter") is not None and counts.get("setup") is None:
                    counts["target"] = counts.get("filter")
            try:
                GLOBAL_STAGE_METRICS.record_stage(
                    name,
                    int(self.states.get(name, 100 if counts.get("entry") is not None else 0)),
                    counts.get("filter"),
                    counts.get("setup"),
                    counts.get("cand"),
                    counts.get("entry"),
                    emit_event=False,
                )
            except Exception:
                pass
        self.refresh_all()

    def apply_exit_counts(self, exit_counts: dict[str, int]) -> None:
        for name, cnt in exit_counts.items():
            if not cnt:
                continue
            snapshot: StageSnapshot | None
            try:
                snapshot = GLOBAL_STAGE_METRICS.record_exit(name, cnt, emit_event=False)
            except Exception:
                snapshot = None
            if snapshot is not None:
                self._apply_snapshot(name, snapshot)
            else:
                self._ensure_counts(name)["exit"] = int(cnt)
        self.refresh_all()

    def refresh_all(self) -> None:
        for name in self.metrics_store.systems():
            self._render_metrics(name)

    def _update_bar(self, key: str, value: int) -> None:
        if not self.show_ui:
            return
        bar = self.bars.get(key)
        if bar is None:
            return
        vv = max(0, min(100, int(value)))
        prev = int(self.states.get(key, 0))
        vv = max(prev, vv)
        self.states[key] = vv
        try:
            bar.progress(vv)
            self.stage_txt[key].text(f"run {vv}%" if vv < 100 else "done 100%")
        except Exception:
            pass

    def _render_metrics(self, key: str) -> None:
        placeholder = self.metrics_txt.get(key)
        if placeholder is None:
            return
        display = self.metrics_store.get_display_metrics(key)
        target_value = (
            self.universe_target if self.universe_target is not None else display.get("target")
        )
        text = "  ".join(
            [
                f"Tgt {self._format_value(target_value)}",
                f"FILpass {self._format_value(display.get('filter'))}",
                f"STUpass {self._format_value(display.get('setup'))}",
                f"TRDlist {self._format_trdlist(display.get('cand'))}",
                f"Entry {self._format_value(display.get('entry'))}",
                f"Exit {self._format_value(display.get('exit'))}",
            ]
        )
        try:
            placeholder.text(text)
        except Exception:
            pass

    def get_display_metrics(self, name: str) -> dict[str, int | None]:
        key = str(name).lower()
        return self.metrics_store.get_display_metrics(key)

    def _ensure_counts(self, name: str) -> dict[str, int | None]:
        return self.metrics_store.ensure_display_metrics(name)

    @staticmethod
    def _format_value(value: Any) -> str:
        return "-" if value is None else str(value)

    @staticmethod
    def _clamp_trdlist(value: Any) -> int | None:
        return StageMetricsStore.clamp_trdlist(value)

    def _format_trdlist(self, value: Any) -> str:
        if value is None:
            return "-"
        try:
            return str(self._clamp_trdlist(value))
        except Exception:
            return "-"


class UILogger:
    """UIとファイル出力の両方へログを書き出す。"""

    def __init__(self, start_time: float, progress_ui: ProgressUI):
        self.start_time = start_time
        self.progress_ui = progress_ui
        self.log_lines: list[str] = []

    def log(self, msg: str) -> None:
        forwarded_from_cli = False
        try:
            forwarding_flag = getattr(_run_today_mod, "_LOG_FORWARDING", None)
            if forwarding_flag is not None:
                forwarded_from_cli = bool(forwarding_flag.get())
        except Exception:
            forwarded_from_cli = False
        try:
            elapsed = max(0, time.time() - self.start_time)
            m, s = divmod(int(elapsed), 60)
        except Exception:
            m, s = 0, 0
        now_txt = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{now_txt} | {m}分{s}秒] {msg}"
        self.log_lines.append(line)
        if _has_st_ctx() and self.progress_ui.show_overall:
            if self._should_display(str(msg)):
                try:
                    self.progress_ui.progress_area.text(line)
                except Exception:
                    pass
        if not forwarded_from_cli:
            self._echo_cli(line)
            try:
                _get_today_logger().info(str(msg))
            except Exception:
                pass

    def _should_display(self, msg: str) -> bool:
        if not self.progress_ui.show_overall:
            return False
        data_load_prefixes = (
            "📦 基礎データロード進捗",
            "🧮 指標データロード進捗",
            "📦 基礎データロード完了",
            "🧮 指標データロード完了",
            "🧮 共有指標 前計算",
        )
        skip_keywords = (
            "進捗",
            "インジケーター",
            "indicator",
            "indicators",
            "指標計算",
            "バッチ時間",
            "batch time",
            "next batch size",
            "候補抽出",
            "候補日数",
            "📊 インジケーター計算",
            "📊 候補抽出",
            "⏱️ ッチ時間",
        )
        if msg.startswith(data_load_prefixes):
            return self.progress_ui.show_data_load
        return not any(keyword in msg for keyword in skip_keywords)

    def _echo_cli(self, line: str) -> None:
        try:
            print(line, flush=True)
            return
        except UnicodeEncodeError:
            try:
                encoding = getattr(sys.stdout, "encoding", "") or "utf-8"
                safe = line.encode(encoding, errors="replace").decode(encoding, errors="replace")
                print(safe, flush=True)
                return
            except Exception:
                pass
        except Exception:
            pass
        try:
            fallback = line.encode("ascii", errors="replace").decode("ascii", errors="replace")
            print(fallback, flush=True)
        except Exception:
            pass


class RunCallbacks:
    """run_all_systems_today へ渡すコールバックをまとめる。"""

    def __init__(
        self, logger: UILogger, progress_ui: ProgressUI, tracker: StageTracker
    ):  # noqa: E501
        self.logger = logger
        self.progress_ui = progress_ui
        self.tracker = tracker

    def ui_log(self, msg: str) -> None:
        self.logger.log(str(msg))

    def overall_progress(self, done: int, total: int, name: str) -> None:
        self.progress_ui.update(done, total, name)

    def per_system_progress(self, name: str, phase: str) -> None:
        self.tracker.update_progress(name, phase)

    def per_system_stage(
        self,
        name: str,
        value: int,
        filter_cnt: int | None = None,
        setup_cnt: int | None = None,
        cand_cnt: int | None = None,
        final_cnt: int | None = None,
    ) -> None:
        self.tracker.update_stage(
            name, value, filter_cnt, setup_cnt, cand_cnt, final_cnt
        )  # noqa: E501

    def per_system_exit(self, name: str, count: int) -> None:
        self.tracker.update_exit(name, count)

    def register_with_module(self) -> None:
        try:
            _run_today_mod._PER_SYSTEM_STAGE = self.per_system_stage  # type: ignore[attr-defined]  # noqa: E501
            _run_today_mod._PER_SYSTEM_EXIT = self.per_system_exit  # type: ignore[attr-defined]  # noqa: E501
            # Universe target setter for display alignment
            _run_today_mod._SET_STAGE_UNIVERSE_TARGET = self.tracker.set_universe_target  # type: ignore[attr-defined]
        except Exception:
            pass


@dataclass
class RunArtifacts:
    final_df: pd.DataFrame
    per_system: dict[str, pd.DataFrame]
    log_lines: list[str]
    total_elapsed: float
    stage_tracker: StageTracker
    logger: UILogger
    debug_mode: bool = False
    missing_report_path: Path | None = None
    missing_details: list[dict[str, Any]] | None = None


@dataclass
class ExitAnalysisResult:
    exits_today: pd.DataFrame
    planned: pd.DataFrame
    exit_counts: dict[str, int]
    error: str | None = None


def _indicator_requirements() -> dict[str, int]:
    """シグナル計算で使用する指標日数を定義する。"""

    return {
        "ROC200": int(200 * 1.1),
        "SMA25": int(25 * 1.1),
        "ATR20": int(20 * 1.1),
        "ADX7": int(7 * 1.1),
        "RETURN6": int(6 * 1.1),
        "Drop3D": int(3 * 1.1),
        "Return6D": int(6 * 1.1),
    }


def _rows_needed(indicator_days: dict[str, int]) -> int:
    if not indicator_days:
        return 0
    return max(indicator_days.values())


def _prepare_symbol_data(
    symbols: list[str],
    rows: int,
    logger: UILogger,
    *,
    debug_scan: bool = False,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    cache_key = (tuple(symbols), rows)
    symbol_cache = st.session_state.get("today_symbol_cache")
    if (
        not debug_scan
        and isinstance(symbol_cache, dict)
        and symbol_cache.get("key") == cache_key
        and isinstance(symbol_cache.get("data"), dict)
    ):
        data_map = symbol_cache.get("data", {})
        try:
            count = len(data_map)
        except Exception:
            count = 0
        logger.log(f"📦 基礎データロード再利用: {count}/{len(symbols)}件 (前回結果を使用)")
        return data_map, []

    logger.log(f"📦 基礎データロード開始: {len(symbols)} 銘柄 (必要日数≒{rows})")
    data_map, missing_details = _collect_symbol_data(
        symbols,
        rows=rows,
        log_fn=logger.log,
        debug_scan=debug_scan,
    )
    if not debug_scan:
        st.session_state["today_symbol_cache"] = {"key": cache_key, "data": data_map}
    return data_map, missing_details


def _save_missing_report(missing_details: list[dict[str, Any]]) -> Path | None:
    if not missing_details:
        return None
    try:
        base_dir = Path(settings.LOGS_DIR)
    except Exception:
        base_dir = Path("logs")
    target_dir = base_dir / "debug"
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
    except Exception:
        timestamp = str(int(time.time()))
    path = target_dir / f"rolling_cache_missing_{timestamp}.csv"
    try:
        try:
            settings2 = get_settings(create_dirs=True)
            round_dec = getattr(settings2.cache, "round_decimals", None)
        except Exception:
            round_dec = None
        try:
            out_df = round_dataframe(pd.DataFrame(missing_details), round_dec)
        except Exception:
            out_df = pd.DataFrame(missing_details)
        out_df.to_csv(path, index=False)
    except Exception:
        return None
    return path


def _store_run_results(
    final_df: pd.DataFrame, per_system: dict[str, pd.DataFrame]
) -> None:  # noqa: E501
    try:
        try:
            settings2 = get_settings(create_dirs=True)
            round_dec = getattr(settings2.cache, "round_decimals", None)
        except Exception:
            round_dec = None
        try:
            st.session_state["today_final_df"] = round_dataframe(final_df.copy(), round_dec)
        except Exception:
            st.session_state["today_final_df"] = final_df.copy()
        stored = {}
        for k, v in per_system.items():
            try:
                stored[k] = round_dataframe(v.copy(), round_dec)
            except Exception:
                stored[k] = v.copy()
        st.session_state["today_per_system"] = stored  # noqa: E501
    except Exception:
        pass


def _postprocess_results(
    final_df: pd.DataFrame, per_system: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    final_df = final_df.reset_index(drop=True)
    per_system = {name: df.reset_index(drop=True) for name, df in per_system.items()}
    final_df = _sort_final_df(final_df)
    if final_df is not None and not final_df.empty:
        try:
            final_df.insert(0, "no", range(1, len(final_df) + 1))
        except Exception:
            pass
    return final_df, per_system


def _sort_final_df(final_df: pd.DataFrame) -> pd.DataFrame:
    if final_df is None or final_df.empty or "system" not in final_df.columns:
        return final_df
    try:
        tmp = final_df.copy()
        tmp["_system_no"] = (
            tmp["system"].astype(str).str.extract(r"(\d+)").fillna(0).astype(int)
        )  # noqa: E501
        sort_cols = [c for c in ["side", "_system_no"] if c in tmp.columns]
        tmp = tmp.sort_values(sort_cols, kind="stable").drop(
            columns=["_system_no"], errors="ignore"
        )
        return tmp.reset_index(drop=True)
    except Exception:
        return final_df


def _log_run_completion(
    final_df: pd.DataFrame, per_system: dict[str, pd.DataFrame], elapsed: float
) -> None:
    try:
        m, s = divmod(int(max(0, elapsed)), 60)
        final_n = 0 if final_df is None or final_df.empty else int(len(final_df))
        per_counts_lines: list[str] = []
        counts_map = {
            str(name).strip().lower(): 0 if df is None or df.empty else int(len(df))
            for name, df in per_system.items()
            if str(name).strip()
        }
        if counts_map:
            per_counts_lines = format_group_counts(counts_map)
        detail = (
            f" | Long/Short別: {', '.join(per_counts_lines)}" if per_counts_lines else ""
        )  # noqa: E501
        _get_today_logger().info(
            f"✅ 本日のシグナル: シグナル検出処理終了 "
            f"(経過 {m}分{s}秒, 最終候補 {final_n} 件)"
            f"{detail}"
        )
    except Exception:
        pass


def _build_per_system_logs(log_lines: list[str]) -> dict[str, list[str]]:
    per_system_logs: dict[str, list[str]] = {f"system{i}": [] for i in range(1, 8)}
    skip_keywords = (
        "📊 指標計算",
        "⏱️ バッチ時間",
        "🧮 指標データ",
        "🧮 指標データロード",
        "🧮 共有指標の前計算",
        "📦 基礎データロード",
        "候補抽出",
        "インジケーター",
        "indicator",
        "indicators",
        "batch time",
        "next batch size",
    )
    for ln in log_lines:
        try:
            if any(k in ln for k in skip_keywords):
                continue
        except Exception:
            pass
        ln_l = ln.lower()
        for i in range(1, 8):
            key = f"system{i}"
            tag_candidates = [f"[system{i}]", f" {key}:", f"{key}:", f" {key}："]
            if any(tag in ln_l for tag in tag_candidates):
                per_system_logs[key].append(ln)
                break
    return per_system_logs


def _display_per_system_logs(per_system_logs: dict[str, list[str]]) -> None:
    if not per_system_logs:
        return
    if not any(per_system_logs[key] for key in per_system_logs):
        return
    tabs = st.tabs([f"system{i}" for i in range(1, 8)])
    for i, key in enumerate([f"system{i}" for i in range(1, 8)]):
        logs = per_system_logs.get(key, [])
        if not logs:
            continue
        with tabs[i]:
            st.text_area(
                label=f"ログ（{key}）",
                key=f"logs_{key}",
                value="\n".join(logs[-1000:]),
                height=380,
                disabled=True,
            )
            if key == "system2":
                _display_system2_filter_breakdown(logs)
            elif key == "system5":
                _display_system5_filter_breakdown(logs)


def _display_system2_filter_breakdown(logs: list[str]) -> None:
    try:
        detail_lines = [x for x in logs if ("フィルタ内訳:" in x or "filter breakdown:" in x)]
        if not detail_lines:
            return
        last_line = str(detail_lines[-1])
        disp = last_line.split("] ", 1)[1] if "] " in last_line else last_line
        st.caption(disp)
    except Exception:
        pass


def _display_system5_filter_breakdown(logs: list[str]) -> None:
    try:
        detail_lines = [
            x for x in logs if ("system5内訳" in x and ("AvgVol50" in x or "avgvol50" in x))
        ]
        if not detail_lines:
            return
        last_line = str(detail_lines[-1])
        disp = last_line.split("] ", 1)[1] if "] " in last_line else last_line
        st.caption(disp)
    except Exception:
        pass


def _configure_today_logger_ui() -> None:
    try:
        mode_env = (os.environ.get("TODAY_SIGNALS_LOG_MODE") or "").strip().lower()
    except Exception:
        mode_env = ""
    sel_mode = "single" if mode_env == "single" else "dated"
    try:
        _run_today_mod._configure_today_logger(mode=sel_mode)  # type: ignore[attr-defined]  # noqa: E501
        sel_path = getattr(_run_today_mod, "_LOG_FILE_PATH", None)
        if sel_path:
            st.caption(f"ログ保存先: {sel_path}")
    except Exception:
        pass


def execute_today_signals(run_config: RunConfig) -> RunArtifacts:
    indicator_days = _indicator_requirements()
    max_days = _rows_needed(indicator_days)
    start_time = time.time()
    ui_vis_raw = st.session_state.get("ui_vis", {})
    ui_vis = ui_vis_raw if isinstance(ui_vis_raw, dict) else {}
    progress_ui = ProgressUI(ui_vis)
    stage_tracker = StageTracker(ui_vis, progress_ui)
    logger = UILogger(start_time, progress_ui)
    callbacks = RunCallbacks(logger, progress_ui, stage_tracker)
    callbacks.register_with_module()
    _configure_today_logger_ui()
    logger.log("▶ 本日のシグナル: シグナル検出処理開始")
    buffer_days = max(20, int(max_days * 0.15))
    rows_needed = max_days + buffer_days
    symbols_for_data = list(dict.fromkeys([*run_config.symbols, "SPY"]))
    progress_ui.set_label("対象読み込み")
    final_df: pd.DataFrame | None = None
    per_system: dict[str, pd.DataFrame] = {}
    debug_result: RunArtifacts | None = None
    with st.spinner("実行中... (経過時間表示あり)"):
        symbol_data_map, missing_details = _prepare_symbol_data(
            symbols_for_data,
            rows_needed,
            logger,
            debug_scan=run_config.scan_missing_only,
        )
        if run_config.scan_missing_only:
            total_elapsed = max(0.0, time.time() - start_time)
            report_path = _save_missing_report(missing_details)
            if missing_details:
                if report_path is not None:
                    logger.log(f"🧪 欠損洗い出し: {len(missing_details)}件 (CSV: {report_path})")
                else:
                    logger.log(f"🧪 欠損洗い出し: {len(missing_details)}件 (CSV保存に失敗)")
            else:
                logger.log("🧪 欠損洗い出し: 欠損は検出されませんでした")
            stage_tracker.finalize_counts(pd.DataFrame(), {})
            debug_result = RunArtifacts(
                final_df=pd.DataFrame(),
                per_system={},
                log_lines=logger.log_lines,
                total_elapsed=total_elapsed,
                stage_tracker=stage_tracker,
                logger=logger,
                debug_mode=True,
                missing_report_path=report_path,
                missing_details=missing_details,
            )
        else:
            final_df, per_system = compute_today_signals(
                run_config.symbols,
                capital_long=run_config.capital_long,
                capital_short=run_config.capital_short,
                save_csv=run_config.save_csv,
                notify=run_config.notify,
                csv_name_mode=run_config.csv_name_mode,
                log_callback=callbacks.ui_log,
                progress_callback=callbacks.overall_progress,
                per_system_progress=callbacks.per_system_progress,
                symbol_data=symbol_data_map,
                parallel=run_config.run_parallel,
            )
    if debug_result is not None:
        return debug_result
    assert final_df is not None  # 安全策: debugモードでは既にreturn済み
    total_elapsed = max(0.0, time.time() - start_time)
    stage_tracker.finalize_counts(final_df, per_system)
    _store_run_results(final_df, per_system)
    return RunArtifacts(
        final_df=final_df,
        per_system=per_system,
        log_lines=logger.log_lines,
        total_elapsed=total_elapsed,
        stage_tracker=stage_tracker,
        logger=logger,
    )


def analyze_exit_candidates(paper_mode: bool) -> ExitAnalysisResult:
    exits_today_rows: list[dict[str, Any]] = []
    planned_rows: list[dict[str, Any]] = []
    exit_counts: dict[str, int] = {f"system{i}": 0 for i in range(1, 8)}
    try:
        client_tmp = ba.get_client(paper=paper_mode)
        positions = list(client_tmp.get_all_positions())
        raw_entry_map = load_entry_dates()
        entry_map: dict[str, str] = {}
        for key, value in raw_entry_map.items():
            try:
                entry_map[str(key).upper()] = str(value)
            except Exception:
                continue
        missing: list[str] = []
        for pos in positions:
            sym = str(getattr(pos, "symbol", "")).upper()
            if sym and sym not in entry_map:
                missing.append(sym)
        if missing:
            fetched = fetch_entry_dates_from_alpaca(client_tmp, missing)
            if fetched:
                for sym, ts in fetched.items():
                    if sym not in entry_map:
                        entry_map[sym] = pd.Timestamp(ts).strftime("%Y-%m-%d")
                try:
                    save_entry_dates(entry_map)
                except Exception:
                    pass
        symbol_system_map = _load_symbol_system_map(Path("data/symbol_system_map.json"))
        latest_trading_day = _latest_trading_day()
        strategy_classes = _strategy_class_map()
        for pos in positions:
            result = _evaluate_position_for_exit(
                pos,
                entry_map,
                symbol_system_map,
                latest_trading_day,
                strategy_classes,
            )
            if result is None:
                continue
            (system, pos_side, qty, exit_when, row_data, exit_today) = result
            when_value = str(exit_when or "")
            when_lower = when_value.lower()
            when_entry = when_lower or when_value
            if exit_today:
                exit_counts[system] = exit_counts.get(system, 0) + 1
                if when_lower == "tomorrow_open":
                    planned_rows.append(row_data | {"when": when_entry})
                else:
                    exits_today_rows.append(row_data | {"when": when_entry})
            else:
                planned_rows.append(row_data | {"when": when_entry})
        exits_today_df = pd.DataFrame(exits_today_rows)
        planned_df = pd.DataFrame(planned_rows)
        return ExitAnalysisResult(
            exits_today=exits_today_df, planned=planned_df, exit_counts=exit_counts
        )
    except Exception as exc:
        return ExitAnalysisResult(
            exits_today=pd.DataFrame(),
            planned=pd.DataFrame(),
            exit_counts=exit_counts,
            error=str(exc),
        )


def _load_symbol_system_map(path: Path) -> dict[str, str]:
    try:
        import json as _json

        if path.exists():
            data = _json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k).upper(): str(v).lower() for k, v in data.items()}
    except Exception:
        pass
    return {}


def _latest_trading_day() -> pd.Timestamp | None:
    calendar_day: pd.Timestamp | None = None
    try:
        calendar_day = get_latest_nyse_trading_day()
    except Exception:
        calendar_day = None

    price_day: pd.Timestamp | None = None
    try:
        spy_df = load_price("SPY", cache_profile="rolling")
        if spy_df is not None and not spy_df.empty:
            price_raw = pd.Timestamp(spy_df.index[-1])
            try:
                price_raw = price_raw.tz_localize(None)
            except (TypeError, ValueError, AttributeError):
                try:
                    price_raw = price_raw.tz_convert(None)
                except Exception:
                    pass
            price_day = pd.Timestamp(price_raw).normalize()
    except Exception:
        price_day = None

    if calendar_day is not None and price_day is not None:
        return max(calendar_day, price_day)
    return calendar_day or price_day


def _strategy_class_map() -> dict[str, Callable[[], Any]]:
    from strategies.system1_strategy import System1Strategy
    from strategies.system2_strategy import System2Strategy
    from strategies.system3_strategy import System3Strategy
    from strategies.system4_strategy import System4Strategy
    from strategies.system5_strategy import System5Strategy
    from strategies.system6_strategy import System6Strategy

    return {
        "system1": System1Strategy,
        "system2": System2Strategy,
        "system3": System3Strategy,
        "system4": System4Strategy,
        "system5": System5Strategy,
        "system6": System6Strategy,
    }


def _evaluate_position_for_exit(
    pos: Any,
    entry_map: dict[str, Any],
    symbol_system_map: dict[str, str],
    latest_trading_day: pd.Timestamp | None,
    strategy_classes: dict[str, Callable[[], Any]],
) -> tuple[str, str, int, str, dict[str, Any], bool] | None:
    try:
        sym = str(getattr(pos, "symbol", "")).upper()
        if not sym:
            return None
        qty = int(abs(float(getattr(pos, "qty", 0)) or 0))
        if qty <= 0:
            return None
        pos_side = str(getattr(pos, "side", "")).lower()
        system = symbol_system_map.get(sym, "").lower()
        if not system:
            if sym == "SPY" and pos_side == "short":
                system = "system7"
            else:
                return None
        if system == "system7":
            return None
        entry_date_str = entry_map.get(sym)
        if not entry_date_str:
            return None
        entry_dt = pd.to_datetime(entry_date_str).normalize()
        df_price = load_price(sym, cache_profile="full")
        if df_price is None or df_price.empty:
            return None
        df = df_price.copy(deep=False)
        if "Date" in df.columns:
            df.index = pd.Index(pd.to_datetime(df["Date"]).dt.normalize())
        else:
            df.index = pd.Index(pd.to_datetime(df.index).normalize())
        if latest_trading_day is None and len(df.index) > 0:
            latest_trading_day = pd.to_datetime(df.index[-1]).normalize()
        entry_idx = _find_entry_index(df.index, entry_dt)
        if entry_idx < 0:
            return None
        strategy_cls = strategy_classes.get(system)
        if strategy_cls is None:
            return None
        strategy = strategy_cls()
        prev_close = float(df.iloc[int(max(0, entry_idx - 1))]["Close"])
        entry_price, stop_price = _entry_and_stop_prices(
            system, strategy, df, entry_idx, prev_close
        )
        if entry_price is None or stop_price is None:
            return None
        _apply_strategy_state(system, strategy, df, entry_idx, prev_close)
        exit_price, exit_date = strategy.compute_exit(
            df, int(entry_idx), float(entry_price), float(stop_price)
        )
        today_norm = pd.to_datetime(df.index[-1]).normalize()
        if latest_trading_day is not None:
            today_norm = latest_trading_day
        is_today_exit, when = decide_exit_schedule(system, exit_date, today_norm)
        row_base = {
            "symbol": sym,
            "qty": qty,
            "position_side": pos_side,
            "system": system,
        }
        return system, pos_side, qty, when, row_base, is_today_exit
    except Exception:
        return None


def _find_entry_index(index: pd.Index, entry_dt: pd.Timestamp) -> int:
    try:
        if entry_dt in index:
            arr = index.get_indexer([entry_dt])
        else:
            arr = index.get_indexer([entry_dt], method="bfill")
        if len(arr) and arr[0] >= 0:
            return int(arr[0])
    except Exception:
        pass
    return -1


def _entry_and_stop_prices(
    system: str,
    strategy: Any,
    df: pd.DataFrame,
    entry_idx: int,
    prev_close: float,
) -> tuple[float | None, float | None]:
    try:
        if system == "system1":
            entry_price = float(df.iloc[int(entry_idx)]["Open"])
            atr20 = float(df.iloc[int(max(0, entry_idx - 1))]["ATR20"])
            stop_mult = float(strategy.config.get("stop_atr_multiple", 5.0))
            return entry_price, entry_price - stop_mult * atr20
        if system == "system2":
            entry_price = float(df.iloc[int(entry_idx)]["Open"])
            atr = float(df.iloc[int(max(0, entry_idx - 1))]["ATR10"])
            stop_mult = float(strategy.config.get("stop_atr_multiple", 3.0))
            return entry_price, entry_price + stop_mult * atr
        if system == "system6":
            ratio = float(strategy.config.get("entry_price_ratio_vs_prev_close", 1.05))
            entry_price = round(prev_close * ratio, 2)
            atr = float(df.iloc[int(max(0, entry_idx - 1))]["ATR10"])
            stop_mult = float(strategy.config.get("stop_atr_multiple", 3.0))
            return entry_price, entry_price + stop_mult * atr
        if system == "system3":
            ratio = float(strategy.config.get("entry_price_ratio_vs_prev_close", 0.93))
            entry_price = round(prev_close * ratio, 2)
            atr = float(df.iloc[int(max(0, entry_idx - 1))]["ATR10"])
            stop_mult = float(strategy.config.get("stop_atr_multiple", 2.5))
            return entry_price, entry_price - stop_mult * atr
        if system == "system4":
            entry_price = float(df.iloc[int(entry_idx)]["Open"])
            atr40 = float(df.iloc[int(max(0, entry_idx - 1))]["ATR40"])
            stop_mult = float(strategy.config.get("stop_atr_multiple", 1.5))
            return entry_price, entry_price - stop_mult * atr40
        if system == "system5":
            ratio = float(strategy.config.get("entry_price_ratio_vs_prev_close", 0.97))
            entry_price = round(prev_close * ratio, 2)
            atr = float(df.iloc[int(max(0, entry_idx - 1))]["ATR10"])
            stop_mult = float(strategy.config.get("stop_atr_multiple", 3.0))
            return entry_price, entry_price - stop_mult * atr
    except Exception:
        return None, None
    return None, None


def _apply_strategy_state(
    system: str,
    strategy: Any,
    df: pd.DataFrame,
    entry_idx: int,
    prev_close: float,
) -> None:
    if system == "system5":
        try:
            atr = float(df.iloc[int(max(0, entry_idx - 1))]["ATR10"])
            strategy._last_entry_atr = atr  # type: ignore[attr-defined]
        except Exception:
            pass
    if system in {"system3", "system5", "system6"}:
        try:
            strategy._last_prev_close = prev_close  # type: ignore[attr-defined]
        except Exception:
            pass


def render_exit_candidates_section(
    trade_options: TradeOptions,
    stage_tracker: StageTracker,
    logger: UILogger,
    notify: bool,
) -> ExitAnalysisResult:
    st.subheader("今日の手仕舞い候補（MOC）")
    result = analyze_exit_candidates(trade_options.paper_mode)
    if result.error:
        st.warning(f"手仕舞い候補の推定に失敗しました: {result.error}")
        return result
    _display_exit_orders_table(result, trade_options, stage_tracker, logger, notify)
    _display_planned_exits_section(result, trade_options)
    return result


def _display_exit_orders_table(
    result: ExitAnalysisResult,
    trade_options: TradeOptions,
    stage_tracker: StageTracker,
    logger: UILogger,
    notify: bool,
) -> None:
    if result.exits_today.empty:
        st.info("本日大引けでの手仕舞い候補はありません。")
        return
    st.dataframe(result.exits_today, use_container_width=True)
    stage_tracker.apply_exit_counts(result.exit_counts)
    if st.button("本日分の手仕舞い注文（MOC）を送信"):
        from common.alpaca_order import submit_exit_orders_df

        res = submit_exit_orders_df(
            result.exits_today,
            paper=trade_options.paper_mode,
            tif="CLS",
            retries=int(trade_options.retries),
            delay=float(max(0.0, trade_options.delay)),
            log_callback=logger.log,
            notify=notify,
        )
        if res is not None and not res.empty:
            st.dataframe(res, use_container_width=True)


def _display_planned_exits_section(
    result: ExitAnalysisResult, trade_options: TradeOptions
) -> None:  # noqa: E501
    if result.planned.empty:
        return
    st.caption("明日発注する手仕舞い計画（保存→スケジューラが実行）")
    st.dataframe(result.planned, use_container_width=True)
    planned_rows = [
        {str(k): v for k, v in row.items()} for row in result.planned.to_dict(orient="records")
    ]
    _auto_save_planned_exits(planned_rows, show_success=False)
    if st.button("計画を保存（JSONL）"):
        _auto_save_planned_exits(planned_rows, show_success=True)
    st.write("")
    dry_run_plan = st.checkbox(
        "ドライラン（予約送信をテストとして実行）",
        value=True,
        key="planned_exits_dry_run",
    )
    col_open, col_close = st.columns(2)
    with col_open:
        if st.button("⏱️ 寄り（OPG）予約を今すぐ送信", key="run_scheduler_open"):
            _run_planned_exit_scheduler("open", dry_run_plan)
    with col_close:
        if st.button("⏱️ 引け（CLS）予約を今すぐ送信", key="run_scheduler_close"):
            _run_planned_exit_scheduler("close", dry_run_plan)


def _auto_save_planned_exits(
    planned_rows: list[dict[str, Any]], show_success: bool
) -> None:  # noqa: E501
    import json as _json

    plan_path = Path("data/planned_exits.jsonl")
    try:
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        with plan_path.open("w", encoding="utf-8") as f:
            for row in planned_rows:
                f.write(_json.dumps(row, ensure_ascii=False) + "\n")
        if show_success:
            st.success(f"保存しました: {plan_path}")
        else:
            st.caption(f"計画を保存しました: {plan_path}")
    except Exception as exc:
        if show_success:
            st.error(f"保存に失敗: {exc}")
        else:
            st.error(f"計画の保存に失敗: {exc}")


def _run_planned_exit_scheduler(kind: str, dry_run: bool) -> None:
    try:
        from schedulers.next_day_exits import submit_planned_exits as _run_sched

        df_exec = _run_sched(kind, dry_run=dry_run)
        if df_exec is not None and not df_exec.empty:
            st.success(
                "寄り（OPG）分の予約送信を実行しました。結果を表示します。"
                if kind == "open"
                else "引け（CLS）分の予約送信を実行しました。結果を表示します。"
            )
            st.dataframe(df_exec, use_container_width=True)
        else:
            st.info(
                "寄り（OPG）対象の予約はありませんでした。"
                if kind == "open"
                else "引け（CLS）対象の予約はありませんでした。"
            )
    except Exception as exc:
        label = "寄り（OPG）" if kind == "open" else "引け（CLS）"
        st.error(f"{label}予約の実行に失敗: {exc}")


def render_today_signals_results(
    artifacts: RunArtifacts,
    run_config: RunConfig,
    trade_options: TradeOptions,
) -> None:
    if artifacts.debug_mode:
        _render_missing_debug_results(artifacts)
        return
    final_df, per_system = _postprocess_results(
        artifacts.final_df, artifacts.per_system
    )  # noqa: E501
    artifacts.stage_tracker.finalize_counts(final_df, per_system)
    _show_total_elapsed(artifacts.total_elapsed)
    _log_run_completion(final_df, per_system, artifacts.total_elapsed)
    per_system_logs = _build_per_system_logs(artifacts.logger.log_lines)
    _display_per_system_logs(per_system_logs)
    render_exit_candidates_section(
        trade_options,
        artifacts.stage_tracker,
        artifacts.logger,
        run_config.notify,
    )
    _render_final_signals_section(
        final_df,
        per_system,
        run_config,
        trade_options,
        artifacts.logger,
    )
    _render_system_details(per_system, artifacts.stage_tracker, per_system_logs)
    _render_previous_results_section()
    _render_previous_run_logs(artifacts.log_lines)


def _render_missing_debug_results(artifacts: RunArtifacts) -> None:
    st.subheader("🧪 欠損洗い出しモードの結果")
    details = artifacts.missing_details or []
    if details:
        st.write(f"検出された銘柄: {len(details)}件")
        try:
            df_details = pd.DataFrame(details)
        except Exception:
            df_details = None
        if df_details is not None and not df_details.empty:
            st.dataframe(df_details, use_container_width=True)
        else:
            st.json(details)
    else:
        st.success("ローリングキャッシュの欠損は検出されませんでした。")
    report_path = artifacts.missing_report_path
    if report_path:
        path_obj = Path(report_path)
        st.info(f"レポート: {path_obj}")
        try:
            data_bytes = path_obj.read_bytes()
        except Exception:
            data_bytes = None
        if data_bytes:
            st.download_button(
                "欠損レポートをダウンロード",
                data=data_bytes,
                file_name=path_obj.name,
                mime="text/csv",
                key=f"missing_report_{int(time.time() * 1000)}",
            )
    st.info("このモードでは基礎データの欠損確認のみを実施しました。シグナル計算は行っていません。")
    _render_previous_results_section()
    _render_previous_run_logs(artifacts.log_lines)


def _show_total_elapsed(total_elapsed: float) -> None:
    total_elapsed = max(0.0, float(total_elapsed))
    m, s = divmod(int(total_elapsed), 60)
    st.info(f"総経過時間: {m}分{s}秒")


def _render_final_signals_section(
    final_df: pd.DataFrame,
    per_system: dict[str, pd.DataFrame],
    run_config: RunConfig,
    trade_options: TradeOptions,
    logger: UILogger,
) -> None:
    st.subheader("最終選定銘柄")
    if final_df is None or final_df.empty:
        st.info("本日のシグナルはありません。")
        return
    _render_final_summary(final_df)
    st.dataframe(final_df, use_container_width=True)
    _render_skip_reports()
    _download_final_csv(final_df)
    st.session_state["today_shown_this_run"] = True
    if run_config.save_csv:
        _auto_save_final_results(final_df, per_system, run_config)
    if trade_options.do_trade:
        _execute_auto_trading(
            final_df,
            trade_options,
            run_config,
            logger,
        )


def _render_final_summary(final_df: pd.DataFrame) -> None:
    summary_lines: list[str] = []
    try:
        if "system" in final_df.columns:
            system_series = final_df["system"].astype(str).str.strip().str.lower()
            counts_map = system_series.value_counts().to_dict()
            values_map: dict[str, float] = {}
            if "position_value" in final_df.columns:
                values_series = (
                    final_df.assign(_system=system_series)[
                        ["_system", "position_value"]
                    ]  # noqa: E501
                    .groupby("_system")["position_value"]
                    .sum()
                )
                values_map = values_series.to_dict()
            if counts_map:
                if values_map:
                    summary_lines = format_group_counts_and_values(
                        counts_map, values_map
                    )  # noqa: E501
                else:
                    summary_lines = format_group_counts(counts_map)
    except Exception:
        summary_lines = []
    if summary_lines:
        font_css = (
            "font-family: 'Noto Sans JP', 'Meiryo', sans-serif; "
            "font-size: 1rem; letter-spacing: 0.02em;"
        )
        html_summary = " / ".join(summary_lines)
        st.markdown(
            f'<div style="{font_css}">サマリー（Long/Short別）: {html_summary}</div>',
            unsafe_allow_html=True,
        )


def _render_skip_reports() -> None:
    try:
        settings2 = get_settings(create_dirs=True)
        results_dir = Path(getattr(settings2.outputs, "results_csv_dir", "results_csv"))
        skip_files = []
        for i in range(1, 8):
            name = f"system{i}"
            fp = results_dir / f"skip_summary_{name}.csv"
            if fp.exists() and fp.is_file():
                skip_files.append((name, fp))
        if skip_files:
            with st.expander("🧪 データスキップ/ショート不可の内訳CSV（本日）", expanded=False):
                _render_skip_file_group(skip_files, "skip")
            detail_files = []
            for i in range(1, 8):
                name = f"system{i}"
                fpd = results_dir / f"skip_details_{name}.csv"
                if fpd.exists() and fpd.is_file():
                    detail_files.append((name, fpd))
            if detail_files:
                st.markdown("---")
                st.caption("スキップ詳細（symbol×reason）")
                _render_skip_file_group(detail_files, "skipdet")
            shortable_files = []
            for i in (2, 6):
                name = f"system{i}"
                fp2 = results_dir / f"shortability_excluded_{name}.csv"
                if fp2.exists() and fp2.is_file():
                    shortable_files.append((name, fp2))
            if shortable_files:
                st.markdown("---")
                st.caption("ショート不可で除外された銘柄（system2/6）")
                _render_skip_file_group(shortable_files, "short_exc")
    except Exception:
        pass


def _render_skip_file_group(files: list[tuple[str, Path]], key_prefix: str) -> None:
    for name, path in files:
        cols = st.columns([4, 1])
        with cols[0]:
            try:
                df_skip = pd.read_csv(path)
            except Exception:
                df_skip = None
            st.caption(f"{name}: {path.name}")
            if df_skip is not None and not df_skip.empty:
                st.dataframe(df_skip, use_container_width=True)
            else:
                st.write("(空) 内訳情報は見つかりませんでした。")
        with cols[1]:
            try:
                data_bytes = path.read_bytes()
            except Exception:
                data_bytes = None
            if data_bytes:
                import time

                st.download_button(
                    label=f"{name} CSV",
                    data=data_bytes,
                    file_name=path.name,
                    mime="text/csv",
                    key=f"dl_{key_prefix}_{name}_{int(time.time() * 1000)}",
                )


def _download_final_csv(final_df: pd.DataFrame) -> None:
    try:
        settings2 = get_settings(create_dirs=True)
        round_dec = getattr(settings2.cache, "round_decimals", None)
    except Exception:
        round_dec = None
    try:
        out_df = round_dataframe(final_df, round_dec)
    except Exception:
        out_df = final_df
    csv = out_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "最終CSVをダウンロード",
        data=csv,
        file_name="today_signals_final.csv",
        on_click=_reset_shown_flag,
    )


def _auto_save_final_results(
    final_df: pd.DataFrame,
    per_system: dict[str, pd.DataFrame],
    run_config: RunConfig,
) -> None:
    try:
        settings2 = get_settings(create_dirs=True)
        sig_dir = Path(settings2.outputs.signals_dir)
        sig_dir.mkdir(parents=True, exist_ok=True)
        from datetime import datetime as _dt

        ts = _dt.now().strftime("%Y-%m-%d")
        if run_config.csv_name_mode == "datetime":
            ts = _dt.now().strftime("%Y-%m-%d_%H%M")
        elif run_config.csv_name_mode == "runid":
            rid = st.session_state.get("last_run_id") or "RUN"
            ts = f"{_dt.now().strftime('%Y-%m-%d')}_{rid}"
        fp = sig_dir / f"today_signals_{ts}.csv"
        try:
            settings2 = get_settings(create_dirs=True)
            round_dec = getattr(settings2.cache, "round_decimals", None)
        except Exception:
            round_dec = None
        try:
            out_df = round_dataframe(final_df, round_dec)
        except Exception:
            out_df = final_df
        out_df.to_csv(fp, index=False)
        st.caption(f"自動保存: {fp}")
        for name, df in per_system.items():
            try:
                if df is None or df.empty:
                    continue
                fp_sys = sig_dir / f"signals_{name}_{ts}.csv"
                try:
                    out_df = round_dataframe(df, round_dec)
                except Exception:
                    out_df = df
                out_df.to_csv(fp_sys, index=False)
                st.caption(f"自動保存: {fp_sys}")
            except Exception as exc:
                st.warning(f"{name} の自動保存に失敗: {exc}")
    except Exception as exc:
        st.warning(f"自動保存に失敗: {exc}")


def _execute_auto_trading(
    final_df: pd.DataFrame,
    trade_options: TradeOptions,
    run_config: RunConfig,
    logger: UILogger,
) -> None:
    st.divider()
    st.subheader("Alpaca自動発注結果")
    system_order_type = {
        "system1": "market",
        "system3": "market",
        "system4": "market",
        "system5": "market",
        "system2": "limit",
        "system6": "limit",
        "system7": "limit",
    }
    results_df = submit_orders_df(
        final_df,
        paper=trade_options.paper_mode,
        order_type=None,
        system_order_type=system_order_type,
        tif="DAY",
        retries=int(trade_options.retries),
        delay=float(max(0.0, trade_options.delay)),
        log_callback=logger.log,
        notify=run_config.notify,
    )
    if results_df is not None and not results_df.empty:
        st.dataframe(results_df, use_container_width=True)
        if trade_options.poll_status and any(
            results_df["order_id"].fillna("").astype(str)
        ):  # noqa: E501
            _poll_order_status(results_df, trade_options)
    if trade_options.update_bp_after:
        _update_buying_power(trade_options)


def _poll_order_status(results_df: pd.DataFrame, trade_options: TradeOptions) -> None:
    st.info("注文状況を10秒間ポーリングします...")
    try:
        client = ba.get_client(paper=trade_options.paper_mode)
    except Exception:
        client = None
    if client is None:
        return
    order_ids = [str(oid) for oid in results_df["order_id"].tolist() if oid]
    end = time.time() + 10
    last: dict[str, Any] = {}
    while time.time() < end:
        status_map = ba.get_orders_status_map(client, order_ids)
        if status_map != last:
            if status_map:
                st.caption("注文状況を更新しました（詳細はログ参照）")
            last = status_map
        time.sleep(1.0)


def _update_buying_power(trade_options: TradeOptions) -> None:
    try:
        client2 = ba.get_client(paper=trade_options.paper_mode)
        acct = client2.get_account()
        bp_raw = getattr(acct, "buying_power", None)
        if bp_raw is None:
            bp_raw = getattr(acct, "cash", None)
        if bp_raw is not None:
            bp = float(bp_raw)
            st.session_state["today_cap_long"] = round(bp / 2.0, 2)
            st.session_state["today_cap_short"] = round(bp / 2.0, 2)
            st.success(
                "約定反映後の資金余力でLong/Shortを再設定しました: "
                f"${st.session_state['today_cap_long']} / "
                f"${st.session_state['today_cap_short']}"
            )
        else:
            st.warning("Alpaca口座情報: buying_power/cashが取得できません（更新なし）")
    except Exception as exc:
        st.error(f"余力の自動更新に失敗: {exc}")


def _render_system_details(
    per_system: dict[str, pd.DataFrame],
    stage_tracker: StageTracker,
    per_system_logs: dict[str, list[str]] | None = None,
) -> None:
    with st.expander("システム別詳細"):
        settings_local = get_settings(create_dirs=True)
        results_dir = Path(
            getattr(settings_local.outputs, "results_csv_dir", "results_csv")
        )  # noqa: E501
        shortable_excluded_map = {}
        for i in (2, 6):
            name = f"system{i}"
            fp = results_dir / f"shortability_excluded_{name}.csv"
            if fp.exists() and fp.is_file():
                try:
                    df_exc = pd.read_csv(fp)
                    if df_exc is not None and not df_exc.empty:
                        shortable_excluded_map[name] = set(
                            df_exc["symbol"].astype(str).str.upper()
                        )  # noqa: E501
                except Exception:
                    pass
        system_order = [f"system{i}" for i in range(1, 8)]
        for name in system_order:
            st.markdown(f"#### {name}")
            display_metrics = stage_tracker.get_display_metrics(name)
            metrics_line = "  ".join(
                [
                    f"Tgt {StageTracker._format_value(display_metrics.get('target'))}",  # noqa: E501
                    f"FILpass {StageTracker._format_value(display_metrics.get('filter'))}",
                    f"STUpass {StageTracker._format_value(display_metrics.get('setup'))}",
                    f"TRDlist {stage_tracker._format_trdlist(display_metrics.get('cand'))}",
                    f"Entry {StageTracker._format_value(display_metrics.get('entry'))}",
                    f"Exit {StageTracker._format_value(display_metrics.get('exit'))}",
                ]
            )
            st.caption(metrics_line)
            df = per_system.get(name)
            if df is None or df.empty:
                # Try to extract explicit zero-reason from per-system logs if available
                reason_text: str | None = None
                try:
                    if per_system_logs and name in per_system_logs:
                        import re as _re

                        logs = per_system_logs.get(name) or []
                        for ln in reversed(logs):
                            if not ln:
                                continue
                            m = _re.search(r"候補0件理由[:：]\s*(.+)$", ln)
                            if m:
                                reason_text = m.group(1).strip()
                                break
                            m2 = _re.search(r"セットアップ不成立[:：]\s*(.+)$", ln)
                            if m2:
                                reason_text = m2.group(1).strip()
                                break
                except Exception:
                    reason_text = None

                st.write("(空) 候補は0件です。")
                if reason_text:
                    st.info(f"候補0件理由: {reason_text}")
                continue
            df_disp = df.copy()
            side_type = None
            if name in LONG_SYSTEMS:
                side_type = "long"
            elif name in SHORT_SYSTEMS:
                side_type = "short"
            if side_type and "side" in df_disp.columns:
                mask = df_disp["side"].str.lower() != side_type
                if mask.any():
                    fill_cols = [
                        col
                        for col in df_disp.columns
                        if col not in {"symbol", "side", "system"}  # noqa: E501
                    ]
                    if fill_cols:
                        df_disp.loc[:, fill_cols] = df_disp.loc[:, fill_cols].astype(
                            "object"
                        )  # noqa: E501
                        df_disp.loc[mask, fill_cols] = "-"
            if name in shortable_excluded_map:
                excluded_syms = shortable_excluded_map[name]
                if excluded_syms:
                    st.caption(f"🚫 ショート不可で除外: {len(excluded_syms)}件")
                    st.write(
                        f"<span style='color:red;font-size:0.95em;'>"
                        f"ショート不可: {', '.join(sorted(excluded_syms)[:10])}"
                        f"{' ...' if len(excluded_syms) > 10 else ''}"  # noqa: E501
                        f"</span>",
                        unsafe_allow_html=True,
                    )
            st.dataframe(df_disp, use_container_width=True)


def _render_previous_results_section() -> None:
    try:
        if (not st.session_state.get("today_shown_this_run", False)) and (
            "today_final_df" in st.session_state
        ):
            prev_df = st.session_state.get("today_final_df")
            if prev_df is not None and not prev_df.empty:
                st.subheader("前回の最終選定銘柄（再表示）")
                st.dataframe(prev_df, use_container_width=True)
                try:
                    settings2 = get_settings(create_dirs=True)
                    round_dec = getattr(settings2.cache, "round_decimals", None)
                except Exception:
                    round_dec = None
                try:
                    prev_out = round_dataframe(prev_df, round_dec)
                except Exception:
                    prev_out = prev_df
                csv_prev = prev_out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "最終CSVをダウンロード（前回）",
                    data=csv_prev,
                    file_name="today_signals_final_prev.csv",
                    key="download_prev_final",
                    on_click=_reset_shown_flag,
                )
                prev_per = st.session_state.get("today_per_system", {})
                if isinstance(prev_per, dict):
                    with st.expander("前回のシステム別CSV", expanded=False):
                        for name, df in prev_per.items():
                            if df is None or df.empty:
                                continue
                            st.markdown(f"#### {name}")
                            st.dataframe(df, use_container_width=True)
    except Exception:
        pass


def _render_previous_run_logs(log_lines: list[str]) -> None:
    prev_msgs = [line for line in log_lines if line and ("(前回結果) system" in line)]
    if not prev_msgs:
        return
    import re as _re

    def _parse_prev_line(ln: str) -> tuple[str, int, str, str]:
        ts = ln.split("] ", 1)[0].strip("[")
        m = _re.search(r"\(前回結果\) (system\d+):\s*(\d+)", ln)
        sys = m.group(1) if m else "system999"
        cnt = int(m.group(2)) if m else 0
        return sys, cnt, ts, ln

    parsed = [_parse_prev_line(x) for x in prev_msgs]
    order = {f"system{i}": i for i in range(1, 8)}
    parsed.sort(key=lambda t: order.get(t[0], 999))
    lines_sorted = [f"{p[2]} | {p[0]}: {p[1]}件\n{p[3]}" for p in parsed]
    with st.expander("前回結果（system別）", expanded=False):
        st.text("\n\n".join(lines_sorted))


with st.sidebar:
    st.header("ユニバース")
    universe: list[str] = []
    try:
        logger = logging.getLogger("today_signals.ui")
        universe = build_symbol_universe_from_settings(settings, logger=logger)
    except Exception as exc:
        universe = []
        st.warning(f"NASDAQ/EODHDの銘柄取得に失敗しました: {exc}")
    if not universe:
        universe = univ.load_universe_file()
    if not universe:
        universe = univ.build_universe_from_cache(limit=None)
        univ.save_universe_file(universe)
    all_syms = universe

    # 任意の件数でユニバースを制限するテスト用オプション
    limit_max = max(1, len(all_syms))
    test_limit = st.number_input(
        "銘柄数 (0は全銘柄)",
        min_value=0,
        max_value=limit_max,
        value=0,
        step=1,
    )
    syms = all_syms[: int(test_limit)] if test_limit else all_syms

    st.write(f"銘柄数: {len(syms)}")
    st.write(", ".join(syms[:10]) + (" ..." if len(syms) > 10 else ""))

    st.header("資産")
    # Alpacaから取得した資産のみを使う
    if "today_cap_long" not in st.session_state:
        st.session_state["today_cap_long"] = 0.0
    if "today_cap_short" not in st.session_state:
        st.session_state["today_cap_short"] = 0.0
    # 口座情報の一時保存領域
    st.session_state.setdefault("alpaca_acct_type", None)
    st.session_state.setdefault("alpaca_buying_power", None)
    st.session_state.setdefault("alpaca_cash", None)
    st.session_state.setdefault("alpaca_multiplier", None)
    st.session_state.setdefault("alpaca_shorting_enabled", None)
    st.session_state.setdefault("alpaca_status", None)

    # Alpacaから取得してフォームに反映
    if st.button("🔍 Alpacaから資産取得してフォームに反映"):
        try:
            client = ba.get_client(paper=True)
            acct = client.get_account()
            # 口座情報を保存（表示用）
            try:
                st.session_state["alpaca_acct_type"] = getattr(
                    acct, "account_type", None
                )  # noqa: E501
                st.session_state["alpaca_multiplier"] = getattr(
                    acct, "multiplier", None
                )  # noqa: E501
                st.session_state["alpaca_shorting_enabled"] = getattr(
                    acct, "shorting_enabled", None
                )
                st.session_state["alpaca_status"] = getattr(acct, "status", None)
            except Exception:
                pass
            bp_raw = getattr(acct, "buying_power", None)
            if bp_raw is None:
                bp_raw = getattr(acct, "cash", None)
            if bp_raw is not None:
                bp = float(bp_raw)
                st.session_state["alpaca_buying_power"] = bp
                try:
                    st.session_state["alpaca_cash"] = float(
                        getattr(acct, "cash", None) or 0.0
                    )  # noqa: E501
                except Exception:
                    pass
                st.session_state["today_cap_long"] = round(bp / 2.0, 2)
                st.session_state["today_cap_short"] = round(bp / 2.0, 2)
                st.success(
                    f"long資産/short資産を{st.session_state['today_cap_long']}ずつに設定"
                    f"（buying_powerの半分={bp}）"
                )
            else:
                st.warning("Alpaca口座情報: buying_power/cashが取得できません")
        except Exception as e:
            st.error(f"Alpaca資産取得エラー: {e}")

    # 口座情報（表示のみの更新ボタン）
    if st.button("ℹ️ Alpaca口座情報を更新（表示のみ）"):
        try:
            client = ba.get_client(paper=True)
            acct = client.get_account()
            st.session_state["alpaca_acct_type"] = getattr(acct, "account_type", None)
            st.session_state["alpaca_buying_power"] = float(
                getattr(acct, "buying_power", getattr(acct, "cash", 0.0)) or 0.0
            )
            st.session_state["alpaca_cash"] = float(getattr(acct, "cash", 0.0))
            st.session_state["alpaca_multiplier"] = getattr(acct, "multiplier", None)
            st.session_state["alpaca_shorting_enabled"] = getattr(
                acct, "shorting_enabled", None
            )  # noqa: E501
            st.session_state["alpaca_status"] = getattr(acct, "status", None)
            st.success("口座情報を更新しました（表示のみ）")
        except Exception as e:
            st.error(f"口座情報の更新に失敗: {e}")

    # 口座情報の表示（タイプ推定 + Buying Power）
    acct_type_raw = st.session_state.get("alpaca_acct_type")
    multiplier = st.session_state.get("alpaca_multiplier")
    try:
        mult_f = float(multiplier) if multiplier is not None else None
    except Exception:
        mult_f = None
    derived_type = (
        "Margin"
        if (mult_f is not None and mult_f > 1.0)
        else ("Cash" if mult_f is not None else "不明")
    )
    bp_val = st.session_state.get("alpaca_buying_power")
    bp_txt = f"${bp_val:,.2f}" if isinstance(bp_val, (int | float)) else "未取得"
    st.caption("Alpaca口座情報")
    st.write(f"アカウント種別（推定）: {derived_type}  |  Buying Power: {bp_txt}")
    if acct_type_raw is not None or mult_f is not None:
        st.caption(
            f"詳細: account_type={acct_type_raw}, "
            f"multiplier={mult_f if mult_f is not None else '-'}"
        )

    # 資産入力フォーム
    st.session_state["today_cap_long"] = st.number_input(
        "long資産 (USD)",
        min_value=0.0,
        step=100.0,
        value=float(st.session_state["today_cap_long"]),
        key="today_cap_long_input",
    )
    st.session_state["today_cap_short"] = st.number_input(
        "short資産 (USD)",
        min_value=0.0,
        step=100.0,
        value=float(st.session_state["today_cap_short"]),
        key="today_cap_short_input",
    )

    st.header("CSV保存")
    st.session_state.setdefault("save_csv", False)
    save_csv = st.checkbox(
        "CSVをsignals_dirに自動保存",
        key="save_csv",
        help="実行後に自動で signals_dir に保存します（他のダウンロードに影響しません）。",
    )
    # CSVファイル名の形式選択（date/datetime/runid）
    st.session_state.setdefault("csv_name_mode", "date")
    csv_name_mode = st.selectbox(
        "CSVファイル名",
        options=["date", "datetime", "runid"],
        index=["date", "datetime", "runid"].index(
            str(st.session_state.get("csv_name_mode", "date"))
        ),
        help="date=YYYY-MM-DD / datetime=YYYY-MM-DD_HHMM / runid=YYYY-MM-DD_RUNID",
        key="csv_name_mode",
    )

    # 既定で並列実行をON（Windowsでも有効化）
    is_windows = platform.system().lower().startswith("win")
    run_parallel_default = True
    run_parallel = st.checkbox("並列実行（システム横断）", value=run_parallel_default)

    st.header("デバッグ")
    scan_missing_only = st.checkbox(
        "🧪 欠損洗い出しモード（ローリングキャッシュ）",
        key="today_scan_missing_only",
        help="rolling キャッシュからの読み込み時に欠損を検出し、CSVに書き出して終了します。",
    )
    if scan_missing_only:
        st.caption("※ このモードではシグナル計算を行いません。欠損レポートのみ出力します。")

    # 通知（Slack Bot Token）設定（チャンネル指定フォームは廃止）
    st.header("通知設定（Slack Bot Token）")
    st.session_state.setdefault("use_slack_notify", True)
    use_slack_notify = st.checkbox(
        "Slack通知を有効化（Bot Token）",
        key="use_slack_notify",
        help="環境変数 SLACK_BOT_TOKEN が設定済みである前提（通知先は既定値を使用）。",
    )
    # 簡易ヘルスチェック表示
    try:
        has_token = bool(os.environ.get("SLACK_BOT_TOKEN", "").strip())
        st.caption("トークン: " + ("検出済み" if has_token else "未設定（.envを設定してください）"))
    except Exception:
        pass

    # 並列実行の詳細設定は削除（初期デフォルト挙動に戻す）
    st.header("Alpaca自動発注")
    paper_mode = st.checkbox("ペーパートレードを使用", value=True)
    retries = st.number_input("リトライ回数", min_value=0, max_value=5, value=2)
    delay = st.number_input("遅延（秒）", min_value=0.0, step=0.5, value=0.5)
    # 既定値は session_state に一度だけ設定し、ウィジェット側では value を渡さない
    st.session_state.setdefault("poll_status", False)
    st.session_state.setdefault("do_trade", False)
    poll_status = st.checkbox("注文状況を10秒ポーリング", key="poll_status")
    do_trade = st.checkbox("Alpacaで自動発注", key="do_trade")
    update_bp_after = st.checkbox("注文後に余力を自動更新", value=True, key="update_bp_after")

    # 注文状況を10秒ポーリングとは？
    # → Alpacaに注文を送信した後、注文IDのステータス（filled, canceled等）を10秒間、
    #    1秒ごとに取得・表示する機能です。
    # これにより、注文が約定したかどうかをリアルタイムで確認できます。

    # キャッシュクリアボタン
    if st.button("キャッシュクリア"):
        st.cache_data.clear()
        st.success("キャッシュをクリアしました")

    if st.button("全注文キャンセル"):
        try:
            client = ba.get_client(paper=paper_mode)
            ba.cancel_all_orders(client)
            st.success("すべての未約定注文をキャンセルしました")
        except Exception as e:
            st.error(f"注文キャンセルエラー: {e}")

    # 未約定注文の表示
    if st.button("未約定注文を表示"):
        try:
            client = ba.get_client(paper=paper_mode)
            try:
                # alpaca-py のAPIに合わせ、リクエストオブジェクトで指定
                # Use runtime importer to avoid hard dependency at static-analysis time
                req_mod = _import_alpaca_requests()
                _GetOrdersRequest = None
                if req_mod is not None:
                    _GetOrdersRequest = getattr(req_mod, "GetOrdersRequest", None)
                try:
                    if _GetOrdersRequest is not None:
                        orders = client.get_orders(filter=_GetOrdersRequest(status="open"))
                    else:
                        orders = client.get_orders()
                except Exception:
                    # フォールバック（古いSDKなど）
                    orders = client.get_orders()
            except Exception:
                # フォールバック（古いSDKなど）
                orders = client.get_orders()
            if not orders:
                st.info("未約定注文はありません。")
            else:
                rows = []
                for o in orders:
                    try:
                        rows.append(
                            {
                                "order_id": str(getattr(o, "id", "")),
                                "symbol": getattr(o, "symbol", None),
                                "side": getattr(o, "side", None),
                                "qty": getattr(o, "qty", None),
                                "status": getattr(o, "status", None),
                                "submitted_at": str(getattr(o, "submitted_at", "")),
                                "type": getattr(o, "type", None),
                                "limit_price": getattr(o, "limit_price", None),
                                "time_in_force": getattr(o, "time_in_force", None),
                            }
                        )
                    except Exception:
                        pass
                if rows:
                    import pandas as _pd

                    df_open = _pd.DataFrame(rows)
                    st.dataframe(df_open, use_container_width=True)
                    # 行ごとにキャンセルボタンを提供
                    st.caption("未約定注文の個別キャンセル")
                    for _i, r in df_open.iterrows():
                        col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 2])
                        with col1:
                            st.write(
                                f"{r.get('symbol')}  {r.get('side')}  qty={r.get('qty')}"
                            )  # noqa: E501
                        with col2:
                            st.write(f"status: {r.get('status')}")
                        with col3:
                            st.write(f"type: {r.get('type')}")
                        with col4:
                            st.write(f"limit: {r.get('limit_price')}")
                        with col5:
                            if st.button("キャンセル", key=f"cancel_{r.get('order_id')}"):
                                try:
                                    client.cancel_order(str(r.get("order_id")))
                                    st.success(f"キャンセルしました: {r.get('order_id')}")
                                except Exception as _e:
                                    st.error(f"キャンセル失敗: {_e}")
                                # 最新のopen ordersを再取得
                                try:
                                    # re-fetch open orders; use runtime importer
                                    req_mod = _import_alpaca_requests()
                                    _GetOrdersRequest = None
                                    if req_mod is not None:
                                        _GetOrdersRequest = getattr(
                                            req_mod, "GetOrdersRequest", None
                                        )
                                    try:
                                        if _GetOrdersRequest is not None:
                                            orders2 = client.get_orders(
                                                filter=_GetOrdersRequest(status="open")
                                            )
                                        else:
                                            orders2 = client.get_orders()
                                    except Exception:
                                        orders2 = client.get_orders()
                                except Exception:
                                    orders2 = client.get_orders()
                                rows2 = []
                                for o2 in orders2:
                                    try:
                                        rows2.append(
                                            {
                                                "order_id": str(getattr(o2, "id", "")),
                                                "symbol": getattr(o2, "symbol", None),
                                                "side": getattr(o2, "side", None),
                                                "qty": getattr(o2, "qty", None),
                                                "status": getattr(o2, "status", None),
                                                "submitted_at": str(
                                                    getattr(o2, "submitted_at", "")
                                                ),
                                                "type": getattr(o2, "type", None),
                                                "limit_price": getattr(o2, "limit_price", None),
                                                "time_in_force": getattr(o2, "time_in_force", None),
                                            }
                                        )
                                    except Exception:
                                        pass
                                if not rows2:
                                    st.info("未約定注文はありません。")
                                else:
                                    df2 = _pd.DataFrame(rows2)
                                    st.dataframe(df2, use_container_width=True)
        except Exception as e:
            st.error(f"未約定注文の取得に失敗: {e}")

    run_config = RunConfig(
        symbols=syms,
        capital_long=float(st.session_state["today_cap_long"]),
        capital_short=float(st.session_state["today_cap_short"]),
        save_csv=bool(save_csv),
        csv_name_mode=str(csv_name_mode),
        notify=bool(use_slack_notify),
        run_parallel=bool(run_parallel),
        scan_missing_only=bool(scan_missing_only),
    )
    trade_options = TradeOptions(
        paper_mode=bool(paper_mode),
        retries=int(retries),
        delay=float(delay),
        poll_status=bool(poll_status),
        do_trade=bool(do_trade),
        update_bp_after=bool(update_bp_after),
    )

    # 表示制御は固定（チェックボックスは廃止）
    st.session_state["ui_vis"] = {
        "overall_progress": True,
        "per_system_progress": True,
        "data_load_progress_lines": True,
        "previous_results": True,
        "system_details": True,
    }

st.subheader("保有ポジションと利益保護判定")
if st.button("🔍 Alpacaから保有ポジション取得"):
    try:
        client = ba.get_client(paper=paper_mode)
        positions = client.get_all_positions()
        st.session_state["positions_df"] = evaluate_positions(positions)
        st.success("ポジションを取得しました")
    except Exception as e:
        st.error(f"ポジション取得エラー: {e}")

if "positions_df" in st.session_state:
    df_pos = st.session_state["positions_df"]
    if df_pos.empty:
        st.info("保有ポジションはありません。")
    else:
        try:
            summary_df = _build_position_summary_table(df_pos)
        except ValueError as exc:
            st.error(f"ポジションサマリーの集計に失敗しました: {exc}")
        else:
            if not summary_df.empty:
                st.caption("ポジションサマリー（件数）")
                st.dataframe(summary_df, use_container_width=True)
        df_disp = df_pos.copy()
        if "holding_days" in df_disp.columns:

            def _normalize_days(value: Any) -> int | None:
                try:
                    if value in ("", None):
                        return None
                except Exception:
                    return None
                try:
                    if pd.isna(value):
                        return None
                except Exception:
                    pass
                try:
                    return int(value)
                except Exception:
                    return None

            df_disp["holding_days"] = df_disp["holding_days"].apply(_normalize_days)
        numeric_cols = [
            "qty",
            "avg_entry_price",
            "current_price",
            "unrealized_pl",
            "unrealized_plpc_percent",
        ]
        for col in numeric_cols:
            if col in df_disp.columns:
                df_disp[col] = pd.to_numeric(df_disp[col], errors="coerce")
        if "unrealized_plpc_percent" in df_disp.columns:
            df_disp["unrealized_plpc_percent"] = df_disp["unrealized_plpc_percent"].round(2)
        rename_map = {
            "symbol": "銘柄",
            "system": "システム",
            "side": "サイド",
            "qty": "数量",
            "entry_date": "取得日",
            "holding_days": "保有日数",
            "avg_entry_price": "平均取得単価",
            "current_price": "現在値",
            "unrealized_pl": "含み損益",
            "unrealized_plpc_percent": "含み損益率(%)",
            "judgement": "判定",
            "next_action": "次のアクション目安",
            "rule_summary": "利確/損切りルール概要",
        }
        df_disp = df_disp.rename(columns=rename_map)
        display_cols = [
            "銘柄",
            "システム",
            "サイド",
            "数量",
            "取得日",
            "保有日数",
            "平均取得単価",
            "現在値",
            "含み損益",
            "含み損益率(%)",
            "判定",
            "次のアクション目安",
            "利確/損切りルール概要",
        ]
        df_disp = df_disp[[col for col in display_cols if col in df_disp.columns]]
        st.dataframe(df_disp, use_container_width=True)

if st.button("▶ 本日のシグナル実行", type="primary"):
    artifacts = execute_today_signals(run_config)
    render_today_signals_results(artifacts, run_config, trade_options)
else:
    _render_previous_results_section()
