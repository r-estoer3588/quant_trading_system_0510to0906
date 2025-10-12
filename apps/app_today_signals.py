from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone, tzinfo

# ruff: noqa: E402
# flake8: noqa: E402
import importlib
import json
import logging
import os
from pathlib import Path
import re
import sys
from threading import Lock
import time
from typing import TYPE_CHECKING, Any, cast
import uuid

try:
    from zoneinfo import ZoneInfo

    def get_zoneinfo(name: str) -> tzinfo:
        return ZoneInfo(name)

except ImportError:
    # Python < 3.9 or Windows without zoneinfo, use UTC as fallback
    def get_zoneinfo(name: str) -> tzinfo:
        _ = name
        return timezone.utc


import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

# ページ設定を最初に実行
st.set_page_config(page_title="本日のシグナル", layout="wide")

# sys.pathを正しく設定してからimport
try:
    # プロジェクトルートがsys.pathにない場合の事前処理
    project_root = Path(__file__).parent.parent.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # scriptsディレクトリも追加
    scripts_dir = project_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
except Exception:
    pass

from common import broker_alpaca as ba  # noqa: E402
from common.alpaca_order import submit_orders_df  # noqa: E402
from common.cache_format import round_dataframe  # noqa: E402
from common.cache_manager import CacheManager  # noqa: E402
from common.data_loader import load_price  # noqa: E402
from common.exit_planner import decide_exit_schedule  # noqa: E402
from common.notifier import create_notifier  # noqa: E402
from common.position_age import (  # noqa: E402
    fetch_entry_dates_from_alpaca,
    load_entry_dates,
    save_entry_dates,
)
from common.profit_protection import evaluate_positions  # noqa: E402
from common.stage_metrics import (  # noqa: E402
    DEFAULT_SYSTEM_ORDER,
    GLOBAL_STAGE_METRICS,
    StageMetricsStore,
    StageSnapshot,
)
from common.system_groups import format_group_counts, format_group_counts_and_values  # noqa: E402
from common.today_signals import LONG_SYSTEMS, SHORT_SYSTEMS  # noqa: E402
from common.today_signals import run_all_systems_today as compute_today_signals  # noqa: E402
from common.utils_spy import (  # noqa: E402
    calculate_trading_days_lag,
    describe_trading_gap,
    get_latest_nyse_trading_day,
    get_signal_target_trading_day,
)
from config.settings import get_settings  # noqa: E402
from core.system1 import summarize_system1_diagnostics  # noqa: E402
from strategies.system1_strategy import System1Strategy  # noqa: E402
from strategies.system2_strategy import System2Strategy  # noqa: E402
from strategies.system3_strategy import System3Strategy  # noqa: E402
from strategies.system4_strategy import System4Strategy  # noqa: E402
from strategies.system5_strategy import System5Strategy  # noqa: E402
from strategies.system6_strategy import System6Strategy  # noqa: E402

# 条件付きインポート - alpaca.trading.requests は実行時のみ必要
AlpacaTradingRequests: Any | None = None


def _import_alpaca_requests():
    """Runtime-safe importer for `alpaca.trading.requests`.

    Returns the module or None if not importable.
    """
    try:
        return importlib.import_module("alpaca.trading.requests")
    except ImportError:
        return None


# 実行時にインポートを試みる
if not TYPE_CHECKING:
    AlpacaTradingRequests = _import_alpaca_requests()


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

# manual_rebuild ログ集約用のモジュール変数（解析器対策として事前定義）
_MANUAL_REBUILD_VERBOSE_LIMIT: int | None = None
_MANUAL_REBUILD_VERBOSE_COUNT: int = 0
_MANUAL_REBUILD_SUPPRESSED: int = 0
_MANUAL_REBUILD_ATEXIT_REGISTERED: bool = False
_MANUAL_REBUILD_AGG = None

if not _IS_STREAMLIT_RUNTIME:
    if __name__ == "__main__":
        print(
            "このスクリプトはStreamlitで実行してください: "
            "`streamlit run apps/dashboards/app_today_signals.py`"
        )
        raise SystemExit

try:
    # Streamlit の実行コンテキスト有無を判定（スレッド外からの UI 呼び出しを防ぐ）
    def _has_st_ctx() -> bool:
        if not _IS_STREAMLIT_RUNTIME:
            return False
        try:
            return get_script_run_ctx() is not None
        except Exception:
            return False

except Exception:

    def _has_st_ctx() -> bool:
        return _IS_STREAMLIT_RUNTIME


# Streamlit checkbox の重複ID対策（key未指定時に自動で一意キーを付与）
try:
    # モジュール属性を安全に処理
    original_checkbox = getattr(st, "checkbox", None)

    if original_checkbox is not None and callable(original_checkbox):

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
            # 念のため呼び出し前に再度チェック
            if callable(original_checkbox):
                return original_checkbox(label, *args, **kwargs)
            else:
                # フォールバック: 元の関数を直接呼び出し
                return st.checkbox(label, *args, **kwargs)

        # 元のチェックボックスを保存して新しい関数を設定
        setattr(st, "_orig_checkbox", original_checkbox)
        setattr(st, "checkbox", _unique_checkbox)
except Exception:
    # 失敗しても従来動作のまま進める
    pass

st.title("📈 本日のシグナル（全システム）")

settings = get_settings(create_dirs=True)
notifier = create_notifier(platform="slack", fallback=True)
# この実行ループで結果を表示したかのフラグ（保存ボタン等でのリラン対策）
st.session_state.setdefault("today_shown_this_run", False)


def _render_freshness_panel() -> None:
    try:
        # 基準日（前営業日）と SPY キャッシュ最終日を推定
        base_day = get_latest_nyse_trading_day(
            get_signal_target_trading_day() - pd.Timedelta(days=1)
        )
        # SPY キャッシュ最終日は utils_spy 側の軽量表示に依存せず、CacheManager経由で取得
        cm = CacheManager(settings)
        spy_df = cm.read("SPY", profile="rolling") or cm.read("SPY", profile="full")
        last_cache = None
        if isinstance(spy_df, pd.DataFrame) and not spy_df.empty:
            try:
                if "Date" in spy_df.columns:
                    last_cache = pd.to_datetime(spy_df["Date"], errors="coerce").max()
                elif "date" in spy_df.columns:
                    last_cache = pd.to_datetime(spy_df["date"], errors="coerce").max()
                else:
                    last_cache = pd.to_datetime(spy_df.index, errors="coerce").max()
            except Exception:
                last_cache = None
        # 許容営業日数（設定値）
        try:
            allowed = int(settings.cache.rolling.max_staleness_days)
        except Exception:
            allowed = 2

        col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 2.4])
        with col1:
            st.caption("基準日（前営業日）")
            st.write(str(pd.Timestamp(base_day).date()))
        with col2:
            st.caption("SPYキャッシュ最終日")
            st.write("—" if last_cache is None else str(pd.Timestamp(last_cache).date()))
        with col3:
            st.caption("営業日差")
            if last_cache is None:
                st.write("—")
                lag_days = None
            else:
                lag_days = calculate_trading_days_lag(
                    pd.Timestamp(last_cache), pd.Timestamp(base_day)
                )
                st.write(f"{lag_days} 日")
        with col4:
            st.caption("許容営業日数 / 理由")
            if last_cache is None:
                st.write("—")
            else:
                reason = describe_trading_gap(pd.Timestamp(last_cache), pd.Timestamp(base_day))
                st.write(f"{allowed} 日 / {reason}")
        st.divider()
    except Exception:
        # UIは失敗しても致命的でない
        pass


# 先頭に鮮度パネルを表示（安全な try/except 内）
try:
    _render_freshness_panel()
except Exception:
    pass


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
            {str(v) for v in work.loc[invalid_side_mask, "side"].values.tolist()}
        )  # noqa: E501
        raise ValueError(f"未対応のsideが含まれています: {invalid_values}")

    invalid_system_mask = work["system_norm"].isna()
    if invalid_system_mask.any():
        invalid_values = sorted(
            {str(v) for v in work.loc[invalid_system_mask, "system"].values.tolist()}
        )  # noqa: E501
        raise ValueError(f"未対応のsystemが含まれています: {invalid_values}")

    long_conflict_mask = (work["side_norm"] == "long") & (
        ~work["system_norm"].isin(LONG_SYSTEMS)
    )  # noqa: E501
    if long_conflict_mask.any():
        conflict = sorted({str(v) for v in work.loc[long_conflict_mask, "system"].values.tolist()})
        raise ValueError(f"Longサイドに想定外のsystemが含まれています: {conflict}")

    short_conflict_mask = (work["side_norm"] == "short") & (
        ~work["system_norm"].isin(SHORT_SYSTEMS)
    )
    if short_conflict_mask.any():
        conflict = sorted({str(v) for v in work.loc[short_conflict_mask, "system"].values.tolist()})
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
        work.columns = pd.Index([str(col) for col in work.columns])  # type: ignore[assignment]
    except Exception:
        work = pd.DataFrame(work)
        work.columns = pd.Index([str(col) for col in work.columns])  # type: ignore[assignment]

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
        work.columns = pd.Index([str(col).lower() for col in work.columns])  # type: ignore[assignment]
    except Exception:
        work.columns = pd.Index([str(col) for col in work.columns])  # type: ignore[assignment]

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
    "return_6d",
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
    "return_6d": 6,
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
            logger = logging.getLogger("today_signals")
            logger.info("lookback未満でスキップされた列は%d件でした", skipped_lookback_count)
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


def _build_manual_rebuild_message(symbol: str, detail: dict[str, Any]) -> str:
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
    message += " （自動スキップ済み）"
    return message


def _log_manual_rebuild_notice(
    symbol: str,
    detail: dict[str, Any],
    log_fn: Callable[[str], None] | None = None,
) -> str:
    """rolling未整備メッセージを出力。

    COMPACT_TODAY_LOGS=1 の場合:
        - 旧仕様: 銘柄ごとに "⛔ rolling未整備: SYMBOL (...) （自動スキップ済み）" を逐次出力し大量に冗長化
        - 新仕様: `common.cache_warnings.RollingIssueAggregator` へカテゴリ manual_rebuild として集約
            * 先頭 N 件 (ROLLING_ISSUES_VERBOSE_HEAD, 既定=5) のみ WARNING
            * 以降は DEBUG にダウングレード（ログ量削減）
            * 集約サマリーは他カテゴリと同じ仕組みで INFO 出力
    COMPACT_TODAY_LOGS!=1 の場合は従来通り全文を log_fn へ出力する。
    """
    message = _build_manual_rebuild_message(symbol, detail)

    # 既定で銘柄ごとの詳細ログは抑制し（過去指示: "1銘柄ごとに出さなくて良い"）
    # 明示的に詳細を見たい場合のみ ROLLING_MANUAL_REBUILD_VERBOSE=1 を設定。
    # 互換のため COMPACT_TODAY_LOGS=1 も引き続き抑制扱い。
    verbose_flag = os.getenv("ROLLING_MANUAL_REBUILD_VERBOSE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    suppress_default_flag = os.getenv(
        "ROLLING_MANUAL_REBUILD_SUPPRESS_PER_SYMBOL", "0"
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    # compact_mode: True => 集約（per-symbolログ抑制）
    # 既定: COMPACT_TODAY_LOGS=1 のとき抑制。明示 suppress 環境変数で強制抑制。
    # ROLLING_MANUAL_REBUILD_VERBOSE=1 が指定されれば、COMPACT でも詳細を出す。
    if suppress_default_flag:
        compact_mode = True
    else:
        compact_mode = (os.getenv("COMPACT_TODAY_LOGS") == "1") and (not verbose_flag)

    # compact モード時は既存 aggregator + 共通 aggregator の二段構え
    if compact_mode:
        try:
            from common.cache_warnings import (  # 遅延 import
                get_rolling_issue_aggregator,
                report_rolling_issue,
            )

            status = str(detail.get("status") or "manual_rebuild")
            agg = get_rolling_issue_aggregator()
            # 既に manual_rebuild か missing_rolling で報告済みなら重複出力を抑止
            if not (
                agg.has_issue("manual_rebuild", symbol) or agg.has_issue("missing_rolling", symbol)
            ):
                report_rolling_issue("manual_rebuild", symbol, status)
            # 共通 aggregator: 簡易ローカル実装（存在しない依存を避ける）
            try:

                class _LocalIssueAgg:
                    def __init__(self) -> None:
                        self.items: set[tuple[str, str]] = set()

                    def add(self, sym: str, st: str) -> None:
                        try:
                            self.items.add((str(sym), str(st)))
                        except Exception:
                            pass

                global _MANUAL_REBUILD_AGG
                if "_MANUAL_REBUILD_AGG" not in globals():
                    _MANUAL_REBUILD_AGG = _LocalIssueAgg()
                if (not agg.has_issue("manual_rebuild", symbol)) and (
                    _MANUAL_REBUILD_AGG is not None
                ):
                    try:
                        _MANUAL_REBUILD_AGG.add(symbol, status)
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception:
            if log_fn:
                try:
                    log_fn(message)
                except Exception:
                    pass
        return message

    # 非コンパクトモード: 大量発生時は環境変数で抑制
    # ROLLING_MANUAL_REBUILD_VERBOSE_LIMIT: 0 または未設定=無制限, N>0 で最初の N 件のみ詳細出力し残りはサマリーへ集約
    global _MANUAL_REBUILD_VERBOSE_LIMIT, _MANUAL_REBUILD_VERBOSE_COUNT
    global _MANUAL_REBUILD_SUPPRESSED, _MANUAL_REBUILD_ATEXIT_REGISTERED
    try:  # 初期化 (例外あっても致命的でない)
        if "_MANUAL_REBUILD_VERBOSE_LIMIT" not in globals():  # 初回
            _MANUAL_REBUILD_VERBOSE_LIMIT = None
            _MANUAL_REBUILD_VERBOSE_COUNT = 0
            _MANUAL_REBUILD_SUPPRESSED = 0
            _MANUAL_REBUILD_ATEXIT_REGISTERED = False
        if _MANUAL_REBUILD_VERBOSE_LIMIT is None:
            import atexit as _atexit
            import os as _os

            try:
                _MANUAL_REBUILD_VERBOSE_LIMIT = int(
                    _os.getenv("ROLLING_MANUAL_REBUILD_VERBOSE_LIMIT", "0")
                )
            except Exception:
                _MANUAL_REBUILD_VERBOSE_LIMIT = 0

            def _flush_manual_rebuild_summary() -> None:  # atexit フラッシュ
                try:
                    limit_val = _MANUAL_REBUILD_VERBOSE_LIMIT or 0
                    if _MANUAL_REBUILD_SUPPRESSED > 0 and limit_val > 0:
                        # 抑制件数の最終サマリー (WARNING でなく INFO 相当が妥当だが log_fn のレベル制御不明なのでそのまま)
                        if log_fn:
                            # 参考として missing_rolling 件数を括弧追加（既報カテゴリの全体感）
                            try:
                                from common.cache_warnings import get_rolling_issue_aggregator

                                _agg_summary = get_rolling_issue_aggregator()
                                _issues_map = getattr(_agg_summary, "_issues", {})
                                _missing_list = _issues_map.get("missing_rolling", [])
                                missing_cnt = len(_missing_list)
                            except Exception:
                                missing_cnt = 0
                            extra = f" missing_rolling:{missing_cnt}件" if missing_cnt else ""
                            log_fn(
                                (
                                    "💡 rolling未整備 追加"
                                    f"{_MANUAL_REBUILD_SUPPRESSED}件 "
                                    f"(閾値{limit_val}超過分) は省略されました"
                                    f"{extra}"
                                )
                            )
                except Exception:
                    pass

            if not _MANUAL_REBUILD_ATEXIT_REGISTERED:
                try:
                    _atexit.register(_flush_manual_rebuild_summary)
                    _MANUAL_REBUILD_ATEXIT_REGISTERED = True
                except Exception:
                    pass

        _MANUAL_REBUILD_VERBOSE_COUNT += 1
        limit = _MANUAL_REBUILD_VERBOSE_LIMIT or 0
        if limit > 0 and _MANUAL_REBUILD_VERBOSE_COUNT > limit:
            _MANUAL_REBUILD_SUPPRESSED += 1
            # 最初の抑制タイミングで 1 度だけ告知行
            if _MANUAL_REBUILD_SUPPRESSED == 1 and log_fn:
                try:
                    log_fn(
                        (
                            "… (以降 rolling未整備 詳細は抑制中: "
                            f"閾値{limit}件を超過。環境変数 "
                            "ROLLING_MANUAL_REBUILD_VERBOSE_LIMIT で変更可能)"
                        )
                    )
                except Exception:
                    pass
            return message  # 呼び出し元には返すが表示しない
    except Exception:  # 失敗時は従来挙動
        pass

    if log_fn is None:
        return message
    try:
        log_fn(message)
    except Exception:
        pass
    return message


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

    try:
        env_parallel = (os.environ.get("TODAY_PREFETCH_PARALLEL") or "").strip().lower()
    except Exception:
        env_parallel = ""
    try:
        env_threshold = int(os.environ.get("TODAY_PREFETCH_PARALLEL_THRESHOLD", "200"))
    except Exception:
        env_threshold = 200

    if env_parallel in {"1", "true", "yes"}:
        use_parallel = total > 1
    elif env_parallel in {"0", "false", "no"}:
        use_parallel = False
    else:
        use_parallel = total >= max(0, env_threshold)

    max_workers: int | None = None
    if use_parallel:
        try:
            env_workers_raw = (os.environ.get("TODAY_PREFETCH_MAX_WORKERS") or "").strip()
            if env_workers_raw:
                max_workers = int(env_workers_raw)
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
        max_workers = max(1, min(int(max_workers), total))
        if log_fn:
            try:
                log_fn(f"🧵 基礎データロード(事前チェック)並列化: workers={max_workers}")
            except Exception:
                pass

    data_lock = Lock()
    missing_lock = Lock()
    malformed_lock = Lock()
    progress_lock = Lock()
    processed = 0
    # 進捗の表示間隔（デフォルト total/20）。環境変数 TODAY_PROGRESS_STEP で上書き可。
    try:
        _env_step_raw = (os.environ.get("TODAY_PROGRESS_STEP") or "").strip()
        step = int(_env_step_raw) if _env_step_raw else max(1, total // 20)
    except Exception:
        step = max(1, total // 20)

    def _emit_progress(current: int) -> None:
        if log_fn is None:
            return
        if current % step != 0 and current != total:
            return
        try:
            elapsed = int(max(0, time.time() - start_ts))
            minutes, seconds = divmod(elapsed, 60)
            # 表示オプション
            use_thousands = os.environ.get("TODAY_PROGRESS_THOUSANDS") == "1"
            style = (os.environ.get("TODAY_PROGRESS_STYLE") or "both").lower()
            # 桁数揺れを避けるため固定幅で整形
            if use_thousands:
                tot_txt = f"{total:,}"
                cur_txt = f"{current:,}"
                w = max(1, len(tot_txt))
                cur_s = f"{cur_txt:>{w}s}"
                tot_s = f"{tot_txt:>{w}s}"
            else:
                w = max(1, len(str(total)))
                cur_s = f"{current:>{w}d}"
                tot_s = f"{total:>{w}d}"
            mm = f"{minutes:02d}"
            ss = f"{seconds:02d}"
            # ETA（単純推定）
            eta_txt = None
            if current > 0:
                try:
                    rate = elapsed / current if current > 0 else 0.0
                    remain = max(0, total - current)
                    eta = int(remain * rate)
                    em, es = divmod(eta, 60)
                    eta_txt = f"{em:02d}分{es:02d}秒"
                except Exception:
                    eta_txt = None
            # スタイル選択
            if style == "elapsed":
                tail = f"経過 {mm}分{ss}秒"
            elif style == "eta" and eta_txt is not None:
                tail = f"ETA {eta_txt}"
            else:
                tail = f"経過 {mm}分{ss}秒" + (f" | ETA {eta_txt}" if eta_txt else "")
            log_fn(f"📦 基礎データロード進捗: {cur_s}/{tot_s} | {tail}")
        except Exception:
            try:
                use_thousands = os.environ.get("TODAY_PROGRESS_THOUSANDS") == "1"
                if use_thousands:
                    tot_txt = f"{total:,}"
                    cur_txt = f"{current:,}"
                    w = max(1, len(tot_txt))
                    cur_s = f"{cur_txt:>{w}s}"
                    tot_s = f"{tot_txt:>{w}s}"
                else:
                    w = max(1, len(str(total)))
                    cur_s = f"{current:>{w}d}"
                    tot_s = f"{total:>{w}d}"
                log_fn(f"📦 基礎データロード進捗: {cur_s}/{tot_s}")
            except Exception:
                pass

    def _process_symbol(
        sym: str,
    ) -> tuple[str, pd.DataFrame | None, dict[str, Any] | None, str | None, bool]:
        manual_msg: str | None = None
        detail: dict[str, Any] | None = None
        malformed_flag = False
        try:
            df = load_price(sym, cache_profile="rolling")
        except Exception:
            df = None
        rows_before = 0 if df is None else int(len(df))
        ok, issues = _analyze_rolling_cache(df)
        if not ok:
            detail = _build_missing_detail(sym, issues, rows_before)
            if debug_scan:
                detail["action"] = "debug_scan"
                detail["note"] = _issues_to_note(issues)
                return sym, None, detail, None, False
            detail["action"] = "manual_rebuild_required"
            manual_note = _merge_note(
                _issues_to_note(issues),
                "自動スキップ",
            )
            detail["note"] = manual_note
            manual_msg = _build_manual_rebuild_message(sym, detail)
            return sym, None, detail, manual_msg, False

        if df is None:
            malformed_flag = True
            return sym, None, None, None, malformed_flag

        norm = _normalize_price_history(df, rows)
        if norm is not None and not norm.empty:
            return sym, norm, None, None, False
        malformed_flag = True
        return sym, None, None, None, malformed_flag

    def _handle_result(
        result: tuple[str, pd.DataFrame | None, dict[str, Any] | None, str | None, bool],
    ) -> None:
        nonlocal processed
        sym, norm, detail, manual_msg, malformed_flag = result
        if norm is not None and not getattr(norm, "empty", True):
            with data_lock:
                fetched[sym] = norm
        elif malformed_flag:
            with malformed_lock:
                malformed.append(sym)
        if detail is not None:
            with missing_lock:
                missing_details.append(detail)
        # per-symbol の "⛔ rolling未整備" は既定で抑制し、必要時のみ詳細表示。
        # 直接ログ出力せず、専用関数で集約・抑制ロジックを適用する。
        if (
            detail
            and log_fn
            and not debug_scan
            and detail.get("action") == "manual_rebuild_required"
        ):
            try:
                _log_manual_rebuild_notice(sym, detail, log_fn=log_fn)
            except Exception:
                # フォールバックとして元のメッセージを出せる場合のみ最小限で出力
                if manual_msg:
                    try:
                        log_fn(manual_msg)
                    except Exception:
                        pass
        with progress_lock:
            processed += 1
            _emit_progress(processed)

    if use_parallel and max_workers and total > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_symbol, sym): sym for sym in symbols}
            for fut in as_completed(futures):
                sym = futures[fut]
                try:
                    result = fut.result()
                except Exception:
                    result = (sym, None, None, None, True)
                _handle_result(result)
    else:
        for sym in symbols:
            result = _process_symbol(sym)
            _handle_result(result)

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
            # より詳細な状況説明を追加
            new_listings = [
                s for s in manual_symbols if len(s) <= 4 and s.isalpha()
            ]  # 新規上場の可能性
            try:
                base_msg = (
                    "⚠️ rolling未整備: "
                    f"{len(manual_symbols)}銘柄 → 手動でキャッシュを更新してください | 例: {sample}"
                )
                if new_listings:
                    base_msg += f" (新規上場含む可能性: {len(new_listings)}件)"
                log_fn(base_msg)
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
    # 動的インポートでエラーを回避
    try:
        import scripts.run_all_systems_today as _run_today_mod

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
                jst_now = datetime.now(get_zoneinfo("Asia/Tokyo"))
            except Exception:
                jst_now = datetime.now(get_zoneinfo("UTC"))
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
        # progress_textは削除（タイトルで表示するため）
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
        # プログレスバー下のテキストは削除（タイトルで表示するため）
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
        # 最後に受け取ったステージ情報のデデュープ用タイムスタンプ
        self._last_event: dict[str, tuple[int, int, int, int, int, float]] = {}
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
        progress_bar = self.bars.get(key)
        if progress_bar is None:
            return

        # フェーズに応じた適切な値を設定
        if phase == "start":
            # 開始時は0%から開始（リセット）
            value = 0
            self.states[key] = 0  # 状態もリセット
        elif phase == "done":
            value = 100
        else:
            # その他のフェーズでは実際の進捗値を取得
            try:
                snapshot = GLOBAL_STAGE_METRICS.get_snapshot(key)
                if snapshot is not None:
                    value = snapshot.progress
                else:
                    value = self.states.get(key, 0)
            except Exception:
                value = self.states.get(key, 0)

        value = max(0, min(100, int(value)))

        # 通常時は進捗後退を防ぐが、開始時（phase="start"）はリセットを許可
        if phase != "start":
            prev = int(self.states.get(key, 0))
            value = max(prev, value)

        self.states[key] = value

        try:
            progress_bar.progress(value)
            self.stage_txt[key].text(f"run {value}%" if value < 100 else "done 100%")
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

        # ターゲット数の設定（優先度順で設定）
        if snapshot.target is not None:
            try:
                target_val = int(snapshot.target)
                counts["target"] = target_val
                self.universe_total = target_val
            except Exception:
                pass
        elif snapshot.filter_pass is not None and counts.get("target") is None:
            try:
                fallback_target = int(snapshot.filter_pass)
                counts["target"] = fallback_target
                if self.universe_total is None:
                    self.universe_total = fallback_target
            except Exception:
                pass

        # 進捗データの設定
        if snapshot.filter_pass is not None:
            try:
                counts["filter"] = int(snapshot.filter_pass)
                # フィルター通過数が設定されている場合、ターゲットがなければフィルター数をターゲットとして使用
                if counts.get("target") is None:
                    counts["target"] = int(snapshot.filter_pass)
                    if self.universe_total is None:
                        self.universe_total = int(snapshot.filter_pass)
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
        # 短時間内に同一内容の更新が来ると UI がフラッタリングするため、
        # 同一システム・同一値・同一カウントの更新は 0.5 秒以内は無視する。
        try:
            last = self._last_event.get(key)
            cur_sig = (
                value,
                int(filter_cnt) if filter_cnt is not None else -1,
                int(setup_cnt) if setup_cnt is not None else -1,
                int(cand_cnt) if cand_cnt is not None else -1,
                int(final_cnt) if final_cnt is not None else -1,
                time.time(),
            )
            if last is not None:
                same = last[0:5] == cur_sig[0:5]
                recent = (cur_sig[5] - last[5]) < 0.5
                if same and recent:
                    return
            self._last_event[key] = cur_sig
        except Exception:
            pass
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
                filter_val = int(filter_cnt)
            except Exception:
                filter_val = None
            if filter_val is not None:
                if value == 0:
                    counts["target"] = filter_val
                    self.universe_total = filter_val
                else:
                    counts["filter"] = filter_val
                    if counts.get("target") is None:
                        counts["target"] = (
                            self.universe_total if self.universe_total is not None else filter_val
                        )
                        if self.universe_total is None:
                            self.universe_total = filter_val
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
                self.universe_total = None
            else:
                self.universe_target = int(tgt)
                self.universe_total = int(tgt)
            GLOBAL_STAGE_METRICS.set_universe_target(self.universe_target)
        except Exception:
            self.universe_target = None
            self.universe_total = None
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
        """最終化：残った候補/エントリー数を補完し、全バーを100%にする。"""
        # AllocationSummary が dict で同梱されている場合、slot_candidates を候補数のフォールバックに使用する
        alloc_slot_candidates: dict[str, int] | None = None
        alloc_final_counts: dict[str, int] | None = None
        system_diagnostics_map: dict[str, dict] | None = None

        try:
            if isinstance(per_system, dict):
                alloc_dict = per_system.get("__allocation_summary__")
            else:
                alloc_dict = None
            if isinstance(alloc_dict, dict):
                # slot_candidates 取得
                cand_map = alloc_dict.get("slot_candidates")
                if isinstance(cand_map, dict):
                    # 正規化: keyは小文字system名に統一し、値はint化
                    alloc_slot_candidates = {}
                    for k, v in cand_map.items():
                        try:
                            key = str(k).strip().lower()
                            val = int(v) if v is not None else 0
                            alloc_slot_candidates[key] = max(0, val)
                        except Exception:
                            continue
                # final_counts 取得（エントリー数の最終確定値）
                final_map = alloc_dict.get("final_counts")
                if isinstance(final_map, dict):
                    alloc_final_counts = {}
                    for k, v in final_map.items():
                        try:
                            key = str(k).strip().lower()
                            val = int(v) if v is not None else 0
                            alloc_final_counts[key] = max(0, val)
                        except Exception:
                            continue
                # system_diagnostics 取得（setup_predicate_count用）
                diag_map = alloc_dict.get("system_diagnostics")
                if isinstance(diag_map, dict):
                    system_diagnostics_map = {}
                    for k, v in diag_map.items():
                        try:
                            key = str(k).strip().lower()
                            if isinstance(v, dict):
                                system_diagnostics_map[key] = v
                        except Exception:
                            continue
        except Exception:
            alloc_slot_candidates = None
            alloc_final_counts = None
            system_diagnostics_map = None
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
            # diagnosticsからsetup_predicate_countを取得して設定
            if counts.get("setup") is None and system_diagnostics_map:
                try:
                    diag = system_diagnostics_map.get(name)
                    if isinstance(diag, dict):
                        setup_count = diag.get("setup_predicate_count")
                        if isinstance(setup_count, (int, float)) and setup_count >= 0:
                            counts["setup"] = int(setup_count)
                except Exception:
                    pass

            if counts.get("cand") is None:
                # 1) AllocationSummary の slot_candidates からフォールバック
                used = False
                try:
                    if alloc_slot_candidates is not None and name in alloc_slot_candidates:
                        counts["cand"] = self._clamp_trdlist(alloc_slot_candidates.get(name))
                        used = True
                except Exception:
                    used = False
                # 2) per_system の DataFrame 件数にフォールバック
                if not used:
                    df_sys = per_system.get(name)
                    if df_sys is None or not isinstance(df_sys, pd.DataFrame) or df_sys.empty:
                        counts["cand"] = 0
                    else:
                        counts["cand"] = self._clamp_trdlist(len(df_sys))
            if counts.get("entry") is None and not system_series.empty:
                try:
                    counts["entry"] = int((system_series == name).sum())
                except Exception:
                    counts["entry"] = 0
            # AllocationSummary.final_counts からの上書き（優先）
            if alloc_final_counts and name in alloc_final_counts:
                try:
                    counts["entry"] = int(alloc_final_counts[name])
                except Exception:
                    pass
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
        progress_bar = self.bars.get(key)
        if progress_bar is None:
            return
        vv = max(0, min(100, int(value)))
        prev = int(self.states.get(key, 0))
        # 進捗が 0/25/50/75 のままでも、下流ステージのカウントが埋まっている場合は補完
        if vv < 100:
            try:
                snap = GLOBAL_STAGE_METRICS.get_snapshot(key)
            except Exception:
                snap = None
            if snap is not None:
                # entry_count が存在 → 75% 以上完了とみなし 100 に丸め
                if snap.entry_count is not None:
                    vv = 100
                # candidate_count のみ → 75%
                elif snap.candidate_count is not None and vv < 75:
                    vv = 75
                # setup_pass があり filter_pass も → 50%
                elif snap.setup_pass is not None and snap.filter_pass is not None and vv < 50:
                    vv = 50
                # filter_pass のみ存在 → 25%
                elif snap.filter_pass is not None and vv < 25:
                    vv = 25
        vv = max(prev, vv)
        self.states[key] = vv
        try:
            progress_bar.progress(vv)
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
        result = self.metrics_store.get_display_metrics(key)
        return cast(dict[str, int | None], result)

    def _ensure_counts(self, name: str) -> dict[str, int | None]:
        result = self.metrics_store.ensure_display_metrics(name)
        return cast(dict[str, int | None], result)

    @staticmethod
    def _format_value(value: Any) -> str:
        """Format value as string, returning '-' for None."""
        result: str = "-" if value is None else str(value)
        return result

    @staticmethod
    def _clamp_trdlist(value: Any) -> int | None:
        result = StageMetricsStore.clamp_trdlist(value)
        return cast(int | None, result)

    def _format_trdlist(self, value: Any) -> str:
        """Format trdlist value with clamping, returning '-' for None."""
        if value is None:
            clamped_str: str = "-"
            return clamped_str
        try:
            clamped_val = self._clamp_trdlist(value)
            return str(clamped_val) if clamped_val is not None else "-"
        except Exception:
            return "-"


class UILogger:
    """UIとファイル出力の両方へログを書き出す。"""

    def __init__(self, start_time: float, progress_ui: ProgressUI):
        self.start_time = start_time
        self.progress_ui = progress_ui
        self.log_lines: list[str] = []
        # ログデデュープ用（短時間に同一メッセージが来たら抑止）
        self._last_log: dict[str, float] = {}

    def log(self, msg: str, no_timestamp: bool = False) -> None:
        forwarded_from_cli = False
        try:
            import scripts.run_all_systems_today as _run_today_mod

            forwarding_flag = getattr(_run_today_mod, "_LOG_FORWARDING", None)
            if forwarding_flag is not None:
                forwarded_from_cli = bool(forwarding_flag.get())
        except Exception:
            forwarded_from_cli = False
        structured_mode = False
        parsed_msg: str | None = None
        iso_ts: str | None = None
        rel_prefix: str | None = None
        if not no_timestamp:
            # STRUCTURED_UI_LOGS=1 のとき、エンジン側から渡される JSON 形式を優先的に解釈
            if os.environ.get("STRUCTURED_UI_LOGS") == "1":
                try:
                    import json as _json

                    if isinstance(msg, str) and msg.startswith("{") and '"msg"' in msg:
                        obj = _json.loads(msg)
                        # 最低限 'msg' があること
                        raw_inner = obj.get("msg")
                        if isinstance(raw_inner, str):
                            structured_mode = True
                            parsed_msg = raw_inner
                            # ISO 時刻
                            iso_candidate = obj.get("iso")
                            if isinstance(iso_candidate, str):
                                iso_ts = iso_candidate
                            # 相対時間（エポックを start_time との差分で計算）
                            ts_val = obj.get("ts")
                            if isinstance(ts_val, (int, float)):
                                try:
                                    rel_elapsed = max(0, (ts_val / 1000.0) - self.start_time)
                                    mm, ss = divmod(int(rel_elapsed), 60)
                                    rel_prefix = f"{mm:02d}分{ss:02d}秒"
                                except Exception:
                                    pass
                except Exception:
                    structured_mode = False

        def _format_rel_compact(elapsed: float) -> str:
            try:
                if elapsed < 0:
                    elapsed = 0.0
                if elapsed < 1:
                    return f"+{int(elapsed * 1000)}ms"
                if elapsed < 60:
                    return f"+{elapsed:.1f}s"
                if elapsed < 3600:
                    m, s = divmod(int(elapsed), 60)
                    return f"+{m}:{s:02d}"
                if elapsed < 86400:
                    h, rem = divmod(int(elapsed), 3600)
                    m, s = divmod(rem, 60)
                    return f"+{h}h{m:02d}m"  # 秒は省略
                d, rem = divmod(int(elapsed), 86400)
                h, _ = divmod(rem, 3600)
                return f"+{d}d{h}h"
            except Exception:
                return "+0.0s"

        compact_mode = os.environ.get("COMPACT_REL_TIME") == "1"

        if structured_mode and parsed_msg is not None:
            # ISO or 現在時刻 fallback
            if iso_ts is None:
                iso_ts = time.strftime("%Y-%m-%d %H:%M:%S")
            if rel_prefix is None:
                try:
                    _elapsed = max(0, time.time() - self.start_time)
                    if compact_mode:
                        rel_prefix = _format_rel_compact(_elapsed)
                    else:
                        mm, ss = divmod(int(_elapsed), 60)
                        rel_prefix = f"{mm:02d}分{ss:02d}秒"
                except Exception:
                    rel_prefix = "0分0秒"
            line = f"[{iso_ts} | {rel_prefix}] {parsed_msg}"
        else:
            try:
                elapsed = max(0, time.time() - self.start_time)
                if compact_mode:
                    rel_prefix = _format_rel_compact(elapsed)
                else:
                    m, s = divmod(int(elapsed), 60)
            except Exception:
                rel_prefix = "0分0秒" if not compact_mode else "+0.0s"
            now_txt = time.strftime("%Y-%m-%d %H:%M:%S")
            if no_timestamp:
                line = str(msg)
            else:
                if compact_mode:
                    if not rel_prefix:
                        rel_prefix = "+0.0s"
                    line = f"[{now_txt} | {rel_prefix}] {msg}"
                else:
                    try:
                        # 非コンパクト時は常に m,s を計算
                        _elapsed2 = max(0, time.time() - self.start_time)
                        m, s = divmod(int(_elapsed2), 60)
                        line = f"[{now_txt} | {m:02d}分{s:02d}秒] {msg}"
                    except Exception:
                        line = f"[{now_txt} | 00分00秒] {msg}"
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
        # ここは比較的限定的なキーワードのみにする（過剰除外を防止）
        skip_keywords = (
            "batch time",
            "next batch size",
        )
        if msg.startswith(data_load_prefixes):
            return self.progress_ui.show_data_load
        # 短時間内の同一ログを抑止（0.3秒以内の重複は無視）
        try:
            now = time.time()
            last = self._last_log.get(msg)
            if last is not None and (now - last) < 0.3:
                return False
            self._last_log[msg] = now
        except Exception:
            pass
        return not any(keyword in msg for keyword in skip_keywords)

    def _echo_cli(self, line: str) -> None:
        # Windows コンソールでの文字化け緩和（任意フラグ）
        try:
            if os.name == "nt" and os.environ.get("FORCE_UTF8_CONSOLE") == "1":
                try:
                    if hasattr(sys.stdout, "reconfigure"):
                        # 既に utf-8 の場合は触らない
                        if (getattr(sys.stdout, "encoding", "") or "").lower() not in (
                            "utf-8",
                            "utf8",
                        ):
                            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore
                except Exception:
                    pass
            # 初回ヒント表示（化けを検知できそうなら）
            if not getattr(self, "_encoding_hint_done", False) and os.name == "nt":
                setattr(self, "_encoding_hint_done", True)
                if os.environ.get("SUPPRESS_ENCODING_HINT") != "1":
                    enc = (getattr(sys.stdout, "encoding", "") or "").lower()
                    # 簡易判定: cp932 / ansi 系で絵文字が含まれそうな行
                    if enc and "utf" not in enc and any(ch for ch in line if ord(ch) > 0x2600):
                        try:
                            print(
                                "[INFO] 文字化けする場合は 'chcp 65001' 実行後に再試行してください (SUPPRESS_ENCODING_HINT=1 で非表示)",
                                flush=True,
                            )
                        except Exception:
                            pass
            try:
                print(line, flush=True)
                return
            except UnicodeEncodeError:
                try:
                    encoding = getattr(sys.stdout, "encoding", "") or "utf-8"
                    safe = line.encode(encoding, errors="replace").decode(
                        encoding, errors="replace"
                    )
                    print(safe, flush=True)
                    return
                except Exception:
                    pass
        except Exception:
            pass
        # 最終フォールバック: ASCII 置換
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
            import scripts.run_all_systems_today as _run_today_mod

            # 安全な属性アクセス方法を使用
            mod = _run_today_mod
            setattr(mod, "_PER_SYSTEM_STAGE", self.per_system_stage)
            setattr(mod, "_PER_SYSTEM_EXIT", self.per_system_exit)
            setattr(mod, "_SET_STAGE_UNIVERSE_TARGET", self.tracker.set_universe_target)
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
        "RETURN_6D": int(6 * 1.1),
        "Drop3D": int(3 * 1.1),
        "return_6d": int(6 * 1.1),
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
    final_df: pd.DataFrame, per_system: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    final_df = final_df.reset_index(drop=True)
    # DataFrame のみ reset_index。辞書などのメタ情報はそのまま保持する
    per_system = {
        name: (df.reset_index(drop=True) if isinstance(df, pd.DataFrame) else df)
        for name, df in per_system.items()
    }
    final_df = _sort_final_df(final_df)
    if final_df is not None and not final_df.empty:
        try:
            final_df.insert(0, "no", range(1, len(final_df) + 1))
        except Exception:
            pass
    # 仕掛け管理の主要列を前面に出し、UI向けの日本語ラベル・ツールチップを適用
    try:
        if isinstance(final_df, pd.DataFrame) and not final_df.empty:
            trade_cols = [
                "entry_type",
                "entry_price_final",
                "stop_price",
                "profit_target_price",
                "use_trailing_stop",
                "trailing_stop_pct",
                "max_holding_days",
                "entry_atr",
                "risk_per_share",
                "total_risk",
            ]
            exist = [c for c in trade_cols if c in final_df.columns]
            # 全NaNの列はノイズになるため除外（ただし 'no' は常に保持）
            exist_non_nan: list[str] = []
            for c in exist:
                try:
                    if not pd.to_numeric(final_df[c], errors="coerce").isna().all():
                        exist_non_nan.append(c)
                except Exception:
                    # 数値変換できないときは、文字列列として非NaN判定
                    try:
                        if not final_df[c].isna().all():
                            exist_non_nan.append(c)
                    except Exception:
                        pass
            # 既存先頭ナンバー列 'no' は温存
            leading = [c for c in (["no"] if "no" in final_df.columns else []) + exist_non_nan]
            if leading:
                other_cols = [c for c in final_df.columns if c not in leading]
                final_df = final_df[leading + other_cols]

            # 表示名（日本語ラベル）を設定
            label_map = {
                "entry_type": "仕掛け種別",
                "entry_price_final": "仕掛け価格",
                "stop_price": "損切価格",
                "profit_target_price": "利食い価格",
                "use_trailing_stop": "利益の保護ON",
                "trailing_stop_pct": "トレーリング幅(%)",
                "max_holding_days": "最大保有日数",
                "entry_atr": "ATR(参照)",
                "risk_per_share": "1株あたりリスク",
                "total_risk": "推定総リスク",
            }
            # 列名の見た目だけを置き換え（内部キーは保持）
            try:
                display_cols = {c: label_map.get(c, c) for c in final_df.columns}
                final_df = final_df.rename(columns=display_cols)
            except Exception:
                pass
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


def _log_run_completion(final_df: pd.DataFrame, per_system: dict[str, Any], elapsed: float) -> None:
    try:
        m, s = divmod(int(max(0, elapsed)), 60)
        final_n = 0 if final_df is None or final_df.empty else int(len(final_df))
        per_counts_lines: list[str] = []
        counts_map: dict[str, int] = {}
        for name, df in per_system.items():
            key = str(name).strip().lower()
            if not key:
                continue
            if isinstance(df, pd.DataFrame):
                counts_map[key] = 0 if df.empty else int(len(df))
            else:
                # 非DataFrame（サマリなど）は件数集計対象外
                continue
        if counts_map:
            per_counts_lines = format_group_counts(counts_map)
        detail = (
            f" | Long/Short別: {', '.join(per_counts_lines)}" if per_counts_lines else ""
        )  # noqa: E501
        _get_today_logger().info(
            "✅ 本日のシグナル: シグナル検出処理終了 (経過 %d分%d秒, 最終候補 %d 件)%s",
            m,
            s,
            final_n,
            detail,
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
        import scripts.run_all_systems_today as _run_today_mod

        _run_today_mod._configure_today_logger(mode=sel_mode)
        sel_path = getattr(_run_today_mod, "_LOG_FILE_PATH", None)
        if sel_path:
            st.caption(f"ログ保存先: {sel_path}")
    except Exception:
        pass


def _interpret_compute_today_result(
    result: Any, logger: Any
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """UI からの呼び出し用に compute_today_signals 戻り値を正規化する。

    戻り値形式:
      - 期待: (DataFrame, AllocationSummary)
      - 後方互換: (DataFrame, dict)
    失敗/不正形式の場合は空 DataFrame / {} を返し、警告をログ。
    """
    empty: tuple[pd.DataFrame, dict[str, Any]] = (pd.DataFrame(), {})
    if not (isinstance(result, (tuple, list)) and len(result) == 2):
        try:
            logger.log(f"⚠️ compute_today_signals の戻り値構造が不正: type={type(result).__name__}")
        except Exception:
            pass
        return empty
    maybe_df, maybe_second = result
    if not isinstance(maybe_df, pd.DataFrame):
        try:
            logger.log(
                (
                    "⚠️ compute_today_signals 戻り値の第1要素が DataFrame でない: "
                    f"{type(maybe_df).__name__}"
                )
            )
        except Exception:
            pass
        return empty

    # dict ならそのまま
    if isinstance(maybe_second, dict):
        return maybe_df, maybe_second

    # AllocationSummary を dict 化
    try:
        from core.final_allocation import to_allocation_summary_dict
    except Exception:
        to_allocation_summary_dict = None

    if to_allocation_summary_dict is not None:
        try:
            summary_dict = to_allocation_summary_dict(maybe_second)
            if summary_dict:
                # ログ出力
                fc = summary_dict.get("final_counts")
                if isinstance(fc, dict):
                    logger.log("🧾 AllocationSummary final_counts=" + str(fc))
                mode = summary_dict.get("mode")
                long_n = len(summary_dict.get("long_allocations", {}))
                short_n = len(summary_dict.get("short_allocations", {}))
                logger.log((f"ℹ️ AllocationSummary mode={mode} " f"long={long_n} short={short_n}"))
                return maybe_df, {"__allocation_summary__": summary_dict}
        except Exception as e:  # pragma: no cover
            try:
                logger.log(f"⚠️ AllocationSummary 解析失敗: {e}")
            except Exception:
                pass

    # 不明な型
    try:
        logger.log(
            (
                "⚠️ compute_today_signals の戻り値型が不正: df=DataFrame, second="
                f"{type(maybe_second).__name__}"
            )
        )
    except Exception:
        pass
    return empty


def execute_today_signals(run_config: RunConfig) -> RunArtifacts:
    # 実行開始時のヘッダーメッセージを表示
    today = get_signal_target_trading_day().normalize()
    try:
        run_id = str(uuid.uuid4())[:8]
    except Exception:
        run_id = "--------"

    # 仮のloggerを作成してヘッダーメッセージを表示
    temp_start_time = time.time()
    # 初期ヘッダーログ用: 本番進捗バーと重複しないよう overall_progress を無効化
    temp_progress_ui = ProgressUI({"overall_progress": False, "data_load_progress_lines": False})
    temp_logger = UILogger(temp_start_time, temp_progress_ui)

    # ヘッダーメッセージの表示
    temp_logger.log(
        "####################################################################",
        no_timestamp=True,
    )
    temp_logger.log("# 🚀🚀🚀  本日のシグナル 実行開始 (Engine)  🚀🚀🚀", no_timestamp=True)

    # 時刻とRUN-ID、銘柄数の表示
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    symbols_count = len(run_config.symbols) if run_config.symbols else 0
    temp_logger.log(
        f"# ⏱️ {now_str} | 銘柄数：{symbols_count}　| RUN-ID: {run_id}",
        no_timestamp=True,
    )
    temp_logger.log(
        "####################################################################",
        no_timestamp=True,
    )

    # 営業日と注意事項の表示
    temp_logger.log(f"📅 対象営業日（NYSE）: {today.date()}", no_timestamp=True)

    # データの新しさをチェックして必要な場合のみ警告を表示
    try:
        settings = get_settings()
        cm = CacheManager(settings)
        # SPYデータでキャッシュの新しさを確認
        spy_df = cm.read("SPY", "rolling")
        if spy_df is not None and not spy_df.empty:
            # last_cache_dateを計算するための簡単な実装
            if "date" in spy_df.columns:
                last_date = pd.to_datetime(spy_df["date"]).max()
            elif spy_df.index.name == "date" or hasattr(spy_df.index, "date"):
                last_date = pd.to_datetime(spy_df.index).max()
            else:
                last_date = None

            if last_date is not None:
                last_cache_date = pd.Timestamp(last_date).normalize()
                days_behind = (today - last_cache_date).days
                if days_behind > 1:  # 1営業日より古い場合のみ警告
                    temp_logger.log(
                        f"ℹ️ 注: キャッシュデータが{days_behind}日古いため、"
                        "直近営業日ベースで計算します。",
                        no_timestamp=True,
                    )
    except Exception:
        # エラー時は従来通り警告を表示
        temp_logger.log(
            "ℹ️ 注: EODHDは当日終値が未反映のため、直近営業日ベースで計算します。",
            no_timestamp=True,
        )

    temp_logger.log("", no_timestamp=True)  # 空行を追加

    # 既存の処理を継続
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
    buffer_days = max(20, int(max_days * 0.15))
    rows_needed = max_days + buffer_days
    symbols_for_data = list(dict.fromkeys([*run_config.symbols, "SPY"]))
    progress_ui.set_label("対象読み込み")
    final_df: pd.DataFrame = pd.DataFrame()
    per_system: dict[str, pd.DataFrame] = {}
    debug_result: RunArtifacts | None = None
    with st.spinner("実行中... (経過時間表示あり)"):
        logger.log("▶ 本日のシグナル: シグナル検出処理開始")
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
            # --- compute_today_signals 実行 & 戻り値解釈 ---
            result = compute_today_signals(
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
            final_df, per_system = _interpret_compute_today_result(result, logger)
            # final_counts=0 の場合の追加デバッグ
            try:
                alloc_dict = (
                    per_system.get("__allocation_summary__")
                    if isinstance(per_system, dict)
                    else None
                )
                final_counts = (
                    (alloc_dict or {}).get("final_counts") if isinstance(alloc_dict, dict) else None
                )
                if (
                    (not final_df.empty)
                    and isinstance(final_counts, dict)
                    and sum(final_counts.values()) == 0
                ):
                    # 最終候補 DataFrame には列があるのに全カウント0 → system 列異常か集計ミス
                    if "system" in final_df.columns:
                        sys_counts = final_df["system"].value_counts().to_dict()
                    else:
                        sys_counts = {"<no system column>": len(final_df)}
                    msg = (
                        "🔍 final_counts=0 だが final_df 行数="
                        f"{len(final_df)} system別={sys_counts}"
                    )
                    logger.log(msg)
                if (
                    final_df.empty
                    and isinstance(final_counts, dict)
                    and sum(final_counts.values()) == 0
                ):
                    # 完全0のとき per_system DataFrame の行数概要
                    if isinstance(per_system, dict):
                        per_rows = {
                            k: (len(v) if hasattr(v, "shape") else None)
                            for k, v in per_system.items()
                            if k.startswith("system")
                        }
                        if per_rows:
                            logger.log(f"🔍 per_system 行数サマリ: {per_rows}")
            except Exception:
                pass

    if debug_result is not None:
        return debug_result
    total_elapsed = max(0.0, time.time() - start_time)

    # AllocationSummary を抽出して stage_tracker に渡す（不要な変数削除）
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
    """現在保有中ポジションの手仕舞い予定を推定する。

    役割:
      1. 保有ポジション取得
      2. エントリー日補完（ローカル→不足分 Alpaca 取得→保存）
      3. システム判定 & Strategy インスタンス生成
      4. ストラテジー exit ロジックを用い本日/将来 exit を分類
    """

    exits_today_rows: list[dict[str, Any]] = []
    planned_rows: list[dict[str, Any]] = []
    exit_counts: dict[str, int] = {f"system{i}": 0 for i in range(1, 8)}
    try:
        client_tmp = ba.get_client(paper=paper_mode)
        try:
            positions = list(client_tmp.get_all_positions())
        except Exception:
            positions = []

        # 1) エントリー日マップ読み込み
        raw_entry_map = load_entry_dates()
        entry_map: dict[str, str] = {}
        for k, v in raw_entry_map.items():
            try:
                entry_map[str(k).upper()] = str(v)
            except Exception:
                continue

        # 2) 不足エントリー日の補完
        missing = [
            str(getattr(p, "symbol", "")).upper()
            for p in positions
            if str(getattr(p, "symbol", "")).upper()
            and str(getattr(p, "symbol", "")).upper() not in entry_map
        ]
        if missing:
            try:
                fetched = fetch_entry_dates_from_alpaca(client_tmp, missing)
            except Exception:
                fetched = None
            if fetched:
                for sym, ts in fetched.items():
                    if sym not in entry_map:
                        try:
                            entry_map[sym] = pd.Timestamp(ts).strftime("%Y-%m-%d")
                        except Exception:
                            continue
                try:
                    save_entry_dates(entry_map)
                except Exception:
                    pass

        symbol_system_map = _load_symbol_system_map(Path("data/symbol_system_map.json"))
        latest_trading_day = _latest_trading_day()
        strategy_classes = STRATEGY_CLASS_MAP

        # 3) 各ポジション解析
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
            system, _pos_side, _qty, exit_when, row_base, exit_today = result
            when_val = str(exit_when or "").strip()
            when_lower = when_val.lower()
            when_display = when_lower or when_val
            if exit_today:
                exit_counts[system] = exit_counts.get(system, 0) + 1
                if when_lower == "tomorrow_open":
                    planned_rows.append(row_base | {"when": when_display})
                else:
                    exits_today_rows.append(row_base | {"when": when_display})
            else:
                planned_rows.append(row_base | {"when": when_display})

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
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
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


STRATEGY_CLASS_MAP: dict[str, Callable[[], Any]] = {
    "system1": System1Strategy,
    "system2": System2Strategy,
    "system3": System3Strategy,
    "system4": System4Strategy,
    "system5": System5Strategy,
    "system6": System6Strategy,
}


# 互換用関数は削除（直接 STRATEGY_CLASS_MAP を参照する実装へ移行済み）


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
        # exit_priceを使用するように修正
        exit_price, exit_date = strategy.compute_exit(
            df, int(entry_idx), float(entry_price), float(stop_price)
        )
        # exit_priceをプロパティに追加
        today_norm = pd.to_datetime(df.index[-1]).normalize()
        if latest_trading_day is not None:
            today_norm = latest_trading_day
        is_today_exit, when = decide_exit_schedule(system, exit_date, today_norm)
        row_base = {
            "symbol": sym,
            "qty": qty,
            "position_side": pos_side,
            "system": system,
            "exit_price": exit_price,  # 追加
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
            strategy._last_entry_atr = atr
        except Exception:
            pass
    if system in {"system3", "system5", "system6"}:
        try:
            strategy._last_prev_close = prev_close
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
    _display_planned_exits_section(result)  # trade_options引数を削除
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
    st.dataframe(result.exits_today, width="stretch")
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
            st.dataframe(res, width="stretch")


def _display_planned_exits_section(
    result: ExitAnalysisResult,
) -> None:  # trade_options引数を削除
    if result.planned.empty:
        return
    st.caption("明日発注する手仕舞い計画（保存→スケジューラが実行）")
    st.dataframe(result.planned, width="stretch")
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
    plan_path = Path("data/planned_exits.jsonl")
    try:
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        with plan_path.open("w", encoding="utf-8") as f:
            for row in planned_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
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
            st.dataframe(df_exec, width="stretch")
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
            st.dataframe(df_details, width="stretch")
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
    st.dataframe(final_df, width="stretch")
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
                st.dataframe(df_skip, width="stretch")
            else:
                st.write("(空) 内訳情報は見つかりませんでした。")
        with cols[1]:
            try:
                data_bytes = path.read_bytes()
            except Exception:
                data_bytes = None
            if data_bytes:
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
        now = datetime.now()
        ts = now.strftime("%Y-%m-%d")
        if run_config.csv_name_mode == "datetime":
            ts = now.strftime("%Y-%m-%d_%H%M")
        elif run_config.csv_name_mode == "runid":
            rid = st.session_state.get("last_run_id") or "RUN"
            ts = f"{now.strftime('%Y-%m-%d')}_{rid}"
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
        st.dataframe(results_df, width="stretch")
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
    order_ids = [str(oid) for oid in results_df["order_id"].values.tolist() if oid]
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
    _SYSTEM1_REASON_LABELS_UI = {
        "filter": "フィルター条件 (filter)",
        "setup": "セットアップ条件 (setup)",
        "roc200": "ROC200≤0",
    }

    def _build_system1_diagnostic_messages(
        diag_payload: Mapping[str, Any] | None,
    ) -> tuple[str | None, str | None]:
        summary = summarize_system1_diagnostics(diag_payload)
        if not summary:
            return None, None

        top_n = summary.get("top_n")
        prefix = f"抽出上限 {top_n} 件, " if isinstance(top_n, int) and top_n > 0 else ""
        reason_line = (
            "候補0件理由: "
            f"{prefix}フィルター通過 {summary.get('filter_pass', 0)} 件, "
            f"セットアップ成立 {summary.get('setup_flag_true', 0)} 件, "
            f"代替判定成立 {summary.get('fallback_pass', 0)} 件, "
            f"ROC200>0 {summary.get('roc200_positive', 0)} 件, "
            f"最終通過 {summary.get('final_pass', 0)} 件。"
        )

        reasons = summary.get("exclude_reasons")
        detail_line: str | None = None
        if isinstance(reasons, Mapping) and reasons:
            parts: list[str] = []
            for key, value in reasons.items():
                if not isinstance(value, int) or value <= 0:
                    continue
                label = _SYSTEM1_REASON_LABELS_UI.get(str(key), str(key))
                parts.append(f"{label} {value} 件")
            if parts:
                detail_line = "除外内訳: " + ", ".join(parts)

        return reason_line, detail_line

    def _build_generic_diagnostic_messages(
        system_name: str, diag_payload: Mapping[str, Any] | None
    ) -> tuple[str | None, str | None]:
        if not isinstance(diag_payload, Mapping):
            return None, None

        # 診断キーの存在確認と安全な型変換
        def _get_int(d: Mapping[str, Any], key: str, default: int = 0) -> int:
            try:
                v = d.get(key, default)
                return int(v) if v is not None else default
            except Exception:
                return default

        def _get_bool(d: Mapping[str, Any], key: str) -> bool | None:
            try:
                v = d.get(key, None)
                if isinstance(v, bool):
                    return v
                if isinstance(v, (int, float)):
                    return bool(v)
                if isinstance(v, str):
                    s = v.strip().lower()
                    if s in {"true", "1", "yes"}:
                        return True
                    if s in {"false", "0", "no"}:
                        return False
                return None
            except Exception:
                return None

        setup_cnt = _get_int(diag_payload, "setup_predicate_count", 0)
        # Read unified key only
        ranked_topn = _get_int(diag_payload, "ranked_top_n_count", 0)
        only_pass = _get_int(diag_payload, "predicate_only_pass_count", 0)
        mismatch = _get_bool(diag_payload, "mismatch_flag")
        ranking_src = str(diag_payload.get("ranking_source", "-") or "-")
        top_n_val = diag_payload.get("top_n")
        try:
            top_n = int(top_n_val) if top_n_val is not None else None
        except Exception:
            top_n = None

        prefix = f"抽出上限 {top_n} 件, " if isinstance(top_n, int) and top_n > 0 else ""
        mismatch_txt = (
            "乖離あり" if mismatch is True else ("乖離なし" if mismatch is False else "乖離不明")
        )
        reason_line = (
            f"候補0件理由: {prefix}セットアップ成立 {setup_cnt} 件, 最終TopN {ranked_topn} 件, "
            f"セットアップのみ通過 {only_pass} 件, ランキング {ranking_src}, {mismatch_txt}。"
        )
        return reason_line, None

    diagnostics_map: dict[str, Mapping[str, Any]] = {}
    try:
        summary_entry = (
            per_system.get("__allocation_summary__") if isinstance(per_system, dict) else None
        )
        if isinstance(summary_entry, Mapping):
            raw_diag = summary_entry.get("system_diagnostics")
            if isinstance(raw_diag, Mapping):
                diagnostics_map = {
                    str(k).strip().lower(): v for k, v in raw_diag.items() if isinstance(k, str)
                }
    except Exception:
        diagnostics_map = {}
    with st.expander("システム別詳細"):
        settings_local = get_settings(create_dirs=True)
        results_dir = Path(getattr(settings_local.outputs, "results_csv_dir", "results_csv"))
        shortable_excluded_map = {}
        for i in (2, 6):
            name = f"system{i}"
            fp = results_dir / f"shortability_excluded_{name}.csv"
            if fp.exists() and fp.is_file():
                try:
                    df_exc = pd.read_csv(fp)
                    if df_exc is not None and not df_exc.empty:
                        shortable_excluded_map[name] = set(df_exc["symbol"].astype(str).str.upper())
                except Exception:
                    pass
        system_order = [f"system{i}" for i in range(1, 8)]
        for name in system_order:
            st.markdown(f"#### {name}")
            display_metrics = stage_tracker.get_display_metrics(name)
            # Line length fix - split formatted string
            metrics_parts = [
                f"Tgt {StageTracker._format_value(display_metrics.get('target'))}",
                f"FILpass {StageTracker._format_value(display_metrics.get('filter'))}",
                f"STUpass {StageTracker._format_value(display_metrics.get('setup'))}",
                f"TRDlist {stage_tracker._format_trdlist(display_metrics.get('cand'))}",
                f"Entry {StageTracker._format_value(display_metrics.get('entry'))}",
                f"Exit {StageTracker._format_value(display_metrics.get('exit'))}",
            ]
            metrics_line = "  ".join(metrics_parts)
            st.caption(metrics_line)
            df = per_system.get(name)
            if df is None or df.empty:
                # Try to extract explicit zero-reason from per-system logs if available
                reason_text: str | None = None
                try:
                    if per_system_logs and name in per_system_logs:
                        logs = per_system_logs.get(name) or []
                        for ln in reversed(logs):
                            if not ln:
                                continue
                            m = re.search(r"候補0件理由[:：]\s*(.+)$", ln)
                            if m:
                                reason_text = m.group(1).strip()
                                break
                            m2 = re.search(r"セットアップ不成立[:：]\s*(.+)$", ln)
                            if m2:
                                reason_text = m2.group(1).strip()
                                break
                except Exception:
                    reason_text = None

                diag_reason: str | None = None
                diag_detail: str | None = None
                diag_payload = diagnostics_map.get(name)
                if name == "system1":
                    diag_reason, diag_detail = _build_system1_diagnostic_messages(diag_payload)
                else:
                    diag_reason, diag_detail = _build_generic_diagnostic_messages(
                        name, diag_payload
                    )

                st.write("(空) 候補は0件です。")
                if diag_reason:
                    st.info(diag_reason)
                elif reason_text:
                    st.info(f"候補0件理由: {reason_text}")
                if diag_detail:
                    st.caption(diag_detail)
                elif reason_text and diag_reason:
                    st.caption(f"ログ補足: {reason_text}")
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
            st.dataframe(df_disp, width="stretch")


def _render_previous_results_section() -> None:
    try:
        if (not st.session_state.get("today_shown_this_run", False)) and (
            "today_final_df" in st.session_state
        ):
            prev_df = st.session_state.get("today_final_df")
            if prev_df is not None and not prev_df.empty:
                st.subheader("前回の最終選定銘柄（再表示）")
                st.dataframe(prev_df, width="stretch")
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
                            st.dataframe(df, width="stretch")
    except Exception:
        pass


def _render_previous_run_logs(log_lines: list[str]) -> None:
    prev_msgs = [line for line in log_lines if line and ("(前回結果) system" in line)]
    if not prev_msgs:
        return

    def _parse_prev_line(ln: str) -> tuple[str, int, str, str]:
        ts = ln.split("] ", 1)[0].strip("[")
        m = re.search(r"\(前回結果\) (system\d+):\s*(\d+)", ln)
        sys = m.group(1) if m else "system999"
        cnt = int(m.group(2)) if m else 0
        return sys, cnt, ts, ln

    parsed = [_parse_prev_line(x) for x in prev_msgs]
    order = {f"system{i}": i for i in range(1, 8)}
    parsed.sort(key=lambda t: order.get(t[0], 999))
    lines_sorted = [f"{p[2]} | {p[0]}: {p[1]}件\n{p[3]}" for p in parsed]
    with st.expander("前回結果（system別）", expanded=False):
        st.text("\n\n".join(lines_sorted))


def _log_and_notify(
    message: str,
    notifier: Callable[[str], None] | None,
    log_callback: Callable[[str], None] | None,
    level: int = logging.INFO,
) -> None:
    """Log to both logger and optional callbacks."""

    _get_today_logger().log(level, message)
    if notifier:
        try:
            notifier(message)
        except Exception as e:
            _get_today_logger().warning("Notifier failed: %s", e)
    if log_callback:
        try:
            log_callback(message)
        except Exception as e:
            _get_today_logger().warning("Log callback failed: %s", e)


# =============================================================================
# メイン UI 実行部分
# =============================================================================

with st.sidebar:
    st.header("ユニバース")
    universe: list[str] = []
    try:
        from common.symbol_universe import build_symbol_universe_from_settings
        import common.universe as univ

        logger = logging.getLogger("today_signals.ui")
        universe = build_symbol_universe_from_settings(settings, logger=logger)
    except Exception as exc:
        universe = []
        st.warning(f"NASDAQ/EODHDの銘柄取得に失敗しました: {exc}")

    if not universe:
        try:
            import common.universe as universe_mod

            universe = universe_mod.load_universe_file()
        except Exception:
            pass

    if not universe:
        try:
            import common.universe as universe_mod

            universe = universe_mod.build_universe_from_cache(limit=None)
            universe_mod.save_universe_file(universe)
        except Exception:
            universe = []

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

    # セッション状態に保存
    st.session_state["universe_symbols"] = syms

    st.write(f"銘柄数: {len(syms)}")
    st.write(", ".join(syms[:10]) + (" ..." if len(syms) > 10 else ""))

    # Alpaca未約定注文表示
    st.header("Alpaca注文状況")

    # デバッグ情報の表示
    with st.expander("🔧 デバッグ情報"):
        st.write("broker_alpaca モジュール属性:")
        ba_attrs = [attr for attr in dir(ba) if not attr.startswith("_")]
        for attr in sorted(ba_attrs):
            if attr == "get_open_orders":
                st.write(f"✅ {attr}: {type(getattr(ba, attr))}")
            elif callable(getattr(ba, attr)):
                st.write(f"📝 {attr}: {type(getattr(ba, attr))}")
            else:
                st.write(f"📦 {attr}: {type(getattr(ba, attr))}")

        st.write(f"get_open_orders 存在確認: {hasattr(ba, 'get_open_orders')}")
        if hasattr(ba, "get_open_orders"):
            st.write(f"get_open_orders 型: {type(ba.get_open_orders)}")
            st.write(f"get_open_orders docstring: {ba.get_open_orders.__doc__}")

    if st.button("📋 未約定注文を表示"):
        try:
            paper_mode = st.session_state.get("paper_mode", True)

            # デバッグ: モジュール状態の確認
            st.info(f"broker_alpaca モジュール: {ba}")
            st.info(f"get_open_orders 存在: {hasattr(ba, 'get_open_orders')}")

            if not hasattr(ba, "get_open_orders"):
                st.error("get_open_orders 関数が見つかりません")
                available_funcs = [
                    attr
                    for attr in dir(ba)
                    if callable(getattr(ba, attr)) and not attr.startswith("_")
                ]
                st.write("利用可能な関数:")
                st.write(available_funcs)
                st.stop()

            client = ba.get_client(paper=paper_mode)
            orders = ba.get_open_orders(client)
            if orders:
                orders_data = []
                for order in orders:
                    orders_data.append(
                        {
                            "注文ID": order.id,
                            "銘柄": order.symbol,
                            "サイド": order.side,
                            "数量": order.qty,
                            "注文価格": getattr(order, "limit_price", "Market"),
                            "注文タイプ": order.order_type,
                            "状況": order.status,
                            "作成日時": order.created_at,
                        }
                    )
                orders_df = pd.DataFrame(orders_data)
                st.dataframe(orders_df, width="stretch")
            else:
                st.info("未約定注文はありません")
        except Exception as e:
            st.error(f"注文取得エラー: {e}")
            st.error(f"エラー詳細: {type(e).__name__}")
            import traceback

            st.code(traceback.format_exc())

    st.header("資産")
    # デフォルト値を設定
    if "today_cap_long" not in st.session_state:
        st.session_state["today_cap_long"] = 10000.0
    if "today_cap_short" not in st.session_state:
        st.session_state["today_cap_short"] = 10000.0

    # （旧）ペーパー資金リセット UI は API 非対応のため削除済み

    # Alpaca資産取得ボタンを追加
    if st.button("💰 Alpacaから現在の資産を取得"):
        try:
            # 接続前の事前チェック
            api_key = os.environ.get("APCA_API_KEY_ID")
            api_secret = os.environ.get("APCA_API_SECRET_KEY")

            if not api_key or not api_secret:
                st.error("❌ Alpaca API認証情報が設定されていません")
                st.info("環境変数 APCA_API_KEY_ID と APCA_API_SECRET_KEY を設定してください")
            else:
                # ネットワーク接続テスト
                with st.spinner("Alpacaサーバーに接続中..."):
                    client = ba.get_client(paper=st.session_state.get("paper_mode", True))
                    acct = client.get_account()

                equity = getattr(acct, "equity", None)
                cash = getattr(acct, "cash", None)
                buying_power = getattr(acct, "buying_power", None)

                if equity is not None:
                    equity_val = float(equity)
                    st.success(f"✅ 総資産: ${equity_val:,.2f}")
                if cash is not None:
                    cash_val = float(cash)
                    st.info(f"💵 現金残高: ${cash_val:,.2f}")
                if buying_power is not None:
                    bp_val = float(buying_power)
                    st.info(f"🚀 買付余力: ${bp_val:,.2f}")

                    # 買付余力を半分ずつロング・ショートに配分
                    half_bp = round(bp_val / 2.0, 2)
                    st.session_state["today_cap_long"] = half_bp
                    st.session_state["today_cap_short"] = half_bp
                    st.success("資金配分を更新しました:")
                    st.success(f"ロング `${half_bp:,.2f}` / ショート `${half_bp:,.2f}`")
                else:
                    st.warning("買付余力が取得できませんでした")

        except Exception as exc:
            ERROR_MSG = str(exc)
            if "getaddrinfo failed" in ERROR_MSG or "Failed to resolve" in ERROR_MSG:
                st.error("🌐 ネットワーク接続エラー")
                st.error("- インターネット接続を確認してください")
                st.error("- DNSサーバー設定を確認してください")
                st.error("- ファイアウォール/プロキシ設定を確認してください")
                with st.expander("詳細エラー情報"):
                    st.code(ERROR_MSG)
            elif "HTTPSConnectionPool" in ERROR_MSG:
                st.error("🔒 HTTPS接続エラー")
                st.error("- SSL証明書の問題の可能性があります")
                st.error("- プロキシ設定を確認してください")
                with st.expander("詳細エラー情報"):
                    st.code(ERROR_MSG)
            elif "401" in ERROR_MSG or "403" in ERROR_MSG:
                st.error("🔑 API認証エラー")
                st.error("- API キーとシークレットを確認してください")
                st.error("- APIキーの権限を確認してください")
            else:
                st.error(f"❌ Alpaca資産取得エラー: {ERROR_MSG}")
                st.info("💡 オフライン環境では手動で資金を設定してください")

    col1, col2 = st.columns(2)
    with col1:
        cap_long = st.number_input(
            "ロング資本 (USD)",
            min_value=0.0,
            step=100.0,
            key="today_cap_long",
        )
    with col2:
        cap_short = st.number_input(
            "ショート資本 (USD)",
            min_value=0.0,
            step=100.0,
            key="today_cap_short",
        )

    st.header("オプション")
    save_csv = st.checkbox("CSVファイルを保存", value=True, key="save_csv")

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
    import platform

    is_windows = platform.system().lower().startswith("win")
    RUN_PARALLEL_DEFAULT = True
    run_parallel = st.checkbox(
        "並列実行（システム横断）", value=RUN_PARALLEL_DEFAULT, key="run_parallel"
    )

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
    paper_mode = st.checkbox("ペーパートレードを使用", value=True, key="paper_mode")
    retries = st.number_input("リトライ回数", min_value=0, max_value=5, value=2, key="retries")
    delay = st.number_input(
        "発注間隔 (秒)", min_value=0.0, max_value=10.0, value=1.0, step=0.1, key="delay"
    )
    poll_status = st.checkbox("注文状況をポーリング", value=False, key="poll_status")
    do_trade = st.checkbox("実際に発注する", value=False, key="do_trade")
    update_bp_after = st.checkbox("約定後に余力を更新", value=False, key="update_bp_after")

# メイン実行部分
# デフォルト値を設定（サイドバーが未実行の場合）
syms = st.session_state.get("universe_symbols", [])
cap_long = st.session_state.get("today_cap_long", 10000.0)
cap_short = st.session_state.get("today_cap_short", 10000.0)
save_csv = st.session_state.get("save_csv", True)
csv_name_mode = st.session_state.get("csv_name_mode", "date")
use_slack_notify = st.session_state.get("use_slack_notify", True)
run_parallel = st.session_state.get("run_parallel", True)
scan_missing_only = st.session_state.get("today_scan_missing_only", False)
paper_mode = st.session_state.get("paper_mode", True)
retries = st.session_state.get("retries", 2)
delay = st.session_state.get("delay", 1.0)
poll_status = st.session_state.get("poll_status", False)
do_trade = st.session_state.get("do_trade", False)
update_bp_after = st.session_state.get("update_bp_after", False)

run_config = RunConfig(
    symbols=syms,
    capital_long=float(cap_long),
    capital_short=float(cap_short),
    save_csv=save_csv,
    csv_name_mode=csv_name_mode,
    notify=use_slack_notify,
    run_parallel=run_parallel,
    scan_missing_only=scan_missing_only,
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
    positions_df = st.session_state["positions_df"]
    if not positions_df.empty:
        try:
            summary_table = _build_position_summary_table(positions_df)
            if not summary_table.empty:
                st.caption("保有ポジション（System × Side別）")
                st.dataframe(summary_table, width="stretch")
        except Exception:
            pass

        # 表示用にカラムを日本語化
        df_disp = positions_df.copy()
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
        st.dataframe(df_disp, width="stretch")

        # 手動手仕舞い機能
        st.subheader("🎯 手動手仕舞い")
        st.caption("選択した銘柄を手動で手仕舞い注文します")

        # 手仕舞い対象の選択
        if not positions_df.empty:
            symbols_list = positions_df["symbol"].values.tolist()
            selected_symbols: list[str] = st.multiselect(
                "手仕舞いする銘柄を選択:",
                options=symbols_list,
                key="manual_exit_symbols",
            )

            if selected_symbols:
                exit_type = st.selectbox(
                    "手仕舞いタイプ:",
                    ["MOC (大引け)", "OPG (寄り付き)", "Market (成行)"],
                    key="exit_type",
                )

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🚀 選択銘柄の手仕舞い注文を送信", type="primary"):
                        try:
                            # 選択された銘柄のポジション情報を取得
                            selected_positions = positions_df[
                                positions_df["symbol"].isin(selected_symbols)
                            ].copy()

                            # 手仕舞い注文の実行
                            exit_orders = []
                            for _, row in selected_positions.iterrows():
                                exit_orders.append(
                                    {
                                        "symbol": row["symbol"],
                                        "side": (
                                            "sell" if str(row["side"]).lower() == "long" else "buy"
                                        ),
                                        "qty": abs(float(row["qty"])),
                                        "order_type": (
                                            "market" if "Market" in exit_type else "limit"
                                        ),
                                        "time_in_force": (
                                            "cls"
                                            if "MOC" in exit_type
                                            else ("opg" if "OPG" in exit_type else "day")
                                        ),
                                    }
                                )

                            if exit_orders:
                                exit_df = pd.DataFrame(exit_orders)
                                results = submit_orders_df(
                                    exit_df,
                                    paper=paper_mode,
                                    tif="DAY",
                                    retries=int(retries),
                                    delay=float(delay),
                                )

                                if results is not None and not results.empty:
                                    st.success(
                                        f"{len(selected_symbols)}銘柄の手仕舞い注文を送信しました"
                                    )
                                    st.dataframe(results, width="stretch")
                                else:
                                    st.warning("注文送信結果が取得できませんでした")

                        except Exception as e:
                            st.error(f"手仕舞い注文エラー: {e}")

                with col2:
                    if st.button("📊 手仕舞い影響を事前確認"):
                        if selected_symbols:
                            selected_positions = positions_df[
                                positions_df["symbol"].isin(selected_symbols)
                            ].copy()
                            total_pl = selected_positions["unrealized_pl"].astype(float).sum()
                            st.info(f"選択銘柄の合計含み損益: ${total_pl:,.2f}")
                            st.dataframe(
                                selected_positions[
                                    [
                                        "symbol",
                                        "side",
                                        "qty",
                                        "unrealized_pl",
                                        "judgement",
                                    ]
                                ],
                                width="stretch",
                            )

if st.button("▶ 本日のシグナル実行", type="primary"):
    artifacts = execute_today_signals(run_config)
    render_today_signals_results(artifacts, run_config, trade_options)
else:
    _render_previous_results_section()
