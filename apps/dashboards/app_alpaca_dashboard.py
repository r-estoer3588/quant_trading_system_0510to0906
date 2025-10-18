"""Alpaca ダッシュボード（UI リフレッシュ＋演出強化）

- アカウント残高/現金/余力をカード表示（前日比、余力ゲージ）
- ポジション一覧は行スタイル（損益で淡い緑/赤）＋スパークライン
- システム別フィルタ（symbol_system_map.json があれば使用）
- 統計チップ（勝ち/負け、平均損益率、最大/合計/中央値の含み損益）
- タイトル直下のスティッキーツールバーに「🔄 手動更新」と最終更新時刻を横並び配置
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
import json
import math
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal
import streamlit as st

# Ensure page config is set before any Streamlit commands execute
try:
    st.set_page_config(page_title="Alpaca Dashboard", layout="wide")
except Exception:
    # If Streamlit already configured (e.g., during tests), ignore
    pass

# プロジェクトルート（apps/dashboards から2階層上）をパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:  # pragma: no cover - optional dependency
    import plotly.graph_objects as go  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    go = None

if TYPE_CHECKING:  # pragma: no cover - help type checkers
    try:
        from plotly.graph_objects import Figure as PlotlyFigure  # type: ignore
    except (ModuleNotFoundError, ImportError):
        PlotlyFigure = Any
else:  # pragma: no cover - runtime fallback when Plotly is missing
    PlotlyFigure = Any

from common import broker_alpaca as ba
from common.cache_manager import load_base_cache
from common.position_age import fetch_entry_dates_from_alpaca, load_entry_dates


# --- Minimal utility helpers (small, safe implementations) ---
def _format_datetime_with_weekday(dt: datetime | None) -> str:
    if dt is None:
        return "-"
    try:
        return dt.strftime("%Y-%m-%d %H:%M (%a)")
    except Exception:
        return str(dt)


def _format_countdown(delta: timedelta) -> str:
    try:
        total = int(delta.total_seconds())
        hours, rem = divmod(total, 3600)
        minutes, seconds = divmod(rem, 60)
        if hours:
            return f"{hours}h{minutes}m"
        if minutes:
            return f"{minutes}m{seconds}s"
        return f"{seconds}s"
    except Exception:
        return "--"


def _safe_float(v: Any | None) -> float | None:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


def _fmt_money(x: Any | None) -> str:
    try:
        val = float(x) if x is not None and x != "" else 0.0
        return f"${val:,.0f}"
    except Exception:
        return str(x or "-")


def _get_nyse_status(now_newyork: datetime) -> str:
    """NYSE の営業状況と次回オープンまでのカウントダウンを返す。"""
    try:
        calendar = mcal.get_calendar("NYSE")
    except Exception:
        return "NYSE: 状態不明"

    start_date = now_newyork.date() - timedelta(days=5)
    end_date = now_newyork.date() + timedelta(days=10)

    try:
        schedule = calendar.schedule(start_date=start_date, end_date=end_date)
    except Exception:
        return "NYSE: 状態不明"

    if schedule.empty:
        return "NYSE: クローズ"

    try:
        is_open = bool(calendar.open_at_time(schedule, pd.Timestamp(now_newyork)))
    except Exception:
        is_open = False

    status = "NYSE: 営業中" if is_open else "NYSE: クローズ"

    if is_open:
        return status

    # If closed, compute the time until next open (countdown)
    try:
        now_ts = pd.Timestamp(now_newyork)
        if now_ts.tz is None:
            try:
                now_ts = now_ts.tz_localize("America/New_York")
            except Exception:
                now_ts = now_ts.tz_localize("UTC")
        now_utc = now_ts.tz_convert("UTC")

        market_open_series = pd.to_datetime(schedule["market_open"], utc=True)
        future_opens = market_open_series[market_open_series > now_utc]
        if getattr(future_opens, "empty", True):
            return status
        next_open = future_opens.iloc[0]
        delta = next_open - now_utc
        if delta.total_seconds() <= 0:
            return status
        countdown = _format_countdown(delta)
        return f"{status}（オープンまで {countdown}）"
    except Exception:
        return status


# --- Minimal constants / paths used by the UI for persistence markers ---
# These are safe placeholders; if the project defines them elsewhere they
# will be used instead. Keep names stable so session_state keys are consistent.
SENT_MARKER_KEY = "ap_sent_markers"
SENT_MARKERS_PATH = Path("data_cache/sent_markers.json")
SCHEDULE_PATH = Path("data_cache/schedule.json")
CONFIG_PATH = Path("data_cache/auto_rule_config.json")
NOTIFY_PATH = Path("data_cache/notify_settings.json")
ORDER_LOG_KEY = "ap_order_log"
EXIT_STATE_KEY = "ap_exit_state"

# Auto-rule defaults (will be merged with disk config if present)
AUTO_RULE_CONFIG: dict[str, dict[str, Any]] = {}

# Hold limits by system (days) - empty by default
HOLD_LIMITS: dict[str, int] = {
    "system1": 50,
    "system2": 50,
    "system3": 50,
    "system4": 50,
    "system5": 50,
    "system6": 50,
    "system7": 50,
}

# Debug mode toggle (can be set via environment variable or settings)
DEBUG_MODE = os.getenv("ALPACA_DASHBOARD_DEBUG", "false").lower() in ("true", "1", "on")


def calculate_business_holding_days(
    entry_dt: datetime | pd.Timestamp | str | None,
) -> int:
    """Fallback for calculating holding days; accepts datetime, pandas Timestamp,
    or ISO date string.

    Returns 0 when unknown or on error.
    """
    try:
        if not entry_dt:
            return 0
        if isinstance(entry_dt, pd.Timestamp):
            entry_dt = entry_dt.to_pydatetime()
        if isinstance(entry_dt, str):
            try:
                entry_dt = datetime.fromisoformat(entry_dt)
            except Exception:
                return 0
        if not isinstance(entry_dt, datetime):
            return 0
        return (datetime.now().date() - entry_dt.date()).days
    except Exception:
        return 0


def submit_exit_orders_df(df, *args, **kwargs):
    """Minimal stub: pretend to submit orders and return an empty result."""
    try:
        # In real usage this submits to broker; here we just return an empty list
        return []
    except Exception:
        return []


class Notifier:
    """Minimal notifier stub used in UI flows during offline/debug runs."""

    def __init__(self, *args, **kwargs):
        pass

    def send(self, *args, **kwargs):
        return True


def _inject_css() -> None:
    """Inject modern dashboard CSS with light/dark theme support."""
    try:
        css = """
        <style>
        /* CSS Variables for theme consistency */
        :root {
            --primary-color: #1f77b4;
            --secondary-color: #ff7f0e;
            --success-color: #2ca02c;
            --danger-color: #d62728;
            --warning-color: #ff9800;
            --info-color: #17a2b8;

            --bg-primary: #ffffff;
            --bg-secondary: #f8f9fa;
            --bg-card: #ffffff;
            --border-color: #dee2e6;
            --text-primary: #212529;
            --text-secondary: #6c757d;
            --text-muted: #adb5bd;

            --shadow-sm: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
            --shadow-md: 0 0.5rem 1rem rgba(0, 0, 0, 0.15);
            --border-radius: 0.5rem;
            --border-radius-sm: 0.375rem;
            --spacing-xs: 0.25rem;
            --spacing-sm: 0.5rem;
            --spacing-md: 1rem;
            --spacing-lg: 1.5rem;
            --spacing-xl: 3rem;
        }

        /* Dark theme support */
        @media (prefers-color-scheme: dark) {
            :root {
                --bg-primary: #1e1e1e;
                --bg-secondary: #2d2d2d;
                --bg-card: #2d2d2d;
                --border-color: #404040;
                --text-primary: #ffffff;
                --text-secondary: #b3b3b3;
                --text-muted: #808080;
            }
        }

        /* Base layout improvements */
        .main > div {
            padding-top: var(--spacing-sm) !important;
        }

        /* Typography */
        .ap-title {
            font-size: 2rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: var(--spacing-md);
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .ap-section {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--text-primary);
            margin: var(--spacing-lg) 0 var(--spacing-md) 0;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: var(--spacing-sm);
        }

        /* Toolbar */
        .ap-toolbar {
            position: sticky;
            top: 0;
            background: var(--bg-primary);
            backdrop-filter: blur(10px);
            z-index: 100;
            padding: var(--spacing-md) 0;
            margin-bottom: var(--spacing-md);
            border-bottom: 1px solid var(--border-color);
            box-shadow: var(--shadow-sm);
        }

        /* Toolbar内のコンポーネント間隔調整 */
        .ap-toolbar .stColumns {
            gap: var(--spacing-md);
        }

        .ap-toolbar .stButton > button {
            height: 2.5rem;
            font-size: 0.9rem;
            font-weight: 600;
            border-radius: var(--border-radius-sm);
        }

        .ap-toolbar .stTimeInput > div > div > input {
            height: 2rem;
            font-size: 0.85rem;
        }

        .ap-toolbar .stCheckbox {
            margin-top: 0.25rem;
        }

        /* Cards */
        .ap-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--border-radius);
            padding: var(--spacing-lg);
            margin: var(--spacing-md) 0;
            box-shadow: var(--shadow-md);
            transition: all 0.3s ease;
        }

        .ap-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 0.75rem 1.5rem rgba(0, 0, 0, 0.2);
        }

        /* Metrics */
        .ap-metric {
            text-align: center;
            padding: var(--spacing-lg);
            background: var(--bg-card);
            border-radius: var(--border-radius);
            border: 1px solid var(--border-color);
            box-shadow: var(--shadow-sm);
            transition: all 0.3s ease;
        }

        .ap-metric:hover {
            transform: translateY(-1px);
            box-shadow: var(--shadow-md);
        }

        .ap-metric .label {
            font-size: 0.875rem;
            color: var(--text-secondary);
            margin-bottom: var(--spacing-xs);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .ap-metric .value {
            font-size: 2rem;
            font-weight: bold;
            color: var(--text-primary);
            margin-bottom: var(--spacing-xs);
            font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
        }

        .ap-metric .delta-pos {
            color: var(--success-color);
            font-size: 0.875rem;
            font-weight: 600;
        }

        .ap-metric .delta-neg {
            color: var(--danger-color);
            font-size: 0.875rem;
            font-weight: 600;
        }

        /* Metric components (for summary cards) */
        .ap-metric-icon {
            font-size: 2rem;
            text-align: center;
            margin-bottom: var(--spacing-sm);
        }

        .ap-metric-value {
            font-size: 2.5rem;
            font-weight: bold;
            color: var(--text-primary);
            text-align: center;
            margin-bottom: var(--spacing-xs);
            font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
            line-height: 1.1;
        }

        .ap-metric-label {
            font-size: 1rem;
            color: var(--text-secondary);
            text-align: center;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Badges */
        .ap-badge {
            display: inline-block;
            padding: var(--spacing-xs) var(--spacing-sm);
            border-radius: var(--border-radius-sm);
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin: var(--spacing-xs);
        }

        .ap-badge.good {
            background: rgba(44, 160, 44, 0.1);
            color: var(--success-color);
            border: 1px solid rgba(44, 160, 44, 0.3);
        }

        .ap-badge.warn {
            background: rgba(255, 152, 0, 0.1);
            color: var(--warning-color);
            border: 1px solid rgba(255, 152, 0, 0.3);
        }

        .ap-badge.danger {
            background: rgba(214, 39, 40, 0.1);
            color: var(--danger-color);
            border: 1px solid rgba(214, 39, 40, 0.3);
        }

        /* Statistics */
        .ap-stat-grid {
            display: grid;
            gap: var(--spacing-md);
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        }

        .ap-stat-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--border-radius);
            padding: var(--spacing-lg);
            box-shadow: var(--shadow-sm);
        }

        .ap-stat-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: var(--spacing-sm) 0;
            border-bottom: 1px solid var(--border-color);
        }

        .ap-stat-item:last-child {
            border-bottom: none;
        }

        .ap-stat-label {
            font-weight: 500;
            color: var(--text-secondary);
        }

        .ap-stat-value {
            font-weight: bold;
            color: var(--text-primary);
            font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
        }

        .ap-stat-value.green {
            color: var(--success-color);
        }

        .ap-stat-value.red {
            color: var(--danger-color);
        }

        /* Position table enhancements */
        .stDataFrame {
            border: 1px solid var(--border-color);
            border-radius: var(--border-radius);
            overflow: hidden;
        }

        /* Animation utilities */
        .ap-fade {
            animation: fadeIn 0.6s ease-out;
        }

        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        /* Button enhancements */
        .stButton > button {
            border-radius: var(--border-radius-sm);
            border: none;
            font-weight: 500;
            transition: all 0.2s ease;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: var(--shadow-sm);
        }

        /* Responsive improvements */
        @media (max-width: 768px) {
            .ap-title {
                font-size: 1.5rem;
            }

            .ap-metric .value {
                font-size: 1.5rem;
            }

            .ap-stat-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    except Exception:
        pass


def _resolve_position_price(position: Any) -> float | str:
    """Return a price preferring last-day close over the current price."""

    for attr in ("lastday_price", "current_price"):
        candidate = getattr(position, attr, None)
        value = _safe_float(candidate)
        if value is not None:
            return value
    fallback = getattr(position, "current_price", None)
    if fallback in (None, ""):
        return ""
    return fallback


def _push_order_log(entry: dict[str, Any]) -> None:
    logs = st.session_state.setdefault(ORDER_LOG_KEY, [])
    # normalize timestamp
    entry = dict(entry)
    entry.setdefault("ts", datetime.now().isoformat())
    logs.insert(0, entry)
    # keep recent 50
    st.session_state[ORDER_LOG_KEY] = logs[:50]


def _load_persistent_sent_markers() -> dict[str, Any]:
    try:
        if not SENT_MARKERS_PATH.exists():
            return {}
        import json

        with SENT_MARKERS_PATH.open("r", encoding="utf8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return {}
        return data
    except Exception:
        return {}


def _save_persistent_sent_markers(markers: dict[str, Any]) -> None:
    try:
        SENT_MARKERS_PATH.parent.mkdir(parents=True, exist_ok=True)
        import json

        with SENT_MARKERS_PATH.open("w", encoding="utf8") as fh:
            json.dump(markers, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _load_schedule() -> dict[str, Any]:
    try:
        if not SCHEDULE_PATH.exists():
            return {}
        import json

        with SCHEDULE_PATH.open("r", encoding="utf8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _save_schedule(data: dict[str, Any]) -> None:
    try:
        SCHEDULE_PATH.parent.mkdir(parents=True, exist_ok=True)
        import json

        with SCHEDULE_PATH.open("w", encoding="utf8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _load_auto_rule_config() -> dict[str, Any]:
    try:
        if not CONFIG_PATH.exists():
            return {}
        import json

        with CONFIG_PATH.open("r", encoding="utf8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _save_auto_rule_config(cfg: dict[str, Any]) -> None:
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        import json

        with CONFIG_PATH.open("w", encoding="utf8") as fh:
            json.dump(cfg, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _load_notify_settings() -> dict[str, Any]:
    try:
        if not NOTIFY_PATH.exists():
            return {}
        import json

        with NOTIFY_PATH.open("r", encoding="utf8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _load_notify_test_log() -> list[dict[str, Any]]:
    p = Path("data") / "notify_test_log.json"
    try:
        if not p.exists():
            return []
        import json

        with p.open("r", encoding="utf8") as fh:
            return json.load(fh)
    except Exception:
        return []


def _save_notify_test_log(rows: list[dict[str, Any]]) -> None:
    p = Path("data") / "notify_test_log.json"
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        import json

        with p.open("w", encoding="utf8") as fh:
            json.dump(rows, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass


# initialize notify test log in session_state (safe after helpers defined)
try:
    if "notify_test_log" not in st.session_state:
        st.session_state["notify_test_log"] = _load_notify_test_log()
except Exception:
    st.session_state.setdefault("notify_test_log", [])


def _save_notify_settings(d: dict[str, Any]) -> None:
    try:
        NOTIFY_PATH.parent.mkdir(parents=True, exist_ok=True)
        import json

        with NOTIFY_PATH.open("w", encoding="utf8") as fh:
            json.dump(d, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _prune_old_sent_markers(days: int = 30) -> None:
    try:
        markers = _load_persistent_sent_markers()
        cutoff = datetime.now().date() - timedelta(days=days)
        keep: dict[str, Any] = {}
        for k, v in markers.items():
            # expecting keys like SYMBOL_today_close_YYYY-MM-DD
            parts = k.rsplit("_", 1)
            if len(parts) == 2:
                try:
                    d = datetime.fromisoformat(parts[1]).date()
                    if d >= cutoff:
                        keep[k] = v
                except Exception:
                    # keep unknown-format keys
                    keep[k] = v
            else:
                keep[k] = v
        _save_persistent_sent_markers(keep)
    except Exception:
        pass


def _today_key_for(symbol: str) -> str:
    today = datetime.now().date().isoformat()
    return f"{symbol}_today_close_{today}"


def _has_sent_today(symbol: str) -> bool:
    key = _today_key_for(symbol)
    ss = st.session_state.setdefault(SENT_MARKER_KEY, {})
    if ss.get(key):
        return True
    persisted = _load_persistent_sent_markers()
    return bool(persisted.get(key))


def _mark_sent_today(symbol: str) -> None:
    key = _today_key_for(symbol)
    ss = st.session_state.setdefault(SENT_MARKER_KEY, {})
    ss[key] = True
    st.session_state[SENT_MARKER_KEY] = ss
    try:
        persisted = _load_persistent_sent_markers()
        persisted[key] = True
        _save_persistent_sent_markers(persisted)
    except Exception:
        pass


def _render_order_logs() -> None:
    logs = st.session_state.get(ORDER_LOG_KEY, [])
    if not logs:
        return
    st.markdown("---")
    st.markdown("#### 発注ログ（直近）")
    for e in logs[:20]:
        ts = e.get("ts", "")
        sym = e.get("symbol", "")
        status = e.get("status") or ("success" if e.get("order_id") else "error")
        msg = e.get("msg") or e.get("error") or ""
        st.write(f"{ts} — {sym} — {status} — {msg}")


def _fetch_account_and_positions() -> tuple[Any, Any, list[Any]]:
    client = ba.get_client()
    account = client.get_account()
    positions = list(client.get_all_positions())
    return client, account, positions


def _days_held(entry_dt: pd.Timestamp | str | datetime | None) -> int | None:
    return calculate_business_holding_days(entry_dt)


@st.cache_data(ttl=300)  # 5分間キャッシュ
def _load_recent_prices(symbol: str, max_points: int = 30) -> list[float] | None:
    if not symbol:
        return None
    try:
        df = load_base_cache(symbol, rebuild_if_missing=False, prefer_precomputed_indicators=True)
    except Exception:
        df = None

    if df is not None and not getattr(df, "empty", True):
        for col in ("Close", "close", "Adj Close", "adj_close", "adj close"):
            if col not in df.columns:
                continue
            try:
                series = pd.to_numeric(df[col], errors="coerce").dropna().tail(max_points)
            except Exception:
                continue
            if not series.empty:
                return list(series.values)

        try:
            numeric_cols = df.select_dtypes(include=["number"])
        except Exception:
            numeric_cols = None
        if numeric_cols is not None and not numeric_cols.empty:
            try:
                series = pd.to_numeric(numeric_cols.iloc[:, 0], errors="coerce").dropna().tail(max_points)
            except Exception:
                series = pd.Series(dtype=float)
            if not series.empty:
                return list(series.values)

    # フォールバック: プロジェクト規約に従った順序でCSVファイルを確認
    candidates = [
        Path("data_cache") / "base" / f"{symbol}.csv",  # 指標付与済み
        Path("data_cache") / "full_backup" / f"{symbol}.csv",  # 原本
        Path("data_cache") / f"{symbol}.csv",  # レガシーパス
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
            cols = {c.lower(): c for c in df.columns}
            close_col = cols.get("close") or cols.get("adj close") or cols.get("adj_close")
            if close_col is None:
                continue
            series = pd.to_numeric(df[close_col], errors="coerce").dropna().tail(max_points)
            if series.empty:
                continue
            return list(series.values)
        except Exception:
            continue
    return None


def _extract_order_prices(order: Any) -> tuple[list[float], list[float], list[str]]:
    stops: list[float] = []
    limits: list[float] = []
    trails: list[str] = []

    def _maybe_add_price(value: Any, bucket: list[float]) -> None:
        try:
            price = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(price):
            return
        bucket.append(price)

    _maybe_add_price(getattr(order, "stop_price", None), stops)
    _maybe_add_price(getattr(order, "limit_price", None), limits)

    take_profit = getattr(order, "take_profit", None)
    if take_profit is not None:
        _maybe_add_price(getattr(take_profit, "limit_price", None), limits)

    stop_loss = getattr(order, "stop_loss", None)
    if stop_loss is not None:
        _maybe_add_price(getattr(stop_loss, "stop_price", None), stops)

    trail_price = getattr(order, "trail_price", None)
    _maybe_add_price(trail_price, stops)

    trail_percent = getattr(order, "trail_percent", None)
    if trail_percent not in (None, ""):
        try:
            perc = float(trail_percent)
        except (TypeError, ValueError):
            pass
        else:
            if math.isfinite(perc):
                trails.append(f"Trail {perc:g}%")

    legs = getattr(order, "legs", None)
    if legs:
        try:
            iterator = list(legs)
        except TypeError:
            iterator = []
        for leg in iterator:
            sub_stops, sub_limits, sub_trails = _extract_order_prices(leg)
            stops.extend(sub_stops)
            limits.extend(sub_limits)
            trails.extend(sub_trails)

    return stops, limits, trails


def _collect_open_exit_levels(client: Any) -> dict[str, dict[str, list[Any]]]:
    if client is None:
        return {}
    try:
        orders_obj = ba.get_open_orders(client)
    except Exception:
        return {}

    try:
        orders = list(orders_obj)
    except TypeError:
        try:
            orders = list(iter(orders_obj))
        except Exception:
            return {}
    except Exception:
        return {}

    levels: dict[str, dict[str, set[Any]]] = {}
    for order in orders:
        sym_raw = getattr(order, "symbol", "")
        try:
            sym_key = str(sym_raw).upper()
        except Exception:
            continue
        if not sym_key:
            continue
        stops, limits, trails = _extract_order_prices(order)
        if not stops and not limits and not trails:
            continue
        bucket = levels.setdefault(
            sym_key,
            {"stops": set(), "limits": set(), "trail": set()},
        )
        for price in stops:
            bucket["stops"].add(price)
        for price in limits:
            bucket["limits"].add(price)
        for note in trails:
            if note:
                bucket["trail"].add(str(note))

    result: dict[str, dict[str, list[Any]]] = {}
    for sym_key, data in levels.items():
        result[sym_key] = {
            "stops": sorted(data["stops"]),
            "limits": sorted(data["limits"]),
            "trail": sorted(data["trail"]),
        }
    return result


def _format_exit_prices(values: Iterable[float] | None) -> str:
    if not values:
        return "-"
    cleaned: list[float] = []
    for value in values:
        try:
            price = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(price):
            continue
        cleaned.append(price)
    if not cleaned:
        return "-"
    dedup: dict[float, float] = {}
    for price in cleaned:
        key = round(price, 6)
        dedup.setdefault(key, price)
    ordered = [dedup[key] for key in sorted(dedup)]
    formatted: list[str] = []
    for price in ordered:
        abs_price = abs(price)
        fmt = "{:,.4f}" if abs_price < 1 else "{:,.2f}"
        formatted.append(fmt.format(price))
    return " / ".join(formatted)


def _render_stop_cell(info: dict[str, list[Any]] | None) -> str:
    if not info:
        return "-"
    parts: list[str] = []
    price_part = _format_exit_prices(info.get("stops"))
    if price_part != "-":
        parts.append(price_part)
    trail_notes = [str(n) for n in info.get("trail", []) if n]
    if trail_notes:
        parts.append(", ".join(sorted(set(trail_notes))))
    return " ・ ".join(parts) if parts else "-"


def _render_limit_cell(info: dict[str, list[Any]] | None) -> str:
    if not info:
        return "-"
    return _format_exit_prices(info.get("limits"))


def _attach_exit_levels(pos_df: pd.DataFrame, client: Any) -> pd.DataFrame:
    if pos_df.empty or "銘柄" not in pos_df.columns:
        return pos_df
    try:
        levels = _collect_open_exit_levels(client)
    except Exception:
        levels = {}
    pos_df = pos_df.copy()
    symbols = pos_df["銘柄"].astype(str).str.upper()
    pos_df["ストップ価格"] = [_render_stop_cell(levels.get(sym)) for sym in symbols]
    pos_df["リミット価格"] = [_render_limit_cell(levels.get(sym)) for sym in symbols]
    return pos_df


def _positions_to_df(positions, client=None) -> pd.DataFrame:
    symbols_upper = [str(getattr(p, "symbol", "")).upper() for p in positions]
    symbol_set = {s for s in symbols_upper if s}
    entry_map: dict[str, Any] = {}
    if client and symbol_set:
        try:
            entry_map.update(fetch_entry_dates_from_alpaca(client, list(symbol_set)))
        except Exception:
            entry_map = {}
    try:
        cached_entries = load_entry_dates()
    except Exception:
        cached_entries = {}
    for sym, value in cached_entries.items():
        try:
            key = str(sym).upper()
        except Exception:
            continue
        if not key or key not in symbol_set:
            continue
        if key not in entry_map or entry_map[key] is None:
            entry_map[key] = value

    mapping_path = Path("data/symbol_system_map.json")
    symbol_map: dict[str, str] = {}
    if mapping_path.exists():
        try:
            raw_map = json.loads(mapping_path.read_text())
            symbol_map = {str(k).upper(): str(v) for k, v in raw_map.items()}
        except Exception:
            symbol_map = {}

    records: list[dict[str, object]] = []
    for pos in positions:
        sym_raw = getattr(pos, "symbol", "")
        sym = str(sym_raw)
        sym_key = sym.upper()
        held = _days_held(entry_map.get(sym_key))
        system_value = symbol_map.get(sym_key, "unknown")
        limit = HOLD_LIMITS.get(str(system_value).lower())
        limit_reached = False
        exit_hint = ""
        if held is not None and limit:
            limit_reached = held >= int(limit)
            if limit_reached:
                exit_hint = f"{limit}日経過で手仕切り検討"
        records.append(
            {
                "銘柄": sym,
                "数量": getattr(pos, "qty", ""),
                "平均取得単価": getattr(pos, "avg_entry_price", ""),
                "現在値": _resolve_position_price(pos),
                "含み損益": getattr(pos, "unrealized_pl", ""),
                "保有日数": held if held is not None else "-",
                "経過日手仕切り": exit_hint,
                "システム": system_value,
                "_limit_days": limit,
                "_limit_reached": limit_reached,
            }
        )
    df = pd.DataFrame(records)
    if df.empty:
        return df

    numeric_cols = ["平均取得単価", "現在値", "含み損益"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "銘柄" in df.columns:
        df["銘柄"] = df["銘柄"].astype(str)
    if "システム" in df.columns:
        df["システム"] = df["システム"].fillna("unknown").astype(str)

    try:
        # ポジション数が多いときは点数を抑えて軽量化
        n_points = 20 if len(df) > 15 else 45
        symbol_series = df["銘柄"].astype(str)
        price_series = [_load_recent_prices(sym, max_points=n_points) or [] for sym in symbol_series]
        df["直近価格チャート"] = price_series
    except Exception:
        pass
    return df


def _build_position_map(positions: Iterable[Any]) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for pos in positions:
        try:
            symbol = str(getattr(pos, "symbol", ""))
        except Exception:
            continue
        symbol_key = symbol.upper()
        if not symbol_key:
            continue
        mapping[symbol_key] = pos
    return mapping


def _parse_exit_quantity(position: Any) -> int | None:
    candidates = [
        getattr(position, "qty_available", None),
        getattr(position, "qty", None),
    ]
    for raw in candidates:
        if raw in (None, "", "-"):
            continue
        try:
            value = Decimal(str(raw).replace(",", ""))
        except (InvalidOperation, ValueError, TypeError):
            continue
        value = abs(value)
        if value == 0:
            continue
        if value != value.to_integral_value():
            # Fractional shares are not supported via this shortcut.
            continue
        qty = int(value)
        if qty > 0:
            return qty
    return None


def _determine_exit_side(position: Any) -> tuple[str, str]:
    side_raw = getattr(position, "side", "")
    side = str(side_raw).lower()
    if side == "short":
        return "buy", "買い戻し"
    return "sell", "売却"


def _render_exit_actions(
    df: pd.DataFrame,
    position_map: dict[str, Any],
    client: Any,
) -> None:
    if df.empty:
        return

    # 経過日数チェック可能なポジションがあるかどうか
    has_limit_info = "_limit_days" in df.columns and "_limit_reached" in df.columns
    if not has_limit_info:
        st.markdown("#### 📅 経過日手仕切り管理")
        st.info(
            "保有日数の情報が不足しています。システムマッピング（`data/symbol_system_map.json`）を確認してください。"
        )
        return

    # デバッグ情報表示（開発時のみ）
    if DEBUG_MODE:
        st.markdown("**デバッグ情報**")
        limit_info_df = df[["銘柄", "システム", "保有日数", "_limit_days", "_limit_reached"]].copy()
        st.dataframe(limit_info_df, width="stretch")

    # 上限日数に近いか、すでに到達したポジションを特定
    eligible_df = df[
        (df["_limit_days"].notna())
        & (
            df["_limit_reached"]
            | (
                df["保有日数"].notna()
                & (
                    pd.to_numeric(df["保有日数"], errors="coerce")
                    >= pd.to_numeric(df["_limit_days"], errors="coerce") * 0.8
                )
            )
        )
    ].copy()

    if eligible_df.empty:
        # 該当ポジションがない場合でも、セクションは表示して情報を提供
        st.markdown("#### 📅 経過日手仕切り管理")
        st.info("現在、上限日数に近づいているポジションはありません。")
        return

    st.markdown("#### 📅 経過日手仕切り管理")
    st.caption("保有日数が上限に近づいているか到達したポジションの決済管理です。")

    # 上限到達ポジション
    limit_reached_df = eligible_df[eligible_df["_limit_reached"]].copy()

    if not limit_reached_df.empty:
        st.markdown("**⚠️ 上限到達（即時決済推奨）**")
        try:
            mask = df["_limit_reached"].astype(bool)
        except Exception:
            mask = df["_limit_reached"].apply(lambda x: bool(x))
        eligible = df[mask].copy()
    else:
        st.markdown("**⏰ 上限接近中のポジション**")
        eligible = eligible_df

    status_map: dict[str, Any] = st.session_state.setdefault(EXIT_STATE_KEY, {})
    is_na = getattr(pd, "isna", None)
    eligible = eligible.reset_index(drop=True)

    # まとめて決済 UI: 対象シンボルを選んで一括で成行決済を送信
    try:
        eligible_symbols = [str(s).upper() for s in eligible["銘柄"].tolist()]
    except Exception:
        eligible_symbols = []
    if eligible_symbols:
        st.markdown("**まとめて決済**")
        cols = st.columns([4, 1])
        with cols[0]:
            to_exit = st.multiselect("決済する銘柄を選択", eligible_symbols, default=eligible_symbols)
            st.selectbox("割合", [100, 75, 50, 25], index=0, key="batch_pct")
        with cols[1]:
            if st.button("まとめて成行決済", key="batch_exit_submit"):
                st.session_state["batch_confirm_request"] = to_exit
        # バッチ確認 UI
        if st.session_state.get("batch_confirm_request"):
            pending = st.session_state.get("batch_confirm_request") or []
            st.info(f"まとめて決済の確認: {', '.join(pending)}")
            c_yes, c_no = st.columns([1, 1])
            with c_yes:
                if st.button("はい、送信する", key="batch_confirm_yes"):
                    rows = []
                    for sym in pending:
                        pos = position_map.get(str(sym).upper())
                        qty = _parse_exit_quantity(pos) if pos is not None else None
                        if qty is None:
                            st.warning(f"{sym}: 決済数量が特定できずスキップしました。")
                            continue
                        side = "long" if getattr(pos, "side", "").lower() == "long" else "short"
                        apply_pct = int(st.session_state.get("batch_pct", 100))
                        apply_qty = max(1, int(qty * apply_pct / 100))
                        rows.append(
                            {
                                "symbol": sym,
                                "qty": apply_qty,
                                "position_side": side,
                                "system": "",
                                "when": "today_close",
                            }
                        )
                    if rows:
                        try:
                            exit_df = pd.DataFrame(rows)
                            res = submit_exit_orders_df(exit_df, paper=True, tif="CLS", notify=True)
                            st.success(f"まとめて決済リクエストを送信しました ({len(res)} 件)")
                            sent = st.session_state.setdefault(SENT_MARKER_KEY, {})
                            for r in rows:
                                _push_order_log(
                                    {
                                        "symbol": r["symbol"],
                                        "status": "submitted",
                                        "msg": "batch exit requested",
                                    }
                                )
                                _mark_sent_today(r["symbol"])
                            st.session_state[SENT_MARKER_KEY] = sent
                            try:
                                _save_persistent_sent_markers(sent)
                            except Exception:
                                pass
                            try:
                                # シンプルな通知ロジック（Slack優先→Discordフォールバック）
                                if st.session_state.get("enable_notifications", True):
                                    notifier = Notifier(platform="auto")
                                    syms = ", ".join([r["symbol"] for r in rows])
                                    notifier.send("まとめて決済実行", f"送信銘柄: {syms}")
                            except Exception:
                                pass
                        except Exception as e:
                            st.error(f"まとめて決済に失敗しました: {e}")
                    st.session_state.pop("batch_confirm_request", None)
            with c_no:
                if st.button("キャンセル", key="batch_confirm_no"):
                    st.session_state.pop("batch_confirm_request", None)

    for _, row in eligible.iterrows():
        symbol_raw = row.get("銘柄", "")
        try:
            symbol = str(symbol_raw).upper()
        except Exception:
            symbol = ""
        if not symbol:
            continue

        position = position_map.get(symbol)
        if position is None:
            st.warning(f"{symbol}: ポジション情報が見つかりませんでした。手動でご確認ください。")
            continue

        qty = _parse_exit_quantity(position)
        if qty is None:
            st.warning(f"{symbol}: 決済数量を特定できませんでした。手動で注文してください。")
            continue

        exit_side, side_label = _determine_exit_side(position)
        system_value = row.get("システム", "unknown")
        limit_value = row.get("_limit_days")
        held_value = row.get("保有日数")

        if held_value in (None, "", "-") or (is_na and is_na(held_value)):
            held_text = "-"
        else:
            try:
                held_text = f"{int(held_value)}日"
            except Exception:
                held_text = str(held_value)

        if limit_value in (None, "") or (is_na and is_na(limit_value)):
            limit_text = "-"
        else:
            try:
                limit_text = f"{int(limit_value)}日"
            except Exception:
                limit_text = str(limit_value)

        # compact row layout: symbol + meta in columns and small action button
        row_cols = st.columns([2, 1, 1, 1])
        with row_cols[0]:
            st.markdown(
                (
                    f"**{symbol}**  "
                    f"<span style='color:#9aa4b2'>システム:{system_value} 保有:{held_text} "
                    f"上限:{limit_text}</span>"
                ),
                unsafe_allow_html=True,
            )
        with row_cols[1]:
            st.caption(f"数量: {qty}")
        with row_cols[2]:
            st.caption(f"保有日数: {held_text}")
        with row_cols[3]:
            existing = status_map.get(symbol)
            # 既に送信済みマーカーがあれば disabled にする
            disabled_sent = _has_sent_today(symbol)
            disabled = bool(existing and existing.get("success")) or disabled_sent
            # 部分決済割合（%）
            pct_key = f"partial_pct_{symbol}"
            pct = st.slider("割合", min_value=10, max_value=100, value=100, step=10, key=pct_key)
            exit_qty = max(1, int(qty * pct / 100))
            button_label = f"{side_label}成行 {exit_qty}株 ({pct}%)"
            clicked = st.button(button_label, key=f"exit_button_{symbol}", disabled=disabled)
            feedback = st.empty()

        if clicked:
            # 個別確認フロー: pending マーカーを立てる
            st.session_state[f"confirm_pending_{symbol}"] = True
        if st.session_state.get(f"confirm_pending_{symbol}"):
            c1, c2 = st.columns([1, 1])
            st.info(f"{symbol} を {qty} 株、{side_label} 成行で決済します。確認してください。")
            with c1:
                if st.button("はい、送信する", key=f"confirm_yes_{symbol}"):
                    confirmed = True
                else:
                    confirmed = False
            with c2:
                if st.button("キャンセル", key=f"confirm_no_{symbol}"):
                    confirmed = False
                    st.session_state.pop(f"confirm_pending_{symbol}", None)
            if confirmed:
                try:
                    if client is None:
                        raise RuntimeError("Alpaca クライアントを初期化できませんでした。")
                    order = ba.submit_order_with_retry(
                        client,
                        symbol,
                        qty,
                        side=exit_side,
                        order_type="market",
                        time_in_force="CLS",
                        retries=2,
                        backoff_seconds=0.5,
                        rate_limit_seconds=0.2,
                    )
                except Exception as exc:  # noqa: BLE001
                    status_map[symbol] = {"success": False, "error": str(exc)}
                    feedback.error(f"{symbol}: 決済注文の送信に失敗しました: {exc}")
                else:
                    order_id = getattr(order, "id", None)
                    status_map[symbol] = {
                        "success": True,
                        "order_id": order_id,
                        "side": exit_side,
                        "qty": qty,
                    }
                    msg = f"{symbol}: 決済注文を送信しました"
                    if order_id:
                        msg += f"（注文ID: {order_id}）"
                    feedback.success(msg)
                    # push order log and mark sent
                    _push_order_log(
                        {
                            "symbol": symbol,
                            "status": "submitted",
                            "order_id": str(order_id),
                            "msg": msg,
                        }
                    )
                    _mark_sent_today(symbol)
            st.session_state[EXIT_STATE_KEY] = status_map
        elif existing:
            if existing.get("success"):
                order_id = existing.get("order_id")
                msg = f"{symbol}: 決済注文済み"
                if order_id:
                    msg += f"（注文ID: {order_id}）"
                feedback.info(msg)
            else:
                feedback.warning(f"{symbol}: 直近の注文送信でエラーが発生しました: {existing.get('error')}")


def _group_by_system(
    df: pd.DataFrame,
    symbol_map: dict[str, str],
) -> dict[str, pd.DataFrame]:
    if df.empty:
        return {}

    work = df.copy()
    try:
        work["評価額"] = work["数量"].astype(float) * work["現在値"].astype(float)
    except Exception:
        return {}
    work["system"] = work["銘柄"].map(symbol_map).fillna("unknown")

    grouped: dict[str, pd.DataFrame] = {}
    for system_value, g in work.groupby("system"):
        cleaned = g[["銘柄", "評価額"]].copy()
        cleaned["評価額"] = pd.to_numeric(cleaned["評価額"], errors="coerce").fillna(0.0)
        grouped[str(system_value)] = cleaned
    return grouped


def main() -> None:
    _inject_css()

    # Debug banner (only in debug mode)
    if DEBUG_MODE:
        try:
            debug_html = (
                "<div style='position:fixed;right:8px;top:8px;background:#111;"
                "padding:6px 10px;border-radius:6px;opacity:0.9;z-index:9999;"
                "color:#9ae6b4;'>DEBUG " + datetime.now().isoformat() + "</div>"
            )
            st.markdown(debug_html, unsafe_allow_html=True)
        except Exception:
            pass

    # タイトル＋ツールバー（右端に 手動更新 と 最終更新 を横並び）
    st.markdown(
        "<div class='ap-title'>Alpaca <span class='accent'>現在状況</span></div>",
        unsafe_allow_html=True,
    )
    tz_tokyo = ZoneInfo("Asia/Tokyo")
    tz_newyork = ZoneInfo("America/New_York")
    now_tokyo = datetime.now(tz_tokyo)
    now_newyork = datetime.now(tz_newyork)
    nyse_status = _get_nyse_status(now_newyork)
    st.caption(
        " / ".join(
            [
                f"日本時間: {_format_datetime_with_weekday(now_tokyo)}",
                (f"ニューヨーク時間: {_format_datetime_with_weekday(now_newyork)} （{nyse_status}）"),
            ]
        )
    )
    # 改善されたツールバー（レイアウト衝突を修正）
    st.markdown("<div class='ap-toolbar ap-fade'>", unsafe_allow_html=True)
    toolbar_cols = st.columns([4, 2, 2, 2])

    # 左側：スケジュール関連
    with toolbar_cols[0]:
        st.caption("🕒 自動ルールスケジュール")
        saved = _load_schedule() or {}
        saved_time = None
        try:
            saved_time_iso = saved.get("time")
            if saved_time_iso:
                saved_time = datetime.fromisoformat(saved_time_iso).time()
        except Exception:
            saved_time = None

        # コンパクトなスケジュール設定
        schedule_inner_cols = st.columns([2, 1.5, 1])
        with schedule_inner_cols[0]:
            run_time = st.time_input(
                "実行時刻",
                value=saved_time or datetime.now().time(),
                label_visibility="collapsed",
            )
        with schedule_inner_cols[1]:
            opt_in = st.checkbox(
                "参加",
                value=bool(saved.get("opt_in", False)),
                key="auto_rule_opt_in",
                help="自動ルールに参加（経過日や損益閾値での自動決済）",
            )
        with schedule_inner_cols[2]:
            if st.button("保存", key="save_schedule", width="stretch"):
                _save_schedule(
                    {
                        "time": datetime.combine(datetime.now().date(), run_time).isoformat(),
                        "opt_in": bool(opt_in),
                    }
                )
                st.success("スケジュール保存済", icon="✅")

    # 中央：手動更新
    with toolbar_cols[1]:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        if st.button("🔄 手動更新", key="manual_refresh", width="stretch"):
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # 自動更新機能
    with toolbar_cols[2]:
        auto_refresh = st.checkbox("⏰ 自動更新", key="auto_refresh")
        if auto_refresh:
            refresh_interval = st.selectbox("更新間隔", [30, 60, 120, 300], index=1, key="refresh_interval")
            # 自動更新用のJavaScript（実際の実装ではst.rerunとタイマーを使用）
            st.caption(f"⏰ {refresh_interval}秒間隔")

    # 右側：時刻表示
    with toolbar_cols[3]:
        st.markdown("<div style='text-align: right;'>", unsafe_allow_html=True)
        st.caption(f"最終更新: {datetime.now().strftime('%H:%M:%S')}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # 自動ルール実行ボタン（スケジュールの下に分離）
    if opt_in:
        st.markdown("---")
        st.markdown("#### 🤖 自動ルール")
        st.caption("設定した時間に経過日や損益閾値による自動決済を実行します。")

        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button(
                "自動ルールを今すぐ実行 (手動)",
                key="auto_rule_run_manual_top",
                type="primary",
            ):
                st.session_state.setdefault("auto_rule_trigger", datetime.now().isoformat())
        with col2:
            last_run = st.session_state.get("last_auto_rule_run")
            st.caption(f"最後の実行: {last_run or '未実行'}")

        # 自動ルールの設定表示
        st.expander("⚙️ 自動ルール設定", expanded=False).markdown(
            f"""
        - **上限保有日数**: {", ".join([f"{k}: {v}日" for k, v in HOLD_LIMITS.items()])}
        - **損益閾値**: 各システムで -20% 以下は自動決済対象
        - **実行時刻**: {run_time.strftime("%H:%M")}
        - **部分決済**: 100% （全量決済）
        """
        )
    else:
        st.markdown("---")
        st.info("🤖 自動ルール機能を使用するには、上記の「参加」チェックボックスを有効にしてください。")

    # 自動スケジュール検出（簡易）: ページロード時に時刻を過ぎていて未実行ならトリガー
    try:
        if opt_in:
            now_local = datetime.now()
            scheduled_dt = datetime.combine(now_local.date(), run_time)
            last_run_iso = st.session_state.get("last_auto_rule_run")
            last_run_dt = None
            if last_run_iso:
                try:
                    last_run_dt = datetime.fromisoformat(str(last_run_iso))
                except Exception:
                    last_run_dt = None
            # if we haven't run today and current time past scheduled time
            cond1 = now_local >= scheduled_dt
            cond2 = last_run_dt is None or last_run_dt.date() < now_local.date()
            if cond1 and cond2:
                st.session_state.setdefault("auto_rule_trigger", datetime.now().isoformat())
    except Exception:
        pass

    try:
        client, account, positions = _fetch_account_and_positions()
    except Exception as exc:  # pragma: no cover
        st.error(f"データ取得に失敗しました: {exc}")
        return
    position_map = _build_position_map(positions)

    # Shortable map: check which symbols are shortable (used for warnings)
    try:
        symbols_for_check = [s.upper() for s in position_map.keys() if s]
        shortable_map = ba.get_shortable_map(client, symbols_for_check) if symbols_for_check else {}
    except Exception:
        shortable_map = {}
    st.session_state.setdefault("shortable_map", shortable_map)

    # Load persisted auto-rule config if present and merge
    try:
        disk_cfg = _load_auto_rule_config() or {}
        for k, v in disk_cfg.items():
            if k in AUTO_RULE_CONFIG and isinstance(v, dict):
                AUTO_RULE_CONFIG[k].update(v)
    except Exception:
        pass

    # Load notify settings for UI defaults
    notify_defaults = _load_notify_settings() or {}
    st.session_state.setdefault("notify_defaults", notify_defaults)

    # Load persistent sent markers and merge into session state to prevent duplicates
    persistent_sent = _load_persistent_sent_markers()
    ss_sent = st.session_state.setdefault(SENT_MARKER_KEY, {})
    for k, v in persistent_sent.items():
        ss_sent.setdefault(k, v)
    st.session_state[SENT_MARKER_KEY] = ss_sent

    # メトリクス行
    st.markdown("<div class='ap-card ap-fade'>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    equity = getattr(account, "equity", "-")
    cash = getattr(account, "cash", "-")
    buying_power = getattr(account, "buying_power", "-")
    last_equity = getattr(account, "last_equity", None)

    equity_value = _safe_float(equity)
    buying_power_value = _safe_float(buying_power)
    last_equity_value = _safe_float(last_equity)

    delta = None
    if equity_value is not None and last_equity_value is not None:
        delta = equity_value - last_equity_value

    ratio = None
    if equity_value not in (None, 0) and buying_power_value is not None:
        try:
            ratio = buying_power_value / equity_value
        except ZeroDivisionError:
            ratio = None

    def _metric_html(label: str, value: str, delta_val: float | None = None) -> str:
        d = ""
        if delta_val is not None:
            klass = "delta-pos" if delta_val >= 0 else "delta-neg"
            arrow = "▲" if delta_val >= 0 else "▼"
            d = f"<div class='{klass}'>{arrow} {_fmt_money(delta_val)}</div>"
        return f"<div class='ap-metric'><div class='label'>{label}</div><div class='value'>{value}</div>{d}</div>"

    with c1:
        st.markdown(
            _metric_html("総資産", _fmt_money(equity), delta),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            _metric_html("現金", _fmt_money(cash)),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            _metric_html("余力", _fmt_money(buying_power)),
            unsafe_allow_html=True,
        )
    with c4:
        ratio_text = f"{ratio * 100:.1f}%" if ratio is not None else "-"
        st.markdown(
            _metric_html("余力比率", ratio_text),
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # 口座状態バッジ
    flags = []
    try:
        if getattr(account, "trading_blocked", False):
            flags.append(("取引停止", "danger"))
        if getattr(account, "pattern_day_trader", False):
            flags.append(("PDT", "warn"))
        if not flags:
            flags.append(("正常", "good"))
    except Exception:
        pass
    st.markdown(
        " ".join([f"<span class='ap-badge {k}'>{t}</span>" for t, k in flags]),
        unsafe_allow_html=True,
    )

    # タブ
    tab_summary, tab_pos, tab_alloc = st.tabs(["サマリー", "ポジション", "配分グラフ"])

    with tab_pos:
        st.markdown("<div class='ap-section'>保有ポジション</div>", unsafe_allow_html=True)

        # ポジション一覧の表示
        pos_df = _positions_to_df(positions, client)
        pos_df = _attach_exit_levels(pos_df, client)

        if not pos_df.empty:
            numeric_cols = ["数量", "平均取得単価", "現在値", "含み損益"]
            for col in numeric_cols:
                if col in pos_df.columns:
                    pos_df[col] = pd.to_numeric(pos_df[col], errors="coerce")

        if pos_df.empty:
            st.info("ポジションはありません。")
        else:
            # ポジション統計サマリ
            total_positions = len(pos_df)
            try:
                total_pnl = pos_df["含み損益"].sum() if "含み損益" in pos_df.columns else 0
                winning_positions = len(pos_df[pos_df["含み損益"] > 0]) if "含み損益" in pos_df.columns else 0
                losing_positions = len(pos_df[pos_df["含み損益"] < 0]) if "含み損益" in pos_df.columns else 0
            except Exception:
                total_pnl = 0
                winning_positions = 0
                losing_positions = 0

            # サマリ表示
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("保有銘柄数", total_positions)
            with col2:
                st.metric("合計含み損益", f"${total_pnl:,.0f}")
            with col3:
                st.metric(
                    "勝ちポジション",
                    f"{winning_positions}銘柄",
                    delta=f"{winning_positions}/{total_positions}",
                )
            with col4:
                st.metric(
                    "負けポジション",
                    f"{losing_positions}銘柄",
                    delta=f"-{losing_positions}/{total_positions}",
                )

            # フィルタリングセクション
            st.markdown("#### 🔍 フィルタリング")
            filter_cols = st.columns([2, 2, 1])

            # 銘柄検索
            with filter_cols[0]:
                search_symbol = st.text_input("銘柄で検索", placeholder="例: AAPL", key="search_symbol")
                if search_symbol:
                    search_upper = search_symbol.upper().strip()
                    pos_df = pos_df[pos_df["銘柄"].str.contains(search_upper, na=False)]

            # システム絞り込み
            with filter_cols[1]:
                if "システム" in pos_df.columns:
                    raw_systems = pos_df["システム"].fillna("unknown").unique()
                    systems = sorted(str(s) for s in raw_systems)
                    selected_systems = st.multiselect(
                        "システム絞り込み",
                        systems,
                        default=systems,
                        key="pos_filter_systems",
                    )
                    pos_df = pos_df[pos_df["システム"].astype(str).isin(selected_systems)]

            # 損益フィルタ
            with filter_cols[2]:
                pnl_filter = st.selectbox(
                    "損益フィルタ",
                    ["全て", "利益のみ", "損失のみ", "大幅損失(-10%以下)"],
                    key="pnl_filter",
                )
                if pnl_filter == "利益のみ" and "含み損益" in pos_df.columns:
                    pos_df = pos_df[pos_df["含み損益"] > 0]
                elif pnl_filter == "損失のみ" and "含み損益" in pos_df.columns:
                    pos_df = pos_df[pos_df["含み損益"] < 0]
                elif pnl_filter == "大幅損失(-10%以下)":
                    try:

                        def calc_pnl_pct(r):
                            try:
                                current = float(r.get("現在値", 0))
                                avg = float(r.get("平均取得単価", 0))
                                return (current / avg - 1) * 100 if avg else 0
                            except Exception:
                                return 0

                        pos_df = pos_df[pos_df.apply(calc_pnl_pct, axis=1) <= -10]
                    except Exception:
                        pass

            # 派生列: 損益率(%)
            try:

                def _pnl_ratio(r):
                    try:
                        p = float(r.get("現在値", 0))
                        a = float(r.get("平均取得単価", 0))
                        return (p / a - 1) * 100 if a else 0.0
                    except Exception:
                        return 0.0

                pos_df["損益率(%)"] = pos_df.apply(_pnl_ratio, axis=1)
            except Exception:
                pass

            # 並び替え
            sort_key = st.selectbox(
                "並び替え",
                ["含み損益", "損益率(%)", "保有日数", "銘柄"],
                index=0,
                key="pos_sort",
            )
            ascending = st.toggle("昇順", value=False, key="pos_asc")
            try:
                pos_df = pos_df.sort_values(sort_key, ascending=ascending)
            except Exception:
                pass

            # 表示用データフレームの準備
            display_df = pos_df.drop(columns=["_limit_days", "_limit_reached"], errors="ignore")

            # カラム設定
            col_cfg: dict[str, Any] = {}
            if "数量" in display_df.columns:
                col_cfg["数量"] = st.column_config.NumberColumn(format="%.0f")
            if "平均取得単価" in display_df.columns:
                col_cfg["平均取得単価"] = st.column_config.NumberColumn(format="%.2f")
            if "現在値" in display_df.columns:
                col_cfg["現在値"] = st.column_config.NumberColumn(format="%.2f")
            if "含み損益" in display_df.columns:
                col_cfg["含み損益"] = st.column_config.NumberColumn(format="%.2f")
            if "損益率(%)" in display_df.columns:
                col_cfg["損益率(%)"] = st.column_config.ProgressColumn(min_value=-20, max_value=20, format="%.1f%%")
            if "ストップ価格" in display_df.columns:
                col_cfg["ストップ価格"] = st.column_config.Column(
                    width="medium",
                    help="未約定のストップ系注文価格（複数は / 区切り表示）。",
                )
            if "リミット価格" in display_df.columns:
                col_cfg["リミット価格"] = st.column_config.Column(
                    width="medium",
                    help="未約定のリミット/テイクプロフィット注文価格（複数は / 区切り表示）。",
                )
            if "直近価格チャート" in display_df.columns:
                col_cfg["直近価格チャート"] = st.column_config.LineChartColumn(
                    label="直近価格チャート",
                    width="small",
                    help="過去数週間の終値推移をスパークラインで表示します。",
                )

            # データフレームの表示
            try:
                st.dataframe(
                    display_df,
                    width="stretch",
                    hide_index=True,
                    column_config=col_cfg,
                )
            except Exception:
                st.dataframe(display_df, width="stretch", hide_index=True)

            # エクスポート機能
            st.markdown("#### 📥 データエクスポート")
            export_cols = st.columns(4)

            # 共通データの準備
            try:
                out_df = pos_df.drop(columns=["_limit_days", "_limit_reached"], errors="ignore")
            except Exception:
                out_df = pos_df.copy() if not pos_df.empty else pd.DataFrame()

            with export_cols[0]:
                try:
                    csv = out_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📊 ポジションCSV",
                        csv,
                        file_name=f"positions_{datetime.now().strftime('%Y%m%d')}.csv",
                    )
                except Exception:
                    pass

            with export_cols[1]:
                # JSON形式でのエクスポート
                try:
                    json_data = out_df.to_json(orient="records", indent=2)
                    st.download_button(
                        "📄 ポジションJSON",
                        json_data,
                        file_name=f"positions_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json",
                    )
                except Exception:
                    pass

            with export_cols[2]:
                # 統計情報のエクスポート
                try:
                    local_stats_data = {
                        "export_date": datetime.now().isoformat(),
                        "account_equity": _fmt_money(equity),
                        "cash": _fmt_money(cash),
                        "buying_power": _fmt_money(buying_power),
                        "total_positions": total_positions,
                        "winning_positions": winning_positions,
                        "losing_positions": losing_positions,
                        "total_pnl": _fmt_money(total_pnl),
                    }
                    stats_json = json.dumps(local_stats_data, indent=2, ensure_ascii=False)
                    st.download_button(
                        "📈 統計JSON",
                        stats_json,
                        file_name=f"stats_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json",
                    )
                except Exception:
                    pass

            with export_cols[3]:
                # 全データの一括エクスポート
                try:
                    all_data = {
                        "export_timestamp": datetime.now().isoformat(),
                        "account_info": {
                            "equity": equity,
                            "cash": cash,
                            "buying_power": buying_power,
                            "last_equity": last_equity,
                        },
                        "positions": (out_df.to_dict("records") if not out_df.empty else []),
                        "statistics": local_stats_data,
                    }
                    all_json = json.dumps(all_data, indent=2, ensure_ascii=False)
                    st.download_button(
                        "🗃️ 全データJSON",
                        all_json,
                        file_name=f"alpaca_dashboard_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                        mime="application/json",
                    )
                except Exception:
                    pass

            # 経過日手仕切りアクション
            _render_exit_actions(pos_df, position_map, client)

        # 自動ルールのトリガー処理（オプトイン + 実行ボタンで動作）
        if st.session_state.get("auto_rule_trigger"):
            trigger_ts = st.session_state.pop("auto_rule_trigger", None)
            st.info(f"自動ルールを実行中 (トリガー: {trigger_ts})")
            auto_rows = []
            try:
                for _, r in pos_df.iterrows():
                    try:
                        limit_reached = bool(r.get("_limit_reached"))
                    except Exception:
                        limit_reached = False
                    pnl_pct = 0.0
                    try:
                        pnl_pct = float(r.get("損益率(%)", 0.0))
                    except Exception:
                        pnl_pct = 0.0
                    system_name = str(r.get("システム", "")).strip() or "unknown"
                    cfg = AUTO_RULE_CONFIG.get(system_name, {})
                    threshold = float(cfg.get("pnl_threshold", -20.0))
                    partial_pct = int(cfg.get("partial_pct", 100))
                    if limit_reached or pnl_pct <= threshold:
                        sym = str(r.get("銘柄", "")).upper()
                        pos = position_map.get(sym)
                        qty = _parse_exit_quantity(pos) if pos is not None else None
                        if qty:
                            apply_qty = max(1, int(qty * partial_pct / 100))
                            auto_rows.append(
                                {
                                    "symbol": sym,
                                    "qty": apply_qty,
                                    "position_side": getattr(pos, "side", ""),
                                    "system": r.get("システム", ""),
                                    "when": "today_close",
                                }
                            )
            except Exception:
                auto_rows = []

            if auto_rows:
                try:
                    df_auto = pd.DataFrame(auto_rows)
                    res = submit_exit_orders_df(df_auto, paper=True, tif="CLS", notify=True)
                    st.success(f"自動ルールによるまとめて決済を送信しました ({len(res)} 件)")
                    for r in auto_rows:
                        _push_order_log(
                            {
                                "symbol": r["symbol"],
                                "status": "auto_submitted",
                                "msg": "auto rule exit",
                            }
                        )
                        _mark_sent_today(r["symbol"])
                    try:
                        # シンプルな通知ロジック（Slack優先→Discordフォールバック）
                        if st.session_state.get("enable_notifications", True):
                            notifier = Notifier(platform="auto")
                            syms = ", ".join([r["symbol"] for r in auto_rows])
                            notifier.send("自動ルール: まとめて決済実行", f"送信銘柄: {syms}")
                    except Exception:
                        pass
                    # 記録: 最終自動実行時刻
                    try:
                        st.session_state["last_auto_rule_run"] = datetime.now().isoformat()
                    except Exception:
                        pass
                except Exception as e:
                    st.error(f"自動ルール決済に失敗しました: {e}")

        # 未約定注文の一覧とキャンセル（ポジションタブ下部）
        st.markdown("---")
        st.markdown("#### 未約定注文の確認とキャンセル")
        try:
            open_orders = list(ba.get_open_orders(client))
        except Exception:
            open_orders = []
        if not open_orders:
            st.info("未約定注文はありません。")
        else:
            try:
                rows = [
                    {
                        "symbol": getattr(o, "symbol", ""),
                        "qty": getattr(o, "qty", ""),
                        "side": getattr(o, "side", ""),
                        "type": getattr(o, "type", ""),
                        "id": getattr(o, "id", ""),
                    }
                    for o in open_orders
                ]
                st.table(pd.DataFrame(rows))
            except Exception:
                st.write(open_orders)
            c1, c2 = st.columns([3, 1])
            with c2:
                if st.button("未約定を全てキャンセル", key="cancel_all_orders"):
                    try:
                        ba.cancel_all_orders(client)
                        st.success("未約定注文をキャンセルしました。")
                        _push_order_log(
                            {
                                "symbol": "ALL",
                                "status": "cancelled",
                                "msg": "cancel all open orders",
                            }
                        )
                    except Exception as e:
                        st.error(f"キャンセルに失敗しました: {e}")

        # ...existing code...
        # 通知設定 UI （シンプル化）
        with st.expander("📢 通知設定"):
            st.markdown(
                """
            **通知ルール**: Slack優先 → Discordフォールバック → 通知なし

            設定が完了したら環境変数またはWebhook URLを設定してください：
            - `SLACK_BOT_TOKEN`: Slack Bot Token （推奨）
            - `DISCORD_WEBHOOK_URL`: Discord Webhook URL （フォールバック）
            """
            )

            # シンプルな通知有効/無効の切り替えのみ
            enable_notifications = st.checkbox("通知を有効にする", value=True, key="enable_notifications")

            if enable_notifications:
                # テスト送信ボタンのみ提供
                if st.button("📨 通知テスト", key="test_notify_simple"):
                    try:
                        # 自動判定で送信試行
                        notifier = Notifier(platform="auto")
                        notifier.send("通知テスト", "Alpacaダッシュボードからのテスト通知です。")
                        st.success("テスト通知を送信しました。Slack/Discordを確認してください。")
                    except Exception as e:
                        st.error(f"通知送信に失敗: {e}")
            else:
                st.info("通知は無効になっています。")
        try:
            items = ", ".join(
                f"{k}={v}日"
                for k, v in sorted(
                    HOLD_LIMITS.items(),
                    key=lambda kv: (
                        int(str(kv[0]).replace("system", ""))
                        if str(kv[0]).startswith("system") and str(kv[0])[6:].isdigit()
                        else 999
                    ),
                )
            )
        except Exception:
            items = ", ".join(f"{k}={v}日" for k, v in HOLD_LIMITS.items())
        st.caption(f"経過日手仕切り（上限日数）: {items}")
    # 自動ルール: オプトイン状態はスケジュール領域のチェックボックスで一元管理
    auto_opt_in = st.session_state.get("auto_rule_opt_in", False)
    st.caption(f"自動ルールオプトイン: {'有効' if auto_opt_in else '無効'}")
    with st.expander("自動ルール設定 (システム別)"):
        for sys_name in sorted(AUTO_RULE_CONFIG.keys()):
            cfg = AUTO_RULE_CONFIG[sys_name]
            cols = st.columns([1, 1])
            with cols[0]:
                v = st.number_input(
                    f"{sys_name} 損益閾値(%)",
                    value=float(cfg.get("pnl_threshold", -20.0)),
                    step=1.0,
                    key=f"cfg_{sys_name}_pnl",
                )
            with cols[1]:
                p = st.selectbox(
                    f"{sys_name} 部分決済%",
                    [100, 75, 50, 25],
                    index=0,
                    key=f"cfg_{sys_name}_pct",
                )
            # apply changes to runtime config
            try:
                AUTO_RULE_CONFIG[sys_name]["pnl_threshold"] = float(v)
                AUTO_RULE_CONFIG[sys_name]["partial_pct"] = int(p)
            except Exception:
                pass
        if st.button("自動ルール設定を保存", key="save_auto_rule_config"):
            try:
                _save_auto_rule_config(AUTO_RULE_CONFIG)
                st.success("自動ルール設定を保存しました。")
            except Exception:
                st.error("自動ルール設定の保存に失敗しました。")

    # expander 終了
    if auto_opt_in:
        st.caption("※自動実行はオプトイン時に手動トリガーされます（将来はスケジューリング対応予定）。")
    if st.button("自動ルールを今すぐ実行 (手動)", key="auto_rule_run_manual_bottom"):
        st.session_state.setdefault("auto_rule_trigger", datetime.now().isoformat())

    with tab_summary:
        st.markdown("<div class='ap-section'>📊 サマリー指標</div>", unsafe_allow_html=True)
        try:
            total_positions = len(positions)
        except Exception:
            total_positions = 0
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f"""
                <div class='ap-card'>
                    <div class='ap-metric-icon'>📈</div>
                    <div class='ap-metric-value'>{total_positions}</div>
                    <div class='ap-metric-label'>保有銘柄数</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            ratio_display = f"{ratio * 100:.1f}%" if ratio is not None else "-"
            st.markdown(
                f"""
                <div class='ap-card'>
                    <div class='ap-metric-icon'>💰</div>
                    <div class='ap-metric-value'>{ratio_display}</div>
                    <div class='ap-metric-label'>余力比率</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col3:
            delta_display = _fmt_money(delta) if delta is not None else "-"
            color = "green" if delta and delta > 0 else "red" if delta and delta < 0 else "gray"
            st.markdown(
                f"""
                <div class='ap-card'>
                    <div class='ap-metric-icon'>📊</div>
                    <div class='ap-metric-value' style='color: {color};'>{delta_display}</div>
                    <div class='ap-metric-label'>前日比</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("---")
        st.markdown("<div class='ap-section'>📈 ポジション統計</div>", unsafe_allow_html=True)
        # 統計計算
        try:
            # 損益率(%)列が存在しない場合は計算
            if pos_df is not None and not pos_df.empty and "損益率(%)" not in pos_df.columns:
                try:
                    # 損益率 = (含み損益 / (平均取得単価 * 数量)) * 100
                    pos_df_copy = pos_df.copy()
                    pos_df_copy["平均取得単価"] = pd.to_numeric(pos_df_copy["平均取得単価"], errors="coerce")
                    pos_df_copy["数量"] = pd.to_numeric(pos_df_copy["数量"], errors="coerce")
                    pos_df_copy["含み損益"] = pd.to_numeric(pos_df_copy["含み損益"], errors="coerce")

                    # 投資額 = 平均取得単価 * 数量
                    investment = pos_df_copy["平均取得単価"] * pos_df_copy["数量"]

                    # 損益率 = (含み損益 / 投資額) * 100
                    pos_df_copy["損益率(%)"] = (pos_df_copy["含み損益"] / investment * 100).fillna(0.0)

                    # 元のpos_dfに追加
                    pos_df = pos_df_copy
                except Exception as calc_error:
                    st.warning(f"損益率計算エラー: {calc_error}")
                    pos_df["損益率(%)"] = 0.0

            if pos_df is not None and not pos_df.empty and "損益率(%)" in pos_df.columns:
                winners = int((pos_df["損益率(%)"] > 0).sum())
                losers = int((pos_df["損益率(%)"] <= 0).sum())
                avg_ret = float(pos_df["損益率(%)"].mean())
                pl_series = pos_df["含み損益"].astype(float) if "含み損益" in pos_df.columns else pd.Series(dtype=float)
                max_pl = float(pl_series.max()) if not pl_series.empty else 0.0
                sum_pl = float(pl_series.sum()) if not pl_series.empty else 0.0
                med_pl = float(pl_series.median()) if not pl_series.empty else 0.0
                stat_col1, stat_col2 = st.columns(2)
                with stat_col1:
                    st.markdown(
                        f"""
                        <div class='ap-stat-grid'>
                            <div class='ap-stat-item'>
                                <span class='ap-stat-label'>勝ち銘柄:</span>
                                <span class='ap-stat-value green'>{winners}</span>
                            </div>
                            <div class='ap-stat-item'>
                                <span class='ap-stat-label'>負け銘柄:</span>
                                <span class='ap-stat-value red'>{losers}</span>
                            </div>
                            <div class='ap-stat-item'>
                                <span class='ap-stat-label'>平均損益率:</span>
                                <span class='ap-stat-value'>{avg_ret:.2f}%</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with stat_col2:
                    st.markdown(
                        f"""
                        <div class='ap-stat-grid'>
                            <div class='ap-stat-item'>
                                <span class='ap-stat-label'>最大含み損益:</span>
                                <span class='ap-stat-value'>{_fmt_money(max_pl)}</span>
                            </div>
                            <div class='ap-stat-item'>
                                <span class='ap-stat-label'>合計含み損益:</span>
                                <span class='ap-stat-value'>{_fmt_money(sum_pl)}</span>
                            </div>
                            <div class='ap-stat-item'>
                                <span class='ap-stat-label'>含み損益中央値:</span>
                                <span class='ap-stat-value'>{_fmt_money(med_pl)}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info("ポジション統計を表示できません。")

            # アラート機能
            st.markdown("---")
            st.markdown("<div class='ap-section'>🚨 アラート</div>", unsafe_allow_html=True)

            alerts = []
            if pos_df is not None and not pos_df.empty:
                # 大幅損失アラート
                try:
                    large_loss_threshold = -15  # -15%以上の損失
                    if "損益率(%)" in pos_df.columns:
                        large_losses = pos_df[pos_df["損益率(%)"] <= large_loss_threshold]
                        if not large_losses.empty:
                            symbols = ", ".join(large_losses["銘柄"].astype(str))
                            alerts.append(
                                {
                                    "type": "critical",
                                    "title": "大幅損失",
                                    "message": (f"以下の銘柄で{large_loss_threshold}%以上の損失: {symbols}"),
                                    "icon": "⚠️",
                                }
                            )

                    # 長期保有アラート
                    if "保有日数" in pos_df.columns:
                        long_holds = pos_df[pd.to_numeric(pos_df["保有日数"], errors="coerce") > 30]
                        if not long_holds.empty:
                            symbols = ", ".join(long_holds["銘柄"].astype(str))
                            alerts.append(
                                {
                                    "type": "warning",
                                    "title": "長期保有",
                                    "message": f"30日以上保有: {symbols}",
                                    "icon": "📅",
                                }
                            )

                    # 集中リスクアラート（単一銘柄が総資産の20%以上）
                    if "含み損益" in pos_df.columns and equity_value:
                        pos_df_temp = pos_df.copy()
                        pos_df_temp["投資額"] = pd.to_numeric(
                            pos_df_temp["平均取得単価"], errors="coerce"
                        ) * pd.to_numeric(pos_df_temp["数量"], errors="coerce")
                        concentration_threshold = equity_value * 0.2
                        concentrated = pos_df_temp[pos_df_temp["投資額"] > concentration_threshold]
                        if not concentrated.empty:
                            symbols = ", ".join(concentrated["銘柄"].astype(str))
                            alerts.append(
                                {
                                    "type": "warning",
                                    "title": "集中リスク",
                                    "message": f"単一銘柄が総資産の20%超: {symbols}",
                                    "icon": "🎯",
                                }
                            )
                except Exception as alert_error:
                    st.warning(f"アラート計算エラー: {alert_error}")

            if alerts:
                for alert in alerts:
                    alert_class = "ap-alert-critical" if alert["type"] == "critical" else "ap-alert"
                    st.markdown(
                        f"""
                    <div class='{alert_class}'>
                        {alert["icon"]} <strong>{alert["title"]}</strong>: {alert["message"]}
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
            else:
                st.success("🟢 現在、アラートはありません。")

        except Exception as e:
            st.error(f"統計計算エラー: {e}")

    with tab_alloc:
        st.markdown("<div class='ap-section'>システム別 配分</div>", unsafe_allow_html=True)
        mapping_path = Path("data/symbol_system_map.json")
        pos_df = _positions_to_df(positions, client)

        if not pos_df.empty and mapping_path.exists():
            try:
                symbol_map = json.loads(mapping_path.read_text())
            except Exception:
                st.info("symbol_system_map.json の読み込みに失敗しました。")
            else:
                grouped = _group_by_system(pos_df, symbol_map)
                if not grouped:
                    st.info("マッピングに該当がありません。")
                else:
                    cols = st.columns(max(1, min(3, len(grouped))))
                    i = 0
                    for system, g in grouped.items():
                        with cols[i % len(cols)]:
                            st.caption(f"{system} の配分")
                            chart_df = g.copy()
                            values = chart_df["評価額"].astype(float).abs().fillna(0.0)
                            labels = chart_df["銘柄"].astype(str)
                            if values.sum() <= 0:
                                st.info("評価額が取得できませんでした。")
                            elif go is None or not hasattr(go, "Figure"):
                                st.info(
                                    "Plotly がインストールされていないため、グラフを表示できません。"  # noqa: E501
                                )
                            else:
                                fig = go.Figure(
                                    data=[
                                        go.Pie(
                                            labels=labels.tolist(),
                                            values=values.tolist(),
                                            textinfo="percent",
                                            hovertemplate=("<b>%{label}</b><br>評価額: %{value:,.0f}<extra></extra>"),
                                            hole=0.35,
                                        )
                                    ]
                                )
                                fig.update_traces(
                                    textfont=dict(color="#f5f7fa"),
                                    marker=dict(line=dict(color="#0f1420", width=1)),
                                )
                                fig.update_layout(
                                    showlegend=True,
                                    legend_title="銘柄",
                                    legend=dict(font=dict(color="#f5f7fa")),
                                    margin=dict(l=0, r=0, t=10, b=10),
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(0,0,0,0)",
                                    font=dict(color="#f5f7fa"),
                                )
                                st.plotly_chart(
                                    fig,
                                    width="stretch",
                                )
                        i += 1
        elif mapping_path.exists():
            st.info("ポジションがないため、グラフを表示できません。")
        else:
            st.info("data/symbol_system_map.json が見つかりません。")


if __name__ == "__main__":  # pragma: no cover - UI entry point
    main()
    try:
        _render_order_logs()
    except Exception:
        pass
