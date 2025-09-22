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
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal
import streamlit as st

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
from common.profit_protection import calculate_business_holding_days

# 経過日手仕切りの上限日数（システム別）
HOLD_LIMITS: dict[str, int] = {
    "system2": 2,
    "system3": 3,
    "system5": 6,
    "system6": 3,
}


WEEKDAY_LABELS_EN = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def _format_datetime_with_weekday(dt: datetime) -> str:
    """Format datetime with English weekday indicator."""

    try:
        weekday = WEEKDAY_LABELS_EN[dt.weekday()]
    except Exception:
        weekday = ""
    date_part = dt.strftime("%Y-%m-%d")
    time_part = dt.strftime("%H:%M:%S")
    if weekday:
        return f"{date_part} ({weekday}) {time_part}"
    return f"{date_part} {time_part}"


DASHBOARD_CSS = """
<style>
:root {
  --bg: #0f1420;
  --panel: #171c2a;
  --panel-alt: #1c2335;
  --text: #f5f7fa;
  --muted: #9aa4b2;
  --accent: #00e6a8;
  --danger: #ff6b6b;
  --warn: #ffd166;
}
body, .stApp {
  background: var(--bg) !important;
  color: var(--text) !important;
}
.main {
  background: var(--bg) !important;
}
.block-container {
  padding-top: 1.6rem !important;
}

.ap-title {
  font-size: 2.2rem;
  font-weight: 800;
  letter-spacing: 0.4px;
  margin: 0.6rem 0 1rem;
}
.ap-title .accent {
  background: linear-gradient(90deg, var(--accent), #12b886);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}
.ap-section {
  font-size: 1.2rem;
  font-weight: 700;
  margin: 1rem 0 0.6rem;
  color: var(--text);
}

.ap-card {
  background: var(--panel);
  border-radius: 16px;
  padding: 1rem;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
}
.ap-card + .ap-card {
  margin-top: 1rem;
}

.ap-metric {
  background:
    linear-gradient(var(--panel-alt), var(--panel-alt)) padding-box,
    linear-gradient(
      135deg,
      rgba(0, 230, 168, 0.45),
      rgba(18, 184, 134, 0.25)
    ) border-box;
  border: 1px solid transparent;
  border-radius: 16px;
  padding: 1rem;
  text-align: center;
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
  transition: transform 0.08s ease-out;
}
.ap-metric:hover {
  transform: translateY(-1px);
}
.ap-metric .label {
  color: var(--muted);
  font-size: 0.95rem;
  margin-bottom: 0.3rem;
}
.ap-metric .value {
  font-size: 2rem;
  font-weight: 800;
  letter-spacing: 0.5px;
}
.ap-metric .delta-pos {
  color: var(--accent);
  font-size: 0.9rem;
  font-weight: 700;
}
.ap-metric .delta-neg {
  color: var(--danger);
  font-size: 0.9rem;
  font-weight: 700;
}

.ap-badge {
  display: inline-block;
  padding: 0.25rem 0.6rem;
  border-radius: 999px;
  background: #0b1625;
  color: var(--muted);
  font-size: 0.78rem;
  margin-right: 0.4rem;
  border: 1px solid rgba(255, 255, 255, 0.08);
}
.ap-badge.good {
  color: var(--accent);
  border-color: rgba(0, 230, 168, 0.4);
}
.ap-badge.warn {
  color: var(--warn);
  border-color: rgba(255, 209, 102, 0.35);
}
.ap-badge.danger {
  color: var(--danger);
  border-color: rgba(255, 107, 107, 0.35);
}
.ap-badge.stat {
  background: rgba(255, 255, 255, 0.06);
  color: var(--text);
  margin-top: 0.25rem;
}
.ap-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  align-items: center;
}
.ap-badges .ap-badge {
  margin-right: 0;
}

.stDataFrame {
  background: var(--panel) !important;
  border-radius: 14px !important;
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25) !important;
}
.stDataFrame [data-testid="StyledFullRow"] {
  background: transparent !important;
}
.stDataFrame tbody tr td,
.stDataFrame thead tr th {
  color: var(--text) !important;
}
.stDataFrame tbody tr td a {
  color: var(--accent) !important;
}

.ap-toolbar {
  position: sticky;
  top: 0.5rem;
  z-index: 20;
  backdrop-filter: blur(6px);
  background: rgba(23, 28, 42, 0.6);
  border-radius: 12px;
  padding: 0.4rem 0.6rem;
  border: 1px solid rgba(255, 255, 255, 0.06);
}
.ap-caption {
  white-space: nowrap;
}

@keyframes apFadeUp {
  from {
    opacity: 0;
    transform: translateY(6px);
  }
  to {
    opacity: 1;
    transform: none;
  }
}
.ap-fade {
  animation: apFadeUp 0.28s ease-out;
}
.ap-card,
.ap-metric,
.stDataFrame,
.stTabs {
  animation: apFadeUp 0.28s ease-out;
}

.ap-toolbar .stButton > button {
  width: 100%;
  border-radius: 10px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: linear-gradient(
    135deg,
    rgba(0, 230, 168, 0.22),
    rgba(18, 184, 134, 0.14)
  );
  color: var(--text);
  font-weight: 700;
}
</style>
"""


def _inject_css() -> None:
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)


def _fmt_money(x: float | int | str | None, prefix: str = "$") -> str:
    try:
        v = float(x) if x is not None else 0.0
        if abs(v) >= 1000:
            return f"{prefix}{v:,.0f}"
        return f"{prefix}{v:,.2f}"
    except Exception:
        return str(x)


def _fmt_number(x: float | int | str | None) -> str:
    try:
        v = float(x) if x is not None else 0.0
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:,.2f}"
    except Exception:
        return str(x)


def _format_countdown(delta: timedelta) -> str:
    """Return countdown text in Japanese."""

    total_seconds = int(delta.total_seconds())
    if total_seconds <= 0:
        return "0秒"
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours}時間")
    if minutes or hours:
        parts.append(f"{minutes}分")
    parts.append(f"{seconds}秒")
    return "".join(parts)


def _safe_float(value: Any) -> float | None:
    """Convert a value to float safely."""

    if value is None:
        return None
    if isinstance(value, (int | float)):
        return float(value)

    try:
        text = str(value).strip()
        if not text or text in {"-", "nan", "NaN"}:
            return None
        cleaned = text.replace(",", "")
        return float(cleaned)
    except (TypeError, ValueError):
        return None


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

    now_ts = pd.Timestamp(now_newyork)
    if now_ts.tz is None:
        try:
            now_ts = now_ts.tz_localize("America/New_York")
        except Exception:
            now_ts = now_ts.tz_localize("UTC")
    now_utc = now_ts.tz_convert("UTC")

    market_open_series = pd.to_datetime(schedule["market_open"], utc=True)

    try:
        future_opens = market_open_series[market_open_series > now_utc]
    except Exception:
        return status

    if getattr(future_opens, "empty", True):
        return status

    next_open = future_opens.iloc[0]
    try:
        delta = next_open - now_utc
    except Exception:
        return status

    if delta.total_seconds() <= 0:
        return status

    countdown = _format_countdown(delta)
    return f"{status}（オープンまで {countdown}）"


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


def _fetch_account_and_positions() -> tuple[Any, Any, list[Any]]:
    client = ba.get_client()
    account = client.get_account()
    positions = list(client.get_all_positions())
    return client, account, positions


def _days_held(entry_dt: pd.Timestamp | str | datetime | None) -> int | None:
    return calculate_business_holding_days(entry_dt)


def _load_recent_prices(symbol: str, max_points: int = 30) -> list[float] | None:
    if not symbol:
        return None
    try:
        df = load_base_cache(symbol, rebuild_if_missing=False)
    except Exception:
        df = None

    if df is not None and not getattr(df, "empty", True):
        for col in ("Close", "close", "Adj Close", "adj_close", "adj close"):
            if col in df.columns:
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
                series = (
                    pd.to_numeric(numeric_cols.iloc[:, 0], errors="coerce")
                    .dropna()
                    .tail(max_points)
                )
            except Exception:
                series = pd.Series(dtype=float)
            if not series.empty:
                return list(series.values)

    candidates = [
        Path("data_cache_recent") / f"{symbol}.csv",
        Path("data_cache") / f"{symbol}.csv",
    ]
    for p in candidates:
        if p.exists():
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
        orders_obj = client.get_orders(status="open")
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
        bucket = levels.setdefault(sym_key, {"stops": set(), "limits": set(), "trail": set()})
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
        exit_hint = ""
        if held is not None and limit and held >= limit:
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
        price_series = [
            _load_recent_prices(sym, max_points=n_points) or [] for sym in symbol_series
        ]
        df["直近価格チャート"] = price_series
    except Exception:
        pass
    return df


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
    st.set_page_config(page_title="Alpaca Dashboard", layout="wide")
    _inject_css()

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
                (
                    "ニューヨーク時間: "
                    f"{_format_datetime_with_weekday(now_newyork)} "
                    f"（{nyse_status}）"
                ),
            ]
        )
    )
    st.markdown("<div class='ap-toolbar ap-fade'>", unsafe_allow_html=True)
    spacer, right = st.columns([7, 3])
    with right:
        bcol, tcol = st.columns([1.2, 1.8])
        with bcol:
            if st.button("🔄 手動更新", use_container_width=True):
                st.rerun()
        with tcol:
            st.caption(f"最終更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown("</div>", unsafe_allow_html=True)

    try:
        client, account, positions = _fetch_account_and_positions()
    except Exception as exc:  # pragma: no cover
        st.error(f"データ取得に失敗しました: {exc}")
        return

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
        return (
            "<div class='ap-metric'>"
            f"<div class='label'>{label}</div>"
            f"<div class='value'>{value}</div>"
            f"{d}"
            "</div>"
        )

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
            # システム絞り込み
            if "システム" in pos_df.columns:
                raw_systems = pos_df["システム"].fillna("unknown").unique()
                systems = sorted(str(s) for s in raw_systems)
                selected = st.multiselect("システム絞り込み", systems, default=systems)
                pos_df = pos_df[pos_df["システム"].astype(str).isin(selected)]

            numeric_cols = ["数量", "平均取得単価", "現在値", "含み損益"]
            for col in numeric_cols:
                if col in pos_df.columns:
                    pos_df[col] = pd.to_numeric(pos_df[col], errors="coerce")

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
                "並び替え", ["含み損益", "損益率(%)", "保有日数", "銘柄"], index=0, key="pos_sort"
            )
            ascending = st.toggle("昇順", value=False, key="pos_asc")
            try:
                pos_df = pos_df.sort_values(sort_key, ascending=ascending)
            except Exception:
                pass

            # 行スタイル（損益で淡い緑/赤, 透明度 0.14）
            def _row_style(row):
                try:
                    pl = float(row.get("含み損益", 0))
                except Exception:
                    pl = 0.0
                bg = (
                    "rgba(0,230,168,.14)"
                    if pl > 0
                    else ("rgba(255,107,107,.14)" if pl < 0 else "transparent")
                )
                return [f"background-color: {bg}"] * len(row)

            display_df = pos_df.copy()
            try:
                display_df["数量"] = pd.to_numeric(display_df["数量"], errors="coerce")
            except Exception:
                pass

            from typing import Any, cast

            format_columns = {
                "数量": "{:,.0f}",
                "平均取得単価": "{:,.2f}",
                "現在値": "{:,.2f}",
                "含み損益": "{:,.2f}",
            }
            styler = display_df.style.apply(_row_style, axis=1)
            # cast to Any to satisfy static type checkers about formatter mapping
            styler = styler.format(cast(Any, format_columns))

            # 表示（スパークライン列は LineChartColumn）
            try:
                st.dataframe(
                    styler,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "数量": st.column_config.NumberColumn(format="%.0f"),
                        "平均取得単価": st.column_config.NumberColumn(format="%.2f"),
                        "現在値": st.column_config.NumberColumn(format="%.2f"),
                        "含み損益": st.column_config.NumberColumn(format="%.2f"),
                        "損益率(%)": st.column_config.ProgressColumn(
                            min_value=-20, max_value=20, format="%.1f%%"
                        ),
                        "ストップ価格": st.column_config.Column(
                            width="medium",
                            help="未約定のストップ系注文価格（複数は / 区切り表示）。",
                        ),
                        "リミット価格": st.column_config.Column(
                            width="medium",
                            help=(
                                "未約定のリミット/テイクプロフィット注文価格"
                                "（複数は / 区切り表示）。"
                            ),
                        ),
                        "直近価格チャート": st.column_config.LineChartColumn(
                            label="直近価格チャート",
                            width="small",
                            help="過去数週間の終値推移をスパークラインで表示します。",
                        ),
                    },
                )
            except Exception:
                st.dataframe(pos_df, use_container_width=True, hide_index=True)

            # CSV ダウンロード
            try:
                try:
                    from config.settings import get_settings

                    settings2 = get_settings(create_dirs=True)
                    round_dec = getattr(settings2.cache, "round_decimals", None)
                except Exception:
                    round_dec = None
                try:
                    from common.cache_manager import round_dataframe

                    out_df = round_dataframe(pos_df, round_dec)
                except Exception:
                    out_df = pos_df
                csv = out_df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇ ポジションCSVをダウンロード", csv, file_name="positions.csv")
            except Exception:
                pass

    with tab_summary:
        st.markdown("<div class='ap-section'>指標</div>", unsafe_allow_html=True)
        try:
            total_positions = len(positions)
        except Exception:
            total_positions = 0
        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown(
                _metric_html("保有銘柄数", f"{total_positions}"),
                unsafe_allow_html=True,
            )
        with s2:
            if ratio is not None:
                st.markdown(
                    _metric_html("余力比率", f"{ratio * 100:.1f}%"),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    _metric_html("余力比率", "-"),
                    unsafe_allow_html=True,
                )
        with s3:
            if delta is not None:
                st.markdown(
                    _metric_html("前日比", _fmt_money(delta)),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    _metric_html("前日比", "-"),
                    unsafe_allow_html=True,
                )

        # 統計チップ
        try:
            winners = (
                int((pos_df["損益率(%)"] > 0).sum())
                if "pos_df" in locals() and "損益率(%)" in pos_df.columns
                else 0
            )
            losers = (
                int((pos_df["損益率(%)"] <= 0).sum())
                if "pos_df" in locals() and "損益率(%)" in pos_df.columns
                else 0
            )
            avg_ret = (
                float(pos_df["損益率(%)"].mean())
                if "pos_df" in locals() and "損益率(%)" in pos_df.columns
                else 0.0
            )
            try:
                pl_series = (
                    pos_df["含み損益"].astype(float)
                    if "含み損益" in pos_df.columns
                    else pd.Series(dtype=float)
                )
                max_pl = float(pl_series.max()) if not pl_series.empty else 0.0
                sum_pl = float(pl_series.sum()) if not pl_series.empty else 0.0
                med_pl = float(pl_series.median()) if not pl_series.empty else 0.0
            except Exception:
                max_pl = sum_pl = med_pl = 0.0
            chips = [
                f"<div class='ap-badge stat'>勝ち銘柄: {winners}</div>",
                f"<div class='ap-badge stat'>負け銘柄: {losers}</div>",
                f"<div class='ap-badge stat'>平均損益率: {avg_ret:.2f}%</div>",
                f"<div class='ap-badge stat'>最大含み損益: {_fmt_money(max_pl)}</div>",
                f"<div class='ap-badge stat'>合計含み損益: {_fmt_money(sum_pl)}</div>",
                f"<div class='ap-badge stat'>含み損益中央値: {_fmt_money(med_pl)}</div>",
            ]
            st.markdown(
                "<div class='ap-badges'>" + "".join(chips) + "</div>",
                unsafe_allow_html=True,
            )
        except Exception:
            pass

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
                                            hovertemplate=(
                                                "<b>%{label}</b><br>評価額: %{value:,.0f}"
                                                "<extra></extra>"
                                            ),
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
                                    use_container_width=True,
                                    config={"displayModeBar": False},
                                )
                        i += 1
        elif mapping_path.exists():
            st.info("ポジションがないため、グラフを表示できません。")
        else:
            st.info("data/symbol_system_map.json が見つかりません。")


if __name__ == "__main__":  # pragma: no cover - UI entry point
    main()
