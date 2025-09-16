"""Alpaca ダッシュボード（UI リフレッシュ＋演出強化）

- アカウント残高/現金/余力をカード表示（前日比、余力ゲージ）
- ポジション一覧は行スタイル（損益で淡い緑/赤）＋スパークライン
- システム別フィルタ（symbol_system_map.json があれば使用）
- 統計チップ（勝ち/負け、平均損益率、最大/合計/中央値の含み損益）
- タイトル直下のスティッキーツールバーに「🔄 手動更新」と最終更新時刻を横並び配置
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal

try:  # pragma: no cover - optional dependency
    import plotly.graph_objects as go
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    go = None

from common import broker_alpaca as ba


# 経過日手仕切りの上限日数（システム別）
HOLD_LIMITS: dict[str, int] = {
    "system2": 2,
    "system3": 3,
    "system5": 6,
    "system6": 3,
}


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
          --bg: #0f1420; --panel: #171c2a; --panel-alt: #1c2335;
          --text: #f5f7fa; --muted: #9aa4b2; --accent: #00e6a8;
          --danger: #ff6b6b; --warn: #ffd166;
        }
        body, .stApp { background: var(--bg) !important; color: var(--text) !important; }
        .main { background: var(--bg) !important; }
        .block-container { padding-top: 1.6rem !important; }

        .ap-title { font-size: 2.2rem; font-weight: 800; letter-spacing: .4px; margin: .6rem 0 1rem; }
        .ap-title .accent { background: linear-gradient(90deg, var(--accent), #12b886); -webkit-background-clip: text; background-clip: text; color: transparent; }
        .ap-section { font-size: 1.2rem; font-weight: 700; margin: 1rem 0 .6rem; color: var(--text); }

        .ap-card { background: var(--panel); border-radius: 16px; padding: 1.0rem; box-shadow: 0 8px 24px rgba(0,0,0,.25); }
        .ap-card + .ap-card { margin-top: 1rem; }

        .ap-metric { background: linear-gradient(var(--panel-alt), var(--panel-alt)) padding-box,
                                 linear-gradient(135deg, rgba(0,230,168,.45), rgba(18,184,134,.25)) border-box;
                      border: 1px solid transparent; border-radius: 16px; padding: 1rem; text-align: center;
                      box-shadow: 0 4px 14px rgba(0,0,0,.25); transition: transform .08s ease-out; }
        .ap-metric:hover { transform: translateY(-1px); }
        .ap-metric .label { color: var(--muted); font-size: .95rem; margin-bottom: .3rem; }
        .ap-metric .value { font-size: 2.0rem; font-weight: 800; letter-spacing: .5px; }
        .ap-metric .delta-pos { color: var(--accent); font-size: .9rem; font-weight: 700; }
        .ap-metric .delta-neg { color: var(--danger); font-size: .9rem; font-weight: 700; }

        .ap-badge { display: inline-block; padding: .25rem .6rem; border-radius: 999px; background: #0b1625; color: var(--muted); font-size: .78rem; margin-right: .4rem; border: 1px solid rgba(255,255,255,.08); }
        .ap-badge.good { color: var(--accent); border-color: rgba(0,230,168,.4); }
        .ap-badge.warn { color: var(--warn); border-color: rgba(255,209,102,.35); }
        .ap-badge.danger { color: var(--danger); border-color: rgba(255,107,107,.35); }
        .ap-badge.stat { background: rgba(255,255,255,.06); color: var(--text); margin-top: .25rem; }
        .ap-badges { display:flex; flex-wrap: wrap; gap:.4rem; align-items:center; }
        .ap-badges .ap-badge { margin-right: 0; }

        .stDataFrame { background: var(--panel) !important; border-radius: 14px !important; box-shadow: 0 6px 18px rgba(0,0,0,.25) !important; }
        .stDataFrame [data-testid="StyledFullRow"] { background: transparent !important; }
        .stDataFrame tbody tr td, .stDataFrame thead tr th { color: var(--text) !important; }
        .stDataFrame tbody tr td a { color: var(--accent) !important; }

        .ap-toolbar { position: sticky; top: .5rem; z-index: 20; backdrop-filter: blur(6px); background: rgba(23,28,42,.6); border-radius: 12px; padding: .4rem .6rem; border: 1px solid rgba(255,255,255,.06); }
        .ap-caption { white-space: nowrap; }

        .ap-ring { --size: 92px; --track: #0b1625; --val: 0.0; width: var(--size); height: var(--size); border-radius: 50%;
                   background: conic-gradient(var(--accent) calc(var(--val) * 1%), rgba(255,255,255,.08) 0), var(--track); display: grid; place-items: center; margin: .25rem auto; }
        .ap-ring > span { font-weight: 800; font-size: 1.0rem; }

        @keyframes apFadeUp { from { opacity: 0; transform: translateY(6px);} to { opacity:1; transform:none; } }
        .ap-fade { animation: apFadeUp .28s ease-out; }
        .ap-card, .ap-metric, .stDataFrame, .stTabs { animation: apFadeUp .28s ease-out; }

        .ap-toolbar .stButton>button { width: 100%; border-radius: 10px; border: 1px solid rgba(255,255,255,.08);
                                       background: linear-gradient(135deg, rgba(0,230,168,.22), rgba(18,184,134,.14));
                                       color: var(--text); font-weight: 700; }
        </style>
        """,
        unsafe_allow_html=True,
    )


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


def _safe_float(value: Any) -> float | None:
    """Convert a value to float safely."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    try:
        text = str(value).strip()
        if not text or text in {"-", "nan", "NaN"}:
            return None
        cleaned = text.replace(",", "")
        return float(cleaned)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    """Convert a value to float safely."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
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
    """NYSE の営業状況を返す。"""

    try:
        calendar = mcal.get_calendar("NYSE")
    except Exception:
        return "NYSE: 状態不明"

    start_date = now_newyork.date() - timedelta(days=5)
    end_date = now_newyork.date() + timedelta(days=5)

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

    return "NYSE: 営業中" if is_open else "NYSE: クローズ"


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
    if entry_dt is None:
        return None
    if isinstance(entry_dt, pd.Timestamp):
        dt = entry_dt
    else:
        try:
            dt = pd.to_datetime(entry_dt, errors="coerce")
            if pd.isna(dt):
                return None
        except Exception:
            return None
    today = pd.Timestamp(datetime.utcnow()).normalize()
    return int((today - dt.normalize()).days)


def _fetch_entry_dates(client, symbols: list[str]) -> dict[str, pd.Timestamp]:
    out: dict[str, pd.Timestamp] = {}
    for sym in symbols:
        try:
            acts = client.get_activities(symbol=sym, activity_types="FILL")
        except Exception:
            continue
        for act in sorted(acts, key=lambda a: getattr(a, "transaction_time", "")):
            t = getattr(act, "transaction_time", None)
            if t:
                out[sym] = pd.Timestamp(t)
                break
    return out


def _load_recent_prices(symbol: str, max_points: int = 30) -> list[float] | None:
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
                s = df[close_col].astype(float).tail(max_points)
                return list(s.values)
            except Exception:
                continue
    return None


def _positions_to_df(positions, client=None) -> pd.DataFrame:
    symbols = [getattr(p, "symbol", "") for p in positions]
    entry_map = _fetch_entry_dates(client, symbols) if client else {}

    mapping_path = Path("data/symbol_system_map.json")
    symbol_map: dict[str, str] = {}
    if mapping_path.exists():
        try:
            symbol_map = json.loads(mapping_path.read_text())
        except Exception:
            symbol_map = {}

    records: list[dict[str, object]] = []
    for pos in positions:
        sym = getattr(pos, "symbol", "")
        held = _days_held(entry_map.get(sym))
        system_value = symbol_map.get(sym, "unknown")
        limit = HOLD_LIMITS.get(str(system_value).lower())
        exit_hint = (
            f"{limit}日経過で手仕切り検討" if held is not None and limit and held >= limit else ""
        )
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
        df["価格ミニ"] = [
            (_load_recent_prices(sym, max_points=n_points) or []) for sym in df["銘柄"].astype(str)
        ]
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
    ny_caption = f"ニューヨーク時間: {now_newyork.strftime('%Y-%m-%d %H:%M:%S')} （{nyse_status}）"
    st.caption(
        " / ".join(
            [
                f"日本時間: {now_tokyo.strftime('%Y-%m-%d %H:%M:%S')}",
                ny_caption,
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
        if ratio is not None:
            fill_ratio = min(max(ratio, 0.0), 1.0)
            ring = (
                f"<div class='ap-ring' style='--val:{fill_ratio*100:.1f};'>"
                f"<span>{ratio*100:.1f}%</span></div>"
            )
            st.markdown(ring, unsafe_allow_html=True)
            st.caption("余力比率")
        else:
            st.caption("余力比率: -")
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
                systems = sorted([str(s) for s in pos_df["システム"].fillna("unknown").unique()])
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

            styler = display_df.style.apply(_row_style, axis=1)
            styler = styler.format(
                {
                    "数量": "{:,.0f}",
                    "平均取得単価": "{:,.2f}",
                    "現在値": "{:,.2f}",
                    "含み損益": "{:,.2f}",
                },
                na_rep="-",
            )

            # 表示（スパークライン列は LineChartColumn）
            try:
                st.dataframe(
                    styler,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "損益率(%)": st.column_config.ProgressColumn(
                            min_value=-20, max_value=20, format="%.1f%%"
                        ),
                        "価格ミニ": st.column_config.LineChartColumn(width="small"),
                    },
                )
            except Exception:
                st.dataframe(pos_df, use_container_width=True, hide_index=True)

            # CSV ダウンロード
            try:
                csv = pos_df.to_csv(index=False).encode("utf-8")
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
                    _metric_html("余力比率", f"{ratio*100:.1f}%"),
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
