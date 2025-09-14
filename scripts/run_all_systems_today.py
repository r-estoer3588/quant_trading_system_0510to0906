from __future__ import annotations

import argparse
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import logging
from pathlib import Path

import pandas as pd

from common import broker_alpaca as ba
from common.alpaca_order import submit_orders_df
from common.cache_manager import CacheManager
from common.notifier import create_notifier
from common.position_age import load_entry_dates, save_entry_dates
from common.signal_merge import Signal, merge_signals
from common.utils_spy import get_latest_nyse_trading_day, get_spy_with_indicators
from config.settings import get_settings

# strategies
from strategies.system1_strategy import System1Strategy
from strategies.system2_strategy import System2Strategy
from strategies.system3_strategy import System3Strategy
from strategies.system4_strategy import System4Strategy
from strategies.system5_strategy import System5Strategy
from strategies.system6_strategy import System6Strategy
from strategies.system7_strategy import System7Strategy

_LOG_CALLBACK = None
_LOG_START_TS = None  # CLI 用の経過時間測定開始時刻


def _get_today_logger() -> logging.Logger:
    """today_signals 用のファイルロガーを取得（logs/today_signals.log）。

    UI 側が log_callback を渡す場合は UI がファイル出力するので、
    本ロガーは CLI 実行や log_callback なしのときのみに使う想定。
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
    try:
        # settings が未初期化でも安全に取得できるようにラップ
        settings = get_settings(create_dirs=True)
        log_dir = Path(settings.LOGS_DIR)
    except Exception:
        log_dir = Path("logs")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    log_path = log_dir / "today_signals.log"

    # 既存の同一ファイルハンドラがあるか確認
    has_handler = False
    for h in list(logger.handlers):
        try:
            if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(
                log_path
            ):
                has_handler = True
                break
        except Exception:
            continue
    if not has_handler:
        try:
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except Exception:
            pass
    return logger


def _log(msg: str, ui: bool = True):
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
        now = _t.strftime("%H:%M:%S")
        elapsed = 0 if _LOG_START_TS is None else max(0, _t.time() - _LOG_START_TS)
        m, s = divmod(int(elapsed), 60)
        prefix = f"[{now} | {m}分{s}秒] "
    except Exception:
        prefix = ""

    # CLI へは整形して出力
    try:
        print(f"{prefix}{msg}", flush=True)
    except Exception:
        pass

    # UI 側のコールバックには原文のまま通知（UI での重複プレフィックス回避）
    try:
        cb = globals().get("_LOG_CALLBACK")
        if cb and callable(cb) and ui:
            try:
                cb(str(msg))
            except Exception:
                pass
    except Exception:
        pass

    # UI コールバックが無いか、ui=False の場合はファイルにINFOで出力（CLI ログ保存）
    try:
        cb = globals().get("_LOG_CALLBACK")
        if not cb or not ui:
            _get_today_logger().info(str(msg))
    except Exception:
        pass


def _asc_by_score_key(score_key: str | None) -> bool:
    return bool(score_key and score_key.upper() in {"RSI4"})


def _filter_ui_logs(lines: list[str]) -> list[str]:
    """Remove verbose log entries for the UI display."""
    skip_keywords = (
        "進捗",
        "インジケーター計算",
        "バッチ時間",
        "batch time",
        "候補抽出",
        "候補日数",
        "銘柄:",
    )
    return [ln for ln in lines if not any(k in ln for k in skip_keywords)]


def _amount_pick(
    per_system: dict[str, pd.DataFrame],
    strategies: dict[str, object],
    total_budget: float,
    weights: dict[str, float],
    side: str,
) -> pd.DataFrame:
    """資金配分に基づいて候補を採用。
    shares と position_value を付与して返す。
    """
    chosen = []
    chosen_symbols = set()

    # システムごとの割当予算
    budgets = {
        name: float(total_budget) * float(weights.get(name, 0.0)) for name in weights
    }  # noqa: E501
    remaining = budgets.copy()

    # システム名の順序を固定（system1..system7）
    sys_order = [f"system{i}" for i in range(1, 8)]
    ordered_names = [n for n in sys_order if n in weights]
    # 各システムの最大ポジション上限（設定 max_positions、既定10）と採用カウンタ
    max_pos_by_system: dict[str, int] = {}
    for _n in ordered_names:
        try:
            _stg = strategies.get(_n)
            _lim = int(getattr(_stg, "config", {}).get("max_positions", 10))
        except Exception:
            _lim = 10
        max_pos_by_system[_n] = max(0, _lim)
    count_by_system: dict[str, int] = {k: 0 for k in ordered_names}
    # システムごとにスコア順で採用。複数周回して1件ずつ拾う（偏りを軽減）
    still = True
    while still:
        still = False
        for name in ordered_names:
            df = per_system.get(name, pd.DataFrame())
            if (
                df is None
                or df.empty
                or remaining.get(name, 0.0) <= 0.0
                or count_by_system.get(name, 0) >= max_pos_by_system.get(name, 0)
            ):
                continue
            stg = strategies[name]
            # 順に探索
            for _, row in df.iterrows():
                sym = row["symbol"]
                if sym in chosen_symbols:
                    continue
                entry = (
                    float(row["entry_price"])
                    if not pd.isna(row.get("entry_price"))
                    else None  # noqa: E501
                )
                stop = (
                    float(row["stop_price"])
                    if not pd.isna(row.get("stop_price"))
                    else None  # noqa: E501
                )
                if not entry or not stop or entry <= 0:
                    continue

                # 望ましい枚数（全システム割当基準）
                try:
                    # stg may be typed as object; call via cast to avoid
                    # static type errors. Call calculate_position_size if available.
                    calc_fn = getattr(stg, "calculate_position_size", None)
                    if callable(calc_fn):
                        try:
                            ds = calc_fn(
                                budgets[name],
                                entry,
                                stop,
                                risk_pct=float(
                                    getattr(stg, "config", {}).get("risk_pct", 0.02)
                                ),  # noqa: E501
                                max_pct=float(
                                    getattr(stg, "config", {}).get("max_pct", 0.10)
                                ),  # noqa: E501
                            )
                            if ds is None:
                                desired_shares = 0
                            else:
                                try:
                                    if isinstance(ds, (int | float | str)):
                                        try:
                                            desired_shares = int(float(ds))
                                        except Exception:
                                            desired_shares = 0
                                    else:
                                        desired_shares = 0
                                except Exception:
                                    desired_shares = 0
                        except Exception:
                            desired_shares = 0
                    else:
                        desired_shares = 0
                except Exception:
                    desired_shares = 0
                if desired_shares <= 0:
                    continue

                # 予算内に収まるよう調整
                max_by_cash = int(remaining[name] // abs(entry))
                shares = min(desired_shares, max_by_cash)
                if shares <= 0:
                    continue
                position_value = shares * abs(entry)
                if position_value <= 0:
                    continue

                # 採用
                rec = row.to_dict()
                rec["shares"] = int(shares)
                rec["position_value"] = float(round(position_value, 2))
                # 採用直前の残余を system_budget に表示（見た目が減っていく）
                rec["system_budget"] = float(round(remaining[name], 2))
                rec["remaining_after"] = float(round(remaining[name] - position_value, 2))
                chosen.append(rec)
                chosen_symbols.add(sym)
                remaining[name] -= position_value
                count_by_system[name] = count_by_system.get(name, 0) + 1
                still = True
                break  # 1件ずつ拾って次のシステムへ

    if not chosen:
        return pd.DataFrame()
    out = pd.DataFrame(chosen)
    out["side"] = side
    return out


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
        _log("(submit) shares 列がありません。" "資金配分モードで実行してください。")
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
                    "order_id": getattr(order, "id", None),
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


def compute_today_signals(
    symbols: list[str] | None,
    *,
    slots_long: int | None = None,
    slots_short: int | None = None,
    capital_long: float | None = None,
    capital_short: float | None = None,
    save_csv: bool = False,
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
    settings = get_settings(create_dirs=True)
    cm = CacheManager(settings)
    # install log callback for helpers
    globals()["_LOG_CALLBACK"] = log_callback
    cache_dir = cm.rolling_dir
    signals_dir = Path(settings.outputs.signals_dir)
    signals_dir.mkdir(parents=True, exist_ok=True)

    # CLI実行時のStreamlit警告を抑制（UIコンテキストが無い場合のみ）
    try:
        import logging as _lg
        import os as _os

        if not _os.environ.get("STREAMLIT_SERVER_ENABLED"):

            class _SilenceBareModeWarnings(_lg.Filter):
                def filter(self, record: _lg.LogRecord) -> bool:  # type: ignore[override]
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

    # 最新営業日（NYSE）
    today = get_latest_nyse_trading_day().normalize()
    _log(f"📅 最新営業日（NYSE）: {today.date()}")
    if progress_callback:
        try:
            progress_callback(0, 8, "init")
        except Exception:
            pass

    # シンボル決定
    if symbols and len(symbols) > 0:
        symbols = [s.upper() for s in symbols]
    else:
        from common.universe import build_universe_from_cache, load_universe_file

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
    if "SPY" not in symbols:
        symbols.append("SPY")

    _log(
        f"🎯 対象シンボル数: {len(symbols)}"
        f"（例: {', '.join(symbols[:10])}"
        f"{'...' if len(symbols) > 10 else ''}）"
    )
    if log_callback:
        try:
            log_callback("🧭 シンボル決定完了。基礎データのロードへ…")
        except Exception:
            pass
    if progress_callback:
        try:
            # 直後に基礎データロードを開始するため、フェーズ名を明確化
            progress_callback(1, 8, "load_basic:start")
        except Exception:
            pass

    # データ読み込み
    # --- フィルター条件で銘柄を絞り込み、
    #     通過銘柄のみデータロード ---
    # 1. まずフィルター条件に必要なデータ
    #    （株価・売買代金・ATR等）を全銘柄分ロード
    # --- フィルター・データロード関数を
    #     ローカル関数として定義 ---

    def load_basic_data(symbols):
        import time as _t

        data = {}
        total_syms = len(symbols)
        start_ts = _t.time()
        CHUNK = 500
        for idx, sym in enumerate(symbols, start=1):
            try:
                # まずは呼び出し元から渡された minimal データを優先
                df = None
                try:
                    if symbol_data and sym in symbol_data:
                        df = symbol_data.get(sym)
                        if df is not None and not df.empty:
                            x = df.copy()
                            if x.index.name is not None:
                                x = x.reset_index()
                            # 日付列の正規化
                            if "date" in x.columns:
                                x["date"] = pd.to_datetime(x["date"], errors="coerce")
                            elif "Date" in x.columns:
                                x["date"] = pd.to_datetime(x["Date"], errors="coerce")
                            # 列名の正規化（存在するもののみ）
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
                            # 最低限の必須列が揃っているか確認
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
                # 受け取りが無い/不足 → キャッシュから取得
                if df is None or df.empty:
                    df = cm.read(sym, "rolling")
                if df is None or df.empty:
                    # rolling 不在 → base から必要分を生成して保存
                    try:
                        from common.cache_manager import load_base_cache
                    except Exception:
                        load_base_cache = None  # type: ignore
                    base_df = (
                        load_base_cache(sym, rebuild_if_missing=True)
                        if load_base_cache is not None
                        else None
                    )
                    if base_df is None or base_df.empty:
                        continue
                    x = base_df.copy()
                    if x.index.name is not None:
                        x = x.reset_index()
                    if "Date" in x.columns:
                        x["date"] = pd.to_datetime(x["Date"], errors="coerce")
                    elif "date" in x.columns:
                        x["date"] = pd.to_datetime(x["date"], errors="coerce")
                    else:
                        continue
                    x = x.dropna(subset=["date"]).sort_values("date")
                    # 列名を rolling 想定へ（存在するもののみ）
                    col_map = {
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "AdjClose": "adjusted_close",
                        "Volume": "volume",
                    }
                    for k, v in list(col_map.items()):
                        if k in x.columns:
                            x = x.rename(columns={k: v})
                    # 必要期間: 設計上 240 営業日（不足時は全量）
                    n = int(settings.cache.rolling.base_lookback_days)
                    sliced = x.tail(n).reset_index(drop=True)
                    cm.write_atomic(sliced, sym, "rolling")
                    df = sliced
                if df is not None and not df.empty:
                    try:
                        if "Date" not in df.columns:
                            if "date" in df.columns:
                                df = df.copy()
                                df["Date"] = pd.to_datetime(df["date"], errors="coerce")
                            else:
                                # 最低限 index を Date 列に昇格
                                df = df.copy()
                                df["Date"] = pd.to_datetime(df.index, errors="coerce")
                        # 正規化（表示/互換性向上）
                        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
                    except Exception:
                        pass
                    data[sym] = df
            except Exception:
                continue
            if idx % CHUNK == 0:
                try:
                    elapsed = max(0.001, _t.time() - start_ts)
                    rate = idx / elapsed
                    remain = max(0, total_syms - idx)
                    eta_sec = int(remain / rate) if rate > 0 else 0
                    m, s = divmod(eta_sec, 60)
                    msg = f"📦 基礎データロード進捗: {idx}/{total_syms} | ETA {m}分{s}秒"
                    _log(msg, ui=False)
                    # UIにも見えるよう適度に流す
                    try:
                        cb = globals().get("_LOG_CALLBACK")
                        if cb and callable(cb):
                            try:
                                cb(msg)
                            except Exception:
                                pass
                    except Exception:
                        pass
                except Exception:
                    _log(f"📦 基礎データロード進捗: {idx}/{total_syms}", ui=False)
                    try:
                        cb = globals().get("_LOG_CALLBACK")
                        if cb and callable(cb):
                            try:
                                cb(f"📦 基礎データロード進捗: {idx}/{total_syms}")
                            except Exception:
                                pass
                    except Exception:
                        pass
        try:
            total_elapsed = int(max(0, _t.time() - start_ts))
            m, s = divmod(total_elapsed, 60)
            done_msg = f"📦 基礎データロード完了: {len(data)}/{total_syms} | 所要 {m}分{s}秒"
            _log(done_msg)
            try:
                cb = globals().get("_LOG_CALLBACK")
                if cb and callable(cb):
                    try:
                        cb(done_msg)
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception:
            _log(f"📦 基礎データロード完了: {len(data)}/{total_syms}")
            try:
                cb = globals().get("_LOG_CALLBACK")
                if cb and callable(cb):
                    try:
                        cb(f"📦 基礎データロード完了: {len(data)}/{total_syms}")
                    except Exception:
                        pass
            except Exception:
                pass
        return data

    def filter_system1(symbols, data):
        result = []
        for sym in symbols:
            df = data.get(sym)
            if df is None or df.empty:
                continue
            # 株価5ドル以上（直近終値）
            if df["close"].iloc[-1] < 5:
                continue
            # 過去20日平均売買代金（厳密: mean(close*volume)）が5000万ドル以上
            if (df["close"] * df["volume"]).tail(20).mean() < 5e7:
                continue
            result.append(sym)
        return result

    def filter_system2(symbols, data):
        result = []
        for sym in symbols:
            df = data.get(sym)
            if df is None or df.empty:
                continue
            if df["close"].iloc[-1] < 5:
                continue
            if (df["close"] * df["volume"]).tail(20).mean() < 2.5e7:
                continue
            # ATR計算（過去10日）
            if "high" in df.columns and "low" in df.columns:
                tr = (df["high"] - df["low"]).tail(10)
                atr = tr.mean()
                if atr < df["close"].iloc[-1] * 0.03:
                    continue
            result.append(sym)
        return result

    def filter_system3(symbols, data):
        result = []
        for sym in symbols:
            df = data.get(sym)
            if df is None or df.empty:
                continue
            low = df.get("Low", df.get("low"))
            if low is None or float(low.iloc[-1]) < 1:
                continue
            av50 = df.get("AvgVolume50")
            if av50 is None or pd.isna(av50.iloc[-1]) or float(av50.iloc[-1]) < 1_000_000:
                continue
            atr_ratio = df.get("ATR_Ratio")
            if atr_ratio is None or pd.isna(atr_ratio.iloc[-1]) or float(atr_ratio.iloc[-1]) < 0.05:
                continue
            result.append(sym)
        return result

    def filter_system4(symbols, data):
        result = []
        for sym in symbols:
            df = data.get(sym)
            if df is None or df.empty:
                continue
            dv50 = df.get("DollarVolume50")
            hv50 = df.get("HV50")
            try:
                if dv50 is None or pd.isna(dv50.iloc[-1]) or float(dv50.iloc[-1]) <= 100_000_000:
                    continue
                if hv50 is None or pd.isna(hv50.iloc[-1]):
                    continue
                hv = float(hv50.iloc[-1])
                if hv < 10 or hv > 40:
                    continue
            except Exception:
                continue
            result.append(sym)
        return result

    def filter_system5(symbols, data):
        result = []
        for sym in symbols:
            df = data.get(sym)
            if df is None or df.empty:
                continue
            av50 = df.get("AvgVolume50")
            dv50 = df.get("DollarVolume50")
            atr_pct = df.get("ATR_Pct")
            try:
                if av50 is None or pd.isna(av50.iloc[-1]) or float(av50.iloc[-1]) <= 500_000:
                    continue
                if dv50 is None or pd.isna(dv50.iloc[-1]) or float(dv50.iloc[-1]) <= 2_500_000:
                    continue
                if atr_pct is None or pd.isna(atr_pct.iloc[-1]) or float(atr_pct.iloc[-1]) <= 0.04:
                    continue
            except Exception:
                continue
            result.append(sym)
        return result

    def filter_system6(symbols, data):
        result = []
        for sym in symbols:
            df = data.get(sym)
            if df is None or df.empty:
                continue
            low = df.get("Low", df.get("low"))
            if low is None or float(low.iloc[-1]) < 5:
                continue
            dv50 = df.get("DollarVolume50")
            if dv50 is None or pd.isna(dv50.iloc[-1]) or float(dv50.iloc[-1]) <= 10_000_000:
                continue
            result.append(sym)
        return result

    def load_indicator_data(symbols):
        import time as _t

        data = {}
        total_syms = len(symbols)
        start_ts = _t.time()
        CHUNK = 500
        for idx, sym in enumerate(symbols, start=1):
            try:
                # 提供された minimal データを優先
                df = None
                try:
                    if symbol_data and sym in symbol_data:
                        df = symbol_data.get(sym)
                        if df is not None and not df.empty:
                            x = df.copy()
                            if x.index.name is not None:
                                x = x.reset_index()
                            if "date" in x.columns:
                                x["date"] = pd.to_datetime(x["date"], errors="coerce")
                            elif "Date" in x.columns:
                                x["date"] = pd.to_datetime(x["Date"], errors="coerce")
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
                    df = cm.read(sym, "rolling")
                if df is None or df.empty:
                    try:
                        from common.cache_manager import load_base_cache
                    except Exception:
                        load_base_cache = None  # type: ignore
                    base_df = (
                        load_base_cache(sym, rebuild_if_missing=True)
                        if load_base_cache is not None
                        else None
                    )
                    if base_df is None or base_df.empty:
                        continue
                    x = base_df.copy()
                    if x.index.name is not None:
                        x = x.reset_index()
                    if "Date" in x.columns:
                        x["date"] = pd.to_datetime(x["Date"], errors="coerce")
                    elif "date" in x.columns:
                        x["date"] = pd.to_datetime(x["date"], errors="coerce")
                    else:
                        continue
                    x = x.dropna(subset=["date"]).sort_values("date")
                    col_map = {
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "AdjClose": "adjusted_close",
                        "Volume": "volume",
                    }
                    for k, v in list(col_map.items()):
                        if k in x.columns:
                            x = x.rename(columns={k: v})
                    n = int(settings.cache.rolling.base_lookback_days)
                    sliced = x.tail(n).reset_index(drop=True)
                    cm.write_atomic(sliced, sym, "rolling")
                    df = sliced
                if df is not None and not df.empty:
                    try:
                        if "Date" not in df.columns:
                            if "date" in df.columns:
                                df = df.copy()
                                df["Date"] = pd.to_datetime(df["date"], errors="coerce")
                            else:
                                df = df.copy()
                                df["Date"] = pd.to_datetime(df.index, errors="coerce")
                        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
                    except Exception:
                        pass
                    data[sym] = df
            except Exception:
                continue
            if total_syms > 0 and idx % CHUNK == 0:
                try:
                    elapsed = max(0.001, _t.time() - start_ts)
                    rate = idx / elapsed
                    remain = max(0, total_syms - idx)
                    eta_sec = int(remain / rate) if rate > 0 else 0
                    m, s = divmod(eta_sec, 60)
                    msg = f"🧮 指標データロード進捗: {idx}/{total_syms} | ETA {m}分{s}秒"
                    _log(msg, ui=False)
                    try:
                        cb = globals().get("_LOG_CALLBACK")
                        if cb and callable(cb):
                            try:
                                cb(msg)
                            except Exception:
                                pass
                    except Exception:
                        pass
                except Exception:
                    _log(f"🧮 指標データロード進捗: {idx}/{total_syms}", ui=False)
                    try:
                        cb = globals().get("_LOG_CALLBACK")
                        if cb and callable(cb):
                            try:
                                cb(f"🧮 指標データロード進捗: {idx}/{total_syms}")
                            except Exception:
                                pass
                    except Exception:
                        pass
        try:
            total_elapsed = int(max(0, _t.time() - start_ts))
            m, s = divmod(total_elapsed, 60)
            done_msg = f"🧮 指標データロード完了: {len(data)}/{total_syms} | 所要 {m}分{s}秒"
            _log(done_msg)
            try:
                cb = globals().get("_LOG_CALLBACK")
                if cb and callable(cb):
                    try:
                        cb(done_msg)
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception:
            _log(f"🧮 指標データロード完了: {len(data)}/{total_syms}")
            try:
                cb = globals().get("_LOG_CALLBACK")
                if cb and callable(cb):
                    try:
                        cb(f"🧮 指標データロード完了: {len(data)}/{total_syms}")
                    except Exception:
                        pass
            except Exception:
                pass
        return data

    # 実行スコープで変数定義
    # --- フィルター・データロード変数を
    #     forループより前に定義 ---
    basic_data = load_basic_data(symbols)
    if progress_callback:
        try:
            progress_callback(2, 8, "load_basic")
        except Exception:
            pass
    # データカバレッジ内訳（rollingに存在する銘柄数）
    try:
        cov_have = len(basic_data)
        cov_total = len(symbols)
        cov_missing = max(0, cov_total - cov_have)
        _log(
            "🧮 データカバレッジ: "
            + f"rolling取得済み {cov_have}/{cov_total} | missing={cov_missing}"
        )
    except Exception:
        pass
    # 共有指標の前計算（ATR/SMA/ADXなど）
    try:
        import os as _os

        from common.indicators_precompute import (
            PRECOMPUTED_INDICATORS,
            precompute_shared_indicators,
        )

        # 実行しきい値（小規模ユニバースでは前計算をスキップしてオーバーヘッド削減）
        try:
            _thr_syms = int(_os.environ.get("PRECOMPUTE_SYMBOLS_THRESHOLD", "300"))
        except Exception:
            _thr_syms = 300
        if len(basic_data) < max(0, _thr_syms):
            _log(
                f"🧮 共有指標の前計算: スキップ（対象銘柄 {len(basic_data)} 件 < 閾値 {_thr_syms}）"
            )
        else:
            try:
                _log(
                    "🧮 共有指標の前計算を開始: "
                    + ", ".join(list(PRECOMPUTED_INDICATORS)[:8])
                    + (" …" if len(PRECOMPUTED_INDICATORS) > 8 else "")
                )
            except Exception:
                _log("🧮 共有指標の前計算を開始 (ATR/SMA/ADX ほか)")
            # 大規模ユニバース時は並列化（環境変数で強制ON/OFF可能）
            force_parallel = _os.environ.get("PRECOMPUTE_PARALLEL", "").lower()
            try:
                _thr_parallel = int(_os.environ.get("PRECOMPUTE_PARALLEL_THRESHOLD", "1000"))
            except Exception:
                _thr_parallel = 1000
            if force_parallel in ("1", "true", "yes"):
                use_parallel = True
            elif force_parallel in ("0", "false", "no"):
                use_parallel = False
            else:
                use_parallel = len(basic_data) >= max(0, _thr_parallel)

            # 前計算のワーカー数は設定値に連動（環境変数の直接指定がある場合は別途関知）
            try:
                _st = get_settings(create_dirs=False)
                _pre_workers = int(getattr(_st, "THREADS_DEFAULT", 12))
            except Exception:
                _pre_workers = 12
            if use_parallel:
                try:
                    _log(f"🧵 前計算 並列ワーカー: {_pre_workers}")
                except Exception:
                    pass
            basic_data = precompute_shared_indicators(
                basic_data,
                log=_log,
                parallel=use_parallel,
                max_workers=_pre_workers if use_parallel else None,
            )
            _log("🧮 共有指標の前計算が完了")
    except Exception as e:
        _log(f"⚠️ 共有指標の前計算に失敗: {e}")
    _log("🧪 事前フィルター実行中 (system1〜system6)…")
    system1_syms = filter_system1(symbols, basic_data)
    system2_syms = filter_system2(symbols, basic_data)
    system3_syms = filter_system3(symbols, basic_data)
    system4_syms = filter_system4(symbols, basic_data)
    system5_syms = filter_system5(symbols, basic_data)
    system6_syms = filter_system6(symbols, basic_data)
    # UI のメトリクスに対象銘柄数を即反映（0%段階）
    try:
        cb2 = globals().get("_PER_SYSTEM_STAGE")
    except Exception:
        cb2 = None
    if cb2 and callable(cb2):
        try:
            cb2("system1", 0, len(system1_syms), None, None, None)
            cb2("system2", 0, len(system2_syms), None, None, None)
            cb2("system3", 0, len(system3_syms), None, None, None)
            cb2("system4", 0, len(system4_syms), None, None, None)
            cb2("system5", 0, len(system5_syms), None, None, None)
            cb2("system6", 0, len(system6_syms), None, None, None)
            cb2("system7", 0, 1 if "SPY" in basic_data else 0, None, None, None)
        except Exception:
            pass
    # System2 フィルター内訳の可視化（価格・売買代金・ATR の段階通過数）
    try:
        s2_total = len(symbols)
        c_price = 0
        c_dv = 0
        c_atr = 0
        for _sym in symbols:
            _df = basic_data.get(_sym)
            if _df is None or _df.empty:
                continue
            try:
                # 価格フィルター
                last_close = float(_df["close"].iloc[-1])
                if last_close >= 5:
                    c_price += 1
                else:
                    continue
                # 売買代金フィルター（20日平均・厳密）
                dv = float((_df["close"] * _df["volume"]).tail(20).mean())
                if dv >= 2.5e7:
                    c_dv += 1
                else:
                    continue
                # ATR 比率フィルター（10日）
                if "high" in _df.columns and "low" in _df.columns:
                    _tr = (_df["high"] - _df["low"]).tail(10)
                    _atr = float(_tr.mean())
                    if _atr >= last_close * 0.03:
                        c_atr += 1
            except Exception:
                continue
        _log(
            "🧪 system2内訳: "
            + f"元={s2_total}, 価格>=5: {c_price}, DV20>=25M: {c_dv}, ATR>=3%: {c_atr}"
        )
    except Exception:
        pass
    # System1 フィルター内訳（価格・売買代金）
    try:
        s1_total = len(symbols)
        s1_price = 0
        s1_dv = 0
        for _sym in symbols:
            _df = basic_data.get(_sym)
            if _df is None or _df.empty:
                continue
            try:
                last_close = float(_df.get("close", _df.get("Close")).iloc[-1])  # type: ignore[index]
                if last_close >= 5:
                    s1_price += 1
                else:
                    continue
                dv20 = float(
                    (_df.get("close", _df.get("Close")) * _df.get("volume", _df.get("Volume")))
                    .tail(20)
                    .mean()  # type: ignore[union-attr]
                )
                if dv20 >= 5e7:
                    s1_dv += 1
            except Exception:
                continue
        _log("🧪 system1内訳: " + f"元={s1_total}, 価格>=5: {s1_price}, DV20>=50M: {s1_dv}")
    except Exception:
        pass
    # System3 フィルター内訳（Low>=1 → AvgVol50>=1M → ATR_Ratio>=5%）
    try:
        s3_total = len(symbols)
        s3_low = 0
        s3_av = 0
        s3_atr = 0
        for _sym in symbols:
            _df = basic_data.get(_sym)
            if _df is None or _df.empty:
                continue
            try:
                _low_ser = _df.get("Low", _df.get("low"))
                if _low_ser is None:
                    continue
                if float(_low_ser.iloc[-1]) >= 1:
                    s3_low += 1
                else:
                    continue
                _av50 = _df.get("AvgVolume50")
                if (
                    _av50 is not None
                    and not pd.isna(_av50.iloc[-1])
                    and float(_av50.iloc[-1]) >= 1_000_000
                ):
                    s3_av += 1
                else:
                    continue
                _atr_ratio = _df.get("ATR_Ratio")
                if (
                    _atr_ratio is not None
                    and not pd.isna(_atr_ratio.iloc[-1])
                    and float(_atr_ratio.iloc[-1]) >= 0.05
                ):
                    s3_atr += 1
            except Exception:
                continue
        _log(
            "🧪 system3内訳: "
            + f"元={s3_total}, Low>=1: {s3_low}, AvgVol50>=1M: {s3_av}, ATR_Ratio>=5%: {s3_atr}"
        )
    except Exception:
        pass
    # System4 フィルター内訳（DV50>=100M → HV50 10〜40）
    try:
        s4_total = len(symbols)
        s4_dv = 0
        s4_hv = 0
        for _sym in symbols:
            _df = basic_data.get(_sym)
            if _df is None or _df.empty:
                continue
            try:
                _dv50 = _df.get("DollarVolume50")
                _hv50 = _df.get("HV50")
                if (
                    _dv50 is not None
                    and not pd.isna(_dv50.iloc[-1])
                    and float(_dv50.iloc[-1]) > 100_000_000
                ):
                    s4_dv += 1
                else:
                    continue
                if _hv50 is not None and not pd.isna(_hv50.iloc[-1]):
                    hv = float(_hv50.iloc[-1])
                    if 10 <= hv <= 40:
                        s4_hv += 1
            except Exception:
                continue
        _log("🧪 system4内訳: " + f"元={s4_total}, DV50>=100M: {s4_dv}, HV50 10〜40: {s4_hv}")
    except Exception:
        pass
    # System5 フィルター内訳（AvgVol50>500k → DV50>2.5M → ATR_Pct>4%）
    try:
        s5_total = len(symbols)
        s5_av = 0
        s5_dv = 0
        s5_atr = 0
        for _sym in symbols:
            _df = basic_data.get(_sym)
            if _df is None or _df.empty:
                continue
            try:
                _av50 = _df.get("AvgVolume50")
                if (
                    _av50 is not None
                    and not pd.isna(_av50.iloc[-1])
                    and float(_av50.iloc[-1]) > 500_000
                ):
                    s5_av += 1
                else:
                    continue
                _dv50 = _df.get("DollarVolume50")
                if (
                    _dv50 is not None
                    and not pd.isna(_dv50.iloc[-1])
                    and float(_dv50.iloc[-1]) > 2_500_000
                ):
                    s5_dv += 1
                else:
                    continue
                _atrp = _df.get("ATR_Pct")
                if (
                    _atrp is not None
                    and not pd.isna(_atrp.iloc[-1])
                    and float(_atrp.iloc[-1]) > 0.04
                ):
                    s5_atr += 1
            except Exception:
                continue
        _log(
            "🧪 system5内訳: "
            + f"元={s5_total}, AvgVol50>500k: {s5_av}, DV50>2.5M: {s5_dv}, ATR_Pct>4%: {s5_atr}"
        )
    except Exception:
        pass
    # System6 フィルター内訳（Low>=5 → DV50>10M）
    try:
        s6_total = len(symbols)
        s6_low = 0
        s6_dv = 0
        for _sym in symbols:
            _df = basic_data.get(_sym)
            if _df is None or _df.empty:
                continue
            try:
                _low_ser = _df.get("Low", _df.get("low"))
                if _low_ser is None:
                    continue
                if float(_low_ser.iloc[-1]) >= 5:
                    s6_low += 1
                else:
                    continue
                _dv50 = _df.get("DollarVolume50")
                if (
                    _dv50 is not None
                    and not pd.isna(_dv50.iloc[-1])
                    and float(_dv50.iloc[-1]) > 10_000_000
                ):
                    s6_dv += 1
            except Exception:
                continue
        _log("🧪 system6内訳: " + f"元={s6_total}, Low>=5: {s6_low}, DV50>10M: {s6_dv}")
    except Exception:
        pass
    # System7 は SPY 固定（参考情報のみ）
    try:
        spyp = (
            1 if ("SPY" in basic_data and not getattr(basic_data.get("SPY"), "empty", True)) else 0
        )
        _log("🧪 system7内訳: SPY固定 | SPY存在=" + str(spyp))
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
    def _subset_data(keys: list[str]) -> dict[str, pd.DataFrame]:
        out = {}
        for s in keys or []:
            v = basic_data.get(s)
            if v is not None and not getattr(v, "empty", True):
                out[s] = v
        return out

    _log("🧮 指標計算用データロード中 (system1)…")
    raw_data_system1 = _subset_data(system1_syms)
    _log(f"🧮 指標データ: system1={len(raw_data_system1)}銘柄")
    _log("🧮 指標計算用データロード中 (system2)…")
    raw_data_system2 = _subset_data(system2_syms)
    _log(f"🧮 指標データ: system2={len(raw_data_system2)}銘柄")
    _log("🧮 指標計算用データロード中 (system3)…")
    raw_data_system3 = _subset_data(system3_syms)
    _log(f"🧮 指標データ: system3={len(raw_data_system3)}銘柄")
    _log("🧮 指標計算用データロード中 (system4)…")
    raw_data_system4 = _subset_data(system4_syms)
    _log(f"🧮 指標データ: system4={len(raw_data_system4)}銘柄")
    _log("🧮 指標計算用データロード中 (system5)…")
    raw_data_system5 = _subset_data(system5_syms)
    _log(f"🧮 指標データ: system5={len(raw_data_system5)}銘柄")
    _log("🧮 指標計算用データロード中 (system6)…")
    raw_data_system6 = _subset_data(system6_syms)
    _log(f"🧮 指標データ: system6={len(raw_data_system6)}銘柄")
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
        System6Strategy(),
        System7Strategy(),
    ]
    strategies = {getattr(s, "SYSTEM_NAME", "").lower(): s for s in strategy_objs}
    # エンジン層はUI依存を排除（UI表示はlog/progressコールバック側に任せる）

    def _run_strategy(name: str, stg) -> tuple[str, pd.DataFrame, str, list[str]]:
        logs: list[str] = []

        def _local_log(message: str) -> None:
            logs.append(str(message))
            # UI コールバックがあれば即時にUIへ転送。無ければCLIへ印字。
            try:
                cb = globals().get("_LOG_CALLBACK")
            except Exception:
                cb = None
            if cb and callable(cb):
                try:
                    cb(f"[{name}] {str(message)}")
                except Exception:
                    pass
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
            base = {"SPY": basic_data.get("SPY")} if "basic_data" in locals() else {}
        else:
            base = basic_data if "basic_data" in locals() else {}
        if name == "system4" and spy_df is None:
            _local_log(
                "⚠️ System4 は SPY 指標が必要ですが "
                + "SPY データがありません。"
                + "スキップします。"
            )
            return name, pd.DataFrame(), f"❌ {name}: 0 件 🚫", logs
        _local_log(f"🔎 {name}: シグナル抽出を開始")
        try:
            # 段階進捗: 0/25/50/75/100 を UI 側に橋渡し
            def _stage(
                v: int,
                f: int | None = None,
                s: int | None = None,
                c: int | None = None,
                fin: int | None = None,
            ) -> None:
                try:
                    cb2 = globals().get("_PER_SYSTEM_STAGE")
                except Exception:
                    cb2 = None
                if cb2 and callable(cb2):
                    try:
                        cb2(name, max(0, min(100, int(v))), f, s, c, fin)
                    except Exception:
                        pass

            import os as _os

            # プロセスプール利用可否（環境変数で上書き可）
            env_pp = _os.environ.get("USE_PROCESS_POOL", "").lower()
            if env_pp in ("0", "false", "no"):
                use_process_pool = False
            elif env_pp in ("1", "true", "yes"):
                use_process_pool = True
            else:
                use_process_pool = True
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
                lb_default = int(settings2.cache.rolling.base_lookback_days)
            except Exception:
                settings2 = None
                lb_default = 240
            # YAMLのstrategiesセクション等からヒントを取得（なければヒューリスティック）
            # ルックバックのマージン/最小日数は環境変数で上書き可能
            try:
                margin = float(_os.environ.get("LOOKBACK_MARGIN", "0.15"))
            except Exception:
                margin = 0.15
            need_map: dict[str, int] = {
                "system1": int(200 * (1 + margin)),
                "system2": int(120 * (1 + margin)),
                "system3": int(60 * (1 + margin)),
                "system4": int(80 * (1 + margin)),
                "system5": int(120 * (1 + margin)),
                "system6": int(80 * (1 + margin)),
                "system7": int(80 * (1 + margin)),
            }
            # 戦略側が get_total_days を実装していれば優先
            custom_need = None
            try:
                fn = getattr(stg, "get_total_days", None)
                if callable(fn):
                    _val = fn(base)
                    if isinstance(_val, (int, float)):
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
            df = stg.get_today_signals(
                base,
                market_df=spy_df,
                today=today,
                progress_callback=None,
                log_callback=_local_log,
                stage_progress=_stage,
                use_process_pool=use_process_pool,
                max_workers=max_workers,
                lookback_days=lookback_days,
            )
            _elapsed = int(max(0, __import__("time").time() - _t0))
            _m, _s = divmod(_elapsed, 60)
            _local_log(f"⏱️ {name}: 経過 {_m}分{_s}秒")
        except Exception as e:  # noqa: BLE001
            _local_log(f"⚠️ {name}: シグナル抽出に失敗しました: {e}")
            df = pd.DataFrame()
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

    _log("🚀 各システムの当日シグナル抽出を開始")
    per_system: dict[str, pd.DataFrame] = {}
    total = len(strategies)
    # 事前に全システムへステージ0%（filter開始）を同時通知（UI同期表示用）
    try:
        cb2 = globals().get("_PER_SYSTEM_STAGE")
    except Exception:
        cb2 = None
    if cb2 and callable(cb2):
        for name in strategies.keys():
            try:
                cb2(name, 0, None, None, None, None)
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
                fut = executor.submit(_run_strategy, name, stg)
                futures[fut] = name
            for _idx, fut in enumerate(as_completed(futures), start=1):
                name, df, msg, logs = fut.result()
                per_system[name] = df
                # 完了通知
                if per_system_progress:
                    try:
                        per_system_progress(name, "done")
                    except Exception:
                        pass
                msg_prev = msg.replace(name, f"(前回結果) {name}", 1)
                _log(f"🧾 {msg_prev}")
                if progress_callback:
                    try:
                        progress_callback(5 + min(_idx, 1), 8, name)
                    except Exception:
                        pass
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
            name, df, msg, logs = _run_strategy(name, stg)
            per_system[name] = df
            if per_system_progress:
                try:
                    per_system_progress(name, "done")
                except Exception:
                    pass
            msg_prev = msg.replace(name, f"(前回結果) {name}", 1)
            _log(f"🧾 {msg_prev}")
        if progress_callback:
            try:
                progress_callback(6, 8, "strategies_done")
            except Exception:
                pass

    # システム別の順序を明示（1..7）に固定
    order_1_7 = [f"system{i}" for i in range(1, 8)]
    per_system = {k: per_system.get(k, pd.DataFrame()) for k in order_1_7 if k in per_system}

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
                if out_fp.exists():
                    metrics_df.to_csv(out_fp, mode="a", header=False, index=False, encoding="utf-8")
                else:
                    metrics_df.to_csv(out_fp, index=False, encoding="utf-8")
                _log(f"📈 メトリクス保存: {out_fp} に {len(metrics_rows)} 行を追記")
            except Exception as e:
                _log(f"⚠️ メトリクス保存に失敗: {e}")
            # 通知: メトリクス概要を送信（環境が用意されていない場合は内部で無害化）
            try:
                fields = {
                    r["system"]: (
                        f"対象→{int(r['prefilter_pass'])}, " f"trade候補数→{int(r['candidates'])}"
                    )
                    for r in metrics_rows
                }
                title = "📈 本日のメトリクス（対象 / 候補）"
                _td = locals().get("today")
                try:
                    _td_str = str(getattr(_td, "date", lambda: None)() or _td)
                except Exception:
                    _td_str = ""
                msg = f"対象日: {_td_str}"
                notifier = create_notifier(platform="auto", fallback=True)
                notifier.send(title, msg, fields=fields)
            except Exception:
                pass
        # 簡易ログ
        try:
            summary = ", ".join(
                [
                    (
                        f"{r['system']}: 対象→{r['prefilter_pass']}, "
                        f"trade候補数→{r['candidates']}"
                    )
                    for r in metrics_rows
                ]
            )
            if summary:
                _log(f"📊 メトリクス概要: {summary}")
        except Exception:
            pass
    except Exception:
        _log("⚠️ メトリクス集計で例外が発生しました（処理続行）")

    # 1) 枠配分（スロット）モード or 2) 金額配分モード
    def _normalize_alloc(d: dict[str, float], default_map: dict[str, float]) -> dict[str, float]:
        try:
            filtered = {k: float(v) for k, v in d.items() if float(v) > 0}
            s = sum(filtered.values())
            if s <= 0:
                filtered = default_map
                s = sum(filtered.values())
            return {k: v / s for k, v in filtered.items()}
        except Exception:
            s = sum(default_map.values())
            return {k: v / s for k, v in default_map.items()}

    defaults_long = {"system1": 0.25, "system3": 0.25, "system4": 0.25, "system5": 0.25}
    defaults_short = {"system2": 0.40, "system6": 0.40, "system7": 0.20}
    try:
        settings_alloc_long = getattr(settings.ui, "long_allocations", {}) or {}
        settings_alloc_short = getattr(settings.ui, "short_allocations", {}) or {}
    except Exception:
        settings_alloc_long, settings_alloc_short = {}, {}
    long_alloc = _normalize_alloc(settings_alloc_long, defaults_long)
    short_alloc = _normalize_alloc(settings_alloc_short, defaults_short)

    _log("🧷 候補の配分（スロット方式 or 金額配分）を実行")
    if capital_long is None and capital_short is None:
        # 旧スロット方式（後方互換）
        max_pos = int(settings.risk.max_positions)
        slots_long = slots_long if slots_long is not None else max_pos
        slots_short = slots_short if slots_short is not None else max_pos

        def _distribute_slots(
            weights: dict[str, float], total_slots: int, counts: dict[str, int]
        ) -> dict[str, int]:
            base = {k: int(total_slots * weights.get(k, 0.0)) for k in weights}
            for k in list(base.keys()):
                if counts.get(k, 0) <= 0:
                    base[k] = 0
                elif base[k] == 0:
                    base[k] = 1
            used = sum(base.values())
            remain = max(0, total_slots - used)
            if remain > 0:
                order = sorted(
                    weights.keys(),
                    key=lambda k: (counts.get(k, 0), weights.get(k, 0.0)),
                    reverse=True,
                )
                idx = 0
                while remain > 0 and order:
                    k = order[idx % len(order)]
                    if counts.get(k, 0) > base.get(k, 0):
                        base[k] += 1
                        remain -= 1
                    idx += 1
                    if idx > 10000:
                        break
            for k in list(base.keys()):
                base[k] = min(base[k], counts.get(k, 0))
            return base

        long_counts = {k: len(per_system.get(k, pd.DataFrame())) for k in long_alloc}
        short_counts = {k: len(per_system.get(k, pd.DataFrame())) for k in short_alloc}
        _log(
            "🧮 枠配分: "
            + ", ".join([f"{k}={long_counts.get(k, 0)}" for k in long_alloc])
            + " | "
            + ", ".join([f"{k}={short_counts.get(k, 0)}" for k in short_alloc])
        )
        long_slots = _distribute_slots(long_alloc, slots_long, long_counts)
        short_slots = _distribute_slots(short_alloc, slots_short, short_counts)

        chosen_frames: list[pd.DataFrame] = []
        for name, slot in {**long_slots, **short_slots}.items():
            df = per_system.get(name, pd.DataFrame())
            if df is None or df.empty or slot <= 0:
                continue
            take = df.head(slot).copy()
            take["alloc_weight"] = (
                long_alloc.get(name) or short_alloc.get(name) or 0.0
            )  # noqa: E501
            chosen_frames.append(take)
        final_df = (
            pd.concat(chosen_frames, ignore_index=True)
            if chosen_frames
            else pd.DataFrame()  # noqa: E501
        )
    else:
        # 金額配分モード
        _settings = get_settings(create_dirs=False)
        _default_cap = float(getattr(_settings.ui, "default_capital", 100000))
        _ratio = float(getattr(_settings.ui, "default_long_ratio", 0.5))

        _cl = None if capital_long is None or float(capital_long) <= 0 else float(capital_long)
        _cs = None if capital_short is None or float(capital_short) <= 0 else float(capital_short)

        if _cl is None and _cs is None:
            total = _default_cap
            capital_long = total * _ratio
            capital_short = total * (1.0 - _ratio)
        elif _cl is None and _cs is not None:
            total = _cs
            capital_long = total * _ratio
            capital_short = total * (1.0 - _ratio)
        elif _cs is None and _cl is not None:
            total = _cl
            capital_long = total * _ratio
            capital_short = total * (1.0 - _ratio)
        else:
            # mypy/pyright対応（この分岐では None にならない）
            from typing import cast as _cast

            capital_long = float(_cast(float, capital_long))
            capital_short = float(_cast(float, capital_short))

        strategies_map = {k: v for k, v in strategies.items()}
        _log(f"💰 金額配分: long=${capital_long}, short=${capital_short}")
        # 参考: システム別の予算内訳を出力
        try:
            long_budgets = {
                k: float(capital_long) * float(long_alloc.get(k, 0.0)) for k in long_alloc
            }
            short_budgets = {
                k: float(capital_short) * float(short_alloc.get(k, 0.0)) for k in short_alloc
            }
            _log(
                "📊 long予算内訳: " + ", ".join([f"{k}=${v:,.0f}" for k, v in long_budgets.items()])
            )
            _log(
                "📊 short予算内訳: "
                + ", ".join([f"{k}=${v:,.0f}" for k, v in short_budgets.items()])
            )
        except Exception:
            pass
        long_df = _amount_pick(
            {k: per_system.get(k, pd.DataFrame()) for k in long_alloc},
            strategies_map,
            float(capital_long),
            long_alloc,
            side="long",
        )
        short_df = _amount_pick(
            {k: per_system.get(k, pd.DataFrame()) for k in short_alloc},
            strategies_map,
            float(capital_short),
            short_alloc,
            side="short",
        )
        parts = [df for df in [long_df, short_df] if df is not None and not df.empty]  # noqa: E501
        final_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()  # noqa: E501

        # 各システムの最大ポジション上限=10 を厳格化
        if not final_df.empty and "system" in final_df.columns:
            final_df = (
                final_df.sort_values(["system", "score"], ascending=[True, True])
                .groupby("system", as_index=False, group_keys=False)
                .head(int(get_settings(create_dirs=False).risk.max_positions))
                .reset_index(drop=True)
            )

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
            tmp = pd.concat(parts2, ignore_index=True)
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
                parts = [
                    f"{r['system']}: {int(r['count'])}件 / ${float(r['sum']):,.0f}"
                    for _, r in grp.iterrows()
                ]
                _log("🧾 system別サマリ: " + ", ".join(parts))
            else:
                grp = final_df.groupby("system").size().to_dict()
                _log("🧾 system別サマリ: " + ", ".join([f"{k}: {v}件" for k, v in grp.items()]))
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

    # 通知は progress_callback の有無に関係なく実行する
    if notify:
        try:
            from tools.notify_signals import send_signal_notification

            send_signal_notification(final_df)
        except Exception:
            _log("⚠️ 通知に失敗しました。")

    # CSV 保存（任意）
    if save_csv and not final_df.empty:
        date_str = today.strftime("%Y-%m-%d")
        out_all = signals_dir / f"signals_final_{date_str}.csv"
        final_df.to_csv(out_all, index=False)
        # システム別
        for name, df in per_system.items():
            if df is None or df.empty:
                continue
            out = signals_dir / f"signals_{name}_{date_str}.csv"
            df.to_csv(out, index=False)
        _log(f"💾 保存: {signals_dir} にCSVを書き出しました")
    if progress_callback:
        try:
            progress_callback(8, 8, "done")
        except Exception:
            pass

    # 終了ログ（UI/CLI 双方で記録される）
    try:
        cnt = 0 if final_df is None else len(final_df)
        _log(f"✅ シグナル検出処理 終了 | 最終候補 {cnt} 件")
    except Exception:
        pass

    # clear callback
    try:
        globals().pop("_LOG_CALLBACK", None)
    except Exception:
        pass

    return final_df, per_system


def main():
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
        help=("買いサイド予算（ドル）。" "指定時は金額配分モード"),
    )
    parser.add_argument(
        "--capital-short",
        type=float,
        default=None,
        help=("売りサイド予算（ドル）。" "指定時は金額配分モード"),
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
    args = parser.parse_args()

    final_df, per_system = compute_today_signals(
        args.symbols,
        slots_long=args.slots_long,
        slots_short=args.slots_short,
        capital_long=args.capital_long,
        capital_short=args.capital_short,
        save_csv=args.save_csv,
        parallel=args.parallel,
    )

    if final_df.empty:
        _log("📭 本日の最終候補はありません。")
    else:
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
        merge_signals([signals_for_merge], portfolio_state={}, market_state={})
        if args.alpaca_submit:
            # CLIでも共通ヘルパーを使用
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


if __name__ == "__main__":
    main()
