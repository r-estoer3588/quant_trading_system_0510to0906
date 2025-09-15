from __future__ import annotations

import argparse
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import no_type_check
import os

import pandas as pd

from common import broker_alpaca as ba
from common.alpaca_order import submit_orders_df
from common.cache_manager import CacheManager, load_base_cache
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

# ワーカー側で観測した cand_cnt(=TRDlist) を保存し、メインスレッドで参照するためのスナップショット
_CAND_COUNT_SNAPSHOT: dict[str, int] = {}

_LOG_CALLBACK = None
_LOG_START_TS = None  # CLI 用の経過時間測定開始時刻

# ログファイル設定（デフォルトは固定ファイル）。必要に応じて日付付きへ切替。
_LOG_FILE_PATH: Path | None = None
_LOG_FILE_MODE: str = "single"  # single | dated


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
            log_path = globals().get("_LOG_FILE_PATH")  # type: ignore[assignment]
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
            cb(str(message))
    except Exception:
        # UI コールバック未設定や例外は黙って無視（CLI 実行時を考慮）
        pass


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

    # キーワードによる除外判定（全体）
    try:
        if any(k in str(msg) for k in _GLOBAL_SKIP_KEYWORDS):
            return
        ui_allowed = ui and not any(k in str(msg) for k in _UI_ONLY_SKIP_KEYWORDS)
    except Exception:
        ui_allowed = ui

    # CLI へは整形して出力
    try:
        print(f"{prefix}{msg}", flush=True)
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
_GLOBAL_SKIP_KEYWORDS = (
    "バッチ時間",
    "batch time",
)
# UI 表示からのみ除外するキーワード
_UI_ONLY_SKIP_KEYWORDS = (
    "進捗",
    "インジケーター計算",
    "候補抽出",
    "候補日数",
    "銘柄:",
)


def _filter_logs(lines: list[str], ui: bool = False) -> list[str]:
    """キーワードに基づいてログ行を除外する。

    Args:
        lines: 対象ログ行のリスト。
        ui: True の場合は UI 限定の除外キーワードも適用。
    """

    skip_keywords = _GLOBAL_SKIP_KEYWORDS + (_UI_ONLY_SKIP_KEYWORDS if ui else ())
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
    # CLI 経由で未設定の場合（UI 等）、既定で日付別ログに切替
    try:
        if globals().get("_LOG_FILE_PATH") is None:
            import os as _os

            _mode_env = (_os.environ.get("TODAY_SIGNALS_LOG_MODE") or "").strip().lower()
            _configure_today_logger(mode=("single" if _mode_env == "single" else "dated"))
    except Exception:
        pass
    # === CLI バナー（開始の明確化）: RUN-ID のみ事前生成 ===
    try:
        import uuid as _uuid

        _run_id = str(_uuid.uuid4())[:8]
    except Exception:
        _run_id = "--------"

    settings = get_settings(create_dirs=True)
    cm = CacheManager(settings)
    # install log callback for helpers
    globals()["_LOG_CALLBACK"] = log_callback
    cache_dir = cm.rolling_dir
    signals_dir = Path(settings.outputs.signals_dir)
    signals_dir.mkdir(parents=True, exist_ok=True)

    # 前回結果の保存/読込ヘルパ
    def _prev_counts_path() -> Path:
        try:
            return signals_dir / "previous_per_system_counts.json"
        except Exception:
            return Path("signals/previous_per_system_counts.json")

    def _load_prev_counts() -> dict[str, int]:
        fp = _prev_counts_path()
        if not fp.exists():
            return {}
        try:
            import json as _json

            data = _json.loads(fp.read_text(encoding="utf-8"))
            counts = data.get("counts", {}) if isinstance(data, dict) else {}
            out: dict[str, int] = {}
            for i in range(1, 8):
                k = f"system{i}"
                try:
                    out[k] = int(counts.get(k, 0))
                except Exception:
                    out[k] = 0
            return out
        except Exception:
            return {}

    def _save_prev_counts(per_system_map: dict[str, pd.DataFrame]) -> None:
        try:
            from datetime import datetime as _dt
            import json as _json

            counts = {
                k: (0 if (v is None or v.empty) else int(len(v))) for k, v in per_system_map.items()
            }
            data = {"timestamp": _dt.utcnow().isoformat() + "Z", "counts": counts}
            fp = _prev_counts_path()
            try:
                fp.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            fp.write_text(_json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

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
    _log("ℹ️ 注: EODHDは当日終値が未反映のため、直近営業日ベースで計算します。")
    # 開始直後に前回結果をまとめて表示
    try:
        prev = _load_prev_counts()
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

    # バナー（開始）: 罫線は print で出力してタイムスタンプを付けない
    try:
        print("#" * 68, flush=True)
    except Exception:
        pass
    # CLI 専用の開始バナー（UI には出さない）
    _log("# 🚀🚀🚀  本日のシグナル 実行開始 (Engine)  🚀🚀🚀", ui=False)
    try:
        import time as _time

        _now = _time.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        _now = ""
    try:
        universe_total = sum(1 for s in symbols if str(s).upper() != "SPY")
    except Exception:
        universe_total = len(symbols)
    _log(
        f"# ⏱️ {_now} | 銘柄数：{universe_total}　| RUN-ID: {_run_id}",
        ui=False,
    )
    try:
        print("#" * 68 + "\n", flush=True)
    except Exception:
        pass

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
            progress_callback(1, 8, "対象読み込み:start")
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
                # 既存 rolling があっても行数不足なら再構築する
                target_len = int(
                    settings.cache.rolling.base_lookback_days + settings.cache.rolling.buffer_days
                )
                if df is None or df.empty or (hasattr(df, "__len__") and len(df) < target_len):
                    # rolling 不在 → base から必要分を生成して保存
                    base_df = load_base_cache(sym, rebuild_if_missing=True)
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
                    # 必要期間: 設計上 base_lookback_days + buffer_days（不足時は全量）
                    n = int(
                        settings.cache.rolling.base_lookback_days
                        + settings.cache.rolling.buffer_days
                    )
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
                    df = _normalize_ohlcv(df)
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
                    _emit_ui_log(msg)
                except Exception:
                    _log(f"📦 基礎データロード進捗: {idx}/{total_syms}", ui=False)
                    _emit_ui_log(f"📦 基礎データロード進捗: {idx}/{total_syms}")
        try:
            total_elapsed = int(max(0, _t.time() - start_ts))
            m, s = divmod(total_elapsed, 60)
            done_msg = f"📦 基礎データロード完了: {len(data)}/{total_syms} | 所要 {m}分{s}秒"
            _log(done_msg)
            _emit_ui_log(done_msg)
        except Exception:
            _log(f"📦 基礎データロード完了: {len(data)}/{total_syms}")
            _emit_ui_log(f"📦 基礎データロード完了: {len(data)}/{total_syms}")
        return data

    # 列名の大小・重複（DataFrame）にも耐える安全な抽出ヘルパー
    def _pick_series(df, names):
        try:
            for nm in names:
                if nm in df.columns:
                    s = df[nm]
                    # 重複列で DataFrame になる場合は先頭列を採用
                    if isinstance(s, pd.DataFrame):
                        try:
                            s = s.iloc[:, 0]
                        except Exception:
                            continue
                    # 数値化（失敗は NaN）
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

    def filter_system1(symbols, data):
        result = []
        for sym in symbols:
            df = data.get(sym)
            if df is None or df.empty:
                continue
            # 株価5ドル以上（直近終値）
            close_s = _pick_series(df, ["close", "Close", "Adj Close", "adj_close"])
            last_close = _last_scalar(close_s)
            if last_close is None or last_close < 5:
                continue
            # 過去20日平均売買代金（厳密: mean(close*volume)）が5000万ドル以上
            vol_s = _pick_series(df, ["volume", "Volume", "Vol", "vol"])
            if vol_s is None or close_s is None:
                continue
            try:
                dollar_vol = (close_s * vol_s).dropna()
            except Exception:
                continue
            if dollar_vol.tail(20).mean() < 5e7:
                continue
            result.append(sym)
        return result

    def filter_system2(symbols, data):
        result = []
        for sym in symbols:
            df = data.get(sym)
            if df is None or df.empty:
                continue
            close_s = _pick_series(df, ["close", "Close", "Adj Close", "adj_close"])
            last_close = _last_scalar(close_s)
            if last_close is None or last_close < 5:
                continue
            vol_s = _pick_series(df, ["volume", "Volume", "Vol", "vol"])
            if vol_s is None or close_s is None:
                continue
            try:
                dollar_vol = (close_s * vol_s).dropna()
            except Exception:
                continue
            if dollar_vol.tail(20).mean() < 2.5e7:
                continue
            # ATR計算（過去10日）
            high_s = _pick_series(df, ["high", "High"]) if df is not None else None
            low_s = _pick_series(df, ["low", "Low"]) if df is not None else None
            if high_s is not None and low_s is not None and close_s is not None:
                try:
                    tr = (high_s - low_s).dropna().tail(10)
                    atr = tr.mean()
                except Exception:
                    atr = None
                if atr is not None and atr < (last_close * 0.03):
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
                target_len = int(
                    settings.cache.rolling.base_lookback_days + settings.cache.rolling.buffer_days
                )
                if df is None or df.empty or (hasattr(df, "__len__") and len(df) < target_len):
                    base_df = load_base_cache(sym, rebuild_if_missing=True)
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
                    n = int(
                        settings.cache.rolling.base_lookback_days
                        + settings.cache.rolling.buffer_days
                    )
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
                    df = _normalize_ohlcv(df)
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
                    _emit_ui_log(msg)
                except Exception:
                    _log(f"🧮 指標データロード進捗: {idx}/{total_syms}", ui=False)
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
        if cov_missing > 0:
            missing_syms = [s for s in symbols if s not in basic_data]
            _log(f"🛠 欠損データ補完中: {len(missing_syms)}銘柄", ui=False)
            fixed = 0
            for sym in missing_syms:
                try:
                    base_df = load_base_cache(sym, rebuild_if_missing=True)
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
                    n = int(
                        settings.cache.rolling.base_lookback_days
                        + settings.cache.rolling.buffer_days
                    )
                    sliced = x.tail(n).reset_index(drop=True)
                    cm.write_atomic(sliced, sym, "rolling")
                    df = _normalize_ohlcv(sliced)
                    basic_data[sym] = df
                    fixed += 1
                except Exception:
                    continue
            try:
                if fixed > 0:
                    _log(f"🧩 補完書き戻し: rolling生成 {fixed}件")
            except Exception:
                pass
            cov_have = len(basic_data)
            cov_missing = max(0, cov_total - cov_have)
            _log(
                "🧮 データカバレッジ(補完後): "
                + f"rolling取得済み {cov_have}/{cov_total} | missing={cov_missing}"
            )
            # 補完後の対象件数を UI の Tgt に即時反映（全system共通）
            try:
                cb2 = globals().get("_PER_SYSTEM_STAGE")
            except Exception:
                cb2 = None
            if cb2 and callable(cb2):
                try:
                    # Tgt はユニバース総数（SPY除外）を採用
                    try:
                        tgt_total = int(cov_total)
                        if any(str(s).upper() == "SPY" for s in (symbols or [])):
                            tgt_total = max(0, tgt_total - 1)
                    except Exception:
                        tgt_total = int(cov_total)
                    for i in range(1, 8):
                        cb2(f"system{i}", 0, tgt_total, None, None, None)
                except Exception:
                    pass
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
                # 安全にカラムを取得して DV20 を計算
                _c = _df["close"] if "close" in _df.columns else _df["Close"]
                _v = _df["volume"] if "volume" in _df.columns else _df["Volume"]
                dv20 = float((_c * _v).tail(20).mean())
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
    # System1 セットアップ内訳（最新日の setup 判定数）を CLI に出力
    try:
        # フィルタ通過は事前フィルター結果（system1_syms）由来で確定
        s1_filter = int(len(system1_syms))
        # 直近日の SMA25>SMA50 を集計（事前計算済み列を参照）
        s1_setup = 0
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
                s1_setup += 1
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
    except Exception:
        pass
    _log("🧮 指標計算用データロード中 (system2)…")
    raw_data_system2 = _subset_data(system2_syms)
    _log(f"🧮 指標データ: system2={len(raw_data_system2)}銘柄")
    # System2 セットアップ内訳: フィルタ通過, RSI3>90, TwoDayUp
    try:
        s2_filter = int(len(system2_syms))
        s2_rsi = 0
        s2_up2 = 0
        for _sym in system2_syms or []:
            _df = raw_data_system2.get(_sym)
            if _df is None or getattr(_df, "empty", True):
                continue
            try:
                last = _df.iloc[-1]
            except Exception:
                continue
            try:
                if float(last.get("RSI3", 0)) > 90:
                    s2_rsi += 1
            except Exception:
                pass
            try:
                if bool(last.get("TwoDayUp", False)):
                    s2_up2 += 1
            except Exception:
                pass
        _log(
            "🧩 system2セットアップ内訳: "
            + f"フィルタ通過={s2_filter}, RSI3>90: {s2_rsi}, "
            + f"TwoDayUp: {s2_up2}"
        )
        try:
            cb2 = globals().get("_PER_SYSTEM_STAGE")
            if cb2 and callable(cb2):
                cb2("system2", 50, int(s2_filter), int(max(s2_rsi, s2_up2)), None, None)
        except Exception:
            pass
    except Exception:
        pass
    _log("🧮 指標計算用データロード中 (system3)…")
    raw_data_system3 = _subset_data(system3_syms)
    _log(f"🧮 指標データ: system3={len(raw_data_system3)}銘柄")
    # System3 セットアップ内訳: フィルタ通過, Close>SMA150, 3日下落率>=12.5%
    try:
        s3_filter = int(len(system3_syms))
        s3_close = 0
        s3_drop = 0
        for _sym in system3_syms or []:
            _df = raw_data_system3.get(_sym)
            if _df is None or getattr(_df, "empty", True):
                continue
            try:
                last = _df.iloc[-1]
            except Exception:
                continue
            try:
                if float(last.get("Close", 0)) > float(last.get("SMA150", float("inf"))):
                    s3_close += 1
            except Exception:
                pass
            try:
                if float(last.get("Drop3D", 0)) >= 0.125:
                    s3_drop += 1
            except Exception:
                pass
        _log(
            "🧩 system3セットアップ内訳: "
            + f"フィルタ通過={s3_filter}, Close>SMA150: {s3_close}, "
            + f"3日下落率>=12.5%: {s3_drop}"
        )
        try:
            cb2 = globals().get("_PER_SYSTEM_STAGE")
            if cb2 and callable(cb2):
                cb2("system3", 50, int(s3_filter), int(max(s3_close, s3_drop)), None, None)
        except Exception:
            pass
    except Exception:
        pass
    _log("🧮 指標計算用データロード中 (system4)…")
    raw_data_system4 = _subset_data(system4_syms)
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
    raw_data_system5 = _subset_data(system5_syms)
    _log(f"🧮 指標データ: system5={len(raw_data_system5)}銘柄")
    # System5 セットアップ内訳: フィルタ通過, Close>SMA100+ATR10, ADX7>55, RSI3<50
    try:
        s5_filter = int(len(system5_syms))
        s5_close = 0
        s5_adx = 0
        s5_rsi = 0
        for _sym in system5_syms or []:
            _df = raw_data_system5.get(_sym)
            if _df is None or getattr(_df, "empty", True):
                continue
            try:
                last = _df.iloc[-1]
            except Exception:
                continue
            try:
                if float(last.get("Close", 0)) > float(last.get("SMA100", 0)) + float(
                    last.get("ATR10", 0)
                ):
                    s5_close += 1
            except Exception:
                pass
            try:
                if float(last.get("ADX7", 0)) > 55:
                    s5_adx += 1
            except Exception:
                pass
            try:
                if float(last.get("RSI3", 100)) < 50:
                    s5_rsi += 1
            except Exception:
                pass
        _log(
            "🧩 system5セットアップ内訳: "
            + f"フィルタ通過={s5_filter}, Close>SMA100+ATR10: {s5_close}, "
            + f"ADX7>55: {s5_adx}, RSI3<50: {s5_rsi}"
        )
        try:
            cb2 = globals().get("_PER_SYSTEM_STAGE")
            if cb2 and callable(cb2):
                cb2("system5", 50, int(s5_filter), int(s5_close), None, None)
        except Exception:
            pass
    except Exception:
        pass
    _log("🧮 指標計算用データロード中 (system6)…")
    raw_data_system6 = _subset_data(system6_syms)
    _log(f"🧮 指標データ: system6={len(raw_data_system6)}銘柄")
    # System6 セットアップ内訳: フィルタ通過, Return6D>20%, UpTwoDays
    try:
        s6_filter = int(len(system6_syms))
        s6_ret = 0
        s6_up2 = 0
        for _sym in system6_syms or []:
            _df = raw_data_system6.get(_sym)
            if _df is None or getattr(_df, "empty", True):
                continue
            try:
                last = _df.iloc[-1]
            except Exception:
                continue
            try:
                if float(last.get("Return6D", 0)) > 0.20:
                    s6_ret += 1
            except Exception:
                pass
            try:
                if bool(last.get("UpTwoDays", False)):
                    s6_up2 += 1
            except Exception:
                pass
        _log(
            "🧩 system6セットアップ内訳: "
            + f"フィルタ通過={s6_filter}, Return6D>20%: {s6_ret}, "
            + f"UpTwoDays: {s6_up2}"
        )
        try:
            cb2 = globals().get("_PER_SYSTEM_STAGE")
            if cb2 and callable(cb2):
                cb2("system6", 50, int(s6_filter), int(max(s6_ret, s6_up2)), None, None)
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
        System6Strategy(),
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
            if cb and callable(cb):
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
                # TRDlist件数スナップショットを更新（後段のメインスレッド通知で使用）
                try:
                    if c is not None:
                        _CAND_COUNT_SNAPSHOT[name] = int(c)
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
                # 既定はプロセスプール無効（UIログ/コールバックを優先）
                use_process_pool = False
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
            # プロセスプール使用時は stage_progress を渡さない（pickle/__main__問題を回避）
            _stage_cb = None if use_process_pool else _stage
            _log_cb = None if use_process_pool else _local_log
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
                except Exception as e2:  # noqa: BLE001
                    _local_log(f"❌ {name}: フォールバックも失敗: {e2}")
                    df = pd.DataFrame()
            else:
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

    # 抽出開始前にセットアップ通過のまとめを出力
    try:
        setup_summary = []
        for name, val in (
            ("system1", locals().get("s1_setup")),
            (
                "system2",
                max(locals().get("s2_rsi", 0), locals().get("s2_up2", 0)),
            ),
            (
                "system3",
                max(locals().get("s3_close", 0), locals().get("s3_drop", 0)),
            ),
            ("system4", locals().get("s4_close")),
            ("system5", locals().get("s5_close")),
            (
                "system6",
                max(locals().get("s6_ret", 0), locals().get("s6_up2", 0)),
            ),
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
            for _idx, fut in enumerate(as_completed(futures), start=1):
                name, df, msg, logs = fut.result()
                per_system[name] = df
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
                        _cand_cnt = (
                            0 if (df is None or getattr(df, "empty", True)) else int(len(df))
                        )
                        if _mx > 0:
                            _cand_cnt = min(int(_cand_cnt), int(_mx))
                        cb2(name, 75, None, None, int(_cand_cnt), None)
                    except Exception:
                        pass
                # UI が無い場合は CLI 向けに簡略ログを集約出力。UI がある場合は完了後に再送。
                # （UI にはワーカー実行中に逐次送信済みのため、ここでの再送は行わない）
                # CLI専用: ワーカー収集ログを常に出力（UIには送らない）
                for line in _filter_logs(logs, ui=False):
                    _log(f"[{name}] {line}", ui=False)
                # UI コールバックがある場合は何もしない（重複防止）
                # 完了通知
                if per_system_progress:
                    try:
                        per_system_progress(name, "done")
                    except Exception:
                        pass
                # CLI専用: 完了を簡潔表示（件数付き。失敗時は件数不明でも続行）
                try:
                    _cnt = 0 if (df is None or getattr(df, "empty", True)) else int(len(df))
                except Exception:
                    _cnt = -1
                try:
                    _log(f"✅ {name} 完了: {('?' if _cnt < 0 else _cnt)}件", ui=False)
                except Exception:
                    pass
                # 前回結果は開始時にまとめて出力するため、ここでは出さない
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
            # CLI専用: 各システム開始を即時表示（UIには出さない）
            try:
                _log(f"▶ {name} 開始", ui=False)
            except Exception:
                pass
            name, df, msg, logs = _run_strategy(name, stg)
            per_system[name] = df
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
                    _cand_cnt = 0 if (df is None or getattr(df, "empty", True)) else int(len(df))
                    if _mx > 0:
                        _cand_cnt = min(int(_cand_cnt), int(_mx))
                    cb2(name, 75, None, None, int(_cand_cnt), None)
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
                    _cand_cnt = int(_CAND_COUNT_SNAPSHOT.get(_name))
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
            merged = pd.concat(all_rows, ignore_index=True)
            merged = merged.sort_values("_sort_val", kind="stable", na_position="last")
            top10 = merged.head(10).drop(columns=["_sort_val"], errors="ignore")
            _log("📝 事前トレードリスト(Top10, メトリクス保存前)")
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
        except Exception:
            pass
    except Exception:
        pass

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
                def _estimate_exit_counts_today() -> dict[str, int]:
                    counts: dict[str, int] = {}
                    try:
                        # 価格ロード関数は共通ローダーを利用
                        from common.data_loader import load_price as _load_price  # lazy import

                        # SPY から本日の基準日（最新営業日）を推定
                        latest_trading_day = None
                        try:
                            spy_df0 = _load_price("SPY", cache_profile="rolling")
                            if spy_df0 is not None and not spy_df0.empty:
                                latest_trading_day = pd.to_datetime(spy_df0.index[-1]).normalize()
                        except Exception:
                            latest_trading_day = None

                        # Alpaca ポジション取得（失敗時は空）
                        try:
                            client0 = ba.get_client(paper=True)
                            positions0 = list(client0.get_all_positions())
                        except Exception:
                            positions0 = []

                        # エントリー日のローカル記録と system 推定マップ
                        entry_map0 = load_entry_dates()
                        sym_map_path0 = Path("data/symbol_system_map.json")
                        try:
                            import json as _json

                            symbol_system_map0 = (
                                _json.loads(sym_map_path0.read_text(encoding="utf-8"))
                                if sym_map_path0.exists()
                                else {}
                            )
                        except Exception:
                            symbol_system_map0 = {}

                        for pos in positions0:
                            try:
                                sym = str(getattr(pos, "symbol", "")).upper()
                                if not sym:
                                    continue
                                qty = int(abs(float(getattr(pos, "qty", 0)) or 0))
                                if qty <= 0:
                                    continue
                                pos_side = str(getattr(pos, "side", "")).lower()
                                # system の推定
                                system0 = str(symbol_system_map0.get(sym, "")).lower()
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
                                            pd.to_datetime(dfp2["Date"]).dt.normalize()
                                        )
                                    else:
                                        dfp2.index = pd.Index(
                                            pd.to_datetime(dfp2.index).normalize()
                                        )
                                except Exception:
                                    continue
                                if latest_trading_day is None and len(dfp2.index) > 0:
                                    latest_trading_day = pd.to_datetime(dfp2.index[-1]).normalize()
                                # エントリー日のインデックス
                                try:
                                    idx = dfp2.index
                                    ent_dt = pd.to_datetime(entry_date_str0).normalize()
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
                                        stg0 = System6Strategy()
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
                                today_norm0 = pd.to_datetime(dfp2.index[-1]).normalize()
                                if latest_trading_day is not None:
                                    today_norm0 = latest_trading_day
                                is_today_exit0 = (
                                    pd.to_datetime(exit_date0).normalize() == today_norm0
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

                exit_counts_map = _estimate_exit_counts_today()
                # UI へも Exit 件数を送る（早期に可視化）
                try:
                    cb_exit = globals().get("_PER_SYSTEM_EXIT")
                except Exception:
                    cb_exit = None
                if cb_exit and callable(cb_exit):
                    try:
                        for _nm, _cnt in (exit_counts_map or {}).items():
                            try:
                                cb_exit(_nm, int(_cnt))
                            except Exception:
                                pass
                    except Exception:
                        pass
                # 既に集計済みの値を再構成
                setup_map = {
                    # System1 は SPY ゲート（Close>SMA100）が偽なら 0 扱い
                    "system1": int(
                        (
                            locals().get("s1_setup")
                            if (
                                (locals().get("_spy_ok") is None)
                                or (int(locals().get("_spy_ok", 0)) == 1)
                            )
                            else 0
                        )
                        or 0
                    ),
                    "system2": int(max(locals().get("s2_rsi", 0), locals().get("s2_up2", 0))),
                    "system3": int(max(locals().get("s3_close", 0), locals().get("s3_drop", 0))),
                    "system4": int(locals().get("s4_close") or 0),
                    "system5": int(locals().get("s5_close") or 0),
                    "system6": int(max(locals().get("s6_ret", 0), locals().get("s6_up2", 0))),
                    "system7": 1 if ("SPY" in (locals().get("basic_data", {}) or {})) else 0,
                }
                final_counts = {}
                try:
                    _final_df = locals().get("final_df")
                    if (
                        _final_df is not None
                        and not getattr(_final_df, "empty", True)
                        and "system" in _final_df.columns
                    ):
                        final_counts = _final_df.groupby("system").size().to_dict()
                except Exception:
                    final_counts = {}
                lines = []
                for sys_name in order_1_7:
                    tgt = tgt_base if sys_name != "system7" else 1
                    fil = int(prefilter_map.get(sys_name, 0))
                    stu = int(setup_map.get(sys_name, 0))
                    try:
                        _df_trd = per_system.get(sys_name, pd.DataFrame())
                        trd = int(
                            0
                            if _df_trd is None or getattr(_df_trd, "empty", True)
                            else len(_df_trd)
                        )
                    except Exception:
                        trd = 0
                    ent = int(final_counts.get(sys_name, 0))
                    exv = exit_counts_map.get(sys_name)
                    ex_txt = "-" if exv is None else str(int(exv))
                    value = (
                        f"Tgt {tgt} / FIL {fil} / STU {stu} / "
                        f"TRD {trd} / Entry {ent} / Exit {ex_txt}"
                    )
                    lines.append({"name": sys_name, "value": value})
                title = "📈 本日の最終メトリクス（system別）"
                _td = locals().get("today")
                try:
                    _td_str = str(getattr(_td, "date", lambda: None)() or _td)
                except Exception:
                    _td_str = ""
                # fields に各systemのメトリクスを添付するため、本文は簡潔にする
                msg = f"対象日: {_td_str}"
                notifier = create_notifier(platform="auto", fallback=True)
                notifier.send(title, msg, fields=lines)
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

    # 最終採用件数（Entry）を100%段階として通知（UI カウンタ整合）
    try:
        cb2 = globals().get("_PER_SYSTEM_STAGE")
    except Exception:
        cb2 = None
    if cb2 and callable(cb2):
        try:
            # per-system 候補（TRDlist）は上で通知済み。ここでは最終採用数を渡す。
            final_counts: dict[str, int] = {}
            try:
                if (
                    final_df is not None
                    and not getattr(final_df, "empty", True)
                    and "system" in final_df.columns
                ):
                    final_counts = (
                        final_df.groupby("system").size().to_dict()  # type: ignore[assignment]
                    )
            except Exception:
                final_counts = {}
            for _name in order_1_7:
                _df_sys = per_system.get(_name, pd.DataFrame())
                _cand_cnt = int(
                    0 if _df_sys is None or getattr(_df_sys, "empty", True) else len(_df_sys)
                )
                _final_cnt = int(final_counts.get(_name, 0))
                cb2(_name, 100, None, None, _cand_cnt, _final_cnt)
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
        # ファイル名モード: date(YYYY-MM-DD) | datetime(YYYY-MM-DD_HHMM) | runid(YYYY-MM-DD_RUNID)
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
            try:
                # _run_id は本関数先頭で採番済み
                suffix = f"{date_str}_{_run_id}"
            except Exception:
                suffix = date_str

        out_all = signals_dir / f"signals_final_{suffix}.csv"
        final_df.to_csv(out_all, index=False)
        # システム別
        for name, df in per_system.items():
            if df is None or df.empty:
                continue
            out = signals_dir / f"signals_{name}_{suffix}.csv"
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

    # === CLI バナー（終了の明確化）===
    try:
        import time as _time

        _end_txt = _time.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        _end_txt = ""
    try:
        print("#" * 68, flush=True)
    except Exception:
        pass
    _log("# 🏁🏁🏁  本日のシグナル 実行終了 (Engine)  🏁🏁🏁", ui=False)
    _log(f"# ⏱️ {_end_txt} | RUN-ID: {_run_id}", ui=False)
    try:
        print("#" * 68 + "\n", flush=True)
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
    args = parser.parse_args()

    # ログ保存形式を決定（CLI > 環境変数 > 既定）
    env_mode = os.environ.get("TODAY_SIGNALS_LOG_MODE", "").strip().lower()
    mode = args.log_file_mode or (env_mode if env_mode in {"single", "dated"} else None) or "dated"
    _configure_today_logger(mode=mode)
    try:
        sel_path = globals().get("_LOG_FILE_PATH")
        _log(f"📝 ログ保存先: {sel_path}", ui=False)
    except Exception:
        pass

    final_df, per_system = compute_today_signals(
        args.symbols,
        slots_long=args.slots_long,
        slots_short=args.slots_short,
        capital_long=args.capital_long,
        capital_short=args.capital_short,
        save_csv=args.save_csv,
        csv_name_mode=args.csv_name_mode,
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
