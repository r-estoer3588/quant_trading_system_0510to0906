from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import inspect
import math
import time as _t
from typing import Any, cast
import numpy as np

import pandas as pd

from config.settings import get_settings
from core.system5 import (
    DEFAULT_ATR_PCT_THRESHOLD,
    format_atr_pct_threshold_label,
)
from common.utils_spy import (
    get_latest_nyse_trading_day,
    get_spy_with_indicators,
    resolve_signal_entry_date,
)


# --- CLI用デフォルトログ関数 -----------------------------------------------
def _default_cli_log(message: str) -> None:
    """log_callback未指定時にCLIへ確実に出力するための簡易プリンタ。

    - 文字化け/エンコード例外を避けるため、失敗時はフォールバックして出力。
    - flush=True でリアルタイムに表示。
    """
    try:
        print(str(message), flush=True)
    except Exception:
        try:
            # 最低限のASCIIにフォールバック
            safe = str(message).encode("ascii", errors="replace").decode("ascii")
            print(safe, flush=True)
        except Exception:
            pass


# --- サイド定義（売買区分）---
# System1/3/5 は買い戦略、System2/4/6/7 は売り戦略として扱う。
LONG_SYSTEMS = {"system1", "system3", "system5"}
SHORT_SYSTEMS = {"system2", "system4", "system6", "system7"}

# fast-path 判定に使用する必須列
_FAST_PATH_REQUIRED_COLUMNS = {"filter", "setup"}

TODAY_SIGNAL_COLUMNS = [
    "symbol",
    "system",
    "side",
    "signal_type",
    "entry_date",
    "entry_price",
    "stop_price",
    "score_key",
    "score",
    "reason",
]

# fast-path 判定に使用する必須列
_FAST_PATH_REQUIRED_COLUMNS = {"filter", "setup"}


@dataclass(frozen=True)
class TodaySignal:
    symbol: str
    system: str
    side: str  # "long" | "short"
    signal_type: str  # "buy" | "sell"
    entry_date: pd.Timestamp
    entry_price: float
    stop_price: float
    score_key: str | None = None
    score: float | None = None
    reason: str | None = None


@dataclass
class SkipStats:
    counts: dict[str, int] = field(default_factory=dict)
    samples: dict[str, list[str]] = field(default_factory=dict)
    details: list[dict[str, str]] = field(default_factory=list)

    def record(self, symbol: str, reason: str) -> None:
        reason = str(reason or "unknown")
        symbol = str(symbol or "")
        self.counts[reason] = self.counts.get(reason, 0) + 1
        if symbol:
            bucket = self.samples.setdefault(reason, [])
            if len(bucket) < 5 and symbol not in bucket:
                bucket.append(symbol)
        try:
            self.details.append({"symbol": symbol, "reason": reason})
        except Exception:
            pass

    def callback(self) -> Callable[..., None]:
        def _on_skip(*args: Any, **kwargs: Any) -> None:
            try:
                if len(args) >= 2:
                    symbol = str(args[0])
                    reason = str(args[1])
                elif len(args) == 1:
                    txt = str(args[0])
                    if ":" in txt:
                        symbol, reason = (txt.split(":", 1) + [""])[:2]
                        symbol = symbol.strip()
                        reason = reason.strip()
                    else:
                        symbol = ""
                        reason = txt.strip()
                else:
                    reason = str(kwargs.get("reason", "unknown"))
                    symbol = str(kwargs.get("symbol", ""))
            except Exception:
                reason = "unknown"
                symbol = ""
            self.record(symbol, reason)

        return _on_skip

    def log_summary(
        self,
        system_name: str,
        log_callback: Callable[[str], None] | None,
    ) -> None:
        if not self.counts:
            return
        try:
            sorted_items = sorted(self.counts.items(), key=lambda x: x[1], reverse=True)
        except Exception:
            sorted_items = list(self.counts.items())
        top = sorted_items[:2]
        if log_callback:
            try:
                details = ", ".join([f"{k}: {v}" for k, v in top])
                log_callback(f"🧪 スキップ統計: {details}")
                for key, _ in top:
                    samples = self.samples.get(key) or []
                    if samples:
                        log_callback(f"  ↳ ({key}): {', '.join(samples)}")
            except Exception:
                pass
        self._persist_to_csv(system_name, log_callback, sorted_items)

    def _persist_to_csv(
        self,
        system_name: str,
        log_callback: Callable[[str], None] | None,
        sorted_items: list[tuple[str, int]],
    ) -> None:
        if not sorted_items:
            return
        try:
            import os
            import pandas as _pd
        except Exception:
            return
        rows: list[dict[str, Any]] = []
        for reason, count in sorted_items:
            rows.append(
                {
                    "reason": reason,
                    "count": int(count),
                    "examples": ", ".join(self.samples.get(reason, [])),
                }
            )
        if not rows:
            return
        try:
            settings = get_settings(create_dirs=True)
            out_dir_obj = getattr(settings.outputs, "results_csv_dir", None)
        except Exception:
            out_dir_obj = None
        out_dir = str(out_dir_obj or "results_csv")
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception:
            pass
        summary_path = os.path.join(out_dir, f"skip_summary_{system_name}.csv")
        try:
            _pd.DataFrame(rows).to_csv(summary_path, index=False, encoding="utf-8")
            if log_callback:
                log_callback(f"📝 スキップ統計CSVを保存 {summary_path}")
        except Exception:
            pass
        if not self.details:
            return
        details_path = os.path.join(out_dir, f"skip_details_{system_name}.csv")
        try:
            _pd.DataFrame(self.details).to_csv(
                details_path, index=False, encoding="utf-8"
            )
            if log_callback:
                log_callback(f"📝 スキップ詳細CSVを保存 {details_path}")
        except Exception:
            pass


@dataclass
class PrepareDataResult:
    prepared: dict[str, pd.DataFrame]
    fast_path_used: bool
    fast_missing: set[str]
    skip_stats: SkipStats
    early_exit_frame: pd.DataFrame | None = None
    early_exit_reason: str | None = None


@dataclass
class CandidateExtraction:
    candidates_by_date: dict | None
    market_df: pd.DataFrame | None
    used_fast_path: bool
    early_exit_frame: pd.DataFrame | None = None
    zero_reason: str | None = None


@dataclass
class CandidateSelection:
    key_map: dict[pd.Timestamp, object]
    target_date: pd.Timestamp | None
    fallback_reason: str | None
    total_candidates_today: int
    zero_reason: str | None = None


def _normalize_today(today: pd.Timestamp | None) -> pd.Timestamp:
    if today is None:
        base = get_latest_nyse_trading_day()
    else:
        try:
            base = pd.Timestamp(today)
        except Exception:
            base = get_latest_nyse_trading_day()
    if getattr(base, "tzinfo", None) is not None:
        try:
            base = base.tz_convert(None)
        except (TypeError, ValueError, AttributeError):
            try:
                base = base.tz_localize(None)
            except Exception:
                base = pd.Timestamp(base.to_pydatetime().replace(tzinfo=None))
    return base.normalize()


def _slice_data_for_lookback(
    raw_data_dict: dict[str, pd.DataFrame],
    lookback_days: int | None,
) -> dict[str, pd.DataFrame]:
    if (
        lookback_days is None
        or lookback_days <= 0
        or not isinstance(raw_data_dict, dict)
    ):
        return raw_data_dict
    sliced: dict[str, pd.DataFrame] = {}
    for sym, df in raw_data_dict.items():
        try:
            if df is None or getattr(df, "empty", True):
                continue
            x = df.copy()
            if "Date" in x.columns:
                idx = pd.to_datetime(x["Date"], errors="coerce").dt.normalize()
                x.index = pd.Index(idx)
            else:
                x.index = pd.to_datetime(x.index, errors="coerce").normalize()
            x = x[~x.index.isna()]
            x = x.tail(int(lookback_days))
            sliced[sym] = x
        except Exception:
            sliced[sym] = df
    return sliced


def _normalize_prepared_dict(
    prepared_dict: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    try:
        if not isinstance(prepared_dict, dict):
            return prepared_dict
        fixed: dict[str, pd.DataFrame] = {}
        for sym, df in prepared_dict.items():
            try:
                x = df.copy()
                if "Date" in x.columns:
                    idx = pd.to_datetime(x["Date"], errors="coerce").dt.normalize()
                else:
                    idx = pd.to_datetime(x.index, errors="coerce").normalize()
                x.index = pd.Index(idx)
                x = x[~x.index.isna()]
                x = x.sort_index()
                if getattr(x.index, "has_duplicates", False):
                    x = x[~x.index.duplicated(keep="last")]
                fixed[sym] = x
            except Exception:
                fixed[sym] = df
        return fixed
    except Exception:
        return prepared_dict


def _prepare_strategy_data(
    strategy,
    sliced_dict: dict[str, pd.DataFrame],
    *,
    progress_callback: Callable[..., None] | None,
    log_callback: Callable[[str], None] | None,
    use_process_pool: bool,
    max_workers: int | None,
    lookback_days: int | None,
) -> PrepareDataResult:
    skip_stats = SkipStats()
    skip_callback = skip_stats.callback()
    prepared_dict: dict[str, pd.DataFrame] | None = None
    fast_path_used = False
    fast_missing: set[str] = set()

    try:
        fast_ok, fast_missing = _is_fast_path_viable(sliced_dict)
    except Exception:
        fast_ok = False
        fast_missing = set()

    if fast_ok:
        try:
            prepared_dict = {
                sym: df.copy()
                for sym, df in sliced_dict.items()
                if df is not None and not getattr(df, "empty", True)
            }
            fast_path_used = True
            if log_callback:
                log_callback("⚡ 高速パス: 既存インジケーターを再利用します")
        except Exception:
            prepared_dict = None
            fast_path_used = False

    if prepared_dict is None:
        if fast_missing and log_callback:
            try:
                missing_list = ", ".join(sorted(fast_missing))
                log_callback(
                    "⚠️ 高速パスを利用できません。必須列不足: "
                    + (missing_list or "不明")
                    + "。再計算します"
                )
            except Exception:
                pass
        try:
            prepared_dict = strategy.prepare_data(
                sliced_dict,
                progress_callback=progress_callback,
                log_callback=log_callback,
                skip_callback=skip_callback,
                use_process_pool=use_process_pool,
                max_workers=max_workers,
                lookback_days=lookback_days,
            )
        except Exception as exc:
            system_name = str(getattr(strategy, "SYSTEM_NAME", ""))
            try:
                if log_callback:
                    log_callback(
                        f"⚠️ {system_name}: 前処理失敗のためフォールバックを再試行します ({exc})"
                    )
            except Exception:
                pass
            try:
                prepared_dict = strategy.prepare_data(
                    sliced_dict,
                    progress_callback=progress_callback,
                    log_callback=log_callback,
                    skip_callback=skip_callback,
                    use_process_pool=False,
                    max_workers=None,
                    lookback_days=lookback_days,
                    reuse_indicators=False,
                )
            except Exception as exc2:
                reason_code = "prepare_fail: 入力不備のため処理中断"
                try:
                    if log_callback:
                        log_callback(
                            f"🛑 {system_name}: {reason_code} ({exc2})"
                        )
                except Exception:
                    pass
                empty = _empty_today_signals_frame(reason_code)
                return PrepareDataResult(
                    {},
                    False,
                    fast_missing,
                    skip_stats,
                    empty,
                    reason_code,
                )

    prepared_dict = prepared_dict or {}
    normalized = _normalize_prepared_dict(prepared_dict)
    return PrepareDataResult(normalized, fast_path_used, fast_missing, skip_stats)


def _log_elapsed(
    log_callback: Callable[[str], None] | None,
    message: str,
    start_time: float,
) -> None:
    if not log_callback:
        return
    try:
        em, es = divmod(int(max(0, _t.time() - start_time)), 60)
        log_callback(f"{message}：経過 {em}分{es}秒")
    except Exception:
        pass


def _compute_filter_pass(
    system_name: str,
    prepared: dict[str, pd.DataFrame] | pd.DataFrame | None,
    today: pd.Timestamp,
    log_callback: Callable[[str], None] | None,
) -> int:
    try:
        prev_trading_day = get_latest_nyse_trading_day(
            pd.Timestamp(today) - pd.Timedelta(days=1)
        )

        def _last_filter_on_date(x: pd.DataFrame) -> bool:
            try:
                if getattr(x, "empty", True) or "filter" not in x.columns:
                    return False
                if "Date" in x.columns:
                    dt_vals = (
                        pd.to_datetime(x["Date"], errors="coerce")
                        .dt.normalize()
                        .to_numpy()
                    )
                    mask = dt_vals == prev_trading_day
                    rows = x.loc[mask]
                else:
                    idx_vals = (
                        pd.to_datetime(x.index, errors="coerce").normalize().to_numpy()
                    )
                    mask = idx_vals == prev_trading_day
                    rows = x.loc[mask]
                if len(rows) == 0:
                    rows = x.tail(1)
                if len(rows) == 0:
                    return False
                return bool(rows.iloc[-1].get("filter"))
            except Exception:
                return False

        if isinstance(prepared, dict):
            filter_pass = sum(int(_last_filter_on_date(df)) for df in prepared.values())
        elif isinstance(prepared, pd.DataFrame):
            filter_pass = int(_last_filter_on_date(prepared))
        else:
            filter_pass = 0
        try:
            if str(system_name).lower() == "system7":
                filter_pass = (
                    1 if (isinstance(prepared, dict) and ("SPY" in prepared)) else 0
                )
        except Exception:
            pass
    except Exception:
        filter_pass = 0
    if log_callback:
        try:
            log_callback(f"🧪 フィルターチェック完了：{filter_pass} 銘柄")
        except Exception:
            pass
    return int(filter_pass)


def _generate_candidates_for_system(
    strategy,
    prepared: dict[str, pd.DataFrame] | pd.DataFrame | None,
    *,
    system_name: str,
    market_df: pd.DataFrame | None,
    filter_pass: int,
    progress_callback: Callable[..., None] | None,
    log_callback: Callable[[str], None] | None,
) -> CandidateExtraction:
    gen_fn = strategy.generate_candidates  # type: ignore[attr-defined]
    params = inspect.signature(gen_fn).parameters
    needs_market_df = "market_df" in params
    market_df_local = market_df
    market_df_arg = market_df
    candidates_by_date: dict | None = None
    used_fast_path = False
    fast_zero_reason: str | None = None

    if str(system_name).lower() == "system4" and isinstance(prepared, dict):
        try:
            top_n_fast = int(get_settings(create_dirs=False).backtest.top_n_rank)
        except Exception:
            top_n_fast = 10

        spy_source: pd.DataFrame | None
        if isinstance(market_df_arg, pd.DataFrame) and not getattr(
            market_df_arg, "empty", False
        ):
            spy_source = market_df_arg.copy()
        else:
            maybe_spy = prepared.get("SPY") if isinstance(prepared, dict) else None
            spy_source = (
                maybe_spy.copy()
                if isinstance(maybe_spy, pd.DataFrame)
                and not getattr(maybe_spy, "empty", True)
                else None
            )
        try:
            spy_with_ind = get_spy_with_indicators(spy_source)
        except Exception:
            spy_with_ind = None
        if spy_with_ind is not None and not getattr(spy_with_ind, "empty", True):
            spy_norm = _normalize_daily_index(spy_with_ind)
        else:
            spy_norm = None

        fast_path_message: str | None = None
        gate_state = _make_spy_gate(spy_norm)
        if gate_state is None:
            fast_path_message = (
                "⚠️ System4 fast path: SPYデータ不足のため従来経路へフォールバックします"
            )
        elif gate_state is False:
            used_fast_path = True
            candidates_by_date = {}
            # fast path: SPY below SMA200 -> explicit zero reason
            fast_zero_reason = "fast_path: SPY below SMA200"
            if spy_norm is not None:
                market_df_local = spy_norm
                market_df_arg = spy_norm
            fast_path_message = (
                "🚫 System4 fast path: SPYがSMA200を下回るため候補は0件です"
            )
        else:
            fast_candidates = None
            if spy_norm is not None:
                fast_candidates = _collect_candidates_for_today(
                    prepared if isinstance(prepared, dict) else {},
                    spy_df=spy_norm,
                    top_n=top_n_fast,
                )
            if fast_candidates is None:
                fast_path_message = "⚠️ System4 fast path: 必須列不足のため従来経路へフォールバックします"
            else:
                used_fast_path = True
                candidates_by_date = fast_candidates
                market_df_local = spy_norm
                market_df_arg = spy_norm
                fast_path_message = (
                    "⚡ System4 fast path: SPYゲート通過の軽量抽出を適用しました"
                )

        if fast_path_message and log_callback:
            try:
                log_callback(fast_path_message)
            except Exception:
                pass
        # return early with zero reason if fast-path produced empty candidates
        if (
            used_fast_path
            and isinstance(candidates_by_date, dict)
            and not candidates_by_date
        ):
            try:
                return CandidateExtraction(
                    candidates_by_date,
                    market_df_local,
                    used_fast_path,
                    None,
                    fast_zero_reason,
                )
            except Exception:
                pass

    if not used_fast_path:
        if needs_market_df and system_name == "system4":
            needs_fallback = market_df_arg is None or getattr(
                market_df_arg, "empty", False
            )
            if needs_fallback and isinstance(prepared, dict):
                maybe_spy = prepared.get("SPY")
                if isinstance(maybe_spy, pd.DataFrame) and not getattr(
                    maybe_spy, "empty", True
                ):
                    market_df_arg = maybe_spy
                    needs_fallback = False
            if needs_fallback:
                try:
                    cached_spy = get_spy_with_indicators()
                except Exception:
                    cached_spy = None
                if cached_spy is not None and not getattr(cached_spy, "empty", True):
                    market_df_arg = cached_spy
                    market_df_local = cached_spy
                    if log_callback:
                        try:
                            log_callback(
                                "🛟 System4: SPYデータをキャッシュから補完しました"
                            )
                        except Exception:
                            pass
            if market_df_arg is None or getattr(market_df_arg, "empty", False):
                if log_callback:
                    try:
                        log_callback(
                            "⚠️ System4: SPYデータが見つからないため候補抽出をスキップします"
                        )
                    except Exception:
                        pass
                return CandidateExtraction(
                    None,
                    market_df_arg,
                    used_fast_path,
                    _empty_today_signals_frame(),
                )
        if log_callback:
            try:
                log_callback(f"🧩 セットアップチェック開始：{filter_pass} 銘柄")
            except Exception:
                pass
        t1 = _t.time()
        if needs_market_df and market_df_arg is not None:
            market_df_local = market_df_arg
            candidates_by_date, _ = gen_fn(
                prepared,
                market_df=market_df_arg,
                progress_callback=progress_callback,
                log_callback=log_callback,
            )
        elif needs_market_df:
            candidates_by_date, _ = gen_fn(
                prepared,
                progress_callback=progress_callback,
                log_callback=log_callback,
            )
        else:
            candidates_by_date, _ = gen_fn(
                prepared,
                progress_callback=progress_callback,
                log_callback=log_callback,
            )
        # if no candidates were produced, return with a reason for diagnostics
        if not candidates_by_date:
            zero_reason = "no_candidates_generated"
            if log_callback:
                try:
                    log_callback("⚠️ 候補が1件も生成されませんでした（戦略抽出側）")
                except Exception:
                    pass
            return CandidateExtraction(
                candidates_by_date,
                market_df_local,
                used_fast_path,
                None,
                zero_reason,
            )

        _log_elapsed(log_callback, "⏱️ セットアップ/候補抽出 完了", t1)
    else:
        if log_callback:
            try:
                log_callback("⏱️ セットアップ/候補抽出 完了（軽量経路）")
            except Exception:
                pass

    return CandidateExtraction(candidates_by_date, market_df_local, used_fast_path)


def _compute_setup_pass(
    system_name: str,
    prepared: dict[str, pd.DataFrame] | pd.DataFrame | None,
    market_df: pd.DataFrame | None,
    filter_pass: int,
    today: pd.Timestamp,
    log_callback: Callable[[str], None] | None,
) -> int:
    try:
        prev_trading_day = get_latest_nyse_trading_day(
            pd.Timestamp(today) - pd.Timedelta(days=1)
        )

        def _last_row(x: pd.DataFrame) -> pd.Series | None:
            try:
                if "Date" in x.columns:
                    dt_vals = (
                        pd.to_datetime(x["Date"], errors="coerce")
                        .dt.normalize()
                        .to_numpy()
                    )
                    mask = dt_vals == prev_trading_day
                    rows = x.loc[mask]
                else:
                    idx_vals = (
                        pd.to_datetime(x.index, errors="coerce").normalize().to_numpy()
                    )
                    mask = idx_vals == prev_trading_day
                    rows = x.loc[mask]
                if len(rows) == 0:
                    rows = x.tail(1)
                if len(rows) == 0:
                    return None
                return rows.iloc[-1]
            except Exception:
                return None

        if isinstance(prepared, dict):
            items = list(prepared.items())
        elif isinstance(prepared, pd.DataFrame):
            items = [("", prepared)]
        else:
            items = []
        latest_rows: dict[str, pd.Series] = {}
        for sym, df in items:
            if df is None or getattr(df, "empty", True):
                continue
            row = _last_row(df)
            if row is None:
                continue
            latest_rows[str(sym)] = row

        def _count_if(rows: list[pd.Series], fn: Callable[[pd.Series], bool]) -> int:
            cnt = 0
            for row in rows:
                try:
                    if fn(row):
                        cnt += 1
                except Exception:
                    continue
            return cnt

        rows_list = list(latest_rows.values())
        name = str(system_name).lower()
        setup_pass = 0

        if name == "system1":
            filtered_rows = [r for r in rows_list if bool(r.get("filter"))]

            def _sma_ok(row: pd.Series) -> bool:
                try:
                    return float(row.get("SMA25", 0)) > float(row.get("SMA50", 0))
                except Exception:
                    return False

            sma_pass = _count_if(filtered_rows, _sma_ok)
            spy_source = market_df if market_df is not None else None
            try:
                spy_df = get_spy_with_indicators(spy_source)
            except Exception:
                spy_df = None

            spy_gate: int | None
            try:
                if spy_df is None or getattr(spy_df, "empty", True):
                    spy_gate = None
                else:
                    last_row = spy_df.iloc[-1]
                    close_val = float(last_row.get("Close", float("nan")))
                    sma_val = float(last_row.get("SMA100", float("nan")))
                    if np.isnan(close_val) or np.isnan(sma_val):
                        spy_gate = None
                    else:
                        spy_gate = 1 if close_val > sma_val else 0
            except Exception:
                spy_gate = None

            setup_pass = sma_pass if spy_gate != 0 else 0

            if log_callback:
                spy_label = "-" if spy_gate is None else str(int(spy_gate))
                try:
                    log_callback(
                        "🧩 system1セットアップ集計: "
                        + f"フィルタ通過={filter_pass}, SPY>SMA100: {spy_label}, "
                        + f"SMA25>SMA50: {sma_pass}"
                    )
                except Exception:
                    pass
        elif name == "system2":

            def _rsi_ok(row: pd.Series) -> bool:
                try:
                    return float(row.get("RSI3", 0)) > 90
                except Exception:
                    return False

            def _two_up_ok(row: pd.Series) -> bool:
                return bool(row.get("TwoDayUp"))

            filtered_rows = [r for r in rows_list if bool(r.get("filter"))]
            rsi_pass = _count_if(filtered_rows, _rsi_ok)
            two_up_pass = _count_if(
                filtered_rows, lambda r: _rsi_ok(r) and _two_up_ok(r)
            )
            setup_pass = two_up_pass
            if log_callback:
                try:
                    log_callback(
                        "🧩 system2セットアップ集計: "
                        + f"フィルタ通過={filter_pass}, RSI3>90: {rsi_pass}, "
                        + f"TwoDayUp: {two_up_pass}"
                    )
                except Exception:
                    pass
        elif name == "system3":
            filtered_rows = [r for r in rows_list if bool(r.get("filter"))]

            def _close_ok(row: pd.Series) -> bool:
                try:
                    return float(row.get("Close", 0)) > float(row.get("SMA150", 0))
                except Exception:
                    return False

            def _drop_ok(row: pd.Series) -> bool:
                try:
                    return float(row.get("Drop3D", 0)) >= 0.125
                except Exception:
                    return False

            close_pass = _count_if(filtered_rows, _close_ok)
            drop_pass = _count_if(filtered_rows, lambda r: _close_ok(r) and _drop_ok(r))
            setup_pass = drop_pass
            if log_callback:
                try:
                    log_callback(
                        "🧩 system3セットアップ集計: "
                        + f"フィルタ通過={filter_pass}, Close>SMA150: {close_pass}, "
                        + f"3日下落率≧12.5%: {drop_pass}"
                    )
                except Exception:
                    pass
        elif name == "system4":

            def _above_sma(row: pd.Series) -> bool:
                try:
                    return bool(row.get("filter")) and (
                        float(row.get("Close", 0)) > float(row.get("SMA200", 0))
                    )
                except Exception:
                    return False

            above_sma = _count_if(rows_list, _above_sma)
            setup_pass = above_sma
            if log_callback:
                try:
                    log_callback(
                        "🧩 system4セットアップ集計: "
                        + f"フィルタ通過={filter_pass}, Close>SMA200: {above_sma}"
                    )
                except Exception:
                    pass
        elif name == "system5":
            threshold_label = format_atr_pct_threshold_label()
            s5_total = len(rows_list)
            s5_av = 0
            s5_dv = 0
            s5_atr = 0
            for row in rows_list:
                try:
                    av_val = row.get("AvgVolume50")
                    if av_val is None or pd.isna(av_val) or float(av_val) <= 500_000:
                        continue
                    s5_av += 1
                    dv_val = row.get("DollarVolume50")
                    if dv_val is None or pd.isna(dv_val) or float(dv_val) <= 2_500_000:
                        continue
                    s5_dv += 1
                    atr_pct_val = row.get("ATR_Pct")
                    if (
                        atr_pct_val is not None
                        and not pd.isna(atr_pct_val)
                        and float(atr_pct_val) > DEFAULT_ATR_PCT_THRESHOLD
                    ):
                        s5_atr += 1
                except Exception:
                    continue
            if log_callback:
                try:
                    log_callback(
                        "🧪 system5集計: "
                        + f"対象={s5_total}, AvgVol50>500k: {s5_av}, "
                        + f"DV50>2.5M: {s5_dv}, {threshold_label}: {s5_atr}"
                    )
                except Exception:
                    pass

            def _price_ok(row: pd.Series) -> bool:
                try:
                    return bool(row.get("filter")) and (
                        float(row.get("Close", 0))
                        > float(row.get("SMA100", 0)) + float(row.get("ATR10", 0))
                    )
                except Exception:
                    return False

            def _adx_ok(row: pd.Series) -> bool:
                try:
                    return float(row.get("ADX7", 0)) > 55
                except Exception:
                    return False

            def _rsi_ok(row: pd.Series) -> bool:
                try:
                    return float(row.get("RSI3", 100)) < 50
                except Exception:
                    return False

            price_pass = _count_if(rows_list, _price_ok)
            adx_pass = _count_if(rows_list, lambda r: _price_ok(r) and _adx_ok(r))
            rsi_pass = _count_if(
                rows_list, lambda r: _price_ok(r) and _adx_ok(r) and _rsi_ok(r)
            )
            setup_pass = rsi_pass
            if log_callback:
                try:
                    log_callback(
                        "🧩 system5セットアップ集計: "
                        + f"フィルタ通過={filter_pass}, Close>SMA100+ATR10: {price_pass}, "
                        + f"ADX7>55: {adx_pass}, RSI3<50: {rsi_pass}"
                    )
                except Exception:
                    pass
        elif name == "system6":
            filtered_rows = [r for r in rows_list if bool(r.get("filter"))]

            def _ret_ok(row: pd.Series) -> bool:
                try:
                    return float(row.get("Return6D", 0)) > 0.20
                except Exception:
                    return False

            def _up_two(row: pd.Series) -> bool:
                return bool(row.get("UpTwoDays"))

            ret_pass = _count_if(filtered_rows, _ret_ok)
            up_pass = _count_if(filtered_rows, lambda r: _ret_ok(r) and _up_two(r))
            setup_pass = up_pass
            if log_callback:
                try:
                    msg = (
                        "🧩 system6セットアップ集計: "
                        f"フィルタ通過={filter_pass}, "
                        f"Return6D>20%: {ret_pass}, "
                        f"UpTwoDays: {up_pass}"
                    )
                    log_callback(msg)
                except Exception:
                    pass
        elif name == "system7":
            spy_present = 1 if "SPY" in latest_rows else 0
            setup_pass = spy_present
            if log_callback:
                try:
                    msg = f"🧩 system7セットアップ集計: SPY存在={spy_present}"
                    if spy_present:
                        try:
                            val = latest_rows.get("SPY", pd.Series())
                            if isinstance(val, pd.Series):
                                setup_flag = bool(val.get("setup", 0))
                            else:
                                setup_flag = False
                            msg += f", setup={int(setup_flag)}"
                        except Exception:
                            pass
                    log_callback(msg)
                except Exception:
                    pass
        else:
            setup_pass = _count_if(
                rows_list,
                lambda r: bool(r.get("setup")) if "setup" in r else False,
            )

        try:
            setup_pass = int(setup_pass)
        except Exception:
            setup_pass = 0
    except Exception:
        setup_pass = 0
    return int(setup_pass)


def _select_candidate_date(
    candidates_by_date: dict | None,
    today: pd.Timestamp,
    filter_pass: int,
    setup_pass: int,
    log_callback: Callable[[str], None] | None,
) -> CandidateSelection:
    try:
        key_map: dict[pd.Timestamp, object] = {}
        cand_keys = list((candidates_by_date or {}).keys())
        for _k in cand_keys:
            try:
                _raw_ts = pd.to_datetime(_k, errors="coerce")
                if pd.isna(_raw_ts):
                    continue
                _ts = pd.Timestamp(_raw_ts)
                if getattr(_ts, "tzinfo", None) is not None:
                    try:
                        _ts = _ts.tz_localize(None)
                    except Exception:
                        try:
                            _ts = pd.Timestamp(_ts.to_pydatetime().replace(tzinfo=None))
                        except Exception:
                            continue
                _ts = _ts.normalize()
                if _ts not in key_map:
                    key_map[_ts] = _k
            except Exception:
                continue
        candidate_dates = sorted(list(key_map.keys()), reverse=True)
    except Exception:
        key_map = {}
        candidate_dates = []

    target_date: pd.Timestamp | None = None
    fallback_reason: str | None = None

    def _collect_recent_days(
        anchor: pd.Timestamp | None, count: int
    ) -> list[pd.Timestamp]:
        if anchor is None or count <= 0:
            return []
        out: list[pd.Timestamp] = []
        seen: set[pd.Timestamp] = set()
        cur = pd.Timestamp(anchor).normalize()
        while len(out) < count:
            if cur in seen:
                break
            out.append(cur)
            seen.add(cur)
            prev = get_latest_nyse_trading_day(cur - pd.Timedelta(days=1))
            prev = pd.Timestamp(prev).normalize()
            if prev >= cur:
                break
            cur = prev
        return out

    try:
        primary_days = _collect_recent_days(today, 3)
        for dt in primary_days:
            if dt in candidate_dates:
                target_date = dt
                break

        if target_date is None:
            try:
                settings = get_settings(create_dirs=False)
                cfg = getattr(settings, "cache", None)
                rolling_cfg = getattr(cfg, "rolling", None)
                max_stale = getattr(
                    rolling_cfg,
                    "max_staleness_days",
                    getattr(rolling_cfg, "max_stale_days", 2),
                )
                stale_limit = int(max_stale)
            except Exception:
                stale_limit = 2
            fallback_window = max(len(primary_days), stale_limit + 3)
            extended_days = _collect_recent_days(today, fallback_window)
            for dt in extended_days:
                if dt in candidate_dates:
                    target_date = dt
                    if dt not in primary_days:
                        fallback_reason = "recent"
                    break

        if target_date is None and candidate_dates:
            today_norm = pd.Timestamp(today).normalize()
            past_candidates = [d for d in candidate_dates if d <= today_norm]
            if past_candidates:
                target_date = max(past_candidates)
                if fallback_reason is None:
                    fallback_reason = "latest_past"
            else:
                target_date = max(candidate_dates)
                if fallback_reason is None:
                    fallback_reason = "latest_any"

        if log_callback:
            try:
                _cands_str = ", ".join([str(d.date()) for d in candidate_dates[:5]])
                _search_str = ", ".join([str(d.date()) for d in primary_days])
                _chosen = str(target_date.date()) if target_date is not None else "None"
                fallback_msg = ""
                if fallback_reason:
                    fallback_labels = {
                        "recent": "直近営業日に候補が無いため過去日を採用",
                        "latest_past": "探索範囲外の最新過去日を採用",
                        "latest_any": "未来日しか存在しないため候補最終日を採用",
                    }
                    label = fallback_labels.get(fallback_reason, fallback_reason)
                    fallback_msg = f" | フォールバック: {label}"
                log_callback(
                    "🗓️ 候補日探索: "
                    f"{_cands_str} | 探索日: {_search_str} | 採用: {_chosen}{fallback_msg}"
                )
            except Exception:
                pass
    except Exception:
        target_date = None
        fallback_reason = None

    try:
        if target_date is not None and target_date in key_map:
            orig_key = key_map[target_date]
            total_candidates_today = len(
                (candidates_by_date or {}).get(orig_key, []) or []
            )
        else:
            total_candidates_today = 0
    except Exception:
        total_candidates_today = 0

    try:
        max_pos_ui = int(get_settings(create_dirs=False).risk.max_positions)
    except Exception:
        max_pos_ui = 10
    if total_candidates_today and max_pos_ui > 0:
        total_candidates_today = min(int(total_candidates_today), int(max_pos_ui))

    if log_callback:
        try:
            log_callback(f"🧩 セットアップチェック完了：{setup_pass} 銘柄")
            log_callback(f"🧮 候補生成済み（セットアップ通過）：{setup_pass} 銘柄")
            log_callback(
                f"🧮 TRDlist相当（直近営業日時点の候補数）：{total_candidates_today} 銘柄"
            )
        except Exception:
            pass

    # If there are no candidates, try to infer a reason for diagnostics
    zero_reason: str | None = None
    try:
        if int(total_candidates_today) == 0:
            if not key_map:
                zero_reason = "no_candidate_dates"
            else:
                zero_reason = "no_candidates_on_target_date"
    except Exception:
        zero_reason = None

    if int(total_candidates_today) == 0 and log_callback:
        try:
            if zero_reason:
                log_callback(f"ℹ️ 候補0件理由: {zero_reason}")
        except Exception:
            pass

    return CandidateSelection(
        key_map,
        target_date,
        fallback_reason,
        int(total_candidates_today),
        zero_reason,
    )


def _build_today_signals_dataframe(
    strategy,
    prepared: dict[str, pd.DataFrame],
    candidates_by_date: dict | None,
    selection: CandidateSelection,
    system_name: str,
    side: str,
    signal_type: str,
    log_callback: Callable[[str], None] | None,
) -> tuple[pd.DataFrame, int]:
    if not candidates_by_date:
        return _empty_today_signals_frame(), 0

    target_date = selection.target_date
    key_map = selection.key_map
    if target_date is not None and target_date in key_map:
        orig_key = key_map[target_date]
        today_candidates = cast(list[dict], candidates_by_date.get(orig_key, []))
    else:
        today_candidates = cast(list[dict], [])
    if not today_candidates:
        return _empty_today_signals_frame(), 0

    rows: list[TodaySignal] = []
    for c in today_candidates:
        sym = c.get("symbol")
        if not sym or sym not in prepared:
            continue
        df = prepared[sym]
        comp = _compute_entry_stop(strategy, df, c, side)
        if not comp:
            continue
        entry, stop = comp
        skey, sval, _asc = _score_from_candidate(system_name, c)

        try:
            if (system_name == "system1") and (
                skey is None or str(skey).upper() != "ROC200"
            ):
                skey = "ROC200"
        except Exception:
            pass

        signal_date_ts: pd.Timestamp | None = None
        try:
            if "Date" in c and c.get("Date") is not None:
                date_arg: Any = c.get("Date")
                tmp = pd.to_datetime(date_arg, errors="coerce")
                if not pd.isna(tmp):
                    signal_date_ts = pd.Timestamp(tmp).normalize()
        except Exception:
            pass
        if signal_date_ts is None:
            try:
                ed_arg: Any = c.get("entry_date")
                ed = pd.to_datetime(ed_arg, errors="coerce")
                if isinstance(ed, pd.Timestamp) and not pd.isna(ed):
                    signal_date_ts = get_latest_nyse_trading_day(
                        pd.Timestamp(ed).normalize() - pd.Timedelta(days=1)
                    )
            except Exception:
                signal_date_ts = None

        rank_val: int | None = None
        total_for_rank: int = 0
        if skey is not None:
            if sval is None or (isinstance(sval, float) and pd.isna(sval)):
                try:
                    if signal_date_ts is not None:
                        xdf = prepared[sym]
                        if "Date" in xdf.columns:
                            dt_vals = (
                                pd.to_datetime(xdf["Date"], errors="coerce")
                                .dt.normalize()
                                .to_numpy()
                            )
                        else:
                            dt_vals = (
                                pd.to_datetime(xdf.index, errors="coerce")
                                .normalize()
                                .to_numpy()
                            )
                        mask = dt_vals == signal_date_ts
                        row = xdf.loc[mask]
                        if not row.empty and skey in row.columns:
                            _v = row.iloc[0][skey]
                            if _v is not None and not pd.isna(_v):
                                sval = float(_v)
                except Exception:
                    pass
            if (system_name == "system1") and (
                sval is None or (isinstance(sval, float) and pd.isna(sval))
            ):
                try:
                    if skey in prepared[sym].columns:
                        _v = pd.Series(prepared[sym][skey]).dropna().tail(1).iloc[0]
                        sval = float(_v)
                except Exception:
                    pass

            try:
                if signal_date_ts is not None:
                    vals: list[tuple[str, float]] = []
                    for psym, pdf in prepared.items():
                        try:
                            if "Date" in pdf.columns:
                                dt_vals = (
                                    pd.to_datetime(pdf["Date"], errors="coerce")
                                    .dt.normalize()
                                    .to_numpy()
                                )
                            else:
                                dt_vals = (
                                    pd.to_datetime(pdf.index, errors="coerce")
                                    .normalize()
                                    .to_numpy()
                                )
                            mask = dt_vals == signal_date_ts
                            row = pdf.loc[mask]
                            if row.empty or skey not in row.columns:
                                continue
                            val = row.iloc[0][skey]
                            if val is None or pd.isna(val):
                                continue
                            vals.append((psym, float(val)))
                        except Exception:
                            continue
                    if vals:
                        vals_sorted = sorted(vals, key=lambda x: x[1], reverse=not _asc)
                        total_for_rank = len(vals_sorted)
                        ranks = {
                            name: idx + 1 for idx, (name, _) in enumerate(vals_sorted)
                        }
                        rank_val = ranks.get(str(sym))
                        if sval is None and rank_val is not None:
                            sval = float(vals_sorted[rank_val - 1][1])
            except Exception:
                pass

        reason_parts: list[str] = []
        if system_name == "system1":
            if rank_val is not None and int(rank_val) <= 10:
                reason_parts = [f"ROC200が{int(rank_val)}位のため"]
            else:
                reason_parts = ["ROC200が上位のため"]
        elif system_name == "system2":
            if rank_val is not None and skey is not None:
                reason_parts = [f"{_label_for_score_key(skey)}が{rank_val}位のため"]
            else:
                reason_parts = ["モメンタムが強く過熱のため"]
        elif system_name == "system3":
            if rank_val is not None and skey is not None:
                reason_parts = [f"{_label_for_score_key(skey)}が{rank_val}位のため"]
            else:
                reason_parts = ["ボラティリティが高く条件一致のため"]
        elif system_name == "system4":
            if rank_val is not None:
                reason_parts = [f"RSI4が{rank_val}位（低水準）のため"]
            else:
                reason_parts = ["SPY上昇局面の押し目候補のため"]
        elif system_name == "system5":
            if rank_val is not None and skey is not None:
                reason_parts = [f"{_label_for_score_key(skey)}が{rank_val}位のため"]
            else:
                reason_parts = ["ADXが強く、反発期待のため"]
        elif system_name == "system6":
            if rank_val is not None:
                reason_parts = [f"過去6日騰落率が{rank_val}位のため"]
            else:
                reason_parts = ["短期下落トレンド（ショート）条件一致のため"]
        elif system_name == "system7":
            reason_parts = ["SPYが50日安値を更新したため（ヘッジ）"]
        else:
            if skey is not None and rank_val is not None:
                if rank_val <= 10:
                    reason_parts = [f"{_label_for_score_key(skey)}が{rank_val}位のため"]
                else:
                    reason_parts = [f"rank={rank_val}/{total_for_rank}"]
            elif skey is not None:
                try:
                    if sval is not None and not (
                        isinstance(sval, float) and pd.isna(sval)
                    ):
                        reason_parts.append("スコア条件を満たしたため")
                except Exception:
                    reason_parts.append("スコア条件を満たしたため")

        if not reason_parts:
            reason_parts.append("条件一致のため")

        reason_text = "; ".join(reason_parts)

        try:
            _ed_raw: Any = c.get("entry_date")
            _ed = pd.Timestamp(_ed_raw) if _ed_raw is not None else None
            if _ed is None or pd.isna(_ed):
                continue
            entry_date_norm = pd.Timestamp(_ed).normalize()
        except Exception:
            continue

        rows.append(
            TodaySignal(
                symbol=str(sym),
                system=system_name,
                side=side,
                signal_type=signal_type,
                entry_date=entry_date_norm,
                entry_price=float(entry),
                stop_price=float(stop),
                score_key=skey,
                score=(
                    None
                    if sval is None or (isinstance(sval, float) and pd.isna(sval))
                    else float(sval)
                ),
                reason=reason_text,
            )
        )

    if not rows:
        return _empty_today_signals_frame(), 0

    out = pd.DataFrame([r.__dict__ for r in rows])

    try:
        max_pos = int(get_settings(create_dirs=False).risk.max_positions)
    except Exception:
        max_pos = 10
    if max_pos > 0 and not out.empty:

        def _sort_val(row: pd.Series) -> float:
            sc = row.get("score")
            sk = row.get("score_key")
            if sc is None or (isinstance(sc, float) and pd.isna(sc)):
                return float("inf")
            return float(sc) if _asc_by_score_key(sk) else -float(sc)

        out["_sort_val"] = out.apply(_sort_val, axis=1)
        out = (
            out.sort_values("_sort_val")
            .head(max_pos)
            .drop(columns=["_sort_val"])
            .reset_index(drop=True)
        )
    final_count = len(out)

    if log_callback:
        try:
            log_callback(f"🧮 トレード候補選定完了（当日）：{final_count} 銘柄")
        except Exception:
            pass

    return out, final_count


def _missing_fast_path_columns(data_dict: dict[str, pd.DataFrame]) -> set[str]:
    """高速経路に必要な列が揃っているかを判定し、不足集合を返す。"""

    if not isinstance(data_dict, dict) or not data_dict:
        return set(_FAST_PATH_REQUIRED_COLUMNS)

    missing: set[str] = set()
    has_valid_frame = False
    for df in data_dict.values():
        if df is None or getattr(df, "empty", True):
            continue
        has_valid_frame = True
        try:
            cols = {str(c).strip().lower() for c in df.columns}
        except Exception:
            missing.update(_FAST_PATH_REQUIRED_COLUMNS)
            continue
        for col in _FAST_PATH_REQUIRED_COLUMNS:
            if col not in cols:
                missing.add(col)

    if not has_valid_frame:
        return set(_FAST_PATH_REQUIRED_COLUMNS)
    return missing


def _is_fast_path_viable(data_dict: dict[str, pd.DataFrame]) -> tuple[bool, set[str]]:
    """高速経路で candidate 抽出が可能か判定し、(bool, 不足列) を返す。"""

    missing = _missing_fast_path_columns(data_dict)
    return len(missing) == 0, missing


def _empty_today_signals_frame(reason: str | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(columns=TODAY_SIGNAL_COLUMNS)
    if reason is not None:
        try:
            frame.attrs["zero_reason"] = str(reason)
        except Exception:
            frame.attrs["zero_reason"] = "unknown"
    return frame


def _normalize_daily_index(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if "Date" in x.columns:
        idx = pd.to_datetime(x["Date"], errors="coerce").dt.normalize()
        x.index = pd.Index(idx)
    else:
        x.index = pd.to_datetime(x.index, errors="coerce").normalize()
    x = x[~x.index.isna()]
    try:
        x = x.sort_index()
    except Exception:
        pass
    try:
        if getattr(x.index, "has_duplicates", False):
            x = x[~x.index.duplicated(keep="last")]
    except Exception:
        pass
    return x


def _make_spy_gate(spy_df: pd.DataFrame | None) -> bool | None:
    if spy_df is None or getattr(spy_df, "empty", True):
        return None
    try:
        last_row = spy_df.iloc[-1]
    except Exception:
        return None
    try:
        close_val = pd.to_numeric(
            pd.Series([last_row.get("Close")]), errors="coerce"
        ).iloc[0]
        sma_val = pd.to_numeric(
            pd.Series([last_row.get("SMA200")]), errors="coerce"
        ).iloc[0]
    except Exception:
        return None
    if pd.isna(close_val) or pd.isna(sma_val):
        return None
    try:
        return bool(float(close_val) > float(sma_val))
    except Exception:
        return None


def _collect_candidates_for_today(
    prepared: dict[str, pd.DataFrame],
    *,
    spy_df: pd.DataFrame,
    top_n: int,
) -> dict[pd.Timestamp, list[dict]] | None:
    try:
        spy_norm = _normalize_daily_index(spy_df)
    except Exception:
        return None
    if spy_norm.empty or "Close" not in spy_norm.columns:
        return None
    if "SMA200" not in spy_norm.columns:
        return None
    spy_norm = spy_norm.copy()
    spy_close = pd.to_numeric(spy_norm["Close"], errors="coerce")
    spy_sma = pd.to_numeric(spy_norm["SMA200"], errors="coerce")
    spy_norm["spy_filter"] = (spy_close > spy_sma).astype(int)
    spy_filter = spy_norm["spy_filter"]

    candidates: dict[pd.Timestamp, list[dict]] = {}
    required_cols = {
        "Close",
        "DollarVolume50",
        "HV50",
        "SMA200",
        "RSI4",
        "ATR40",
    }

    for sym, df in prepared.items():
        if str(sym).upper() == "SPY":
            continue
        if not isinstance(df, pd.DataFrame) or getattr(df, "empty", True):
            continue
        try:
            norm = _normalize_daily_index(df)
        except Exception:
            return None
        if required_cols - set(norm.columns):
            return None

        close = pd.to_numeric(norm["Close"], errors="coerce")
        sma200 = pd.to_numeric(norm["SMA200"], errors="coerce")
        dv50 = pd.to_numeric(norm["DollarVolume50"], errors="coerce")
        hv50 = pd.to_numeric(norm["HV50"], errors="coerce")

        setup_mask = (dv50 > 100_000_000) & hv50.between(10, 40) & (close > sma200)
        setup_mask = setup_mask.fillna(False)
        if not setup_mask.any():
            continue

        try:
            last_close_series = close.dropna()
            last_close = (
                float(last_close_series.iloc[-1])
                if not last_close_series.empty
                else None
            )
        except Exception:
            last_close = None

        for ts in norm.index[setup_mask]:
            ts_norm = pd.Timestamp(ts).normalize()
            if ts_norm not in spy_filter.index:
                continue
            try:
                gate_val = spy_filter.loc[ts_norm]
                if isinstance(gate_val, pd.Series):
                    gate_val = gate_val.iloc[-1]
            except Exception:
                continue
            if pd.isna(gate_val) or int(gate_val) == 0:
                continue
            entry_date = resolve_signal_entry_date(ts_norm)
            if pd.isna(entry_date):
                continue
            row = norm.loc[ts_norm]
            candidate = {
                "symbol": sym,
                "entry_date": entry_date,
                "RSI4": row.get("RSI4"),
                "ATR40": row.get("ATR40"),
            }
            if last_close is not None and not pd.isna(last_close):
                candidate["entry_price"] = last_close
            candidates.setdefault(entry_date, []).append(candidate)

    if not candidates:
        return {}

    limited: dict[pd.Timestamp, list[dict]] = {}
    try:
        limit_n = max(0, int(top_n))
    except Exception:
        limit_n = 0
    for dt, rows in candidates.items():
        sorted_rows = sorted(
            rows,
            key=lambda c: (
                (
                    float("inf")
                    if c.get("RSI4") is None or pd.isna(c.get("RSI4"))
                    else float(c["RSI4"])
                ),
                str(c.get("symbol") or ""),
            ),
        )
        limited[dt] = sorted_rows[:limit_n]
    return limited


def _infer_side(system_name: str) -> str:
    name = (system_name or "").lower()
    if name in SHORT_SYSTEMS:
        return "short"
    return "long"


def _score_from_candidate(
    system_name: str, candidate: dict
) -> tuple[str | None, float | None, bool]:
    """
    候補レコードからスコア項目と並び順（昇順か）を推定して返す。
    戻り値: (score_key, score_value, asc)
    """
    name = (system_name or "").lower()
    # System7 は SPY 専用ヘッジ。ATR50 はストップ計算用のため、
    # スコア/理由には使用しない（スコア欄は空にする）。
    if name == "system7":
        return None, None, False
    # システム別の代表スコア
    key_order: list[tuple[list[str], bool]] = [
        (["ROC200"], False),  # s1: 大きいほど良い
        (["ADX7"], False),  # s2,s5: 大きいほど良い
        (["Drop3D"], False),  # s3: 大きいほど良い（下落率）
        (["RSI4"], True),  # s4: 小さいほど良い
        (["Return6D"], False),  # s6: 大きいほど良い
        (["ATR50"], False),  # s7: 参考
    ]
    # system 固有優先順位
    if name == "system4":
        key_order = [(["RSI4"], True), (["ATR40"], True)] + key_order
    elif name == "system2":
        key_order = [(["ADX7"], False), (["RSI3"], False)] + key_order
    elif name == "system5":
        key_order = [(["ADX7"], False), (["ATR10"], True)] + key_order
    elif name == "system6":
        key_order = [(["Return6D"], False), (["ATR10"], True)] + key_order

    for keys, asc in key_order:
        for k in keys:
            if k in candidate:
                v = candidate.get(k)
                if v is None:
                    return k, None, asc
                if isinstance(v, (int, float, str)):
                    try:
                        return k, float(v), asc
                    except Exception:
                        return k, None, asc
                else:
                    return k, None, asc
    # 見つからない場合
    return None, None, False


def _label_for_score_key(key: str | None) -> str:
    """スコアキーの日本語ラベルを返す（既知のもののみ簡潔表示）。"""
    if key is None:
        return "スコア"
    k = str(key).upper()
    mapping = {
        "ROC200": "ROC200",
        "ADX7": "ADX",
        "RSI4": "RSI4",
        "RSI3": "RSI3",
        "DROP3D": "3日下落率",
        "RETURN6D": "過去6日騰落率",
        "ATR10": "ATR10",
        "ATR20": "ATR20",
        "ATR40": "ATR40",
        "ATR50": "ATR50",
    }
    return mapping.get(k, k)


def _asc_by_score_key(score_key: str | None) -> bool:
    """スコアキーごとの昇順/降順を判定。"""
    return bool(score_key and score_key.upper() in {"RSI4"})


def _pick_atr_col(df: pd.DataFrame) -> str | None:
    for col in ("ATR20", "ATR10", "ATR40", "ATR50", "ATR14"):
        if col in df.columns:
            return col
    return None


def _compute_entry_stop(
    strategy, df: pd.DataFrame, candidate: dict, side: str
) -> tuple[float, float] | None:
    # strategy 独自の compute_entry があれば優先
    try:
        _fn = strategy.compute_entry  # type: ignore[attr-defined]
    except Exception:
        _fn = None
    if callable(_fn):
        try:
            res = _fn(df, candidate, 0.0)
            if res and isinstance(res, tuple) and len(res) == 2:
                entry, stop = float(res[0]), float(res[1])
                if entry > 0 and (
                    (side == "short" and stop > entry)
                    or (side == "long" and entry > stop)
                ):
                    return round(entry, 4), round(stop, 4)
        except Exception:
            pass

    # フォールバック: 当日始値 ± 3*ATR
    def _as_positive(value: Any) -> float | None:
        try:
            val = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(val) or val <= 0:
            return None
        return val

    def _latest_positive(series: pd.Series | None) -> float | None:
        if series is None:
            return None
        try:
            numeric = pd.to_numeric(series, errors="coerce").dropna()
        except Exception:
            return None
        numeric = numeric[numeric > 0]
        if numeric.empty:
            return None
        val = float(numeric.iloc[-1])
        if not math.isfinite(val) or val <= 0:
            return None
        return val

    def _infer_atr_window(name: str | None, default: int = 14) -> int:
        if not name:
            return default
        digits = "".join(ch for ch in str(name) if ch.isdigit())
        if not digits:
            return default
        try:
            window = int(digits)
        except ValueError:
            return default
        return max(1, window)

    def _fallback_atr(frame: pd.DataFrame, window: int) -> float | None:
        required = {"High", "Low", "Close"}
        if frame is None or frame.empty:
            return None
        if any(col not in frame.columns for col in required):
            return None
        try:
            high = pd.to_numeric(frame["High"], errors="coerce")
            low = pd.to_numeric(frame["Low"], errors="coerce")
            close = pd.to_numeric(frame["Close"], errors="coerce")
        except Exception:
            return None
        tr = pd.concat(
            [
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        if tr.empty:
            return None
        window = max(1, int(window or 14))
        min_periods = min(window, max(2, min(5, len(tr))))
        atr_series = tr.rolling(window, min_periods=min_periods).mean()
        return _latest_positive(atr_series)

    entry_ts = None
    if isinstance(candidate, dict):
        try:
            raw = candidate.get("entry_date")
            if raw is not None:
                tmp = pd.to_datetime(raw, errors="coerce")
                if not pd.isna(tmp):
                    entry_ts = tmp
        except Exception:
            entry_ts = None
    else:
        try:
            tmp = pd.to_datetime(candidate, errors="coerce")
            if not pd.isna(tmp):
                entry_ts = tmp
        except Exception:
            entry_ts = None

    entry_idx = -1
    if entry_ts is not None:
        try:
            idxer = df.index.get_indexer([entry_ts])
            entry_idx = int(idxer[0]) if len(idxer) else -1
        except Exception:
            entry_idx = -1

    atr_candidates = [
        col
        for col in getattr(df, "columns", [])
        if isinstance(col, str) and col.upper().startswith("ATR")
    ]
    atr_column = atr_candidates[0] if atr_candidates else None
    atr_window = _infer_atr_window(atr_column)

    entry = None
    atr_val = None
    if 0 <= entry_idx < len(df):
        row = df.iloc[entry_idx]
        try:
            entry = _as_positive(row.get("Open"))
        except Exception:
            entry = None
        if entry_idx > 0:
            prev_row = df.iloc[max(entry_idx - 1, 0)]
            for col in atr_candidates:
                try:
                    candidate_val = _as_positive(prev_row.get(col))
                except Exception:
                    candidate_val = None
                if candidate_val is not None:
                    atr_val = candidate_val
                    atr_column = col
                    atr_window = _infer_atr_window(col, atr_window)
                    break

    if isinstance(candidate, dict):
        if entry is None:
            for key in ("entry_price", "open", "close", "price", "last_price"):
                if key in candidate:
                    entry_candidate = _as_positive(candidate.get(key))
                    if entry_candidate is not None:
                        entry = entry_candidate
                        break
        if atr_val is None:
            for key, value in candidate.items():
                if not isinstance(key, str):
                    continue
                if "atr" not in key.lower():
                    continue
                atr_candidate = _as_positive(value)
                if atr_candidate is not None:
                    atr_val = atr_candidate
                    atr_window = _infer_atr_window(key, atr_window)
                    break

    if entry is None:
        entry = _latest_positive(df.get("Close"))
    if entry is None:
        entry = _latest_positive(df.get("Open"))

    if atr_val is None and atr_column:
        atr_val = _latest_positive(df.get(atr_column))
    if atr_val is None:
        atr_val = _fallback_atr(df, atr_window)

    if entry is None or atr_val is None:
        return None

    mult = 3.0
    stop = entry - mult * atr_val if side == "long" else entry + mult * atr_val
    if (side == "long" and stop >= entry) or (side == "short" and stop <= entry):
        return None

    return round(entry, 4), round(stop, 4)


def get_today_signals_for_strategy(
    strategy,
    raw_data_dict: dict[str, pd.DataFrame],
    *,
    market_df: pd.DataFrame | None = None,
    today: pd.Timestamp | None = None,
    progress_callback: Callable[..., None] | None = None,
    log_callback: Callable[[str], None] | None = None,
    stage_progress: (
        Callable[[int, int | None, int | None, int | None, int | None], None] | None
    ) = None,
    use_process_pool: bool = False,
    max_workers: int | None = None,
    lookback_days: int | None = None,
) -> pd.DataFrame:
    """
    各 Strategy の prepare_data / generate_candidates を流用し、
    最新営業日の候補のみを DataFrame で返す。

    戻り値カラム:
        - symbol, system, side, signal_type,
          entry_date, entry_price, stop_price,
          score_key, score
    """
    try:
        system_name = str(strategy.SYSTEM_NAME).lower()  # type: ignore[attr-defined]
    except Exception:
        system_name = ""
    side = _infer_side(system_name)
    signal_type = "sell" if side == "short" else "buy"

    # CLI実行時などでlog_callback未指定の場合は、標準出力へ出すデフォルトを適用
    if log_callback is None:
        log_callback = _default_cli_log

    today_ts = _normalize_today(today)

    total_symbols = len(raw_data_dict)
    if log_callback:
        try:
            log_callback(f"🧪 フィルターチェック開始：{total_symbols} 銘柄")
        except Exception:
            pass
    try:
        if stage_progress:
            stage_progress(0, total_symbols, None, None, None)
    except Exception:
        pass

    t0 = _t.time()
    sliced_dict = _slice_data_for_lookback(raw_data_dict, lookback_days)

    prepare_result = _prepare_strategy_data(
        strategy,
        sliced_dict,
        progress_callback=progress_callback,
        log_callback=log_callback,
        use_process_pool=use_process_pool,
        max_workers=max_workers,
        lookback_days=lookback_days,
    )
    if prepare_result.early_exit_frame is not None:
        if log_callback and prepare_result.early_exit_reason:
            try:
                log_callback(
                    f"🛈 中断理由コード: {prepare_result.early_exit_reason}"
                )
            except Exception:
                pass
        return prepare_result.early_exit_frame

    prepared = prepare_result.prepared

    _log_elapsed(log_callback, "⏱️ フィルター/前処理完了", t0)
    prepare_result.skip_stats.log_summary(system_name, log_callback)

    filter_pass = _compute_filter_pass(system_name, prepared, today_ts, log_callback)
    try:
        if stage_progress:
            stage_progress(25, filter_pass, None, None, None)
    except Exception:
        pass

    candidates = _generate_candidates_for_system(
        strategy,
        prepared,
        system_name=system_name,
        market_df=market_df,
        filter_pass=filter_pass,
        progress_callback=progress_callback,
        log_callback=log_callback,
    )
    if candidates.early_exit_frame is not None:
        try:
            if stage_progress:
                stage_progress(75, filter_pass, None, None, None)
                stage_progress(100, filter_pass, None, 0, 0)
        except Exception:
            pass
        return candidates.early_exit_frame

    market_df = candidates.market_df
    candidates_by_date = candidates.candidates_by_date

    setup_pass = _compute_setup_pass(
        system_name,
        prepared,
        market_df,
        filter_pass,
        today_ts,
        log_callback,
    )
    try:
        if stage_progress:
            stage_progress(50, filter_pass, setup_pass, None, None)
    except Exception:
        pass

    selection = _select_candidate_date(
        candidates_by_date,
        today_ts,
        filter_pass,
        setup_pass,
        log_callback,
    )

    # If setup_pass is zero, try to derive a more specific zero reason (e.g. SPY gate)
    try:
        if int(setup_pass) == 0:
            setup_zero_reason: str | None = None
            # If there were filter passes but setup failed, check SPY gate for system1
            try:
                if int(filter_pass) > 0 and str(system_name).lower() == "system1":
                    spy_source = (
                        market_df
                        if isinstance(market_df, pd.DataFrame)
                        else (
                            prepared.get("SPY") if isinstance(prepared, dict) else None
                        )
                    )
                    try:
                        spy_with = get_spy_with_indicators(spy_source)
                    except Exception:
                        spy_with = None
                    gate = _make_spy_gate(
                        _normalize_daily_index(spy_with)
                        if spy_with is not None
                        else None
                    )
                    if gate is False:
                        setup_zero_reason = "setup_fail: SPY close <= SMA100"
            except Exception:
                setup_zero_reason = None

            if setup_zero_reason is not None:
                # attach to selection for downstream logging
                selection = CandidateSelection(
                    selection.key_map,
                    selection.target_date,
                    selection.fallback_reason,
                    selection.total_candidates_today,
                    setup_zero_reason,
                )
                if log_callback:
                    try:
                        log_callback(f"🛈 セットアップ不成立: {setup_zero_reason}")
                    except Exception:
                        pass
    except Exception:
        pass
    try:
        if stage_progress:
            stage_progress(
                75,
                filter_pass,
                setup_pass,
                selection.total_candidates_today,
                None,
            )
    except Exception:
        pass

    signals_df, final_count = _build_today_signals_dataframe(
        strategy,
        prepared,
        candidates_by_date,
        selection,
        system_name,
        side,
        signal_type,
        log_callback,
    )

    # Emit diagnostic log if selection or extraction indicated zero candidates
    try:
        if getattr(selection, "zero_reason", None):
            if log_callback:
                try:
                    log_callback(f"🛈 選定結果: 候補0件理由: {selection.zero_reason}")
                except Exception:
                    pass
        elif hasattr(candidates, "zero_reason") and getattr(
            candidates, "zero_reason", None
        ):
            if log_callback:
                try:
                    log_callback(f"🛈 抽出結果: 候補0件理由: {candidates.zero_reason}")
                except Exception:
                    pass
    except Exception:
        pass

    try:
        if stage_progress:
            stage_progress(
                100,
                filter_pass,
                setup_pass,
                selection.total_candidates_today,
                final_count,
            )
    except Exception:
        pass

    return signals_df


def run_all_systems_today(
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
    per_system_progress: Callable[[str, str], None] | None = None,
    symbol_data: dict[str, pd.DataFrame] | None = None,
    parallel: bool = False,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """scripts.run_all_systems_today.compute_today_signals のラッパー。"""
    from scripts.run_all_systems_today import compute_today_signals as _compute

    # log_callback が未指定なら CLI へ出すデフォルトを使う
    if log_callback is None:
        log_callback = _default_cli_log

    return _compute(
        symbols,
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


compute_today_signals = run_all_systems_today


__all__ = [
    "get_today_signals_for_strategy",
    "LONG_SYSTEMS",
    "SHORT_SYSTEMS",
    "run_all_systems_today",
    "compute_today_signals",
]
