from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import inspect
import time as _t
from typing import Any, cast
import numpy as np

import pandas as pd

from config.settings import get_settings
from core.system5 import (
    DEFAULT_ATR_PCT_THRESHOLD,
    format_atr_pct_threshold_label,
)
from common.utils_spy import get_spy_with_indicators

# --- サイド定義（売買区分）---
# System1/3/5 は買い戦略、System2/4/6/7 は売り戦略として扱う。
LONG_SYSTEMS = {"system1", "system3", "system5"}
SHORT_SYSTEMS = {"system2", "system4", "system6", "system7"}

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


def _is_fast_path_viable(
    data_dict: dict[str, pd.DataFrame]
) -> tuple[bool, set[str]]:
    """高速経路で candidate 抽出が可能か判定し、(bool, 不足列) を返す。"""

    missing = _missing_fast_path_columns(data_dict)
    return len(missing) == 0, missing


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
    try:
        entry_ts = pd.Timestamp(candidate["entry_date"])
    except Exception:
        return None
    try:
        idxer = df.index.get_indexer([entry_ts])
        entry_idx = int(idxer[0]) if len(idxer) else -1
    except Exception:
        return None
    if entry_idx <= 0 or entry_idx >= len(df):
        return None
    try:
        entry = float(df.iloc[entry_idx]["Open"])
    except Exception:
        return None
    atr_col = _pick_atr_col(df)
    if not atr_col:
        return None
    try:
        atr = float(df.iloc[entry_idx - 1][atr_col])
    except Exception:
        return None
    mult = 3.0
    stop = entry - mult * atr if side == "long" else entry + mult * atr
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
    from common.utils_spy import get_latest_nyse_trading_day

    try:
        system_name = str(strategy.SYSTEM_NAME).lower()  # type: ignore[attr-defined]
    except Exception:
        system_name = ""
    side = _infer_side(system_name)
    signal_type = "sell" if side == "short" else "buy"

    # 取引日
    if today is None:
        today = get_latest_nyse_trading_day()
    try:
        today_ts = pd.Timestamp(today)
    except Exception:
        today_ts = get_latest_nyse_trading_day()
    if getattr(today_ts, "tzinfo", None) is not None:
        try:
            today_ts = today_ts.tz_convert(None)
        except (TypeError, ValueError, AttributeError):
            try:
                today_ts = today_ts.tz_localize(None)
            except Exception:
                today_ts = pd.Timestamp(today_ts.to_pydatetime().replace(tzinfo=None))
    today = today_ts.normalize()

    # 準備
    total_symbols = len(raw_data_dict)
    if log_callback:
        try:
            log_callback(f"🧪 フィルターチェック開始：{total_symbols} 銘柄")
        except Exception:
            pass
    # 0% -> 25%
    try:
        if stage_progress:
            # 0% ステージでは対象銘柄数を第1引数に渡す（UI 側で "対象→n" 表示に使用）
            stage_progress(0, total_symbols, None, None, None)
    except Exception:
        pass
    t0 = _t.time()
    # ルックバック最適化：必要日数が指定されていれば各DFを末尾N行にスライス
    sliced_dict = raw_data_dict
    try:
        if (
            lookback_days is not None
            and lookback_days > 0
            and isinstance(raw_data_dict, dict)
        ):
            sliced: dict[str, pd.DataFrame] = {}
            for _sym, _df in raw_data_dict.items():
                try:
                    if _df is None or getattr(_df, "empty", True):
                        continue
                    x = _df.copy()
                    if "Date" in x.columns:
                        idx = pd.to_datetime(x["Date"], errors="coerce").dt.normalize()
                        x.index = pd.Index(idx)
                    else:
                        x.index = pd.to_datetime(x.index, errors="coerce").normalize()
                    # 不正な日時は除外
                    x = x[~x.index.isna()]
                    # 末尾N営業日相当を抽出
                    x = x.tail(int(lookback_days))
                    sliced[_sym] = x
                except Exception:
                    sliced[_sym] = _df
            sliced_dict = sliced
    except Exception:
        sliced_dict = raw_data_dict

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

    # スキップ理由の収集（systemごとに集計）
    _skip_counts: dict[str, int] = {}
    _skip_samples: dict[str, list[str]] = {}
    _skip_details: list[dict[str, str]] = []

    def _on_skip(*args, **kwargs):
        try:
            if len(args) >= 2:
                _sym = str(args[0])
                _reason = str(args[1])
            elif len(args) == 1:
                # "SYM: reason" 形式から理由だけ抽出
                txt = str(args[0])
                _sym, _reason = (
                    (txt.split(":", 1) + [""])[:2] if ":" in txt else ("", txt)
                )
                _sym = _sym.strip()
                _reason = _reason.strip()
            else:
                _reason = str(kwargs.get("reason", "unknown"))
                _sym = str(kwargs.get("symbol", ""))
        except Exception:
            _reason = "unknown"
            _sym = ""
        _skip_counts[_reason] = _skip_counts.get(_reason, 0) + 1
        if _sym:
            if _reason not in _skip_samples:
                _skip_samples[_reason] = []
            if len(_skip_samples[_reason]) < 5 and _sym not in _skip_samples[_reason]:
                _skip_samples[_reason].append(_sym)
        try:
            _skip_details.append(
                {"symbol": str(_sym or ""), "reason": str(_reason or "")}
            )
        except Exception:
            pass

    if not fast_path_used or prepared_dict is None:
        if fast_missing and log_callback:
            try:
                missing_list = ", ".join(sorted(fast_missing))
                log_callback(
                    "⚠️ 高速パスを利用できません（必須列不足: "
                    + (missing_list or "不明")
                    + "）。再計算します。"
                )
            except Exception:
                pass
        try:
            prepared_dict = strategy.prepare_data(
                sliced_dict,
                progress_callback=progress_callback,
                log_callback=log_callback,
                skip_callback=_on_skip,
                use_process_pool=use_process_pool,
                max_workers=max_workers,
                lookback_days=lookback_days,
            )
        except Exception as e:
            # フォールバック: 非プール + 再計算（reuse_indicators=False）で再試行
            try:
                if log_callback:
                    log_callback(
                        f"⚠️ {system_name}: 前処理失敗のためフォールバック（非プール・再計算）: {e}"
                    )
            except Exception:
                pass
            try:
                prepared_dict = strategy.prepare_data(
                    sliced_dict,
                    progress_callback=progress_callback,
                    log_callback=log_callback,
                    skip_callback=_on_skip,
                    use_process_pool=False,
                    max_workers=None,
                    lookback_days=lookback_days,
                    reuse_indicators=False,
                )
            except Exception as e2:
                # ここで失敗したら空の結果を返す（後段は0件で流れる）
                try:
                    if log_callback:
                        log_callback(
                            f"⚠️ {system_name}: フォールバックも失敗（中断）: {e2}"
                        )
                except Exception:
                    pass
                return pd.DataFrame(
                    columns=[
                        "symbol",
                        "system",
                        "side",
                        "signal_type",
                        "entry_date",
                        "entry_price",
                        "stop_price",
                        "score_key",
                        "score",
                    ]
                )
    if prepared_dict is None:
        prepared_dict = {}

    # インデックスを正規化・昇順・重複除去（pandas の再インデックス関連エラー対策）
    try:
        if isinstance(prepared_dict, dict):
            _fixed: dict[str, pd.DataFrame] = {}
            for _sym, _df in prepared_dict.items():
                try:
                    x = _df.copy()
                    if "Date" in x.columns:
                        idx = pd.to_datetime(x["Date"], errors="coerce").dt.normalize()
                    else:
                        idx = pd.to_datetime(x.index, errors="coerce").normalize()
                    x.index = pd.Index(idx)
                    # 欠損・非単調・重複を整理
                    x = x[~x.index.isna()]
                    x = x.sort_index()
                    if getattr(x.index, "has_duplicates", False):
                        x = x[~x.index.duplicated(keep="last")]
                    _fixed[_sym] = x
                except Exception:
                    _fixed[_sym] = _df
            prepared_dict = _fixed
    except Exception:
        pass

    prepared = prepared_dict
    try:
        if log_callback:
            em, es = divmod(int(max(0, _t.time() - t0)), 60)
            log_callback(f"⏱️ フィルター/前処理 完了（経過 {em}分{es}秒）")
    except Exception:
        pass
    # スキップ内訳の要約（存在時のみ）
    try:
        if log_callback and _skip_counts:
            # 上位2件のみを簡潔に表示
            sorted_items = sorted(
                _skip_counts.items(), key=lambda x: x[1], reverse=True
            )
            top = sorted_items[:2]
            details = ", ".join([f"{k}: {v}" for k, v in top])
            log_callback(f"🧪 スキップ内訳: {details}")
            # サンプル銘柄出力
            for k, _ in top:
                samples = _skip_samples.get(k) or []
                if samples:
                    log_callback(f"  ↳ 例({k}): {', '.join(samples)}")
            # 追加: 全スキップのCSVを保存（デバッグ用）。UI/CLI両方でパスを出力。
            try:
                import pandas as _pd
                from config.settings import get_settings as _gs

                _rows = []
                for _reason, _cnt in sorted_items:
                    _rows.append(
                        {
                            "reason": _reason,
                            "count": int(_cnt),
                            "examples": ", ".join(_skip_samples.get(_reason, [])),
                        }
                    )
                if _rows:
                    _df = _pd.DataFrame(_rows)
                    try:
                        _settings = _gs(create_dirs=True)
                        _dir = getattr(_settings.outputs, "results_csv_dir", None)
                    except Exception:
                        _dir = None
                    import os as _os

                    _out_dir = str(_dir or "results_csv")
                    try:
                        _os.makedirs(_out_dir, exist_ok=True)
                    except Exception:
                        pass
                    _fp = _os.path.join(_out_dir, f"skip_summary_{system_name}.csv")
                    try:
                        _df.to_csv(_fp, index=False, encoding="utf-8")
                        log_callback(f"📝 スキップ内訳CSVを保存: {_fp}")
                    except Exception:
                        pass
                    # per-symbol の詳細（symbol, reason）も保存
                    try:
                        if _skip_details:
                            _df2 = _pd.DataFrame(_skip_details)
                            _fp2 = _os.path.join(
                                _out_dir, f"skip_details_{system_name}.csv"
                            )
                            _df2.to_csv(_fp2, index=False, encoding="utf-8")
                            log_callback(f"📝 スキップ詳細CSVを保存: {_fp2}")
                    except Exception:
                        pass
            except Exception:
                pass
    except Exception:
        pass
    # フィルター通過件数（NYSEカレンダーの前営業日を優先。無い場合は最終行）。
    try:
        # 前営業日（当日エントリーのシグナルは前営業日の終値で判定）
        prev_trading_day = get_latest_nyse_trading_day(
            pd.Timestamp(today) - pd.Timedelta(days=1)
        )

        def _last_filter_on_date(x: pd.DataFrame) -> bool:
            try:
                if getattr(x, "empty", True) or "filter" not in x.columns:
                    return False
                # Date列があれば優先、無ければindexで比較
                if "Date" in x.columns:
                    dt_vals = (
                        pd.to_datetime(x["Date"], errors="coerce")
                        .dt.normalize()
                        .to_numpy()
                    )
                    mask = dt_vals == prev_trading_day
                    sel = pd.Series(np.asarray(x.loc[mask, "filter"]))
                else:
                    idx_vals = (
                        pd.to_datetime(x.index, errors="coerce").normalize().to_numpy()
                    )
                    mask = idx_vals == prev_trading_day
                    sel = pd.Series(np.asarray(x.loc[mask, "filter"]))
                if sel.size > 0:
                    v = sel.iloc[-1]
                    return bool(False if pd.isna(v) else bool(v))
                # フォールバック: 最終行
                v = pd.Series(x["filter"]).tail(1).iloc[0]
                return bool(False if pd.isna(v) else bool(v))
            except Exception:
                return False

        filter_pass = sum(int(_last_filter_on_date(df)) for df in prepared.values())
        # System7 は SPY 固定のため、SPYが存在する場合はフィルタ通過=1として扱う
        try:
            if str(system_name).lower() == "system7":
                filter_pass = 1 if ("SPY" in (prepared or {})) else 0
        except Exception:
            pass
    except Exception:
        filter_pass = 0
    if log_callback:
        try:
            log_callback(f"🧪 フィルターチェック完了：{filter_pass} 銘柄")
        except Exception:
            pass
    try:
        if stage_progress:
            stage_progress(25, filter_pass, None, None, None)
    except Exception:
        pass

    # 候補生成（market_df を必要とする実装に配慮）
    gen_fn = strategy.generate_candidates  # type: ignore[attr-defined]
    params = inspect.signature(gen_fn).parameters
    if log_callback:
        try:
            log_callback(f"🧩 セットアップチェック開始：{filter_pass} 銘柄")
        except Exception:
            pass
    t1 = _t.time()
    if "market_df" in params and market_df is not None:
        candidates_by_date, _ = gen_fn(
            prepared,
            market_df=market_df,
            progress_callback=progress_callback,
            log_callback=log_callback,
        )
    else:
        candidates_by_date, _ = gen_fn(
            prepared,
            progress_callback=progress_callback,
            log_callback=log_callback,
        )
    try:
        if log_callback:
            em, es = divmod(int(max(0, _t.time() - t1)), 60)
            log_callback(f"⏱️ セットアップ/候補抽出 完了（経過 {em}分{es}秒）")
    except Exception:
        pass

    # セットアップ通過件数（NYSEカレンダーの前営業日を優先。無ければ最終行）
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
                        pd.to_datetime(x.index, errors="coerce")
                        .normalize()
                        .to_numpy()
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
                        "🧩 system1セットアップ内訳: "
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
                        "🧩 system2セットアップ内訳: "
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
            drop_pass = _count_if(
                filtered_rows, lambda r: _close_ok(r) and _drop_ok(r)
            )
            setup_pass = drop_pass
            if log_callback:
                try:
                    log_callback(
                        "🧩 system3セットアップ内訳: "
                        + f"フィルタ通過={filter_pass}, Close>SMA150: {close_pass}, "
                        + f"3日下落率>=12.5%: {drop_pass}"
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
                        "🧩 system4セットアップ内訳: "
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
                        "🧪 system5内訳: "
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
                        "🧩 system5セットアップ内訳: "
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
                        "🧩 system6セットアップ内訳: "
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
                    msg = f"🧩 system7セットアップ内訳: SPY存在={spy_present}"
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
    try:
        if stage_progress:
            stage_progress(50, filter_pass, setup_pass, None, None)
    except Exception:
        pass
    # トレード候補件数（当日のみ）→ UI表示は最大ポジション数に合わせて上限10に丸める
    # 候補キー型のゆらぎ（str/date/Timestamp）を吸収するため、
    # 正規化Timestamp→元キーのマップを作成してから選択・参照する
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
                            _ts = pd.Timestamp(
                                _ts.to_pydatetime().replace(tzinfo=None)
                            )
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
            today_norm = (
                pd.Timestamp(today).normalize() if today is not None else None
            )
            past_candidates = [
                d
                for d in candidate_dates
                if today_norm is None or d <= today_norm
            ]
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
                    "🗓️ 候補日（最新上位）: "
                    f"{_cands_str} | 探索順: {_search_str} | 採用: {_chosen}{fallback_msg}"
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
    # UIのTRDlistは各systemの最大ポジション数を超えないように表示
    try:
        _max_pos_ui = int(get_settings(create_dirs=False).risk.max_positions)
    except Exception:
        _max_pos_ui = 10
    if total_candidates_today and _max_pos_ui > 0:
        total_candidates_today = min(int(total_candidates_today), int(_max_pos_ui))
    try:
        if stage_progress:
            stage_progress(75, filter_pass, setup_pass, total_candidates_today, None)
    except Exception:
        pass
    if log_callback:
        try:
            log_callback(f"🧩 セットアップチェック完了：{setup_pass} 銘柄")
            # 誤解回避: ここでの件数は『候補生成の母集団（セットアップ通過）』
            log_callback(f"🧮 候補生成元（セットアップ通過）：{setup_pass} 銘柄")
            # TRDlist 相当（直近営業日時点の候補数。最大{_max_pos_ui}に丸め）
            log_callback(
                f"🧮 TRDlist相当（直近営業日時点の候補数）：{total_candidates_today} 銘柄"
            )
        except Exception:
            pass

    if not candidates_by_date:
        return pd.DataFrame(
            columns=[
                "symbol",
                "system",
                "side",
                "signal_type",
                "entry_date",
                "entry_price",
                "stop_price",
                "score_key",
                "score",
            ]
        )

    # 当日または直近過去日の候補のみ抽出
    if target_date is not None and target_date in key_map:
        orig_key2 = key_map[target_date]
        today_candidates = cast(
            list[dict], candidates_by_date.get(orig_key2, [])
        )
    else:
        today_candidates = cast(list[dict], [])
    if not today_candidates:
        return pd.DataFrame(
            columns=[
                "symbol",
                "system",
                "side",
                "signal_type",
                "entry_date",
                "entry_price",
                "stop_price",
                "score_key",
                "score",
            ]
        )
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

        # System1 は ROC200 を必ずスコアに採用できるよう堅牢化
        try:
            if (system_name == "system1") and (
                skey is None or str(skey).upper() != "ROC200"
            ):
                skey = "ROC200"
        except Exception:
            pass

        # signal 日（通常は entry_date の前営業日を想定）
        signal_date_ts: pd.Timestamp | None = None
        try:
            # candidate["Date"] があれば優先
            if "Date" in c and c.get("Date") is not None:
                date_arg: Any = c.get("Date")
                tmp = pd.to_datetime(date_arg, errors="coerce")
                if not pd.isna(tmp):
                    signal_date_ts = pd.Timestamp(tmp).normalize()
        except Exception:
            # フォールバックは後段の entry_date 補完に任せる
            pass
        if signal_date_ts is None:
            try:
                ed_arg: Any = c.get("entry_date")
                ed = pd.to_datetime(ed_arg, errors="coerce")
                if isinstance(ed, pd.Timestamp) and not pd.isna(ed):
                    # エントリー日の前「NYSE営業日」を推定
                    signal_date_ts = get_latest_nyse_trading_day(
                        pd.Timestamp(ed).normalize() - pd.Timedelta(days=1)
                    )
            except Exception:
                signal_date_ts = None

        # 欠損スコアの補完（まず値、次に順位）
        rank_val: int | None = None
        total_for_rank: int = 0
        if skey is not None:
            # 1) 欠損なら prepared から同日値を補完
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
            # System1 用のフォールバック（前日が見つからない場合は直近値）
            if (system_name == "system1") and (
                sval is None or (isinstance(sval, float) and pd.isna(sval))
            ):
                try:
                    if skey in prepared[sym].columns:
                        _v = pd.Series(prepared[sym][skey]).dropna().tail(1).iloc[0]
                        sval = float(_v)
                except Exception:
                    pass

            # 2) 値がまだ欠損なら、同日全銘柄の順位を算出してスコアに設定
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
                            if not row.empty and skey in row.columns:
                                v = row.iloc[0][skey]
                                if v is not None and not pd.isna(v):
                                    vals.append((psym, float(v)))
                        except Exception:
                            continue
                    total_for_rank = len(vals)
                    if total_for_rank:
                        # 並び順: system の昇降順推定に合わせる（ROC200 などは降順）
                        reverse = not _asc
                        # 値が同一のときはシンボルで安定ソート
                        vals_sorted = sorted(
                            vals, key=lambda t: (t[1], t[0]), reverse=reverse
                        )
                        # 自銘柄の順位を決定
                        symbols_sorted = [s for s, _ in vals_sorted]
                        if sym in symbols_sorted:
                            rank_val = symbols_sorted.index(sym) + 1
                        # スコアが欠損なら順位をそのままスコアに採用
                        if sval is None or (isinstance(sval, float) and pd.isna(sval)):
                            if rank_val is not None:
                                sval = float(rank_val)
            except Exception:
                pass

        # 選定理由（順位を最優先、なければ簡潔かつシステム固有の文言）
        reason_parts: list[str] = []
        # System1 は日本語で「ROC200がn位のため」に統一（順位が取れない場合のみ汎用文言）
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
            # ATR50 は損切り計算用。理由は「50日安値更新」に限定する。
            reason_parts = ["SPYが50日安値を更新したため（ヘッジ）"]
        else:
            if skey is not None and rank_val is not None:
                if rank_val <= 10:
                    reason_parts = [f"{_label_for_score_key(skey)}が{rank_val}位のため"]
                else:
                    reason_parts = [f"rank={rank_val}/{total_for_rank}"]
            elif skey is not None:
                # 値は原則非表示（冗長回避）。必要最小限だけ示す。
                try:
                    if sval is not None and not (
                        isinstance(sval, float) and pd.isna(sval)
                    ):
                        reason_parts.append("スコア条件を満たしたため")
                except Exception:
                    reason_parts.append("スコア条件を満たしたため")

        # fallback generic info
        if not reason_parts:
            reason_parts.append("条件一致のため")

        reason_text = "; ".join(reason_parts)

        try:
            _ed_raw: Any = c.get("entry_date")
            _ed = pd.Timestamp(_ed_raw) if _ed_raw is not None else None
            if _ed is None or pd.isna(_ed):
                # entry_date が欠損する候補は無効
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
                # スコアは値があれば値、無ければ順位（上記で補完済み）
                score=(
                    None
                    if sval is None or (isinstance(sval, float) and pd.isna(sval))
                    else float(sval)
                ),
                reason=reason_text,
            )
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "symbol",
                "system",
                "side",
                "signal_type",
                "entry_date",
                "entry_price",
                "stop_price",
                "score_key",
                "score",
            ]
        )

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

    try:
        if log_callback:
            log_callback(f"🧮 トレード候補選定完了（当日）：{final_count} 銘柄")
    except Exception:
        pass
    try:
        if stage_progress:
            stage_progress(
                100, filter_pass, setup_pass, total_candidates_today, final_count
            )
    except Exception:
        pass
    return out


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
