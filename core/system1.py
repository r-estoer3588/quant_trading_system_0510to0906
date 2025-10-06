"""System1 core logic (Long ROC200 momentum).

ROC200-based momentum strategy:
- Indicators: ROC200, SMA200, DollarVolume20 (precomputed only)
- Setup conditions: Close>5, DollarVolume20>25M, Close>SMA200, ROC200>0
- Candidate generation: ROC200 descending ranking by date, extract top_n
- Optimization: Removed all indicator calculations, using precomputed indicators only
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
import os
from typing import Any, Callable, DefaultDict, Mapping, cast

import pandas as pd

from common.batch_processing import process_symbols_batch
from common.system_common import check_precomputed_indicators, get_total_days
from common.system_constants import REQUIRED_COLUMNS
from common.system_setup_predicates import system1_setup_predicate


@dataclass(slots=True)
class System1Diagnostics:
    """Lightweight diagnostics payload for System1 candidate generation."""

    mode: str = "full_scan"  # or "latest_only"
    top_n: int = 20

    # counts
    symbols_total: int = 0
    symbols_with_data: int = 0
    total_symbols: int = 0
    filter_pass: int = 0
    setup_flag_true: int = 0
    fallback_pass: int = 0
    roc200_positive: int = 0
    final_pass: int = 0
    setup_predicate_count: int = 0
    final_top_n_count: int = 0
    predicate_only_pass_count: int = 0
    mismatch_flag: int = 0
    date_fallback_count: int = 0

    # ranking source marker
    ranking_source: str | None = None

    # reason histogram
    exclude_reasons: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int))

    def as_dict(self) -> dict[str, Any]:
        # normalize defaultdict to plain dict of ints
        return {
            "mode": self.mode,
            "top_n": int(self.top_n),
            "symbols_total": int(self.symbols_total),
            "symbols_with_data": int(self.symbols_with_data),
            "total_symbols": int(self.total_symbols),
            "filter_pass": int(self.filter_pass),
            "setup_flag_true": int(self.setup_flag_true),
            "fallback_pass": int(self.fallback_pass),
            "roc200_positive": int(self.roc200_positive),
            "final_pass": int(self.final_pass),
            "setup_predicate_count": int(self.setup_predicate_count),
            "final_top_n_count": int(self.final_top_n_count),
            "predicate_only_pass_count": int(self.predicate_only_pass_count),
            "mismatch_flag": int(self.mismatch_flag),
            "date_fallback_count": int(self.date_fallback_count),
            "ranking_source": self.ranking_source,
            "exclude_reasons": {k: int(v) for k, v in dict(self.exclude_reasons).items()},
        }


def summarize_system1_diagnostics(
    diag: Mapping[str, Any] | None,
    *,
    max_reasons: int = 3,
) -> dict[str, Any]:
    """Normalize raw diagnostics payload for display/log output.

    Args:
        diag: Raw diagnostics mapping emitted by ``generate_candidates_system1``.
        max_reasons: Maximum number of exclusion reasons to keep.

    Returns:
        Dictionary containing integer-normalized counters and (optionally)
        a trimmed ``exclude_reasons`` mapping sorted by descending count.
    """

    if not isinstance(diag, Mapping):
        return {}

    def _coerce_int(value: Any) -> int:
        if value is None:
            return 0
        try:
            return int(value)
        except Exception:
            try:
                return int(float(value))
            except Exception:
                return 0

    summary_keys = (
        "symbols_total",
        "symbols_with_data",
        "total_symbols",
        "filter_pass",
        "setup_flag_true",
        "fallback_pass",
        "roc200_positive",
        "final_pass",
        "setup_predicate_count",
        "final_top_n_count",
        "predicate_only_pass_count",
        "mismatch_flag",
        "date_fallback_count",
    )

    summary: dict[str, Any] = {key: _coerce_int(diag.get(key)) for key in summary_keys}

    top_n_val = diag.get("top_n")
    if top_n_val is not None:
        top_n_int = _coerce_int(top_n_val)
        if top_n_int > 0:
            summary["top_n"] = top_n_int

    exclude_raw = diag.get("exclude_reasons")
    if isinstance(exclude_raw, Mapping):
        reasons: list[tuple[str, int]] = []
        for key, value in exclude_raw.items():
            count = _coerce_int(value)
            if count <= 0:
                continue
            reasons.append((str(key), count))
        if reasons:
            reasons.sort(key=lambda item: item[1], reverse=True)
            if max_reasons >= 0:
                reasons = reasons[:max_reasons]
            summary["exclude_reasons"] = {name: count for name, count in reasons}

    return summary


def _to_float(value: Any) -> float:
    try:
        v = float(value)
        if math.isnan(v):
            return math.nan
        return v
    except Exception:
        return math.nan


def system1_row_passes_setup(
    row: pd.Series, *, allow_fallback: bool = True
) -> tuple[bool, dict[str, bool], str | None]:
    filter_ok = bool(row.get("filter", True))
    setup_flag = bool(row.get("setup", False))
    fallback_ok = False
    if allow_fallback and not setup_flag:
        sma25 = _to_float(row.get("sma25"))
        sma50 = _to_float(row.get("sma50"))
        fallback_ok = not math.isnan(sma25) and not math.isnan(sma50) and sma25 > sma50
    roc200_val = _to_float(row.get("roc200"))
    roc200_positive = not math.isnan(roc200_val) and roc200_val > 0

    passes = filter_ok and roc200_positive and (setup_flag or fallback_ok)
    reason: str | None = None
    if not filter_ok:
        reason = "filter"
    elif not (setup_flag or fallback_ok):
        reason = "setup"
    elif not roc200_positive:
        reason = "roc200"

    flags = {
        "filter_ok": filter_ok,
        "setup_flag": setup_flag,
        "fallback_ok": fallback_ok,
        "roc200_positive": roc200_positive,
    }
    return passes, flags, reason


def _rename_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy(deep=False)
    rename_map = {}
    for low, up in (
        ("open", "Open"),
        ("high", "High"),
        ("low", "Low"),
        ("close", "Close"),
        ("volume", "Volume"),
    ):
        if low in x.columns and up not in x.columns:
            rename_map[low] = up
    if rename_map:
        try:
            x = x.rename(columns=rename_map)
        except Exception:
            pass
    return x


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" in df.columns:
        idx = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    elif "date" in df.columns:
        idx = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    else:
        idx = pd.to_datetime(df.index, errors="coerce").normalize()
    x = df.copy(deep=False)
    x.index = pd.Index(idx, name="Date")
    x = x[~x.index.isna()]
    try:
        x = x.sort_index()
        if getattr(x.index, "has_duplicates", False):
            x = x[~x.index.duplicated(keep="last")]
    except Exception:
        pass
    return x


def _prepare_source_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("empty_frame")
    x = _rename_ohlcv(df)
    missing = [c for c in REQUIRED_COLUMNS if c not in x.columns]
    if missing:
        raise ValueError(f"missing_cols:{','.join(missing)}")
    x = _normalize_index(x)
    for col in REQUIRED_COLUMNS:
        if col in x.columns:
            try:
                x[col] = pd.to_numeric(x[col], errors="coerce")
            except Exception:
                pass
    x = x.dropna(subset=[c for c in ("High", "Low", "Close") if c in x.columns])
    return x


def _compute_indicators_frame(df: pd.DataFrame) -> pd.DataFrame:
    # System1 now relies exclusively on precomputed indicators (fast path).
    x = df.copy(deep=False)
    required_indicators = ["sma25", "sma50", "roc200", "atr20", "dollarvolume20"]
    missing_indicators = [c for c in required_indicators if c not in x.columns]
    if missing_indicators:
        raise ValueError(f"missing precomputed indicators: {missing_indicators}")
    # Derive filter/setup (legacy style) for tests expecting them here.
    try:
        if "filter" not in x.columns:
            x["filter"] = (x["Low"] >= 5) & (x["dollarvolume20"] > 50_000_000)
        if "setup" not in x.columns:
            x["setup"] = x["filter"] & (x["sma25"] > x["sma50"])
    except Exception:
        # Fallback safe defaults
        x["filter"] = False
        x["setup"] = False
    return x


def _compute_indicators(symbol: str) -> tuple[str, pd.DataFrame | None]:
    """Check precomputed indicators and apply System1-specific filters.

    Args:
        symbol: Target symbol to process

    Returns:
        (symbol, processed DataFrame | None)

    Note:
        This function uses global cache access for batch processing.
        Indicators must be precomputed.
    """
    from common.cache_manager import CacheManager, load_base_cache
    from config.settings import get_settings

    try:
        settings = get_settings()
        cache_mgr = CacheManager(settings)
        df = load_base_cache(symbol, cache_manager=cache_mgr)
        if df is None or df.empty:
            return symbol, None

        # Only require 'roc200' for ranking; other indicators are optional here.
        if "roc200" not in df.columns:
            return symbol, None

        # Apply System1-specific filters and setup (best-effort). If required
        # columns are missing, set conservative defaults (False) to avoid
        # accidental passes while still keeping the symbol in prepared data
        # for ranking by roc200.
        x = df.copy()

        # Filter: Close>=5, DollarVolume20>25M
        try:
            filt = (x["Close"] >= 5.0) & (
                x.get("dollarvolume20", pd.Series(index=x.index, dtype=float)) > 25_000_000
            )
        except Exception:
            # If columns not available, mark filter False
            filt = pd.Series(False, index=x.index)
        x["filter"] = filt

        # Setup: Filter + Close>SMA200 + ROC200>0
        try:
            setup = (
                filt
                & (
                    x.get("Close", pd.Series(index=x.index, dtype=float))
                    > x.get("sma200", pd.Series(index=x.index, dtype=float))
                )
                & (x["roc200"] > 0)
            )
        except Exception:
            setup = pd.Series(False, index=x.index)
        x["setup"] = setup

        return symbol, x

    except Exception:
        return symbol, None


def prepare_data_vectorized_system1(
    raw_data_dict: dict[str, pd.DataFrame] | None,
    *,
    progress_callback: Callable[[str], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
    skip_callback: Callable[[str, str], None] | None = None,
    batch_size: int | None = None,
    reuse_indicators: bool = True,
    symbols: list[str] | None = None,
    use_process_pool: bool = False,
    max_workers: int | None = None,
    **_kwargs: object,
) -> dict[str, pd.DataFrame]:
    """System1 data preparation processing (ROC200 momentum strategy).

    Execute high-speed processing using precomputed indicators.

    Args:
        raw_data_dict: Raw data dictionary (None to fetch from cache)
        progress_callback: Progress reporting callback
        log_callback: Log output callback
        skip_callback: Error skip callback
        batch_size: Batch size
        reuse_indicators: Reuse existing indicators (for speed)
        symbols: Target symbol list
        use_process_pool: Process pool usage flag
        max_workers: Maximum worker count

    Returns:
        Processed data dictionary
    """

    def _substep(msg: str) -> None:
        if not log_callback:
            return
        try:
            if (os.environ.get("ENABLE_SUBSTEP_LOGS") or "").lower() in {
                "1",
                "true",
                "yes",
            }:
                log_callback(f"System1: {msg}")
        except Exception:
            pass

    _substep("enter prepare_data")
    # Fast path: reuse precomputed indicators
    if reuse_indicators and raw_data_dict:
        _substep("fast-path check start")
        try:
            # Early check - verify required indicators exist (roc200 のみ必須)
            valid_data_dict, error_symbols = check_precomputed_indicators(
                raw_data_dict, ["roc200"], "System1", skip_callback
            )

            if valid_data_dict:
                # Apply System1-specific filters
                prepared_dict = {}
                for symbol, df in valid_data_dict.items():
                    x = df.copy()

                    # Filter: Close>=5, DollarVolume20>25M（欠損は安全側に False）
                    try:
                        dv = x.get("dollarvolume20", pd.Series(index=x.index, dtype=float))
                        filt = (x["Close"] >= 5.0) & (dv > 25_000_000)
                    except Exception:
                        filt = pd.Series(False, index=x.index)
                    x["filter"] = filt

                    # Setup: Filter + Close>SMA200 + ROC200>0（欠損は False）
                    try:
                        close_s = x.get("Close", pd.Series(index=x.index, dtype=float))
                        sma200_s = x.get("sma200", pd.Series(index=x.index, dtype=float))
                        roc_ok = x.get("roc200", pd.Series(index=x.index, dtype=float)) > 0
                        setup = filt & (close_s > sma200_s) & roc_ok
                    except Exception:
                        setup = pd.Series(False, index=x.index)
                    x["setup"] = setup

                    prepared_dict[symbol] = x

                _substep(f"fast-path processed symbols={len(prepared_dict)}")

                return prepared_dict

        except RuntimeError:
            # Re-raise error immediately if required indicators are missing
            raise
        except Exception:
            # Fall back to normal processing for other errors
            _substep("fast-path failed fallback to normal path")

    # Normal processing path: batch processing from symbol list
    if symbols:
        target_symbols = symbols
    elif raw_data_dict:
        target_symbols = list(raw_data_dict.keys())
    else:
        _substep("no symbols provided -> empty dict")
        return {}

    _substep(f"normal path start symbols={len(target_symbols)}")
    if log_callback:
        try:
            log_callback(f"System1: Starting normal processing for {len(target_symbols)} symbols")
        except Exception:
            pass

    # Execute batch processing
    _substep("batch processing start")
    results, error_symbols = process_symbols_batch(
        target_symbols,
        _compute_indicators,
        batch_size=batch_size,
        use_process_pool=use_process_pool,
        max_workers=max_workers,
        progress_callback=progress_callback,
        log_callback=log_callback,
        skip_callback=skip_callback,
        system_name="System1",
    )
    _substep(f"batch processing done ok={len(results)} err={len(error_symbols)}")

    # Cast batch results to the expected typed mapping
    typed_results: dict[str, pd.DataFrame] = (
        cast(dict[str, pd.DataFrame], results) if isinstance(results, dict) else {}
    )
    return typed_results

    # NOTE: predicate 検証呼び出しは結果返却前に行う設計だが、
    # 上方で return 済みのため通常経路は到達しない。後続統合時に
    # fast-path/normal-path の共通ポスト処理へリファクタ予定。


def generate_candidates_system1(
    prepared_dict: dict[str, pd.DataFrame] | None,
    *,
    top_n: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
    batch_size: int | None = None,
    latest_only: bool = False,
    include_diagnostics: bool = False,
    diagnostics: System1Diagnostics | Mapping[str, Any] | None = None,
    **kwargs: object,
) -> (
    tuple[dict[pd.Timestamp, object], pd.DataFrame | None]
    | tuple[dict[pd.Timestamp, object], pd.DataFrame | None, dict[str, object]]
):
    """System1 candidate generation (ROC200 descending ranking).

    Returns a tuple of (per-date candidates, merged dataframe,
    diagnostics when requested).
    """

    if kwargs and log_callback:
        ignored = ", ".join(sorted(map(str, kwargs.keys())))
        log_callback(f"System1: Ignoring unsupported kwargs -> {ignored}")

    # batch_size is unused for candidate generation; silently ignore

    resolved_top_n = 20 if top_n is None else top_n
    mode = "latest_only" if latest_only else "full_scan"

    if isinstance(diagnostics, System1Diagnostics):
        diag = diagnostics
        diag.mode = mode
        diag.top_n = resolved_top_n
    else:
        diag = System1Diagnostics(mode=mode, top_n=resolved_top_n)
        if isinstance(diagnostics, Mapping):
            prev_reasons = diagnostics.get("exclude_reasons")
            if isinstance(prev_reasons, Mapping):
                for key, value in prev_reasons.items():
                    try:
                        # value は Mapping[Any, Any] 由来のため型が不明だが、
                        # int() で正規化してから加算する。
                        diag.exclude_reasons[str(key)] += int(value)
                    except Exception:
                        continue

    def finalize(
        by_date: Mapping[pd.Timestamp, object],
        merged: pd.DataFrame | None,
    ) -> (
        tuple[dict[pd.Timestamp, object], pd.DataFrame | None]
        | tuple[dict[pd.Timestamp, object], pd.DataFrame | None, dict[str, object]]
    ):
        diag_payload = diag.as_dict()
        normalized = dict(by_date)
        if include_diagnostics:
            return normalized, merged, diag_payload
        # Maintain backward compatibility: always include diagnostics payload
        return normalized, merged, diag_payload

    if not isinstance(prepared_dict, dict) or not prepared_dict:
        diag.symbols_total = len(prepared_dict or {})
        diag.ranking_source = mode
        if log_callback:
            log_callback("System1: No data provided for candidate generation")
        return finalize({}, None)

    diag.symbols_total = len(prepared_dict)
    diag.symbols_with_data = sum(
        1 for df in prepared_dict.values() if isinstance(df, pd.DataFrame) and not df.empty
    )

    # Fast path: evaluate only the most recent bar per symbol
    if latest_only:
        try:
            rows: list[dict] = []
            date_counter: dict[pd.Timestamp, int] = {}
            # Optional: orchestrator may specify a target trading date fallback
            target_date = None
            try:
                maybe = kwargs.get("latest_mode_date")
                if maybe is not None:
                    try:
                        target_date = pd.Timestamp(str(maybe)).normalize()
                    except Exception:
                        td = pd.to_datetime(str(maybe), errors="coerce")
                        if (td is not None) and not pd.isna(td):
                            target_date = pd.Timestamp(str(td)).normalize()
            except Exception:
                target_date = None
            max_date_lag_days = 1
            lag_override = kwargs.get("max_date_lag_days")
            if lag_override is not None:
                try:
                    max_date_lag_days = max(0, int(float(str(lag_override))))
                except Exception:
                    pass

            def _ensure_series(obj: Any) -> pd.Series | None:
                if obj is None:
                    return None
                if isinstance(obj, pd.DataFrame):
                    try:
                        return obj.iloc[-1]
                    except Exception:
                        return None
                if isinstance(obj, pd.Series):
                    return obj
                return None

            fallback_log_remaining = 5
            for sym, df in prepared_dict.items():
                if df is None or df.empty:
                    continue
                diag.total_symbols += 1
                row_obj: pd.Series | pd.DataFrame | None
                date_val: pd.Timestamp | None
                row_obj = None
                date_val = None
                fallback_used = False
                if target_date is not None:
                    if target_date in df.index:
                        row_obj = df.loc[target_date]
                        date_val = target_date
                    else:
                        latest_idx_raw = df.index[-1]
                        latest_idx_norm = pd.Timestamp(str(latest_idx_raw)).normalize()
                        lag_days: int | None = None
                        try:
                            lag_delta = target_date - latest_idx_norm
                            lag_days = int(lag_delta.days)
                        except Exception:
                            lag_days = None
                        if lag_days is not None and lag_days >= 0 and lag_days <= max_date_lag_days:
                            try:
                                row_obj = df.loc[latest_idx_raw]
                            except Exception:
                                row_obj = None
                            date_val = latest_idx_norm
                            fallback_used = True
                            diag.date_fallback_count += 1
                            if fallback_used and log_callback and fallback_log_remaining > 0:
                                try:
                                    msg = (
                                        "System1: latest_only missing target bar -> "
                                        f"fallback to {latest_idx_norm.date()} "
                                        f"for {sym}"
                                    )
                                    log_callback(msg)
                                except Exception:
                                    pass
                                fallback_log_remaining -= 1
                        else:
                            diag.exclude_reasons["missing_date"] += 1
                            continue
                else:
                    latest_idx_raw = df.index[-1]
                    date_val = pd.Timestamp(str(latest_idx_raw)).normalize()
                    try:
                        row_obj = df.loc[latest_idx_raw]
                    except Exception:
                        row_obj = None
                last_row = _ensure_series(row_obj)
                if last_row is None or date_val is None:
                    diag.exclude_reasons["invalid_row"] += 1
                    continue
                # latest_only: シンプルに setup==True のみで候補化（診断は最小限）
                if not bool(last_row.get("setup", False)):
                    diag.exclude_reasons["setup"] += 1
                    continue
                diag.setup_flag_true += 1
                diag.final_pass += 1
                roc200_val = _to_float(last_row.get("roc200"))
                close_val = _to_float(last_row.get("Close", 0))
                date_counter[date_val] = date_counter.get(date_val, 0) + 1
                rows.append(
                    {
                        "symbol": sym,
                        "date": date_val,
                        "roc200": roc200_val,
                        "close": 0.0 if math.isnan(close_val) else close_val,
                        "setup": bool(last_row.get("setup", False)),
                    }
                )
            else:
                df_all = pd.DataFrame(rows)
                try:
                    if target_date is not None:
                        df_all = df_all[df_all["date"] == target_date]
                    else:
                        mode_date = max(date_counter.items(), key=lambda kv: kv[1])[0]
                        df_all = df_all[df_all["date"] == mode_date]
                except Exception:
                    pass
                df_all = df_all.sort_values("roc200", ascending=False, kind="stable").head(
                    resolved_top_n
                )
                diag.final_top_n_count = len(df_all)
                diag.ranking_source = "latest_only"
                by_date: dict[pd.Timestamp, dict[str, dict]] = {}
                # If no candidates, emit 1-2 sample rows for quick triage (DEBUG-like)
                if diag.final_top_n_count == 0 and log_callback:
                    try:
                        samples: list[str] = []
                        count = 0
                        for s_sym, s_df in prepared_dict.items():
                            if s_df is None or getattr(s_df, "empty", True):
                                continue
                            try:
                                s_last = s_df.iloc[-1]
                                s_dt = pd.to_datetime(str(s_df.index[-1])).normalize()
                                s_setup = bool(s_last.get("setup", False))
                                s_roc = _to_float(s_last.get("roc200"))
                                samples.append(
                                    (
                                        f"{s_sym}: date={s_dt.date()} "
                                        f"setup={s_setup} roc200={s_roc:.4f}"
                                    )
                                )
                                count += 1
                                if count >= 2:
                                    break
                            except Exception:
                                continue
                        if samples:
                            log_callback(
                                ("System1: DEBUG latest_only 0 candidates. " + " | ".join(samples))
                            )
                    except Exception:
                        pass
                for dt_any, sub in df_all.groupby("date"):
                    assert isinstance(dt_any, pd.Timestamp)
                    dt = dt_any
                    payload: dict[str, dict] = {}
                    for rec in sub.to_dict("records"):
                        sym_val = rec.get("symbol")
                        if not sym_val:
                            continue
                        payload[str(sym_val)] = {
                            k: v for k, v in rec.items() if k not in {"symbol", "date"}
                        }
                    by_date[dt] = payload
                if log_callback:
                    candidate_count = sum(len(v) for v in by_date.values())
                    log_callback(
                        f"System1: latest_only -> {candidate_count} candidates "
                        f"(symbols={len(rows)})"
                    )
                out_df = df_all.copy()
                return finalize(by_date, out_df)
        except Exception as fast_err:
            if log_callback:
                log_callback(f"System1: fast-path failed -> fallback ({fast_err})")

    # Fallback: evaluate full history per date
    all_dates = sorted(
        {
            date
            for df in prepared_dict.values()
            if isinstance(df, pd.DataFrame) and not df.empty
            for date in df.index
        }
    )

    if not all_dates:
        diag.ranking_source = "latest_only" if latest_only else "full_scan"
        if log_callback:
            log_callback("System1: No valid dates found in data")
        return finalize({}, None)

    # mypy/py311 での var-annotated 誤検出を避けるため、明示型の空 dict を生成
    candidates_by_date: dict[pd.Timestamp, list[dict[str, object]]] = {}
    all_candidates: list[dict] = []

    diag_target_date = all_dates[-1]
    if log_callback:
        log_callback(f"System1: Generating candidates for {len(all_dates)} dates")

    for i, date in enumerate(all_dates):
        date_candidates: list[dict] = []

        for symbol, df in prepared_dict.items():
            if df is None or date not in df.index:
                continue
            try:
                row_obj = df.loc[date]
            except Exception:
                continue

            if isinstance(row_obj, pd.DataFrame):
                row_obj = row_obj.iloc[-1]
            if not isinstance(row_obj, pd.Series):
                continue
            row = cast(pd.Series, row_obj)

            if date == diag_target_date:
                diag.total_symbols += 1
                passed, flags, reason = system1_row_passes_setup(row, allow_fallback=False)
                pred_val = system1_setup_predicate(row)
                if pred_val:
                    diag.setup_predicate_count += 1
                setup_flag = bool(row.get("setup", False))
                if pred_val and not setup_flag:
                    diag.predicate_only_pass_count += 1
                    diag.mismatch_flag = 1
                if flags["filter_ok"]:
                    diag.filter_pass += 1
                if flags["setup_flag"]:
                    diag.setup_flag_true += 1
                if flags["fallback_ok"]:
                    diag.fallback_pass += 1
                if flags["roc200_positive"]:
                    diag.roc200_positive += 1
                if not passed:
                    if reason:
                        diag.exclude_reasons[reason] += 1
                    continue
                diag.final_pass += 1

            setup_flag = bool(row.get("setup", False))
            if not setup_flag:
                continue

            roc200_val = _to_float(row.get("roc200"))
            if pd.isna(roc200_val) or roc200_val <= 0:
                continue

            close_val = _to_float(row.get("Close", 0))
            sma200_val = _to_float(row.get("sma200", 0))

            date_candidates.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "roc200": roc200_val,
                    "close": 0.0 if math.isnan(close_val) else close_val,
                    "sma200": 0.0 if math.isnan(sma200_val) else sma200_val,
                }
            )

        if date_candidates:
            date_candidates.sort(key=lambda x: x["roc200"], reverse=True)
            top_candidates = date_candidates[:resolved_top_n]
            candidates_by_date[date] = top_candidates
            all_candidates.extend(top_candidates)

        if progress_callback and (i + 1) % max(1, len(all_dates) // 10) == 0:
            progress_callback(f"Processed {i + 1}/{len(all_dates)} dates")

    if all_candidates:
        candidates_df = pd.DataFrame(all_candidates)
        candidates_df["date"] = pd.to_datetime(candidates_df["date"])
        candidates_df = candidates_df.sort_values(["date", "roc200"], ascending=[True, False])
        last_date = max(candidates_by_date.keys()) if candidates_by_date else None
        if last_date is not None:
            diag.final_top_n_count = len(candidates_by_date.get(last_date, []))
        diag.ranking_source = "full_scan"
    else:
        candidates_df = None
        diag.ranking_source = "latest_only" if latest_only else "full_scan"

    if log_callback:
        total_candidates = len(all_candidates)
        unique_dates = len(candidates_by_date)
        log_callback(
            ("System1: Generated " f"{total_candidates} candidates across {unique_dates} dates")
        )

    return finalize(candidates_by_date, candidates_df)


def get_total_days_system1(data_dict: dict[str, pd.DataFrame]) -> int:
    """Get total days count for System1 data.

    Args:
        data_dict: Data dictionary

    Returns:
        Maximum day count
    """
    return int(get_total_days(data_dict))


def generate_roc200_ranking_system1(
    data_dict: dict[str, pd.DataFrame],
    date: str,
    top_n: int = 20,
    log_callback: Callable[[str], None] | None = None,
) -> list[dict]:
    """Generate ROC200-based ranking for a specific date.

    Args:
        data_dict: Dictionary of prepared data
        date: Target date (YYYY-MM-DD format)
        top_n: Number of top candidates to return
        log_callback: Optional logging callback

    Returns:
        List of candidate dictionaries with symbol, ROC200, and other metrics
    """
    if not data_dict:
        if log_callback:
            log_callback("System1: No data available for ranking")
        return []

    target_date = pd.to_datetime(date)
    candidates = []

    for symbol, df in data_dict.items():
        try:
            if df is None or target_date not in df.index:
                continue

            row = df.loc[target_date]

            # Check setup conditions
            setup_flag = bool(row.get("setup", False))
            if not setup_flag:
                continue

            # Get ROC200 value
            roc200_val = _to_float(row.get("roc200"))
            if pd.isna(roc200_val) or roc200_val <= 0:
                continue

            close_val = _to_float(row.get("Close", 0))
            sma200_val = _to_float(row.get("sma200", 0))

            candidates.append(
                {
                    "symbol": symbol,
                    "roc200": roc200_val,
                    "close": 0.0 if math.isnan(close_val) else close_val,
                    "sma200": 0.0 if math.isnan(sma200_val) else sma200_val,
                    "setup": setup_flag,
                }
            )

        except Exception:
            continue

    # Sort by ROC200 descending and take top_n
    candidates.sort(key=lambda x: cast(float, x["roc200"]), reverse=True)
    result = candidates[:top_n]

    if log_callback:
        log_callback(f"System1: Generated {len(result)} ROC200 candidates for {date}")

    return result


__all__ = [
    "prepare_data_vectorized_system1",
    "generate_candidates_system1",
    "get_total_days_system1",
    "generate_roc200_ranking_system1",
    "system1_row_passes_setup",
    "System1Diagnostics",
    "summarize_system1_diagnostics",
]
