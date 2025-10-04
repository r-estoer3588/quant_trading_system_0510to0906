"""System1 core logic (Long ROC200 momentum).

ROC200-based momentum strategy:
- Indicators: ROC200, SMA200, DollarVolume20 (precomputed only)
- Setup conditions: Close>5, DollarVolume20>25M, Close>SMA200, ROC200>0
- Candidate generation: ROC200 descending ranking by date, extract top_n
- Optimization: Removed all indicator calculations, using precomputed indicators only
"""

from __future__ import annotations

import math
import os
from collections import Counter
from collections.abc import Mapping, Callable
from dataclasses import dataclass, field
from typing import Any, cast

import pandas as pd

from common.batch_processing import process_symbols_batch
from common.system_common import check_precomputed_indicators, get_total_days
from common.system_constants import SYSTEM1_REQUIRED_INDICATORS

# --- Backward compatibility helpers for legacy direct tests ---
# Some tests (tests/test_system1_direct.py) expect internal helper functions
# that existed in the previous refactored version (see system1_backup.py).
# We reintroduce lightweight versions here without altering the new fast-path
# design. These are intentionally minimal and rely only on current imports.

REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


@dataclass(slots=True)
class System1Diagnostics:
    mode: str
    top_n: int | None = None
    symbols_total: int = 0
    symbols_with_data: int = 0
    total_symbols: int = 0
    filter_pass: int = 0
    setup_flag_true: int = 0
    fallback_pass: int = 0
    roc200_positive: int = 0
    final_pass: int = 0
    # --- Enrichment (ID8) -------------------------------------------------
    # predicate による True 判定の総数 (行ベース / latest_only ではシンボルベース)
    setup_predicate_count: int = 0
    # 最終 top_n に入った数 (latest_only では単一日候補数)
    final_top_n_count: int = 0
    # setup 列 False だが predicate True (ギャップ検知)
    predicate_only_pass_count: int = 0
    # mismatch_flag: 1 回でも列 vs predicate 不一致が観測されたら 1
    mismatch_flag: int = 0
    # ranking_source: 'latest_only' or 'full_scan'
    ranking_source: str | None = None
    exclude_reasons: Counter[str] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "mode": self.mode,
            "top_n": self.top_n,
            "symbols_total": self.symbols_total,
            "symbols_with_data": self.symbols_with_data,
            "total_symbols": self.total_symbols,
            "filter_pass": self.filter_pass,
            "setup_flag_true": self.setup_flag_true,
            "fallback_pass": self.fallback_pass,
            "roc200_positive": self.roc200_positive,
            "final_pass": self.final_pass,
            "setup_predicate_count": self.setup_predicate_count,
            "final_top_n_count": self.final_top_n_count,
            "predicate_only_pass_count": self.predicate_only_pass_count,
            "mismatch_flag": self.mismatch_flag,
            "ranking_source": self.ranking_source,
        }
        if self.exclude_reasons:
            data["exclude_reasons"] = dict(self.exclude_reasons)
        return data


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
    from common.cache_manager import CacheManager
    from config.settings import get_settings

    try:
        settings = get_settings()
        cache_mgr = CacheManager(settings)
        df = cache_mgr.load_base_cache(symbol)  # type: ignore[attr-defined]
        if df is None or df.empty:
            return symbol, None

        # Check for required indicators
        missing_indicators = [col for col in SYSTEM1_REQUIRED_INDICATORS if col not in df.columns]
        if missing_indicators:
            return symbol, None

        # Apply System1-specific filters and setup
        x = df.copy()

        # Filter: Close>=5, DollarVolume20>25M
        x["filter"] = (x["Close"] >= 5.0) & (x["dollarvolume20"] > 25_000_000)

        # Setup: Filter + Close>SMA200 + ROC200>0
        x["setup"] = x["filter"] & (x["Close"] > x["sma200"]) & (x["roc200"] > 0)

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
            if (os.environ.get("ENABLE_SUBSTEP_LOGS") or "").lower() in {"1", "true", "yes"}:
                log_callback(f"System1: {msg}")
        except Exception:
            pass

    _substep("enter prepare_data")
    # Fast path: reuse precomputed indicators
    if reuse_indicators and raw_data_dict:
        _substep("fast-path check start")
        try:
            # Early check - verify required indicators exist
            valid_data_dict, error_symbols = check_precomputed_indicators(
                raw_data_dict, SYSTEM1_REQUIRED_INDICATORS, "System1", skip_callback
            )

            if valid_data_dict:
                # Apply System1-specific filters
                prepared_dict = {}
                for symbol, df in valid_data_dict.items():
                    x = df.copy()

                    # Filter: Close>=5, DollarVolume20>25M
                    x["filter"] = (x["Close"] >= 5.0) & (x["dollarvolume20"] > 25_000_000)

                    # Setup: Filter + Close>SMA200 + ROC200>0
                    x["setup"] = x["filter"] & (x["Close"] > x["sma200"]) & (x["roc200"] > 0)

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

    return results

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
    **kwargs: object,
) -> tuple[dict, pd.DataFrame | None, dict[str, object]]:
    """System1 candidate generation (ROC200 descending ranking).

    Returns a tuple of (per-date candidates, merged dataframe, diagnostics).
    """

    if not isinstance(prepared_dict, dict) or not prepared_dict:
        mode = "latest_only" if latest_only else "full_scan"
        diag = System1Diagnostics(mode=mode, top_n=top_n)
        diag.symbols_total = len(prepared_dict or {})
        diag.ranking_source = mode  # 明示的に設定
        if log_callback:
            log_callback("System1: No data provided for candidate generation")
        return {}, None, diag.as_dict()

    if top_n is None:
        top_n = 20

    mode = "latest_only" if latest_only else "full_scan"
    diagnostics = System1Diagnostics(mode=mode, top_n=top_n)
    diagnostics.symbols_total = len(prepared_dict)
    diagnostics.symbols_with_data = sum(
        1 for df in prepared_dict.values() if isinstance(df, pd.DataFrame) and not df.empty
    )

    # Fast path: evaluate only the most recent bar per symbol
    if latest_only:
        try:
            rows: list[dict] = []
            date_counter: dict[pd.Timestamp, int] = {}
            for sym, df in prepared_dict.items():
                if df is None or df.empty:
                    continue
                diagnostics.total_symbols += 1
                last_row = df.iloc[-1]
                passed, flags, reason = system1_row_passes_setup(last_row)
                # 共通 predicate による再評価（fallback を含めない純粋な setup 条件）
                from common.system_setup_predicates import (
                    system1_setup_predicate as _s1_pred,
                )

                pred_val = _s1_pred(last_row)
                if pred_val:
                    diagnostics.setup_predicate_count += 1
                # setup 列 False だが predicate True (System1 では原則稀)
                if pred_val and not bool(last_row.get("setup", False)):
                    diagnostics.predicate_only_pass_count += 1
                    diagnostics.mismatch_flag = 1
                if flags["filter_ok"]:
                    diagnostics.filter_pass += 1
                if flags["setup_flag"]:
                    diagnostics.setup_flag_true += 1
                if flags["fallback_ok"]:
                    diagnostics.fallback_pass += 1
                if flags["roc200_positive"]:
                    diagnostics.roc200_positive += 1
                if not passed:
                    if reason:
                        diagnostics.exclude_reasons[reason] += 1
                    continue
                diagnostics.final_pass += 1
                roc200_val = last_row.get("roc200", 0)
                date_val = df.index[-1]
                date_counter[date_val] = date_counter.get(date_val, 0) + 1
                rows.append(
                    {
                        "symbol": sym,
                        "date": date_val,
                        "roc200": roc200_val,
                        "close": last_row.get("Close", 0),
                        "setup": bool(last_row.get("setup", False)),
                    }
                )

            if not rows:
                if log_callback:
                    log_callback(
                        "System1: latest_only fast-path produced 0 rows (after gating) — fallback"
                    )
                # fall through to full scan
            else:
                df_all = pd.DataFrame(rows)
                try:
                    mode_date = max(date_counter.items(), key=lambda kv: kv[1])[0]
                    df_all = df_all[df_all["date"] == mode_date]
                except Exception:
                    pass
                df_all = df_all.sort_values("roc200", ascending=False, kind="stable").head(top_n)
                diagnostics.final_top_n_count = len(df_all)
                diagnostics.ranking_source = "latest_only"
                by_date: dict[pd.Timestamp, dict[str, dict]] = {}
                for dt, sub in df_all.groupby("date"):
                    payload: dict[str, dict] = {}
                    for rec in sub.to_dict("records"):
                        sym = rec.get("symbol")
                        if not sym:
                            continue
                        payload[str(sym)] = {
                            k: v for k, v in rec.items() if k not in {"symbol", "date"}
                        }
                    by_date[dt] = payload
                if log_callback:
                    candidate_count = sum(len(v) for v in by_date.values())
                    log_callback(
                        f"System1: latest_only -> {candidate_count} candidates (symbols={len(rows)})"
                    )
                out_df = df_all.copy()
                return by_date, out_df, diagnostics.as_dict()
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
        diagnostics.ranking_source = "latest_only" if latest_only else "full_scan"
        if log_callback:
            log_callback("System1: No valid dates found in data")
        return {}, None, diagnostics.as_dict()

    candidates_by_date: dict[pd.Timestamp, list[dict]] = {}
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
                row = df.loc[date]
            except Exception:
                continue

            if date == diag_target_date:
                diagnostics.total_symbols += 1
                passed, flags, reason = system1_row_passes_setup(row, allow_fallback=False)
                # 共通 predicate による正式 predicate（fallback 無し）
                from common.system_setup_predicates import (
                    system1_setup_predicate as _s1_pred,
                )

                pred_val = _s1_pred(row)
                if pred_val:
                    diagnostics.setup_predicate_count += 1
                if pred_val and not bool(row.get("setup", False)):
                    diagnostics.predicate_only_pass_count += 1
                    diagnostics.mismatch_flag = 1
                if flags["filter_ok"]:
                    diagnostics.filter_pass += 1
                if flags["setup_flag"]:
                    diagnostics.setup_flag_true += 1
                if flags["fallback_ok"]:
                    diagnostics.fallback_pass += 1
                if flags["roc200_positive"]:
                    diagnostics.roc200_positive += 1
                if not passed:
                    if reason:
                        diagnostics.exclude_reasons[reason] += 1
                    continue
                if passed:
                    diagnostics.final_pass += 1

            if not row.get("setup", False):
                continue

            roc200_val = row.get("roc200", 0)
            if pd.isna(roc200_val) or roc200_val <= 0:
                continue

            date_candidates.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "roc200": roc200_val,
                    "close": row.get("Close", 0),
                }
            )

        if date_candidates:
            date_candidates.sort(key=lambda x: x["roc200"], reverse=True)
            top_candidates = date_candidates[:top_n]
            candidates_by_date[date] = top_candidates
            all_candidates.extend(top_candidates)

        if progress_callback and (i + 1) % max(1, len(all_dates) // 10) == 0:
            progress_callback(f"Processed {i + 1}/{len(all_dates)} dates")

    if all_candidates:
        candidates_df = pd.DataFrame(all_candidates)
        candidates_df["date"] = pd.to_datetime(candidates_df["date"])
        candidates_df = candidates_df.sort_values(["date", "roc200"], ascending=[True, False])
        # 最終日の top_n 数を格納
        last_date = max(candidates_by_date.keys()) if candidates_by_date else None
        if last_date is not None:
            diagnostics.final_top_n_count = len(candidates_by_date.get(last_date, []))
        diagnostics.ranking_source = "full_scan"
    else:
        candidates_df = None
        # full scan で候補がない場合も ranking_source を設定
        diagnostics.ranking_source = "latest_only" if latest_only else "full_scan"

    if log_callback:
        total_candidates = len(all_candidates)
        unique_dates = len(candidates_by_date)
        log_callback(
            f"System1: Generated {total_candidates} candidates across {unique_dates} dates"
        )

    return candidates_by_date, candidates_df, diagnostics.as_dict()


def get_total_days_system1(data_dict: dict[str, pd.DataFrame]) -> int:
    """Get total days count for System1 data.

    Args:
        data_dict: Data dictionary

    Returns:
        Maximum day count
    """
    return get_total_days(data_dict)


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
            if not row.get("setup", False):
                continue

            # Get ROC200 value
            roc200_val = row.get("roc200", 0)
            if pd.isna(roc200_val) or roc200_val <= 0:
                continue

            candidates.append(
                {
                    "symbol": symbol,
                    "roc200": float(roc200_val),
                    "close": float(row.get("Close", 0)),
                    "sma200": float(row.get("sma200", 0)),
                    "setup": bool(row.get("setup", False)),
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
