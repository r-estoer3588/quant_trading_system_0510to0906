"""System1 core logic (Long ROC200 momentum).

ROC200-based momentum strategy:
- Indicators: ROC200, SMA200, DollarVolume20 (precomputed only)
- Setup conditions: Close>5, DollarVolume20>50M, Close>SMA200, ROC200>0
- Candidate generation: ROC200 descending ranking by date, extract top_n
- Optimization: Removed all indicator calculations, using precomputed indicators only
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
from typing import Any, Callable, DefaultDict, Mapping, Optional, cast

import pandas as pd

from common.batch_processing import process_symbols_batch
from common.system_common import check_precomputed_indicators, get_total_days
from common.system_constants import REQUIRED_COLUMNS
from common.system_setup_predicates import (
    system1_setup_predicate,
    validate_predicate_equivalence,
)
from strategies.constants import STOP_ATR_MULTIPLE_SYSTEM1


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
    ranked_top_n_count: int = 0
    predicate_only_pass_count: int = 0
    mismatch_flag: int = 0
    date_fallback_count: int = 0

    # ranking source marker
    ranking_source: str | None = None

    # reason histogram
    exclude_reasons: DefaultDict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )

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
            "ranked_top_n_count": int(self.ranked_top_n_count),
            "predicate_only_pass_count": int(self.predicate_only_pass_count),
            "mismatch_flag": int(self.mismatch_flag),
            "date_fallback_count": int(self.date_fallback_count),
            "ranking_source": self.ranking_source,
            "exclude_reasons": {
                k: int(v) for k, v in dict(self.exclude_reasons).items()
            },
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
    """System1 setup evaluation using SMA trend and ROC200.

    Conditions:
    - SMA trend: SMA25 > SMA50 (individual stock condition)
    - ROC200 > 0 (momentum confirmation)

    Note: Market condition (SPY > SMA100) is checked at orchestrator level,
    not within this function. Phase 2 filter (Price>=5, DV20>=50M) is assumed
    to have already passed for rows reaching this function.

    Args:
        row: DataFrame row containing indicators
        allow_fallback: Legacy parameter for backward compatibility (ignored)

    Returns:
        (passes, flags, reason) tuple where:
        - passes: True if all conditions met
        - flags: dict with individual condition results
        - reason: exclusion reason if passes is False
    """
    # SMA trend condition (正式なセットアップ条件)
    sma25 = _to_float(row.get("sma25"))
    sma50 = _to_float(row.get("sma50"))
    sma_trend_ok = not math.isnan(sma25) and not math.isnan(sma50) and sma25 > sma50

    # ROC200 momentum condition
    roc200_val = _to_float(row.get("roc200"))
    roc200_positive = not math.isnan(roc200_val) and roc200_val > 0

    # Combined evaluation
    passes = sma_trend_ok and roc200_positive
    reason: str | None = None
    if not sma_trend_ok:
        reason = "sma_trend"
    elif not roc200_positive:
        reason = "roc200"

    flags = {
        "sma_trend_ok": sma_trend_ok,
        "roc200_positive": roc200_positive,
        # Legacy keys for backward compatibility
        "setup_flag": sma_trend_ok,
        "fallback_ok": False,  # No longer used
        "filter_ok": True,  # Phase 2 filter already passed
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
        x["filter"] = pd.Series(False, index=x.index, dtype=bool)
        x["setup"] = pd.Series(False, index=x.index, dtype=bool)
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

        # Filter: Close>=5, DollarVolume20>50M
        try:
            filt = (x["Close"] >= 5.0) & (
                x.get(
                    "dollarvolume20",
                    pd.Series(index=x.index, dtype=float),
                )
                > 50_000_000
            )
        except Exception:
            # If columns not available, mark filter False
            filt = pd.Series(False, index=x.index, dtype=bool)
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
            setup = pd.Series(False, index=x.index, dtype=bool)
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
            # 型安全な環境変数アクセスに統一
            from config.environment import get_env_config  # 遅延importで循環回避

            if getattr(get_env_config(), "enable_substep_logs", False):
                log_callback(f"System1: {msg}")
        except Exception:
            # 失敗時は静かに無視（従来挙動と同等の安全側）
            pass

    _substep("enter prepare_data")
    # 軽量化ヒント: latest_only のときは最小列 + 末尾数行のみを処理
    latest_only = bool(_kwargs.get("latest_only", False))
    # Fast path: reuse precomputed indicators
    if reuse_indicators and raw_data_dict:
        _substep("fast-path check start")
        try:
            # latest_only の場合は必須列チェックを緩め、末尾数行 + 最小列に絞る
            if latest_only:
                prepared_dict: dict[str, pd.DataFrame] = {}
                minimal_cols = {
                    # OHLCV 最小限（フォールバック計算や整合のため）
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    # 参照指標
                    "sma25",
                    "sma50",
                    "sma200",
                    "roc200",
                    "atr20",
                    "dollarvolume20",
                }

                # OHLC 不足時にキャッシュから補完するヘルパ
                def _augment_ohlc_if_missing(
                    sym: str, df_in: pd.DataFrame
                ) -> pd.DataFrame:
                    x2 = df_in
                    try:
                        need_any = any(
                            col not in x2.columns
                            for col in ("Open", "High", "Low", "Close")
                        )
                        if not need_any:
                            return x2
                        # 遅延 import（循環回避）
                        from common.cache_manager import CacheManager, load_base_cache
                        from config.settings import get_settings

                        try:
                            settings = get_settings()
                            cache_mgr = CacheManager(settings)
                            base_df = load_base_cache(sym, cache_manager=cache_mgr)
                        except Exception:
                            base_df = None
                        if base_df is None or getattr(base_df, "empty", True):
                            return x2

                        base_df = _rename_ohlcv(base_df)
                        base_df = _normalize_index(base_df)
                        cols = [
                            c
                            for c in ("Open", "High", "Low", "Close")
                            if c in getattr(base_df, "columns", [])
                        ]
                        if not cols:
                            return x2
                        price_df = base_df.loc[:, cols].copy()
                        # 既存列は優先し、欠損はキャッシュ値で埋める
                        for col in ("Open", "High", "Low", "Close"):
                            if col in price_df.columns:
                                if col in x2.columns:
                                    try:
                                        x2[col] = x2[col].fillna(price_df[col])
                                    except Exception:
                                        # インデックス不一致時は結合してから埋める
                                        try:
                                            joined = x2.join(
                                                price_df[[col]], how="left"
                                            )
                                            x2[col] = joined[col]
                                        except Exception:
                                            pass
                                else:
                                    try:
                                        joined = x2.join(price_df[[col]], how="left")
                                        x2[col] = joined[col]
                                    except Exception:
                                        # インデックス不一致があっても諦めない
                                        try:
                                            x2[col] = price_df[col]
                                        except Exception:
                                            pass
                    except Exception:
                        # 失敗しても静かに元の df を返す
                        return x2
                    return x2

                for symbol, df in raw_data_dict.items():
                    try:
                        if df is None or getattr(df, "empty", True):
                            continue
                        x = df.copy(deep=False)

                        # OHLCV 列名を正規化（小文字 → PascalCase）
                        x = _rename_ohlcv(x)

                        # 日付インデックス正規化（'Date' だけでなく 'date' も考慮）
                        # 既存のヘルパーを利用して安全に正規化する。
                        try:
                            x = _normalize_index(x)
                        except Exception:
                            # フォールバック（万一の安全策）
                            try:
                                if "Date" in x.columns:
                                    idx = pd.to_datetime(
                                        x["Date"], errors="coerce"
                                    ).dt.normalize()
                                    x.index = pd.Index(idx, name="Date")
                                elif "date" in x.columns:
                                    idx = pd.to_datetime(
                                        x["date"], errors="coerce"
                                    ).dt.normalize()
                                    x.index = pd.Index(idx, name="Date")
                                else:
                                    x.index = pd.to_datetime(
                                        x.index, errors="coerce"
                                    ).normalize()
                                x = x[~x.index.isna()]
                                x = x.sort_index()
                                if getattr(x.index, "has_duplicates", False):
                                    x = x[~x.index.duplicated(keep="last")]
                            except Exception:
                                pass

                        # OHLC が欠けている場合はキャッシュから補完
                        missing_ohlc = [
                            c
                            for c in ("Open", "High", "Low", "Close")
                            if c not in x.columns
                        ]
                        if missing_ohlc:
                            try:
                                if log_callback and symbol in ("SPY", "A"):
                                    log_callback(
                                        f"[DEBUG_PREPARED] {symbol}: missing OHLC "
                                        f"before augment => {missing_ohlc}"
                                    )
                            except Exception:
                                pass
                            x = _augment_ohlc_if_missing(symbol, x)

                        # 列の最小化（存在するもののみ残す）— 補完後に実行
                        keep_cols = [c for c in minimal_cols if c in x.columns]
                        if keep_cols:
                            x = x.loc[:, keep_cols].copy()

                        # 数値変換（必要列のみ）
                        for col in (
                            "Close",
                            "Low",
                            "High",
                            "Open",
                            "sma25",
                            "sma50",
                            "sma200",
                            "roc200",
                            "atr20",
                            "dollarvolume20",
                        ):
                            if col in x.columns:
                                try:
                                    x[col] = pd.to_numeric(x[col], errors="coerce")
                                except Exception:
                                    pass

                        # 末尾数行だけ残す（前日/当日推定に十分）
                        x = x.tail(3).copy()
                        if x.empty:
                            continue

                        # 軽量 filter/setup 付与
                        try:
                            dv = x.get(
                                "dollarvolume20",
                                pd.Series(index=x.index, dtype=float),
                            )
                            close_s = x.get(
                                "Close", pd.Series(index=x.index, dtype=float)
                            )
                            filt = (close_s >= 5.0) & (dv > 50_000_000)
                        except Exception:
                            filt = pd.Series(
                                False,
                                index=x.index,
                                dtype=bool,
                            )  # type: ignore[assignment]
                        x["filter"] = filt

                        try:
                            close_s = x.get(
                                "Close", pd.Series(index=x.index, dtype=float)
                            )
                            sma200_s = x.get(
                                "sma200", pd.Series(index=x.index, dtype=float)
                            )
                            roc_ok = (
                                x.get(
                                    "roc200",
                                    pd.Series(index=x.index, dtype=float),
                                )
                                > 0
                            )
                            setup = filt & (close_s > sma200_s) & roc_ok
                        except Exception:
                            setup = pd.Series(
                                False,
                                index=x.index,
                                dtype=bool,
                            )  # type: ignore[assignment]
                        x["setup"] = setup

                        # DEBUG: prepared_dict 格納前に Close/Open 値を確認
                        if log_callback and symbol in ("SPY", "A"):
                            try:
                                has_close = "Close" in x.columns
                                has_open = "Open" in x.columns
                                close_val = None
                                open_val = None
                                if has_close and len(x) > 0:
                                    close_val = x["Close"].iloc[-1]
                                if has_open and len(x) > 0:
                                    open_val = x["Open"].iloc[-1]
                                msg = (
                                    f"[DEBUG_PREPARED] {symbol}: "
                                    f"Close={close_val} Open={open_val} shape={x.shape}"
                                )
                                log_callback(msg)
                            except Exception as e:
                                log_callback(f"[DEBUG_PREPARED] {symbol}: ERROR={e}")

                        prepared_dict[symbol] = x
                    except Exception:
                        continue

                _substep(
                    f"fast-path (latest_only) processed symbols={len(prepared_dict)}"
                )

                # predicate 検証は検証フラグが有効なときのみ（速度最優先）
                try:
                    from config.environment import get_env_config

                    if getattr(get_env_config(), "validate_setup_predicate", False):
                        validate_predicate_equivalence(
                            prepared_dict, "System1", log_fn=log_callback
                        )
                except Exception:
                    pass

                return prepared_dict

            # 通常 fast-path: precomputed インジケーター前提で安全にチェック
            valid_data_dict, error_symbols = check_precomputed_indicators(
                raw_data_dict, ["roc200"], "System1", skip_callback
            )

            if valid_data_dict:
                prepared_dict = {}
                for symbol, df in valid_data_dict.items():
                    x = df.copy()

                    # Ensure date index (if not already set)
                    if "date" in x.columns and not isinstance(
                        x.index, pd.DatetimeIndex
                    ):
                        try:
                            x["date"] = pd.to_datetime(x["date"])
                            x = x.set_index("date", drop=False)
                        except Exception:
                            pass  # Keep original index if conversion fails

                    # Filter: Close>=5, DollarVolume20>50M（欠損は安全側に False）
                    try:
                        dv = x.get(
                            "dollarvolume20", pd.Series(index=x.index, dtype=float)
                        )
                        filt = (x["Close"] >= 5.0) & (dv > 50_000_000)
                    except Exception:
                        filt = pd.Series(
                            False,
                            index=x.index,
                            dtype=bool,
                        )  # type: ignore[assignment]
                    x["filter"] = filt

                    # Setup: Filter + Close>SMA200 + ROC200>0（欠損は False）
                    try:
                        close_s = x.get("Close", pd.Series(index=x.index, dtype=float))
                        sma200_s = x.get(
                            "sma200", pd.Series(index=x.index, dtype=float)
                        )
                        roc_ok = (
                            x.get("roc200", pd.Series(index=x.index, dtype=float)) > 0
                        )
                        setup = filt & (close_s > sma200_s) & roc_ok
                    except Exception:
                        setup = pd.Series(
                            False,
                            index=x.index,
                            dtype=bool,
                        )  # type: ignore[assignment]
                    x["setup"] = setup

                    prepared_dict[symbol] = x

                _substep(f"fast-path processed symbols={len(prepared_dict)}")

                # Validate setup column vs predicate equivalence
                validate_predicate_equivalence(
                    prepared_dict, "System1", log_fn=log_callback
                )

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
            log_callback(
                "System1: Starting normal processing for "
                f"{len(target_symbols)} symbols"
            )
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

    # Validate setup column vs predicate equivalence（環境設定が有効な時のみ）
    try:
        from config.environment import get_env_config

        if getattr(get_env_config(), "validate_setup_predicate", False):
            validate_predicate_equivalence(
                typed_results, "System1", log_fn=log_callback
            )
    except Exception:
        pass

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
        1
        for df in prepared_dict.values()
        if isinstance(df, pd.DataFrame) and not df.empty
    )

    # Fast path: evaluate only the most recent bar per symbol
    if latest_only:
        if log_callback:
            log_callback(
                f"[DEBUG_S1_LATEST] Entering latest_only block, "
                f"prepared_dict keys={len(prepared_dict)}"
            )
        try:
            rows: list[dict] = []
            date_counter: dict[pd.Timestamp, int] = {}
            # Optional: orchestrator may specify a target trading date fallback
            target_date = None
            # Optional max lag (calendar days). If provided and latest bar is older than
            # target_date by more than this lag, the symbol will be skipped.
            raw_lag = kwargs.get("max_date_lag_days", None)
            max_lag_days: Optional[int] = None
            try:
                if raw_lag is not None:
                    max_lag_days = int(str(raw_lag).strip())
            except Exception:
                max_lag_days = None
            if (
                isinstance(max_lag_days, int)
                and max_lag_days is not None
                and max_lag_days < 0
            ):
                max_lag_days = 0
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
            # Date lag override accepted but currently unused
            # (kept for API compatibility)

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

            # trading-day lag helper removed (used to call exchange calendars).
            # Calendar-day comparison is enough for our lag<=1 fallback.

            # Predicate validation toggle (avoid duplicate evaluation in production)
            validate_predicates = False
            try:
                from config.environment import get_env_config

                env = get_env_config()
                validate_predicates = bool(
                    getattr(env, "validate_setup_predicate", False)
                )
            except Exception:
                # If environment config is not available, keep default False
                validate_predicates = False

            for sym, df in prepared_dict.items():
                if df is None or df.empty:
                    continue
                diag.total_symbols += 1
                if log_callback and diag.total_symbols <= 5:
                    log_callback(
                        f"[DEBUG_S1_LOOP] Processing symbol {sym}, "
                        f"total_symbols={diag.total_symbols}"
                    )
                row_obj: pd.Series | pd.DataFrame | None
                date_val: pd.Timestamp | None
                row_obj = None
                date_val = None
                fallback_used = False
                # latest index guard (e.g., SPY cache tail issues)
                try:
                    if getattr(df.index, "size", 0) == 0 or df.index[-1] is None:
                        diag.exclude_reasons["latest_index_missing"] += 1
                        continue
                except Exception:
                    diag.exclude_reasons["latest_index_missing"] += 1
                    continue
                if target_date is not None:
                    if target_date in df.index:
                        row_obj = df.loc[target_date]
                        date_val = target_date
                    else:
                        latest_idx_raw = df.index[-1]
                        try:
                            latest_idx_norm = pd.Timestamp(
                                str(latest_idx_raw)
                            ).normalize()
                        except Exception:
                            diag.exclude_reasons["invalid_date"] += 1
                            continue

                        # Fast path: if dates equal, accept immediately
                        if latest_idx_norm == target_date:
                            try:
                                row_obj = df.loc[latest_idx_raw]
                            except Exception:
                                row_obj = None
                            date_val = latest_idx_norm
                            fallback_used = True
                            diag.date_fallback_count += 1
                        else:
                            # Unconditional fallback to latest available bar.
                            # Final selection by date is handled after the loop
                            # (target_date → mode_date among collected rows).
                            try:
                                row_obj = df.loc[latest_idx_raw]
                            except Exception:
                                row_obj = None
                            date_val = latest_idx_norm
                            fallback_used = True
                            diag.date_fallback_count += 1
                else:
                    latest_idx_raw = df.index[-1]
                    try:
                        date_val = pd.Timestamp(str(latest_idx_raw)).normalize()
                    except Exception:
                        diag.exclude_reasons["invalid_date"] += 1
                        continue
                    try:
                        row_obj = df.loc[latest_idx_raw]
                    except Exception:
                        row_obj = None

                last_row = _ensure_series(row_obj)
                if last_row is None or date_val is None:
                    diag.exclude_reasons["invalid_row"] += 1
                    continue

                # Staleness guard (calendar-day based): when target_date is defined and
                # max_lag_days is provided, drop symbols whose latest bar is too old.
                try:
                    if (
                        target_date is not None
                        and isinstance(max_lag_days, int)
                        and max_lag_days >= 0
                        and date_val < target_date
                    ):
                        delta_days = int(
                            (pd.Timestamp(target_date) - pd.Timestamp(date_val)).days
                        )
                        if delta_days > max_lag_days:
                            diag.exclude_reasons["too_stale"] += 1
                            continue
                except Exception:
                    # On any failure, do not exclude based on staleness.
                    pass

                # Primary evaluation via predicate (single source)
                res_pred = system1_setup_predicate(last_row, return_reason=True)
                if isinstance(res_pred, tuple):
                    pred_ok, pred_reason = res_pred
                else:
                    pred_ok, pred_reason = bool(res_pred), None
                if not pred_ok:
                    if pred_reason:
                        diag.exclude_reasons[str(pred_reason)] += 1
                    else:
                        diag.exclude_reasons["predicate_failed"] += 1
                    continue

                # Count Phase2 filter pass (Close>=5 & DollarVolume20>=50M)
                try:
                    close_v = _to_float(last_row.get("Close"))
                    dv20_v = _to_float(last_row.get("dollarvolume20"))
                    if (
                        not math.isnan(close_v)
                        and not math.isnan(dv20_v)
                        and close_v >= 5.0
                        and dv20_v >= 50_000_000
                    ):
                        diag.filter_pass += 1
                except Exception:
                    pass

                # Optional legacy check (diagnostics only)
                try:
                    if validate_predicates:
                        passed_legacy, flags, legacy_reason = system1_row_passes_setup(
                            last_row, allow_fallback=True
                        )
                        if bool(passed_legacy) != bool(pred_ok):
                            diag.exclude_reasons["_predicate_mismatch"] += 1
                except Exception:
                    # do not block on diagnostics
                    pass

                # Update diagnostics counters
                # For compatibility, infer flags from row directly when available
                try:
                    sma25_v = _to_float(last_row.get("sma25"))
                    sma50_v = _to_float(last_row.get("sma50"))
                    if (
                        not math.isnan(sma25_v)
                        and not math.isnan(sma50_v)
                        and sma25_v > sma50_v
                    ):
                        diag.setup_flag_true += 1
                except Exception:
                    pass
                try:
                    roc200_v = _to_float(last_row.get("roc200"))
                    if not math.isnan(roc200_v) and roc200_v > 0:
                        diag.roc200_positive += 1
                except Exception:
                    pass

                diag.setup_predicate_count += 1
                diag.final_pass += 1
                roc200_val = _to_float(last_row.get("roc200"))
                close_val = _to_float(last_row.get("Close", 0))
                atr20_val = _to_float(last_row.get("atr20", 0))

                # エントリー価格とストップ価格の計算
                # System1: 翌日寄り付きで買い、損切りは買値 - 5*ATR20
                entry_price = close_val if close_val > 0 else 0.0
                stop_price = (
                    entry_price - (STOP_ATR_MULTIPLE_SYSTEM1 * atr20_val)
                    if (entry_price > 0 and atr20_val > 0)
                    else 0.0
                )

                # fallback時は target_date ラベルで集計（フィルタ時に消えないように）
                raw_label_dt = (
                    target_date
                    if (fallback_used and (target_date is not None))
                    else date_val
                )
                # 異常年（例: 8237年）や NaT を除去する安全サニタイズ

                def _sanitize_signal_date(
                    dt_obj: object, fallback: pd.Timestamp | None
                ) -> pd.Timestamp | None:
                    try:
                        ts = pd.Timestamp(str(dt_obj)).normalize()
                    except Exception:
                        return fallback
                    if pd.isna(ts):
                        return fallback
                    y = int(getattr(ts, "year", 0) or 0)
                    if y < 1900 or y > 2262:
                        return fallback
                    return ts

                label_dt = _sanitize_signal_date(raw_label_dt, fallback=date_val)
                if label_dt is None:
                    # ラベル日付が解決できない場合は除外（診断カウント）
                    try:
                        diag.exclude_reasons["invalid_date_label"] += 1
                    except Exception:
                        pass
                    continue
                # 明示エントリー日（翌営業日）
                try:
                    from common.utils_spy import (
                        resolve_signal_entry_date as _resolve_entry,
                    )

                    entry_dt = _resolve_entry(label_dt)
                except Exception:
                    entry_dt = None
                date_counter[label_dt] = date_counter.get(label_dt, 0) + 1

                # ATR20 を配分計算用に保持
                atr20_val = 0.0
                try:
                    atr20_raw = last_row.get("atr20")
                    if atr20_raw is not None and not math.isnan(float(atr20_raw)):
                        atr20_val = float(atr20_raw)
                except Exception:
                    pass

                row_dict = {
                    "symbol": sym,
                    "date": label_dt,
                    "entry_date": entry_dt,
                    "roc200": roc200_val,
                    "close": 0.0 if math.isnan(close_val) else close_val,
                    "entry_price": entry_price,
                    "stop_price": stop_price,
                    "atr20": atr20_val,
                    "setup": bool(last_row.get("setup", False)),
                }
                rows.append(row_dict)

            # After latest_only loop: create DataFrame and rank
            if log_callback:
                log_callback(f"[DEBUG_S1] Collected rows={len(rows)}")
            if len(rows) == 0:
                diag.ranking_source = "latest_only_empty"
                return finalize({}, None)

            df_all = pd.DataFrame(rows)
            df_all_original = df_all.copy()
            if log_callback:
                log_callback(
                    f"[DEBUG_S1] df_all created: {len(df_all)} rows, "
                    f"columns={list(df_all.columns)}"
                )
                # Log date statistics
                if "date" in df_all.columns and len(df_all) > 0:
                    try:
                        date_sample = df_all["date"].head(5).tolist()
                        log_callback(
                            f"[DEBUG_S1] date_sample={date_sample}, "
                            f"target_date={target_date}"
                        )
                    except Exception:
                        pass

            # Sanitize date column
            try:
                df_all["date"] = pd.to_datetime(
                    df_all["date"], errors="coerce"
                ).dt.normalize()
                df_all = df_all.dropna(subset=["date"]).copy()
            except Exception:
                pass

            # Filter by target_date or most frequent date
            try:
                filtered = df_all
                final_label_date: pd.Timestamp | None = None
                if target_date is not None:
                    filtered = df_all[df_all["date"] == target_date]
                    final_label_date = target_date
                    if log_callback:
                        log_callback(
                            f"[DEBUG_S1_FILTER] target_date={target_date}, "
                            f"filtered={len(filtered)} from df_all={len(df_all)}"
                        )
                    if filtered.empty and len(df_all) > 0:
                        try:
                            mode_date = max(date_counter.items(), key=lambda kv: kv[1])[
                                0
                            ]
                        except Exception:
                            mode_date = None
                        if mode_date is not None:
                            filtered = df_all[df_all["date"] == mode_date]
                            final_label_date = mode_date
                        if filtered.empty:
                            tmp = df_all.copy()
                            tmp.loc[:, "date"] = target_date
                            try:
                                from common.utils_spy import (
                                    resolve_signal_entry_date as _resolve_entry_dt,
                                )

                                tmp.loc[:, "entry_date"] = _resolve_entry_dt(
                                    target_date
                                )
                            except Exception:
                                tmp.loc[:, "entry_date"] = target_date
                            filtered = tmp
                            final_label_date = target_date
                else:
                    mode_date = max(date_counter.items(), key=lambda kv: kv[1])[0]
                    filtered = df_all[df_all["date"] == mode_date]
                    final_label_date = mode_date
            except Exception:
                filtered = df_all
                final_label_date = None

            # Rank by ROC200 and take top N
            ranked = filtered.sort_values(
                "roc200", ascending=False, kind="stable"
            ).copy()
            top_cut = ranked.head(resolved_top_n)

            # Top-off:補完
            missing = max(0, resolved_top_n - len(top_cut))
            if log_callback:
                log_callback(
                    f"[DEBUG_S1_TOPOFF] filtered={len(filtered)} "
                    f"top_cut={len(top_cut)} missing={missing} "
                    f"df_all_orig={len(df_all_original)}"
                )
            if missing > 0 and len(df_all_original) > 0:
                try:
                    exists = (
                        set(top_cut["symbol"].astype(str))
                        if not top_cut.empty
                        else set()
                    )
                    extras_pool = (
                        df_all_original.sort_values(
                            "roc200", ascending=False, kind="stable"
                        )
                        .loc[~df_all_original["symbol"].astype(str).isin(exists)]
                        .copy()
                    )
                    if not extras_pool.empty:
                        if final_label_date is None:
                            try:
                                final_label_date = (
                                    target_date
                                    if target_date is not None
                                    else max(
                                        date_counter.items(), key=lambda kv: kv[1]
                                    )[0]
                                )
                            except Exception:
                                final_label_date = None
                        if final_label_date is not None:
                            extras_pool.loc[:, "date"] = final_label_date
                            try:
                                from common.utils_spy import (
                                    resolve_signal_entry_date as _resolve_entry_dt2,
                                )

                                extras_pool.loc[:, "entry_date"] = _resolve_entry_dt2(
                                    final_label_date
                                )
                            except Exception:
                                extras_pool.loc[:, "entry_date"] = final_label_date
                        extras_take = extras_pool.head(missing)
                        top_cut = (
                            pd.concat([top_cut, extras_take], ignore_index=True)
                            .drop_duplicates(subset=["symbol"], keep="first")
                            .head(resolved_top_n)
                        )
                except Exception:
                    pass

            df_all = top_cut
            diag.ranked_top_n_count = len(df_all)
            diag.ranking_source = "latest_only"

            # Build by_date structure
            by_date: dict[pd.Timestamp, dict[str, dict]] = {}
            for _, row in df_all.iterrows():
                dt = row.get("date")
                if pd.isna(dt):
                    continue
                dt_norm = pd.Timestamp(str(dt)).normalize()
                if dt_norm not in by_date:
                    by_date[dt_norm] = {}
                sym = str(row.get("symbol", ""))
                by_date[dt_norm][sym] = {
                    "symbol": sym,
                    "date": dt_norm,
                    "entry_date": row.get("entry_date"),
                    "roc200": row.get("roc200", 0.0),
                    "close": row.get("close", 0.0),
                    "setup": bool(row.get("setup", False)),
                }

            return finalize(by_date, df_all)

        except Exception as e_latest:
            if log_callback:
                log_callback(f"System1 latest_only error: {e_latest}")
            diag.ranking_source = "error_latest"
            return finalize({}, None)

    # Original else block (latest_only=False) is now unreachable because we always
    # use latest_only=True in production. Keeping for reference:
    # (Deleted duplicate ranking logic - all moved into latest_only block above)

    # Fallback: evaluate full history per date (unreachable in practice)
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
                passed, flags, reason = system1_row_passes_setup(
                    row, allow_fallback=False
                )
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
                    "entry_date": (
                        __import__(
                            "common.utils_spy", fromlist=["resolve_signal_entry_date"]
                        ).resolve_signal_entry_date(date)
                    ),
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
        try:
            candidates_df["entry_date"] = pd.to_datetime(candidates_df["entry_date"])
        except Exception:
            pass
        candidates_df = candidates_df.sort_values(
            ["date", "roc200"], ascending=[True, False]
        )
        last_date = max(candidates_by_date.keys()) if candidates_by_date else None
        if last_date is not None:
            diag.ranked_top_n_count = len(candidates_by_date.get(last_date, []))
        diag.ranking_source = "full_scan"
    else:
        candidates_df = None
        diag.ranking_source = "latest_only" if latest_only else "full_scan"

    if log_callback:
        total_candidates = len(all_candidates)
        unique_dates = len(candidates_by_date)
        log_callback(
            (
                "System1: Generated "
                f"{total_candidates} candidates across {unique_dates} dates"
            )
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
    candidates: list[dict] = []

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
