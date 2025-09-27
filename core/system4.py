"""System4 core logic (Long trend low-vol pullback).

Trend low-volatility pullback strategy:
- Indicators: SMA200, ATR40, RSI4 (precomputed only)
- Setup conditions: Close>=5, Close>SMA200, RSI4<30, ATR_Ratio<0.05
- Candidate generation: RSI4 ascending ranking by date, extract top_n
- Optimization: Removed all indicator calculations, using precomputed indicators only
"""

from __future__ import annotations

import pandas as pd

from common.batch_processing import process_symbols_batch
from common.system_common import check_precomputed_indicators, get_total_days
from common.system_constants import (
    SYSTEM4_REQUIRED_INDICATORS,
)
from common.utils import get_cached_data


def _compute_indicators(symbol: str) -> tuple[str, pd.DataFrame | None]:
    """Check precomputed indicators and apply System4-specific filters.

    Args:
        symbol: Target symbol to process

    Returns:
        (symbol, processed DataFrame | None)
    """
    try:
        df = get_cached_data(symbol)
        if df is None or df.empty:
            return symbol, None

        # Check for required indicators
        missing_indicators = [col for col in SYSTEM4_REQUIRED_INDICATORS if col not in df.columns]
        if missing_indicators:
            return symbol, None

        # Apply System4-specific filters and setup
        x = df.copy()

        # Calculate ATR ratio
        x["atr_ratio"] = x["atr40"] / x["Close"]

        # Filter: Close>=5, Close>SMA200, ATR_Ratio<0.05 (low volatility)
        x["filter"] = (x["Close"] >= 5.0) & (x["Close"] > x["sma200"]) & (x["atr_ratio"] < 0.05)

        # Setup: Filter + RSI4<30 (oversold pullback)
        x["setup"] = x["filter"] & (x["rsi4"] < 30.0)

        return symbol, x

    except Exception:
        return symbol, None


def prepare_data_vectorized_system4(
    raw_data_dict: dict[str, pd.DataFrame] | None,
    *,
    progress_callback=None,
    log_callback=None,
    skip_callback=None,
    batch_size: int | None = None,
    reuse_indicators: bool = True,
    symbols: list[str] | None = None,
    use_process_pool: bool = False,
    max_workers: int | None = None,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """System4 data preparation processing (trend low-volatility pullback strategy).

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
    # Fast path: reuse precomputed indicators
    if reuse_indicators and raw_data_dict:
        try:
            # Early check - verify required indicators exist
            valid_data_dict, error_symbols = check_precomputed_indicators(
                raw_data_dict, SYSTEM4_REQUIRED_INDICATORS, "System4", skip_callback
            )

            if valid_data_dict:
                # Apply System4-specific filters
                prepared_dict = {}
                for symbol, df in valid_data_dict.items():
                    x = df.copy()

                    # Calculate ATR ratio
                    x["atr_ratio"] = x["atr40"] / x["Close"]

                    # Filter: Close>=5, Close>SMA200, ATR_Ratio<0.05 (low volatility)
                    x["filter"] = (
                        (x["Close"] >= 5.0) & (x["Close"] > x["sma200"]) & (x["atr_ratio"] < 0.05)
                    )

                    # Setup: Filter + RSI4<30 (oversold pullback)
                    x["setup"] = x["filter"] & (x["rsi4"] < 30.0)

                    prepared_dict[symbol] = x

                if log_callback:
                    log_callback(f"System4: Fast-path processed {len(prepared_dict)} symbols")

                return prepared_dict

        except RuntimeError:
            # Re-raise error immediately if required indicators are missing
            raise
        except Exception:
            # Fall back to normal processing for other errors
            if log_callback:
                log_callback("System4: Fast-path failed, falling back to normal processing")

    # Normal processing path: batch processing from symbol list
    if symbols:
        target_symbols = symbols
    elif raw_data_dict:
        target_symbols = list(raw_data_dict.keys())
    else:
        if log_callback:
            log_callback("System4: No symbols provided, returning empty dict")
        return {}

    if log_callback:
        log_callback(f"System4: Starting normal processing for {len(target_symbols)} symbols")

    # Execute batch processing
    results, error_symbols = process_symbols_batch(
        target_symbols,
        _compute_indicators,
        batch_size=batch_size,
        use_process_pool=use_process_pool,
        max_workers=max_workers,
        progress_callback=progress_callback,
        log_callback=log_callback,
        skip_callback=skip_callback,
        system_name="System4",
    )

    return results


def generate_candidates_system4(
    prepared_dict: dict[str, pd.DataFrame],
    *,
    top_n: int | None = None,
    progress_callback=None,
    log_callback=None,
) -> tuple[dict, pd.DataFrame | None]:
    """System4 candidate generation (RSI4 ascending ranking).

    Args:
        prepared_dict: Prepared data dictionary
        top_n: Number of top entries to extract
        progress_callback: Progress reporting callback
        log_callback: Log output callback

    Returns:
        (Daily candidate dictionary, Integrated candidate DataFrame)
    """
    if not prepared_dict:
        if log_callback:
            log_callback("System4: No data provided for candidate generation")
        return {}, None

    if top_n is None:
        top_n = 20  # Default value

    # Aggregate all dates
    all_dates = set()
    for df in prepared_dict.values():
        if df is not None and not df.empty:
            all_dates.update(df.index)

    if not all_dates:
        if log_callback:
            log_callback("System4: No valid dates found in data")
        return {}, None

    all_dates = sorted(all_dates)

    candidates_by_date = {}
    all_candidates = []

    if log_callback:
        log_callback(f"System4: Generating candidates for {len(all_dates)} dates")

    # Execute RSI4 ranking by date (ascending - lowest RSI4 first for oversold)
    for i, date in enumerate(all_dates):
        date_candidates = []

        for symbol, df in prepared_dict.items():
            try:
                if df is None or date not in df.index:
                    continue

                row = df.loc[date]

                # Check setup conditions
                if not row.get("setup", False):
                    continue

                # Get RSI4 value
                rsi4_val = row.get("rsi4", 100)
                if pd.isna(rsi4_val) or rsi4_val >= 30.0:
                    continue

                date_candidates.append(
                    {
                        "symbol": symbol,
                        "date": date,
                        "rsi4": rsi4_val,
                        "atr_ratio": row.get("atr_ratio", 0),
                        "close": row.get("Close", 0),
                        "sma200": row.get("sma200", 0),
                    }
                )

            except Exception:
                continue

        # Sort by RSI4 ascending (lowest first) and extract top_n
        if date_candidates:
            date_candidates.sort(key=lambda x: x["rsi4"])
            top_candidates = date_candidates[:top_n]

            candidates_by_date[date] = top_candidates
            all_candidates.extend(top_candidates)

        # Progress reporting
        if progress_callback and (i + 1) % max(1, len(all_dates) // 10) == 0:
            progress_callback(f"Processed {i + 1}/{len(all_dates)} dates")

    # Create integrated DataFrame
    if all_candidates:
        candidates_df = pd.DataFrame(all_candidates)
        candidates_df["date"] = pd.to_datetime(candidates_df["date"])
        candidates_df = candidates_df.sort_values(["date", "rsi4"], ascending=[True, True])
    else:
        candidates_df = None

    if log_callback:
        total_candidates = len(all_candidates)
        unique_dates = len(candidates_by_date)
        log_callback(
            f"System4: Generated {total_candidates} candidates across {unique_dates} dates"
        )

    return candidates_by_date, candidates_df


def get_total_days_system4(data_dict: dict[str, pd.DataFrame]) -> int:
    """Get total days count for System4 data.

    Args:
        data_dict: Data dictionary

    Returns:
        Maximum day count
    """
    return get_total_days(data_dict)


__all__ = [
    "prepare_data_vectorized_system4",
    "generate_candidates_system4",
    "get_total_days_system4",
]
