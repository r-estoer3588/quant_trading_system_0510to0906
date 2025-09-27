"""System2 core logic (Short RSI spike).

RSI3-based short spike strategy:
- Indicators: RSI3, ADX7, ATR10, DollarVolume20, ATR_Ratio, TwoDayUp (precomputed only)
- Setup conditions: Close>5, DollarVolume20>25M, ATR_Ratio>0.03, RSI3>90, TwoDayUp
- Candidate generation: ADX7 descending ranking by date, extract top_n
- Optimization: Removed all indicator calculations, using precomputed indicators only
"""

from __future__ import annotations

import pandas as pd

from common.batch_processing import process_symbols_batch
from common.system_common import check_precomputed_indicators, get_total_days
from common.system_constants import (
    SYSTEM2_REQUIRED_INDICATORS,
)
from common.utils import get_cached_data


def _compute_indicators(symbol: str) -> tuple[str, pd.DataFrame | None]:
    """Check precomputed indicators and apply System2-specific filters.

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
        missing_indicators = [col for col in SYSTEM2_REQUIRED_INDICATORS if col not in df.columns]
        if missing_indicators:
            return symbol, None

        # Apply System2-specific filters and setup
        x = df.copy()

        # Filter: Close>=5, DollarVolume20>25M, ATR_Ratio>0.03
        x["filter"] = (
            (x["Close"] >= 5.0) & (x["dollarvolume20"] > 25_000_000) & (x["atr_ratio"] > 0.03)
        )

        # Setup: Filter + RSI3>90 + TwoDayUp
        x["setup"] = x["filter"] & (x["rsi3"] > 90) & x["twodayup"]

        return symbol, x

    except Exception:
        return symbol, None


def prepare_data_vectorized_system2(
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
    """System2 data preparation processing (RSI3 spike strategy).

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
                raw_data_dict, SYSTEM2_REQUIRED_INDICATORS, "System2", skip_callback
            )

            if valid_data_dict:
                # Apply System2-specific filters
                prepared_dict = {}
                for symbol, df in valid_data_dict.items():
                    x = df.copy()

                    # Filter: Close>=5, DollarVolume20>25M, ATR_Ratio>0.03
                    x["filter"] = (
                        (x["Close"] >= 5.0)
                        & (x["dollarvolume20"] > 25_000_000)
                        & (x["atr_ratio"] > 0.03)
                    )

                    # Setup: Filter + RSI3>90 + TwoDayUp
                    x["setup"] = x["filter"] & (x["rsi3"] > 90) & x["twodayup"]

                    prepared_dict[symbol] = x

                if log_callback:
                    log_callback(f"System2: Fast-path processed {len(prepared_dict)} symbols")

                return prepared_dict

        except RuntimeError:
            # Re-raise error immediately if required indicators are missing
            raise
        except Exception:
            # Fall back to normal processing for other errors
            if log_callback:
                log_callback("System2: Fast-path failed, falling back to normal processing")

    # Normal processing path: batch processing from symbol list
    if symbols:
        target_symbols = symbols
    elif raw_data_dict:
        target_symbols = list(raw_data_dict.keys())
    else:
        if log_callback:
            log_callback("System2: No symbols provided, returning empty dict")
        return {}

    if log_callback:
        log_callback(f"System2: Starting normal processing for {len(target_symbols)} symbols")

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
        system_name="System2",
    )

    return results


def generate_candidates_system2(
    prepared_dict: dict[str, pd.DataFrame],
    *,
    top_n: int | None = None,
    progress_callback=None,
    log_callback=None,
) -> tuple[dict, pd.DataFrame | None]:
    """System2 candidate generation (ADX7 descending ranking).

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
            log_callback("System2: No data provided for candidate generation")
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
            log_callback("System2: No valid dates found in data")
        return {}, None

    all_dates = sorted(all_dates)

    candidates_by_date = {}
    all_candidates = []

    if log_callback:
        log_callback(f"System2: Generating candidates for {len(all_dates)} dates")

    # Execute ADX7 ranking by date
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

                # Get ADX7 value
                adx7_val = row.get("adx7", 0)
                if pd.isna(adx7_val) or adx7_val <= 0:
                    continue

                date_candidates.append(
                    {
                        "symbol": symbol,
                        "date": date,
                        "adx7": adx7_val,
                        "rsi3": row.get("rsi3", 0),
                        "close": row.get("Close", 0),
                    }
                )

            except Exception:
                continue

        # Sort by ADX7 descending and extract top_n
        if date_candidates:
            date_candidates.sort(key=lambda x: x["adx7"], reverse=True)
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
        candidates_df = candidates_df.sort_values(["date", "adx7"], ascending=[True, False])
    else:
        candidates_df = None

    if log_callback:
        total_candidates = len(all_candidates)
        unique_dates = len(candidates_by_date)
        log_callback(
            f"System2: Generated {total_candidates} candidates across {unique_dates} dates"
        )

    return candidates_by_date, candidates_df


def get_total_days_system2(data_dict: dict[str, pd.DataFrame]) -> int:
    """Get total days count for System2 data.

    Args:
        data_dict: Data dictionary

    Returns:
        Maximum day count
    """
    return get_total_days(data_dict)


__all__ = [
    "prepare_data_vectorized_system2",
    "generate_candidates_system2",
    "get_total_days_system2",
]
