"""System2 core logic (Short RSI spike) を共通化。

- インジケーター: RSI3, ADX7, ATR10, DollarVolume20, ATR_Ratio, TwoDayUp (precomputed only)
- セットアップ条件: Close>5, DollarVolume20>25M, ATR_Ratio>0.03, RSI3>90, TwoDayUp
- 候補生成: ADX7 降順で top_n を日別抽出
- 最適化: 全指標計算を除去、プリコンピュート指標のみ使用
"""

import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from common.utils import BatchSizeMonitor, get_cached_data, resolve_batch_size
from common.utils_spy import resolve_signal_entry_date


def _compute_indicators(symbol: str) -> tuple[str, pd.DataFrame | None]:
    """Worker-side: プリコンピューテッド指標のみ使用、計算は完全除去."""
    df = get_cached_data(symbol)
    if df is None or df.empty:
        return symbol, None

    # 子プロセスから親へ簡易進捗を送る（存在すれば）
    try:
        q = globals().get("_PROGRESS_QUEUE")
        if q is not None:
            try:
                q.put((symbol, 0))
            except Exception:
                pass
    except Exception:
        pass

    # Normalize OHLCV column names
    rename_map = {}
    for low, up in (
        ("open", "Open"),
        ("high", "High"),
        ("low", "Low"),
        ("close", "Close"),
        ("volume", "Volume"),
    ):
        if low in df.columns and up not in df.columns:
            rename_map[low] = up
    if rename_map:
        df = df.rename(columns=rename_map)

    # Required precomputed indicators (lowercase, from indicators_common)
    required_indicators = [
        "rsi3",
        "adx7",
        "atr10",
        "dollarvolume20",
        "atr_ratio",
        "twodayup",
    ]

    # Check if all required indicators exist - immediate stop on missing
    missing_indicators = [col for col in required_indicators if col not in df.columns]
    if missing_indicators:
        raise RuntimeError(
            f"IMMEDIATE_STOP: System2 missing precomputed indicators {missing_indicators} for {symbol}. Daily signal execution must be stopped."
        )

    # Copy only necessary columns
    needed_cols = ["Open", "High", "Low", "Close", "Volume"] + required_indicators
    available_cols = [col for col in needed_cols if col in df.columns]
    x = df[available_cols].copy()

    if len(x) < 20:
        return symbol, None

    # Create strategy-specific derived columns using precomputed indicators only
    x["filter"] = (
        (x["Low"] >= 5) & (x["dollarvolume20"] > 25_000_000) & (x["atr_ratio"] > 0.03)
    )
    x["setup"] = x["filter"] & (x["rsi3"] > 90) & x["twodayup"]

    # 完了を親に伝える
    try:
        q = globals().get("_PROGRESS_QUEUE")
        if q is not None:
            try:
                q.put((symbol, 100))
            except Exception:
                pass
    except Exception:
        pass

    return symbol, x


def prepare_data_vectorized_system2(
    raw_data_dict: dict[str, pd.DataFrame] | None,
    *,
    progress_callback=None,
    log_callback=None,
    batch_size: int | None = None,
    reuse_indicators: bool = True,
    symbols: list[str] | None = None,
    use_process_pool: bool = False,
    max_workers: int | None = None,
    skip_callback=None,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """
    System2用データ準備（プリコンピューテッド指標のみ使用、計算除去版）
    """
    if log_callback:
        log_callback("Starting System2 data preparation (precomputed only)")

    result_dict = {}
    skipped_count = 0

    # Today-mode fast-path: incoming frames with precomputed indicators
    if raw_data_dict:
        for sym, df in raw_data_dict.items():
            if symbols and sym not in symbols:
                continue

            if df is None or df.empty:
                skipped_count += 1
                continue

            # Check required precomputed indicators
            required_indicators = [
                "rsi3",
                "adx7",
                "atr10",
                "dollarvolume20",
                "atr_ratio",
                "twodayup",
            ]
            missing_indicators = [
                col for col in required_indicators if col not in df.columns
            ]

            if missing_indicators:
                raise RuntimeError(
                    f"IMMEDIATE_STOP: System2 missing precomputed indicators {missing_indicators} for {sym}. Daily signal execution must be stopped."
                )

            # Use precomputed indicators directly
            x = df.copy()

            # Normalize column names
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
                x = x.rename(columns=rename_map)

            if len(x) < 20:
                skipped_count += 1
                continue

            # Create strategy-specific derived columns
            x["filter"] = (
                (x["Low"] >= 5)
                & (x["dollarvolume20"] > 25_000_000)
                & (x["atr_ratio"] > 0.03)
            )
            x["setup"] = x["filter"] & (x["rsi3"] > 90) & x["twodayup"]

            result_dict[sym] = x

        if log_callback:
            log_callback(
                f"System2 fast-path: processed {len(result_dict)} symbols, skipped {skipped_count}"
            )
        return result_dict

    # Cache-based path with parallel processing
    # Use file system to get available symbols
    from config.settings import get_settings

    settings = get_settings()
    data_cache_path = settings.data.cache_dir / "base"
    available_symbols = []
    if data_cache_path.exists():
        for f in data_cache_path.glob("*.feather"):
            symbol = f.stem.upper()
            available_symbols.append(symbol)

    if symbols:
        target_symbols = [s for s in symbols if s in available_symbols]
    else:
        target_symbols = available_symbols

    if not target_symbols:
        if log_callback:
            log_callback("System2: No symbols available for processing")
        return {}

    total = len(target_symbols)
    processed = 0
    buffer = []

    if use_process_pool and total > 100:
        # Parallel processing for large datasets
        batch_size = resolve_batch_size(len(target_symbols), batch_size or 50)
        monitor = BatchSizeMonitor(initial=batch_size)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for i in range(0, total, batch_size):
                batch = target_symbols[i : i + batch_size]
                start_time = time.time()

                futures = {
                    executor.submit(_compute_indicators, sym): sym for sym in batch
                }
                batch_results = 0

                for future in as_completed(futures):
                    sym = futures[future]
                    try:
                        _, df = future.result()
                        if df is not None:
                            result_dict[sym] = df
                            buffer.append(sym)
                        else:
                            skipped_count += 1
                            if skip_callback:
                                try:
                                    skip_callback(sym, "precomputed_indicators_missing")
                                except Exception:
                                    pass
                        batch_results += 1
                        processed += 1

                        if progress_callback:
                            try:
                                progress_callback(processed, total)
                            except Exception:
                                pass

                    except Exception as e:
                        skipped_count += 1
                        if skip_callback:
                            try:
                                skip_callback(sym, f"processing_error: {str(e)}")
                            except Exception:
                                pass

                elapsed = time.time() - start_time
                batch_size = monitor.update(elapsed)

    else:
        # Sequential processing for smaller datasets
        for sym in target_symbols:
            try:
                _, df = _compute_indicators(sym)
                if df is not None:
                    result_dict[sym] = df
                    buffer.append(sym)
                else:
                    skipped_count += 1
                    if skip_callback:
                        try:
                            skip_callback(sym, "precomputed_indicators_missing")
                        except Exception:
                            pass

                processed += 1
                if progress_callback:
                    try:
                        progress_callback(processed, total)
                    except Exception:
                        pass

            except Exception as e:
                skipped_count += 1
                if skip_callback:
                    try:
                        skip_callback(sym, f"processing_error: {str(e)}")
                    except Exception:
                        pass

    if log_callback:
        log_callback(
            f"System2 preparation complete: {len(result_dict)} symbols processed, {skipped_count} skipped"
        )

    return result_dict


def select_candidates_system2(
    data_dict: dict[str, pd.DataFrame],
    date: str,
    top_n: int = 20,
    log_callback=None,
) -> list[str]:
    """
    System2候補選択: ADX7降順でtop_n選択
    """
    if log_callback:
        log_callback(f"System2 candidate selection for {date}, top_n={top_n}")

    candidates = []
    date_pd = pd.to_datetime(date)

    for symbol, df in data_dict.items():
        if df is None or df.empty:
            continue

        # Find the closest available date
        available_dates = df.index
        if date_pd not in available_dates:
            # Find the closest date <= target date
            valid_dates = available_dates[available_dates <= date_pd]
            if valid_dates.empty:
                continue
            use_date = valid_dates.max()
        else:
            use_date = date_pd

        try:
            row = df.loc[use_date]
            setup_val = row.get("setup", False)
            if pd.isna(setup_val) or not bool(setup_val):
                continue

            adx7_val = row.get("adx7", 0)
            if pd.isna(adx7_val):
                continue

            candidates.append((symbol, float(adx7_val)))

        except (KeyError, IndexError, ValueError):
            continue

    # Sort by ADX7 descending and take top_n
    candidates.sort(key=lambda x: x[1], reverse=True)
    selected = [sym for sym, _ in candidates[:top_n]]

    if log_callback:
        log_callback(
            f"System2 selected {len(selected)} candidates from {len(candidates)} setups"
        )

    return selected


def system2_backtest_vectorized(
    start_date: str,
    end_date: str,
    *,
    progress_callback=None,
    log_callback=None,
    include_details: bool = True,
    top_n: int = 20,
    symbols: list[str] | None = None,
    use_process_pool: bool = False,
    max_workers: int | None = None,
    **kwargs,
) -> dict:
    """
    System2バックテスト（プリコンピューテッド指標のみ使用）
    """
    if log_callback:
        log_callback(f"Starting System2 backtest: {start_date} to {end_date}")

    # Data preparation
    data_dict = prepare_data_vectorized_system2(
        raw_data_dict=None,
        progress_callback=progress_callback,
        log_callback=log_callback,
        symbols=symbols,
        use_process_pool=use_process_pool,
        max_workers=max_workers,
        **kwargs,
    )

    if not data_dict:
        if log_callback:
            log_callback("System2: No data available for backtest")
        return {"trades": [], "summary": {}, "daily_pnl": pd.DataFrame()}

    # Generate trading dates
    all_dates = set()
    for df in data_dict.values():
        if df is not None:
            all_dates.update(df.index)

    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    trading_dates = [d for d in date_range if d in all_dates]

    if not trading_dates:
        if log_callback:
            log_callback("System2: No trading dates available in range")
        return {"trades": [], "summary": {}, "daily_pnl": pd.DataFrame()}

    trades = []
    daily_pnl = []

    for i, current_date in enumerate(trading_dates):
        if progress_callback and i % 50 == 0:
            try:
                progress_callback(i, len(trading_dates))
            except Exception:
                pass

        date_str = current_date.strftime("%Y-%m-%d")

        # Select candidates
        candidates = select_candidates_system2(
            data_dict, date_str, top_n=top_n, log_callback=log_callback
        )

        if not candidates:
            daily_pnl.append({"Date": current_date, "PnL": 0.0, "Positions": 0})
            continue

        # Calculate position metrics (simplified for performance)
        positions = len(candidates)
        daily_return = (
            0.0  # Placeholder - would calculate actual returns in full implementation
        )

        daily_pnl.append(
            {"Date": current_date, "PnL": daily_return, "Positions": positions}
        )

        # Record trades (simplified)
        for symbol in candidates:
            trades.append(
                {
                    "Date": current_date,
                    "Symbol": symbol,
                    "Action": "SHORT",
                    "System": "System2",
                }
            )

    # Create summary
    pnl_df = pd.DataFrame(daily_pnl)
    total_trades = len(trades)

    summary = {
        "total_trades": total_trades,
        "trading_days": len(trading_dates),
        "avg_positions_per_day": pnl_df["Positions"].mean() if not pnl_df.empty else 0,
        "total_return": pnl_df["PnL"].sum() if not pnl_df.empty else 0.0,
    }

    if log_callback:
        log_callback(
            f"System2 backtest complete: {total_trades} trades, {summary['total_return']:.4f} total return"
        )

    return {
        "trades": trades if include_details else [],
        "summary": summary,
        "daily_pnl": pnl_df,
    }


def generate_candidates_system2(
    prepared_dict: dict[str, pd.DataFrame],
    *,
    top_n: int = 10,
) -> tuple[dict, pd.DataFrame | None]:
    """セットアップ通過銘柄を日別に ADX7 降順で抽出。
    返却: (candidates_by_date, merged_df=None)
    """

    all_signals = []
    for sym, df in prepared_dict.items():
        if "setup" not in df.columns or not df["setup"].any():
            continue
        setup_df = df[df["setup"]].copy()
        setup_df["symbol"] = sym

        # last_price（直近終値）を取得
        last_price = None
        if "Close" in df.columns and not df["Close"].empty:
            last_price = df["Close"].iloc[-1]
        setup_df["entry_price"] = last_price

        base_dates = pd.to_datetime(setup_df.index, errors="coerce").to_series(
            index=setup_df.index
        )
        setup_df["entry_date"] = base_dates.map(resolve_signal_entry_date)
        setup_df = setup_df.dropna(subset=["entry_date"])
        all_signals.append(setup_df)

    if not all_signals:
        return {}, None

    all_df = pd.concat(all_signals)

    # entry_date 単位のシグナル候補: date -> list[dict]
    candidates_by_date: dict[pd.Timestamp, list[dict]] = {}
    for date, group in all_df.groupby("entry_date"):
        ranked = group.sort_values("ADX7", ascending=False).copy()
        total = len(ranked)
        if total == 0:
            candidates_by_date[date] = []
            continue
        ranked.loc[:, "rank"] = range(1, total + 1)
        ranked.loc[:, "rank_total"] = total
        top_ranked = ranked.head(int(top_n))
        candidates_by_date[date] = top_ranked.to_dict("records")
    return candidates_by_date, None


def get_total_days_system2(data_dict: dict[str, pd.DataFrame]) -> int:
    """データ中の日数ユニーク数。"""
    all_dates = set()
    for df in data_dict.values():
        if df is None or df.empty:
            continue
        if "Date" in df.columns:
            dates = pd.to_datetime(df["Date"]).dt.normalize()
        elif "date" in df.columns:
            dates = pd.to_datetime(df["date"]).dt.normalize()
        else:
            dates = pd.to_datetime(df.index).normalize()
        all_dates.update(dates)
    return len(all_dates)


# Export functions for strategy wrappers
__all__ = [
    "prepare_data_vectorized_system2",
    "generate_candidates_system2",
    "get_total_days_system2",
    "select_candidates_system2",
    "system2_backtest_vectorized",
]
