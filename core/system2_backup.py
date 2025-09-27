"""System2 core logic (Short RSI spike).

RSI3-based short spike strategy:
- Indicators: RSI3, ADX7, ATR10, DollarVolume20, ATR_Ratio, TwoDayUp (precomputed only)
- Setup conditions: Close>5, DollarVolume20>25M, ATR_Ratio>0.03, RSI3>90, TwoDayUp
- Candidate generation: ADX7 descending ranking by date, extract top_n
- Optimization: Removed all indicator calculations, using precomputed indicators only
"""

from __future__ import annotations

import pandas as pd
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from common.batch_processing import process_symbols_batch
from common.batch_size_manager import BatchSizeMonitor, resolve_batch_size
from common.system_common import check_precomputed_indicators, get_total_days
from common.system_constants import (
    SYSTEM2_REQUIRED_INDICATORS,
    MIN_ROWS_SYSTEM2,
    get_system_config,
)
from common.utils import get_cached_data
from common.utils_spy import resolve_signal_entry_date
stem2-specific filters.

def _compute_indicators(symbol: str) -> tuple[str, pd.DataFrame | None]:
    """Worker-side: プリコンピューテッド指標のみ使用、計算は完全除去."""ol to process
    df = get_cached_data(symbol)        
    if df is None or df.empty:
        return symbol, None(symbol, processed DataFrame | None)

    # 子プロセスから親へ簡易進捗を送る（存在すれば）
    try:_cached_data(symbol)
        q = globals().get("_PROGRESS_QUEUE")
        if q is not None:ne
            try:
                q.put((symbol, 0))equired indicators
            except Exception:ing_indicators = [col for col in SYSTEM2_REQUIRED_INDICATORS if col not in df.columns]
                pass        if missing_indicators:
    except Exception:
        pass
em2-specific filters and setup
    # Normalize OHLCV column names
    rename_map = {}
    for low, up in (>=5, DollarVolume20>25M, ATR_Ratio>0.03
        ("open", "Open"),
        ("high", "High"),0) & 
        ("low", "Low"),      (x["dollarvolume20"] > 25_000_000) &
        ("close", "Close"),
        ("volume", "Volume"),
    ):
        if low in df.columns and up not in df.columns:Up
            rename_map[low] = up        x["setup"] = (
    if rename_map:
        df = df.rename(columns=rename_map)
            (x["twodayup"] == True)
    # Required precomputed indicators (lowercase, from indicators_common)
    required_indicators = ["rsi3", "adx7", "atr10", "dollarvolume20", "atr_ratio", "twodayup"]

    # Check if all required indicators exist - immediate stop on missing
    missing_indicators = [col for col in required_indicators if col not in df.columns]
    if missing_indicators:eturn symbol, None
        raise RuntimeError(
            f"IMMEDIATE_STOP: System2 missing precomputed indicators {missing_indicators} for {symbol}. Daily signal execution must be stopped."ators exist - immediate stop on missing
        )mns]

    # Copy only necessary columns
    needed_cols = ["Open", "High", "Low", "Close", "Volume"] + required_indicators            f"IMMEDIATE_STOP: System2 missing precomputed indicators {missing_indicators} for {symbol}. Daily signal execution must be stopped."
    available_cols = [col for col in needed_cols if col in df.columns]
    x = df[available_cols].copy()
    # Copy only necessary columns
    if len(x) < 20:rs
        return symbol, None

    # Create strategy-specific derived columns using precomputed indicators only
    x["filter"] = (x["Low"] >= 5) & (x["dollarvolume20"] > 25_000_000) & (x["atr_ratio"] > 0.03)< 20:
    x["setup"] = x["filter"] & (x["rsi3"] > 90) & x["twodayup"]return symbol, None

    # 完了を親に伝えるcific derived columns using precomputed indicators only
    try:= (x["Low"] >= 5) & (x["dollarvolume20"] > 25_000_000) & (x["atr_ratio"] > 0.03)
        q = globals().get("_PROGRESS_QUEUE")si3"] > 90) & x["twodayup"]
        if q is not None:
            try:
                q.put((symbol, 100))
            except Exception:globals().get("_PROGRESS_QUEUE")
                pass        if q is not None:
    except Exception:
        pass                q.put((symbol, 100))
            except Exception:
    return symbol, x

  pass
def prepare_data_vectorized_system2(
    raw_data_dict: dict[str, pd.DataFrame] | None,
    *,
    progress_callback=None,
    log_callback=None,
    batch_size: int | None = None,aFrame] | None,
    reuse_indicators: bool = True,
    symbols: list[str] | None = None,one,
    use_process_pool: bool = False,ack=None,
    max_workers: int | None = None,None,
    skip_callback=None,se_indicators: bool = True,
    **kwargs,
) -> dict[str, pd.DataFrame]:_process_pool: bool = False,
    """ | None = None,
    System2用データ準備（プリコンピューテッド指標のみ使用、計算除去版）
    """    **kwargs,
    if log_callback:taFrame]:
        log_callback("Starting System2 data preparation (precomputed only)")
    System2用データ準備（プリコンピューテッド指標のみ使用、計算除去版）
    result_dict = {}
    skipped_count = 0
reparation (precomputed only)")
    # Today-mode fast-path: incoming frames with precomputed indicators
    if raw_data_dict:
        for sym, df in raw_data_dict.items():    skipped_count = 0
            if symbols and sym not in symbols:
                continueng frames with precomputed indicators

            if df is None or df.empty:        for sym, df in raw_data_dict.items():
                skipped_count += 1
                continue

            # Check required precomputed indicatorsne or df.empty:
            required_indicators = [count += 1
                "rsi3",
                "adx7",
                "atr10",d precomputed indicators
                "dollarvolume20",equired_indicators = [
                "atr_ratio",
                "twodayup",                "adx7",
            ]
            missing_indicators = [col for col in required_indicators if col not in df.columns]

            if missing_indicators:twodayup",
                raise RuntimeError(            ]
                    f"IMMEDIATE_STOP: System2 missing precomputed indicators {missing_indicators} for {sym}. Daily signal execution must be stopped."required_indicators if col not in df.columns]
                )
            if missing_indicators:
            # Use precomputed indicators directly
            x = df.copy()IATE_STOP: System2 missing precomputed indicators {missing_indicators} for {sym}. Daily signal execution must be stopped."

            # Normalize column names
            rename_map = {}icators directly
            for low, up in (
                ("open", "Open"),
                ("high", "High"),
                ("low", "Low"),name_map = {}
                ("close", "Close"),
                ("volume", "Volume"),
            ):High"),
                if low in x.columns and up not in x.columns:
                    rename_map[low] = up                ("close", "Close"),
            if rename_map:"Volume"),
                x = x.rename(columns=rename_map)
n x.columns and up not in x.columns:
            if len(x) < 20:                    rename_map[low] = up
                skipped_count += 1
                continuee(columns=rename_map)

            # Create strategy-specific derived columnsf len(x) < 20:
            x["filter"] = (
                (x["Low"] >= 5) & (x["dollarvolume20"] > 25_000_000) & (x["atr_ratio"] > 0.03)                continue
            )
            x["setup"] = x["filter"] & (x["rsi3"] > 90) & x["twodayup"]            # Create strategy-specific derived columns
= (
            result_dict[sym] = x >= 5) & (x["dollarvolume20"] > 25_000_000) & (x["atr_ratio"] > 0.03)

        if log_callback:["setup"] = x["filter"] & (x["rsi3"] > 90) & x["twodayup"]
            log_callback(
                f"System2 fast-path: processed {len(result_dict)} symbols, skipped {skipped_count}"            result_dict[sym] = x
            )
        return result_dict

    # Cache-based path with parallel processing                f"System2 fast-path: processed {len(result_dict)} symbols, skipped {skipped_count}"
    # Use file system to get available symbols
    from config.settings import get_settings

    settings = get_settings()llel processing
    data_cache_path = settings.data.cache_dir / "base"
    available_symbols = []_settings
    if data_cache_path.exists():
        for f in data_cache_path.glob("*.feather"):    settings = get_settings()
            symbol = f.stem.upper()path = settings.data.cache_dir / "base"
            available_symbols.append(symbol)
ta_cache_path.exists():
    if symbols:eather"):
        target_symbols = [s for s in symbols if s in available_symbols]            symbol = f.stem.upper()
    else:ols.append(symbol)
        target_symbols = available_symbols

    if not target_symbols:mbols = [s for s in symbols if s in available_symbols]
        if log_callback:    else:
            log_callback("System2: No symbols available for processing")ble_symbols
        return {}
et_symbols:
    total = len(target_symbols)        if log_callback:
    processed = 0mbols available for processing")
    buffer = []

    if use_process_pool and total > 100:
        # Parallel processing for large datasets    processed = 0
        batch_size = resolve_batch_size(len(target_symbols), batch_size or 50)
        monitor = BatchSizeMonitor(initial=batch_size)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:datasets
            for i in range(0, total, batch_size):        batch_size = resolve_batch_size(len(target_symbols), batch_size or 50)
                batch = target_symbols[i : i + batch_size]
                start_time = time.time()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_compute_indicators, sym): sym for sym in batch}
                batch_results = 0: i + batch_size]
me = time.time()
                for future in as_completed(futures):
                    sym = futures[future](_compute_indicators, sym): sym for sym in batch}
                    try:
                        _, df = future.result()
                        if df is not None: as_completed(futures):
                            result_dict[sym] = df
                            buffer.append(sym)
                        else:re.result()
                            skipped_count += 1
                            if skip_callback:
                                try:d(sym)
                                    skip_callback(sym, "precomputed_indicators_missing")
                                except Exception:unt += 1
                                    pass                            if skip_callback:
                        batch_results += 1
                        processed += 1    skip_callback(sym, "precomputed_indicators_missing")

                        if progress_callback:
                            try:s += 1
                                progress_callback(processed, total)                        processed += 1
                            except Exception:
                                passck:

                    except Exception as e:progress_callback(processed, total)
                        skipped_count += 1
                        if skip_callback:
                            try:
                                skip_callback(sym, f"processing_error: {str(e)}")                    except Exception as e:
                            except Exception:
                                pass
                            try:
                elapsed = time.time() - start_time                       skip_callback(sym, f"processing_error: {str(e)}")
                batch_size = monitor.update(elapsed)
ss
    else:
        # Sequential processing for smaller datasetsme
        for sym in target_symbols:or.update(elapsed)
            try:
                _, df = _compute_indicators(sym)
                if df is not None:processing for smaller datasets
                    result_dict[sym] = df
                    buffer.append(sym)
                else:pute_indicators(sym)
                    skipped_count += 1
                    if skip_callback:
                        try:d(sym)
                            skip_callback(sym, "precomputed_indicators_missing")                else:
                        except Exception:unt += 1
                            pass
try:
                processed += 1_indicators_missing")
                if progress_callback:ion:
                    try:pass
                        progress_callback(processed, total)
                    except Exception:
                        passck:

            except Exception as e:progress_callback(processed, total)
                skipped_count += 1
                if skip_callback:
                    try:
                        skip_callback(sym, f"processing_error: {str(e)}")            except Exception as e:
                    except Exception:ped_count += 1
                        passip_callback:

    if log_callback:               skip_callback(sym, f"processing_error: {str(e)}")
        log_callback(                    except Exception:
            f"System2 preparation complete: {len(result_dict)} symbols processed, {skipped_count} skipped"  pass
        )
    if log_callback:
    return result_dict
ete: {len(result_dict)} symbols processed, {skipped_count} skipped"

def select_candidates_system2(
    data_dict: dict[str, pd.DataFrame],
    date: str,
    top_n: int = 20,
    log_callback=None,
) -> list[str]:a_dict: dict[str, pd.DataFrame],
    """
    System2候補選択: ADX7降順でtop_n選択
    """    log_callback=None,
    if log_callback:
        log_callback(f"System2 candidate selection for {date}, top_n={top_n}")
    System2候補選択: ADX7降順でtop_n選択
    candidates = []
    date_pd = pd.to_datetime(date)
(f"System2 candidate selection for {date}, top_n={top_n}")
    for symbol, df in data_dict.items():
        if df is None or df.empty:
            continue

        # Find the closest available date
        available_dates = df.index
        if date_pd not in available_dates:
            # Find the closest date <= target date
            valid_dates = available_dates[available_dates <= date_pd]e
            if valid_dates.empty:able_dates = df.index
                continuelable_dates:
            use_date = valid_dates.max()            # Find the closest date <= target date
        else:valid_dates = available_dates[available_dates <= date_pd]
            use_date = date_pd

        try:
            row = df.loc[use_date]
            setup_val = row.get("setup", False)            use_date = date_pd
            if pd.isna(setup_val) or not bool(setup_val):
                continue
[use_date]
            adx7_val = row.get("adx7", 0)            setup_val = row.get("setup", False)
            if pd.isna(adx7_val)::
                continue                continue

            candidates.append((symbol, float(adx7_val))) = row.get("adx7", 0)
            if pd.isna(adx7_val):
        except (KeyError, IndexError, ValueError):
            continue
)))
    # Sort by ADX7 descending and take top_n
    candidates.sort(key=lambda x: x[1], reverse=True)rror, IndexError, ValueError):
    selected = [sym for sym, _ in candidates[:top_n]]

    if log_callback:descending and take top_n
        log_callback(f"System2 selected {len(selected)} candidates from {len(candidates)} setups")    candidates.sort(key=lambda x: x[1], reverse=True)
    selected = [sym for sym, _ in candidates[:top_n]]
    return selected

ck(f"System2 selected {len(selected)} candidates from {len(candidates)} setups")
def system2_backtest_vectorized(
    start_date: str,
    end_date: str,
    *,
    progress_callback=None,_vectorized(
    log_callback=None,
    include_details: bool = True,
    top_n: int = 20,
    symbols: list[str] | None = None,callback=None,
    use_process_pool: bool = False,llback=None,
    max_workers: int | None = None,lude_details: bool = True,
    **kwargs,
) -> dict:bols: list[str] | None = None,
    """: bool = False,
    System2バックテスト（プリコンピューテッド指標のみ使用）
    """    **kwargs,
    if log_callback:
        log_callback(f"Starting System2 backtest: {start_date} to {end_date}")
ド指標のみ使用）
    # Data preparation
    data_dict = prepare_data_vectorized_system2(
        raw_data_dict=None,tarting System2 backtest: {start_date} to {end_date}")
        progress_callback=progress_callback,
        log_callback=log_callback,
        symbols=symbols,repare_data_vectorized_system2(
        use_process_pool=use_process_pool,   raw_data_dict=None,
        max_workers=max_workers,        progress_callback=progress_callback,
        **kwargs,log_callback,
    )

    if not data_dict:
        if log_callback:        **kwargs,
            log_callback("System2: No data available for backtest")
        return {"trades": [], "summary": {}, "daily_pnl": pd.DataFrame()}

    # Generate trading dates
    all_dates = set()data available for backtest")
    for df in data_dict.values():        return {"trades": [], "summary": {}, "daily_pnl": pd.DataFrame()}
        if df is not None:
            all_dates.update(df.index)
    all_dates = set()
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")alues():
    trading_dates = [d for d in date_range if d in all_dates]e:

    if not trading_dates:
        if log_callback:    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
            log_callback("System2: No trading dates available in range")es = [d for d in date_range if d in all_dates]
        return {"trades": [], "summary": {}, "daily_pnl": pd.DataFrame()}
    if not trading_dates:
    trades = []
    daily_pnl = [] dates available in range")
"trades": [], "summary": {}, "daily_pnl": pd.DataFrame()}
    for i, current_date in enumerate(trading_dates):
        if progress_callback and i % 50 == 0:
            try:
                progress_callback(i, len(trading_dates))
            except Exception:
                pass        if progress_callback and i % 50 == 0:

        date_str = current_date.strftime("%Y-%m-%d")g_dates))

        # Select candidates       pass
        candidates = select_candidates_system2(
            data_dict, date_str, top_n=top_n, log_callback=log_callback_date.strftime("%Y-%m-%d")
        )
didates
        if not candidates:        candidates = select_candidates_system2(
            daily_pnl.append({"Date": current_date, "PnL": 0.0, "Positions": 0})llback
            continue

        # Calculate position metrics (simplified for performance)        if not candidates:
        positions = len(candidates)
        daily_return = 0.0  # Placeholder - would calculate actual returns in full implementation            continue

        daily_pnl.append({"Date": current_date, "PnL": daily_return, "Positions": positions})ics (simplified for performance)
ndidates)
        # Record trades (simplified)n
        for symbol in candidates:
            trades.append(        daily_pnl.append({"Date": current_date, "PnL": daily_return, "Positions": positions})
                {"Date": current_date, "Symbol": symbol, "Action": "SHORT", "System": "System2"}
            )
es:
    # Create summary            trades.append(
    pnl_df = pd.DataFrame(daily_pnl) {"Date": current_date, "Symbol": symbol, "Action": "SHORT", "System": "System2"}
    total_trades = len(trades)

    summary = {
        "total_trades": total_trades,
        "trading_days": len(trading_dates),otal_trades = len(trades)
        "avg_positions_per_day": pnl_df["Positions"].mean() if not pnl_df.empty else 0,
        "total_return": pnl_df["PnL"].sum() if not pnl_df.empty else 0.0,
    }": total_trades,

    if log_callback:avg_positions_per_day": pnl_df["Positions"].mean() if not pnl_df.empty else 0,
        log_callback(        "total_return": pnl_df["PnL"].sum() if not pnl_df.empty else 0.0,
            f"System2 backtest complete: {total_trades} trades, {summary['total_return']:.4f} total return"
        )

    return {
        "trades": trades if include_details else [],       f"System2 backtest complete: {total_trades} trades, {summary['total_return']:.4f} total return"
        "summary": summary,        )
        "daily_pnl": pnl_df,
    }
 else [],
  "summary": summary,
def generate_candidates_system2( pnl_df,
    prepared_dict: dict[str, pd.DataFrame],
    *,
    top_n: int = 10,
) -> tuple[dict, pd.DataFrame | None]:erate_candidates_system2(
    """セットアップ通過銘柄を日別に ADX7 降順で抽出。    prepared_dict: dict[str, pd.DataFrame],
    返却: (candidates_by_date, merged_df=None)
    """

    all_signals = []に ADX7 降順で抽出。
    for sym, df in prepared_dict.items():ne)
        if "setup" not in df.columns or not df["setup"].any():
            continue
        setup_df = df[df["setup"]].copy()
        setup_df["symbol"] = symed_dict.items():
():
        # last_price（直近終値）を取得
        last_price = None
        if "Close" in df.columns and not df["Close"].empty:        setup_df["symbol"] = sym
            last_price = df["Close"].iloc[-1]
        setup_df["entry_price"] = last_price

        base_dates = pd.to_datetime(setup_df.index, errors="coerce").to_series(index=setup_df.index) not df["Close"].empty:
        setup_df["entry_date"] = base_dates.map(resolve_signal_entry_date)            last_price = df["Close"].iloc[-1]
        setup_df = setup_df.dropna(subset=["entry_date"])_price"] = last_price
        all_signals.append(setup_df)
        base_dates = pd.to_datetime(setup_df.index, errors="coerce").to_series(index=setup_df.index)
    if not all_signals:se_dates.map(resolve_signal_entry_date)
        return {}, None        setup_df = setup_df.dropna(subset=["entry_date"])

    all_df = pd.concat(all_signals)

    # entry_date 単位のシグナル候補: date -> list[dict]
    candidates_by_date: dict[pd.Timestamp, list[dict]] = {}
    for date, group in all_df.groupby("entry_date"):(all_signals)
        ranked = group.sort_values("ADX7", ascending=False).copy()
        total = len(ranked)シグナル候補: date -> list[dict]
        if total == 0:t]] = {}
            candidates_by_date[date] = []y_date"):
            continuescending=False).copy()
        ranked.loc[:, "rank"] = range(1, total + 1)
        ranked.loc[:, "rank_total"] = total
        top_ranked = ranked.head(int(top_n))            candidates_by_date[date] = []
        candidates_by_date[date] = top_ranked.to_dict("records")            continue
    return candidates_by_date, None
rank_total"] = total
ranked.head(int(top_n))
def get_total_days_system2(data_dict: dict[str, pd.DataFrame]) -> int:= top_ranked.to_dict("records")
    """データ中の日数ユニーク数。"""e
    all_dates = set()
    for df in data_dict.values():
        if df is None or df.empty:) -> int:
            continue
        if "Date" in df.columns:
            dates = pd.to_datetime(df["Date"]).dt.normalize() data_dict.values():
        elif "date" in df.columns:
            dates = pd.to_datetime(df["date"]).dt.normalize()
        else:olumns:
            dates = pd.to_datetime(df.index).normalize()            dates = pd.to_datetime(df["Date"]).dt.normalize()
        all_dates.update(dates)        elif "date" in df.columns:
    return len(all_dates)ate"]).dt.normalize()
e:
index).normalize()
# Export functions for strategy wrappers
__all__ = [
    "prepare_data_vectorized_system2",
    "generate_candidates_system2",
    "get_total_days_system2", Export functions for strategy wrappers
    "select_candidates_system2",__all__ = [



]    "system2_backtest_vectorized",    "prepare_data_vectorized_system2",
    "generate_candidates_system2",
    "get_total_days_system2",
    "select_candidates_system2",
    "system2_backtest_vectorized",
]
