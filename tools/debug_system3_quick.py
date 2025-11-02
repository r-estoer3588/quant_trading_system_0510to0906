"""Quick debug runner for System3.

Run from repository root:
    python tools/debug_system3_quick.py [YYYY-MM-DD]

Prints strategy output count, collected log messages, and last_diagnostics.
"""

from __future__ import annotations

import os
import pprint
import sys
import traceback
from pathlib import Path
from typing import Any, cast

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    args = sys.argv[1:]
    # Parse optional first argument as date; remaining args are symbols
    date_arg = None
    symbols: list[str] = []
    if args:
        # try first arg as date
        try:
            candidate = pd.to_datetime(str(args[0]), errors="coerce")
            if candidate is not pd.NaT and not pd.isna(candidate):
                date_arg = args[0]
                symbols = args[1:]
            else:
                symbols = args
        except Exception:
            symbols = args

    date_arg = date_arg or os.environ.get("DEBUG_DATE", "2025-10-20")
    try:
        today = pd.to_datetime(str(date_arg))
    except Exception:
        today = pd.Timestamp("2025-10-20")

    from strategies.system3_strategy import System3Strategy

    stg = System3Strategy()

    logs: list[str] = []

    def log_callback(msg: str) -> None:
        try:
            txt = str(msg)
            print(txt)
            logs.append(txt)
        except Exception:
            print(repr(msg))

    # If no symbols supplied, pick a small sample from data_cache/rolling so
    # that the prepare path will load cached data and produce meaningful
    # diagnostics. This mirrors typical run_all_systems_today behavior.
    if not symbols:
        try:
            from pathlib import Path

            rc = Path("data_cache/rolling")
            if rc.exists():
                feather_files = [p.stem for p in rc.glob("*.feather")]
                csv_files = [p.stem for p in rc.glob("*.csv")]
                files = sorted(feather_files + csv_files)
                symbols = files[:3] if files else ["AAPL"]
            else:
                symbols = ["AAPL"]
        except Exception:
            symbols = ["AAPL"]

    print(f"--- RUN system3 (symbols={symbols}) ---")
    try:
        # 1) Prepare data for the provided symbol list. This uses the
        #    strategy.prepare_data() fast-path which will load cached
        #    data for the symbols and compute/validate indicators.
        prepared = stg.prepare_data(
            cast(Any, symbols),
            reuse_indicators=True,
            log_callback=log_callback,
        )

        # 2) Generate candidates from the prepared data. The strategy
        #    stores extended diagnostics in `last_diagnostics`.
        gen_result = stg.generate_candidates(
            prepared,
            market_df=None,
            latest_only=True,
            top_n=10,
            latest_mode_date=today,
        )

        # Report prepared-data diagnostics so we can inspect precomputed
        # indicator column names and last-row values for a sample symbol.
        if prepared:
            try:
                sample_sym = next(iter(prepared.keys()))
                sample_df = prepared.get(sample_sym)
                if sample_df is not None and not getattr(sample_df, "empty", True):
                    print("Prepared symbols:", len(prepared), "; sample=", sample_sym)
                    try:
                        print("sample columns:", list(sample_df.columns))
                    except Exception:
                        pass
                    try:
                        print("sample last row:", sample_df.iloc[-1].to_dict())
                    except Exception:
                        pass
            except Exception:
                pass

            # When FILTER_DEBUG is enabled, also run the lightweight
            # filter_system3 pass over the prepared frames to collect
            # per-symbol reason tags (df.attrs['_fdbg_reasons3']) and
            # print a small summary of last-row key fields so debugging
            # is easy without running the whole pipeline.
            try:
                from common.today_filters import filter_system3

                stats: dict = {}
                # Ensure we pass the prepared dict keys (symbols) in a stable order
                prepared_keys = list(prepared.keys())
                _ = filter_system3(prepared_keys, prepared, stats)
                print("--- FILTER DEBUG: system3 per-symbol summary ---")
                print("filter_system3 stats:", stats)
                for sym in sorted(prepared_keys):
                    try:
                        df_sym = prepared.get(sym)
                        reason = None
                        if hasattr(df_sym, "attrs"):
                            try:
                                reasons = df_sym.attrs.get(
                                    "_fdbg_reasons3"
                                )  # type: ignore[attr-defined]
                                if isinstance(reasons, list) and reasons:
                                    reason = reasons[-1]
                            except Exception:
                                reason = None
                        # collect last-row values for common fields
                        last_vals: dict = {}
                        for col in (
                            "Close",
                            "dollarvolume20",
                            "atr_ratio",
                            "drop3d",
                            "sma150",
                        ):
                            val = None
                            try:
                                if df_sym is not None and not getattr(
                                    df_sym, "empty", True
                                ):
                                    # Use get to avoid KeyError
                                    row = df_sym.iloc[-1]
                                    val = row.get(col) if hasattr(row, "get") else None
                            except Exception:
                                val = None
                            last_vals[col] = val
                        print(f"[FDBG system3] {sym} reason={reason} last={last_vals}")
                    except Exception:
                        # Per-symbol best-effort: continue to next symbol on error
                        pass
                print("--- END FILTER DEBUG ---")
            except Exception:
                # Best-effort: ignore debug printing failures
                pass

        # If the strategy returned the tuple form, try to extract today's
        # candidates for simple reporting.
        df = None
        if isinstance(gen_result, tuple) and len(gen_result) >= 1:
            candidates_by_date = gen_result[0]
            # Determine the label_date from diagnostics if available
            diag = getattr(stg, "last_diagnostics", {}) or {}
            label_date = diag.get("label_date")
            if label_date and isinstance(candidates_by_date, dict):
                try:
                    label_ts = pd.Timestamp(label_date)
                except Exception:
                    label_ts = None
                if label_ts is not None:
                    df = candidates_by_date.get(label_ts)
                if df is None:
                    df = candidates_by_date.get(str(label_date))
    except Exception as e:
        print("system3 get_today_signals raised:", e)
        traceback.print_exc()
        return

    print("--- RESULT ---")
    if df is None or getattr(df, "empty", True):
        print("system3 -> empty DataFrame")
    else:
        print(f"system3 -> rows: {len(df)}")
        try:
            print("sample row:", df.iloc[0].to_dict())
        except Exception:
            pass

    print("--- LAST DIAGNOSTICS ---")
    pprint.pprint(getattr(stg, "last_diagnostics", None))

    print("--- COLLECTED LOGS (filtered) ---")
    for log_line in logs:
        condition_a = "System3: DEBUG latest_only 0 candidates." in log_line
        condition_b = log_line.startswith("[DEBUG_S3")
        if condition_a or condition_b:
            print(log_line)

    print("done")


if __name__ == "__main__":
    main()
