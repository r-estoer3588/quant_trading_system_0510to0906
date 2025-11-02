#!/usr/bin/env python3
"""
Small helper to extract examples from the System3 FILTER_DEBUG CSV where
pre-filters passed but the 3-day drop (drop3d) is below the System3 setup
threshold (default 0.125) or missing. Prints overall counts and up to N
representative example rows.
"""
import argparse
import csv
from pathlib import Path
import statistics


def parse_float(s):
    if s is None:
        return None
    s = s.strip()
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "csvfile",
        nargs="?",
        default="results_csv_test/system3_filter_debug_all_20251021_145535.csv",
    )
    p.add_argument("--threshold", type=float, default=0.125)
    p.add_argument("-n", type=int, default=10)
    args = p.parse_args()

    path = Path(args.csvfile)
    if not path.exists():
        print(f"ERROR: CSV not found: {path}")
        return 2

    total_rows = 0
    total_pass = 0
    pass_below = []
    pass_missing = []
    pass_drop_vals = []

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        _ = next(reader, None)
        for row in reader:
            total_rows += 1
            if len(row) < 10:
                continue
            reason = (row[5] or "").strip().lower()
            if not reason.startswith("pass"):
                continue
            total_pass += 1
            symbol = row[0]
            close = row[6] if len(row) > 6 else ""
            dollarvol = parse_float(row[7]) if len(row) > 7 else None
            atr_ratio = parse_float(row[8]) if len(row) > 8 else None
            drop3d = parse_float(row[9]) if len(row) > 9 else None
            sma150 = row[10] if len(row) > 10 else ""

            entry = {
                "symbol": symbol,
                "reason": reason,
                "atr_ratio": atr_ratio,
                "drop3d": drop3d,
                "dollarvol20": dollarvol,
                "sma150": sma150,
                "close": close,
            }
            if drop3d is None:
                pass_missing.append(entry)
            else:
                pass_drop_vals.append(drop3d)
                if drop3d < args.threshold:
                    pass_below.append(entry)

    print(f"Total CSV rows: {total_rows}")
    print(
        f"Total rows where pre-filters passed (reason startswith 'pass'): {total_pass}"
    )
    print(f"pass rows with numeric drop3d: {len(pass_drop_vals)}")
    print(f"pass rows where drop3d < {args.threshold}: {len(pass_below)}")
    print(f"pass rows where drop3d is missing (NaN): {len(pass_missing)}")
    print()

    if pass_drop_vals:

        def q(v, q=0.5):
            return statistics.median(v) if v else None

        print("drop3d (pass rows) stats:")
        print(f"  min: {min(pass_drop_vals):.5f}")
        print(f"  median: {statistics.median(pass_drop_vals):.5f}")
        print(f"  mean: {statistics.mean(pass_drop_vals):.5f}")
        print(f"  max: {max(pass_drop_vals):.5f}")
        print()

    if pass_below:
        # Pick representative examples: sort by dollar volume desc (treat None as 0)
        pass_below_sorted = sorted(
            pass_below, key=lambda r: (r["dollarvol20"] or 0.0), reverse=True
        )
        print(
            f"Top {args.n} examples where prefilters passed but drop3d < "
            f"{args.threshold} (sorted by 20-day $ volume):"
        )
        print("symbol, atr_ratio, drop3d, dollarvol20, sma150, close")
        for e in pass_below_sorted[: args.n]:
            print(
                f"{e['symbol']}, {e['atr_ratio']}, {e['drop3d']}, "
                f"{e['dollarvol20']}, {e['sma150']}, {e['close']}"
            )
        print()
    else:
        print(f"No rows found where prefilters passed but drop3d < {args.threshold}.")
        print()

    if pass_missing:
        # show examples with missing drop3d
        # (sorted by atr_ratio ascending to show low atr)
        pass_missing_sorted = sorted(
            pass_missing, key=lambda r: (r["atr_ratio"] is None, r["atr_ratio"] or 0.0)
        )
        print(f"Top {args.n} examples where prefilters passed but drop3d is missing:")
        print("symbol, atr_ratio, drop3d, dollarvol20, sma150, close")
        for e in pass_missing_sorted[: args.n]:
            print(
                f"{e['symbol']}, {e['atr_ratio']}, {e['drop3d']}, "
                f"{e['dollarvol20']}, {e['sma150']}, {e['close']}"
            )
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
