# c:\Repos\quant_trading_system\tools\analyze_system3_csv_stage_counts.py
# Quick helper to load the CSV produced by debug_system3_all and simulate the pipeline filters
import argparse
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument(
    "csvfile",
    nargs="?",
    default="results_csv_test/system3_filter_debug_all_20251021_145535.csv",
)
p.add_argument("--drop3d-threshold", type=float, default=0.125)
args = p.parse_args()

csv = args.csvfile
print("Reading:", csv)

df = pd.read_csv(csv)
print("Total rows:", len(df))

# prefilters: low_ok & avgvol_ok & atr_ok from CSV
prefilter = df[
    (df["low_ok"] == True) & (df["avgvol_ok"] == True) & (df["atr_ok"] == True)
]
print("Prefilter pass (all three true):", len(prefilter))

# setup conditions: close > sma150 (if sma150 present)
close_vs_sma = prefilter.dropna(subset=["sma150"])
close_vs_sma = close_vs_sma[close_vs_sma["close"] > close_vs_sma["sma150"]]
print("Close > SMA150 (on rows with sma150):", len(close_vs_sma))

# drop3d >= threshold (drop3d may be negative/positive; treat as numeric)
prefilter_numeric_drop3d = prefilter[
    pd.to_numeric(prefilter["drop3d"], errors="coerce").notnull()
]
print("Prefilter rows with numeric drop3d:", len(prefilter_numeric_drop3d))

pass_drop3d = prefilter_numeric_drop3d[
    pd.to_numeric(prefilter_numeric_drop3d["drop3d"]) >= args.drop3d_threshold
]
print(f"Prefilter rows with drop3d >= {args.drop3d_threshold}:", len(pass_drop3d))

# combine: prefilter & close_vs_sma150 & drop3d>=threshold
# must ensure sma150 present for close_vs_sma
combined = prefilter.copy()
combined = combined[pd.to_numeric(combined["drop3d"], errors="coerce").notnull()]
combined = combined[combined["close"] > combined["sma150"]]
combined = combined[pd.to_numeric(combined["drop3d"]) >= args.drop3d_threshold]
print("Combined (prefilter & close>sma150 & drop3d>=threshold):", len(combined))

# show few examples
print(
    "\nExamples (top 20 by dollarvolume20) where prefilter passed but drop3d < threshold:"
)
low_drop = prefilter_numeric_drop3d[
    pd.to_numeric(prefilter_numeric_drop3d["drop3d"]) < args.drop3d_threshold
]
low_drop = low_drop.sort_values("dollarvolume20", ascending=False)
print(
    low_drop[["symbol", "atr_ratio", "drop3d", "dollarvolume20", "sma150", "close"]]
    .head(20)
    .to_string(index=False)
)

print("\nDone.")
