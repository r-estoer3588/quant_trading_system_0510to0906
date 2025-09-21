import pandas as pd
from pathlib import Path

p = Path("C:/Users/stair/Downloads/rolling_cache_missing_20250921_155609.csv")
df = pd.read_csv(p)
print("rows", len(df))
# detect symbol column
sym = None
for c in ["Symbol", "symbol", "Ticker"]:
    if c in df.columns:
        sym = c
        break
print("symbol col", sym)
vc = df[sym].value_counts()
top500 = vc.head(500)
print("unique symbols", df[sym].nunique())
# date distribution
if "Date" in df.columns:
    s = pd.to_datetime(df["Date"], errors="coerce")
    print("date min/max/na", s.min(), s.max(), s.isna().sum())
    dd = s.dt.date.value_counts().sort_index()
    print("dates with most misses")
    print(dd.sort_values(ascending=False).head(10))
# write outputs
outdir = Path("data_cache/rolling/_missing_reports")
outdir.mkdir(parents=True, exist_ok=True)
(
    top500.reset_index()
    .rename(columns={"index": sym, sym: "count"})
    .to_csv(outdir / "top500_missing.csv", index=False)
)

with open(outdir / "distro_top_dates.txt", "w", encoding="utf-8") as f:
    if "Date" in df.columns:
        for d, c in dd.items():
            f.write(f"{d}, {c}\n")
print("wrote outputs to", outdir)
