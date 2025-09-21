import pandas as pd

p = r"C:/Users/stair/Downloads/rolling_cache_missing_20250921_155609.csv"
df = pd.read_csv(p)
# find symbol column
sym = None
for c in ["Symbol", "symbol", "Ticker"]:
    if c in df.columns:
        sym = c
        break
print("rows", len(df))
print("symbol_col", sym)
if "Date" in df.columns:
    s = pd.to_datetime(df["Date"], errors="coerce")
    dist = s.dt.date.value_counts().sort_index()
    print("\nDate range:", s.min(), "->", s.max(), " NaNs=", s.isna().sum())
    print("\nTop dates by missing count:")
    print(dist.tail(20))
else:
    print("No Date column")
if sym:
    vc = df[sym].value_counts()
    top = vc.head(20)
    print("\nTop 20 symbols by missing count:")
    print(top)

    # simple liquidity proxy
    def lp(sym):
        if sym == "SPY":
            return 1000
        if len(sym) == 1:
            return 500
        return 1

    impact = vc.reset_index().rename(columns={"index": sym, sym: "missing_count"})
    impact["liquidity_proxy"] = impact[sym].apply(lp)
    impact["impact_score"] = impact["missing_count"] * impact["liquidity_proxy"]
    print("\nTop 20 by impact_score:")
    print(impact.sort_values("impact_score", ascending=False).head(20))
