import pandas as pd

p = "results_csv_test/system3_filter_debug_all_test.csv"
df = pd.read_csv(p)
total = len(df)
print("total symbols:", total)
# clean columns
for col in ["atr_ratio", "drop3d"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
# counts
atr_ge_005 = df["atr_ratio"] >= 0.05
drop_ge_0125 = df["drop3d"] >= 0.125
close_ok = df.get("low_ok", False).astype(bool)
av_ok = df.get("avgvol_ok", False).astype(bool)
filter_ok = close_ok & av_ok & (df["atr_ratio"] >= 0.05)
setup_ok = filter_ok & drop_ge_0125
print("\nCounts:")
print("atr_ratio >= 0.05 :", int(atr_ge_005.sum()), f"({atr_ge_005.mean()*100:.1f}%)")
print("drop3d >= 0.125 :", int(drop_ge_0125.sum()), f"({drop_ge_0125.mean()*100:.3f}%)")
print("close_ok (Low>=1) :", int(close_ok.sum()))
print("avgvol_ok (AvgVol50>=1M) :", int(av_ok.sum()))
print("filter_ok (close & avgvol & atr>=0.05) :", int(filter_ok.sum()))
print("setup_ok (filter & drop3d>=0.125) :", int(setup_ok.sum()))
print("\nExamples of setup_ok symbols:")
print(df.loc[setup_ok, "symbol"].dropna().head(20).to_list())
print("\nTop 20 by drop3d:")
print(
    df.sort_values("drop3d", ascending=False)
    .loc[:, ["symbol", "drop3d", "atr_ratio"]]
    .head(20)
    .to_string(index=False)
)
print("\natr_ratio quantiles:")
print(df["atr_ratio"].quantile([0.25, 0.5, 0.75, 0.9, 0.99]).to_string())
print("\ndrop3d quantiles:")
print(df["drop3d"].quantile([0.25, 0.5, 0.75, 0.9, 0.99]).to_string())
