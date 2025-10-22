import pandas as pd

p = "results_csv_test/system3_filter_debug_all_test.csv"
df = pd.read_csv(p)
print("rows", len(df))
print(df.columns.tolist())
print(df[["symbol", "atr_ratio", "drop3d", "reason"]].head(20).to_string(index=False))
print("\natr_ratio describe:")
print(df["atr_ratio"].describe())
print("\ndrop3d describe:")
print(df["drop3d"].describe())
print("\nreason value counts:")
print(df["reason"].value_counts().to_string())
