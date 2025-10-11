import pandas as pd

from core.system6 import generate_candidates_system6

dates = pd.date_range("2023-01-01", periods=5, freq="D")
data_dict = {
    "AAPL": pd.DataFrame(
        {
            "Close": [150.0, 152.0, 148.0, 151.0, 153.0],
            "return_6d": [0.05, 0.08, 0.12, 0.10, 0.15],
            "atr10": [2.0, 2.1, 2.05, 2.15, 2.2],
            "setup": [True] * 5,
        },
        index=dates,
    ),
}

print("DataFrame info:")
df = data_dict["AAPL"]
print(f"Index: {df.index}")
print(f"Index type: {type(df.index)}")
print(f"Columns: {df.columns.tolist()}")
print("\nLast row:")
print(df.iloc[-1])
print(f'\nsetup column type: {type(df.iloc[-1]["setup"])}')
print(f'setup value: {df.iloc[-1]["setup"]}')
print(f'bool(setup): {bool(df.iloc[-1].get("setup"))}')

result = generate_candidates_system6(data_dict, top_n=10, latest_only=True)
candidates = result[0]
print(f"\nCandidates count: {len(candidates)}")
print(f"Candidates: {candidates}")
