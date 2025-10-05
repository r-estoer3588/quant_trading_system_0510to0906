"""compute_base_indicators 内部処理の詳細確認"""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))

from scripts.update_from_bulk_last_day import _merge_existing_full

# テストデータ
new_data = {
    "date": ["2025-10-04"],
    "open": [142.0],
    "high": [143.0],
    "low": [141.0],
    "close": [142.5],
    "volume": [1000000],
}
new_full = pd.DataFrame(new_data)
existing_full = pd.read_csv("data_cache/full_backup/A.csv")

# マージ実行
merged = _merge_existing_full(new_full, existing_full)
print(f"Merged shape: {merged.shape}")
print(f"Merged columns: {list(merged.columns)[:10]}...")

# compute_base_indicatorsの内部処理を再現
x = merged.copy()

# Step 1: Normalize column names
print("\n=== Step 1: Lowercase rename ===")
rename_map = {c: c.lower() for c in x.columns}
x = x.rename(columns=rename_map)
print(f"Columns: {list(x.columns)[:10]}...")
print(f'close type: {type(x["close"])}')
print(f'close is Series: {isinstance(x["close"], pd.Series)}')

# Step 2: Ensure 'Date' column and set as index
print("\n=== Step 2: Date processing ===")
if "date" in x.columns:
    x = x.rename(columns={"date": "Date"})
print(f'Date column exists: {"Date" in x.columns}')
x["Date"] = pd.to_datetime(x["Date"], errors="coerce")
x = x.dropna(subset=["Date"]).sort_values("Date")
print(f"Shape after dropna: {x.shape}")
print(f'close type: {type(x["close"])}')

print("\n=== Step 3: Set index ===")
x = x.set_index("Date")
print(f"Shape: {x.shape}")
print(f"Index type: {type(x.index)}")
print(f'close type: {type(x["close"])}')
print(f'close is Series: {isinstance(x["close"], pd.Series)}')
if isinstance(x["close"], pd.Series):
    print(f'close length: {len(x["close"])}')
    print(f'close sample (last 3): {x["close"].tail(3).tolist()}')

# Step 4: Standardize OHLCV column names
print("\n=== Step 4: OHLCV rename ===")
ohlcv_map = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "adjusted_close": "Close",
    "adj_close": "Close",
    "adjclose": "Close",
    "close": "Close",
    "volume": "Volume",
}
final_rename = {c: ohlcv_map[c] for c in x.columns if c in ohlcv_map}
print(f"Rename map: {final_rename}")
x = x.rename(columns=final_rename)
print(f"Columns after rename: {list(x.columns)[:10]}...")
print(f'Close in columns: {"Close" in x.columns}')

# Step 5: Try pd.to_numeric
print("\n=== Step 5: pd.to_numeric ===")
print(f'x["Close"] type: {type(x["Close"])}')
print(f'x["Close"] is Series: {isinstance(x["Close"], pd.Series)}')

# 問題の箇所を直接確認
print("\nDirect access test:")
print(f"  x is DataFrame: {isinstance(x, pd.DataFrame)}")
print(f"  x.shape: {x.shape}")

close_col = x["Close"]
print(f"  close_col type: {type(close_col)}")
print(f'  close_col shape: {getattr(close_col, "shape", "N/A")}')

# Try to_numeric
try:
    result = pd.to_numeric(close_col, errors="coerce")
    print("✅ pd.to_numeric SUCCESS")
    print(f"  Result type: {type(result)}")
    print(f"  Result length: {len(result)}")
except Exception as e:
    print(f"❌ pd.to_numeric ERROR: {e}")
    print(f"  close_col value: {close_col}")
    print(f"  close_col.__class__: {close_col.__class__}")
