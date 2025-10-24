from pathlib import Path

import pandas as pd

p = Path("data_cache/signals/signals_final_2025-10-21.csv")
print("exists", p.exists(), p)
if p.exists():
    df = pd.read_csv(p)
    print("shape", df.shape)
    print(df.to_dict(orient="records"))
else:
    print("file not found")
