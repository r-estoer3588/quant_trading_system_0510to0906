import json
from pathlib import Path

import pandas as pd

files = [
    Path("results_csv_test/per_system_system1.feather"),
    Path("results_csv_test/per_system_system4.feather"),
]

out = {}
for p in files:
    key = p.name
    out[key] = {"path": str(p), "exists": p.exists()}
    if p.exists():
        try:
            df = pd.read_feather(p)
            recs = df.to_dict(orient="records")
            # stringify non-serializable values (Timestamp)
            for r in recs:
                for k, v in r.items():
                    if hasattr(v, "isoformat"):
                        try:
                            r[k] = v.isoformat()
                        except Exception:
                            r[k] = str(v)
            out[key].update({"shape": df.shape, "columns": list(df.columns), "records": recs})
        except Exception as e:
            out[key].update({"error": repr(e)})

print(json.dumps(out, ensure_ascii=False, indent=2))
