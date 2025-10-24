import json
from pathlib import Path

p = Path("results_csv")
if not p.exists():
    print("results_csv not exists")
else:
    files = list(p.glob("exit_counts_*.json"))
    print("found", len(files))
    for f in files[:5]:
        try:
            print(f, json.loads(f.read_text(encoding="utf-8")))
        except Exception as e:
            print("err", f, e)
