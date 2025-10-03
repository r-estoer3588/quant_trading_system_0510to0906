import random
from config.settings import get_settings
from common.cache_manager import CacheManager

st = get_settings()
cm = CacheManager(st)
paths = list(cm.rolling_dir.glob("*.feather"))
print("total rolling files:", len(paths))
if not paths:
    raise SystemExit("No rolling files present.")
random.shuffle(paths)
sample = paths[:12]
cols_interest = [
    "Date",
    "Close",
    "Volume",
    "DollarVolume50",
    "HV50",
    "AvgVolume50",
    "ATR_Pct",
    "ATR_Ratio",
]
report = []
for p in sample:
    sym = p.stem
    try:
        df = cm.read(sym, "rolling")
    except Exception as e:
        print(f"!! read fail {sym}: {e}")
        continue
    if df is None or df.empty:
        print(f"!! empty {sym}")
        continue
    present = [c for c in cols_interest if c in df.columns]
    tail = df[present].tail(3).copy()
    last = tail.iloc[-1]

    def fmt(v):
        try:
            if v != v:
                return "NaN"
            if isinstance(v, (int, float)):
                if abs(v) >= 1e6:
                    return f"{v/1e6:.2f}M"
                return f"{v:.4g}"
            return str(v)
        except Exception:
            return "?"

    summary = {
        "sym": sym,
        "dv_raw": fmt(last.get("DollarVolume50")),
        "hv_raw": fmt(last.get("HV50")),
        "av_raw": fmt(last.get("AvgVolume50")),
        "atr_raw": fmt(last.get("ATR_Pct", last.get("ATR_Ratio"))),
        "has_ATR_Pct": "ATR_Pct" in df.columns,
        "has_ATR_Ratio": "ATR_Ratio" in df.columns,
        "row_date": str(last.get("Date", "?")),
    }
    report.append((sym, tail, summary))

for sym, tail, summary in report:
    print("\n===", sym, "===")
    print(tail)
    print("summary:", summary)

nan_counts = {"dv": 0, "hv": 0, "av": 0, "atr": 0, "total": 0}
for _, _, s in report:
    nan_counts["total"] += 1
    if s["dv_raw"] == "NaN":
        nan_counts["dv"] += 1
    if s["hv_raw"] == "NaN":
        nan_counts["hv"] += 1
    if s["av_raw"] == "NaN":
        nan_counts["av"] += 1
    if s["atr_raw"] == "NaN":
        nan_counts["atr"] += 1
print("\nNaN summary:", nan_counts)
