"""Report mapping of persisted -> final assignment for duplicated symbols.

Reads per-system persisted files (`results_csv_test/per_system_*.feather`) and
the final signals CSV, then reports for each symbol that was present in one or
more persisted lists but appears in final under a different system.

Run from repo root:
  python tools/resolve_allocation_conflicts.py
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    results_dir = ROOT / "results_csv_test"
    per_system = {}
    for p in results_dir.glob("per_system_*.feather"):
        name = p.stem.replace("per_system_", "")
        try:
            df = pd.read_feather(p)
        except Exception:
            continue
        if "symbol" in df.columns:
            syms = set(df["symbol"].astype(str).tolist())
        else:
            possible = [c for c in df.columns if c.lower() == "symbol"]
            syms = set(df[possible[0]].astype(str).tolist()) if possible else set()
        per_system[name] = syms

    # load final CSV (latest)
    final_dir = ROOT / "data_cache" / "signals"
    files = sorted(final_dir.glob("signals_final_*.csv"))
    if not files:
        print("No final signals CSV found")
        return 2
    final = pd.read_csv(files[-1])

    final_map = defaultdict(set)
    for _, r in final.iterrows():
        final_map[str(r["system"])].add(str(r["symbol"]))

    # build inverse: symbol -> final system(s)
    sym_to_final = defaultdict(set)
    for sysn, syms in final_map.items():
        for s in syms:
            sym_to_final[s].add(sysn)

    # find symbols that were persisted but not in final for that same system
    report = {}
    for sysn, persisted in per_system.items():
        kept = sorted(list(persisted & final_map.get(sysn, set())))
        dropped = sorted(list(persisted - final_map.get(sysn, set())))
        moved = []
        for s in dropped:
            finals = sorted(list(sym_to_final.get(s, set())))
            moved.append({"symbol": s, "final_systems": finals})
        report[sysn] = {
            "persisted_count": len(persisted),
            "kept": kept,
            "dropped": dropped,
            "moved_info": moved,
        }

    out = (
        ROOT
        / "results_csv_test"
        / (
            f"allocation_conflicts_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        )
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        from common.io_utils import write_json

        write_json(out, report, ensure_ascii=False, indent=2)
    except Exception:
        out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8"
        )
    print(f"Wrote report: {out}")
    # also print concise summary for user
    for sysn, info in report.items():
        print("\n" + "=" * 40)
        print(
            f"System {sysn}: persisted={info['persisted_count']} "
            f"kept={len(info['kept'])} dropped={len(info['dropped'])}"
        )
        if info["dropped"]:
            for m in info["moved_info"]:
                finals = m["final_systems"] or ["(none)"]
                print(f"  - {m['symbol']} -> final: {', '.join(finals)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
