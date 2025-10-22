"""Run System3 candidate generation for symbols listed in a diagnostics snapshot.

This uses the official core functions (prepare_data_vectorized_system3 and
generate_candidates_system3) with default thresholds (no changes) and writes the
returned diagnostics to a JSON file for inspection.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.system3 import prepare_data_vectorized_system3, generate_candidates_system3


def _parse_setlike(s):
    if s is None:
        return []
    if isinstance(s, (list, tuple, set)):
        return list(map(str, s))
    if isinstance(s, str):
        t = s.strip()
        if t.startswith("{") and t.endswith("}"):
            inner = t[1:-1].strip()
            if not inner:
                return []
            parts = [p.strip().strip("'\"") for p in inner.split(",")]
            return [p for p in parts if p]
    return []


def main():
    snap_p = Path(
        "results_csv_test/diagnostics_test/diagnostics_snapshot_20251022_135717.json"
    )
    if not snap_p.exists():
        print("snapshot missing", snap_p)
        return
    snap = json.loads(snap_p.read_text(encoding="utf-8"))
    systems = snap.get("systems") or []
    sys3 = next((s for s in systems if s.get("system_id") == "system3"), None)
    if not sys3:
        print("no system3 in snapshot")
        return
    excl = (sys3.get("diagnostics_extra") or {}).get("exclude_symbols") or {}
    drop = _parse_setlike(excl.get("drop3d"))
    close = _parse_setlike(excl.get("close_vs_sma150"))
    symbols = sorted(set(drop + close))

    print(f"Preparing data for {len(symbols)} symbols: {symbols}")
    prepared = prepare_data_vectorized_system3(
        None, reuse_indicators=True, symbols=symbols
    )
    print(f"Prepared dict size: {len(prepared)}")

    _res = generate_candidates_system3(
        prepared, latest_only=True, include_diagnostics=True
    )
    # include_diagnostics=True returns a 3-tuple
    if isinstance(_res, tuple) and len(_res) == 3:
        by_date, df_all, diagnostics = _res
    else:
        by_date, df_all = _res
        diagnostics = {}

    out_p = Path("results_csv_test/system3_diagnostics_snapshot_symbols_20251022.json")
    out_p.write_text(json.dumps(diagnostics, default=str, indent=2), encoding="utf-8")
    print(f"Wrote diagnostics: {out_p}")


if __name__ == "__main__":
    main()
