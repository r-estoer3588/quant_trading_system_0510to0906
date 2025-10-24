"""Summarize the per-symbol trace files created by
`tools/extract_system3_symbol_traces.py`.

Writes a CSV to stdout and a JSON summary to
`results_csv_test/system3_symbol_traces_20251022.summary.json`.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional


def parse_trace_file(p: Path) -> Dict[str, Optional[str]]:
    text = p.read_text(encoding="utf-8")
    # split sections
    parts = text.split("--- tail rows (csv) ---")
    if len(parts) < 2:
        return {
            "symbol": p.stem,
            "reason_present": None,
            "close": None,
            "drop3d": None,
            "atr_ratio": None,
        }
    tail_and_rest = parts[1]
    if "--- _fdbg_reasons3 ---" in tail_and_rest:
        tail_block, rest = tail_and_rest.split("--- _fdbg_reasons3 ---", 1)
    else:
        # no reasons section
        tail_block = tail_and_rest
        rest = ""

    # parse CSV tail_block: find first newline then header+rows
    tail_lines = [ln for ln in tail_block.splitlines() if ln.strip()]
    if not tail_lines:
        header = None
        last_vals: dict[str, str] = {}
    else:
        # The CSV block includes header at index 0 and rows thereafter
        header = tail_lines[0].split(",")
        # find last non-empty row
        last_row = None
        for ln in reversed(tail_lines[1:]):
            if ln.strip():
                last_row = ln
                break
        last_vals_local: dict[str, str] = {}
        if last_row is not None:
            vals = [v for v in last_row.split(",")]
            for i, col in enumerate(header):
                try:
                    last_vals_local[col.strip()] = vals[i].strip()
                except Exception:
                    last_vals_local[col.strip()] = ""
        last_vals = last_vals_local

    # Prefer a persistent debug column if present in the tail CSV
    reason_present = None
    if last_vals and "_dbg_reasons3" in last_vals:
        reason_present = last_vals.get("_dbg_reasons3")
    else:
        # fallback to the legacy _fdbg_reasons3 section
        if rest:
            r = rest.strip().splitlines()
            if r:
                # first non-empty line is representation of reasons
                reason_present = r[0].strip()

    return {
        "symbol": p.stem,
        "reason_present": reason_present,
        "close": last_vals.get("Close") if last_vals else None,
        "drop3d": last_vals.get("drop3d") if last_vals else None,
        "atr_ratio": last_vals.get("atr_ratio") if last_vals else None,
    }


def main() -> None:
    out_dir = Path("results_csv_test/system3_symbol_traces_20251022")
    if not out_dir.exists():
        print("Trace dir not found:", out_dir)
        raise SystemExit(1)
    rows: List[Dict[str, Optional[str]]] = []
    for p in sorted(out_dir.iterdir()):
        if p.suffix.lower() != ".txt":
            continue
        rows.append(parse_trace_file(p))

    # print CSV to stdout
    fieldnames = ["symbol", "reason_present", "close", "drop3d", "atr_ratio"]
    w = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow({k: (r.get(k) or "") for k in fieldnames})

    # save JSON summary
    summary_p = out_dir.parent / (out_dir.name + ".summary.json")
    with summary_p.open("w", encoding="utf-8") as fh:
        json.dump({"count": len(rows), "rows": rows}, fh, ensure_ascii=False, indent=2)
    print(f"Wrote summary JSON: {summary_p}")


if __name__ == "__main__":
    main()
