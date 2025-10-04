#!/usr/bin/env python
"""Compare two perf snapshot JSON files and print delta summary.

Usage (example):
  python scripts/compare_perf_snapshots.py path/to/perf_A.json path/to/perf_B.json

Output:
  - Header with file timestamps & latest_only flags
  - Overall total_time delta (abs & percent)
  - Cache IO diff table
  - Per-system table (elapsed, symbol_count, candidate_count, delta sec, delta %)

Design principles:
  - Pure stdlib (json, pathlib)
  - Safe if keys missing (treat as 0 / None)
  - No external side effects other than stdout
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any, Dict


def _load(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # pragma: no cover - defensive
        raise SystemExit(f"Failed to load JSON {p}: {e}") from e


def _fmt_float(v: float | None, nd=4) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "-"
    return f"{v:.{nd}f}"


def _percent(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    if a == 0:
        return None
    return (b - a) / a * 100.0


def compare(a: Dict[str, Any], b: Dict[str, Any]) -> None:
    print("=== Perf Snapshot Compare ===")
    print(
        f"File A: ts={a.get('timestamp')} latest_only={a.get('latest_only')} total_time={_fmt_float(a.get('total_time_sec'))}"
    )
    print(
        f"File B: ts={b.get('timestamp')} latest_only={b.get('latest_only')} total_time={_fmt_float(b.get('total_time_sec'))}"
    )
    tA = a.get("total_time_sec")
    tB = b.get("total_time_sec")
    pct = _percent(tA, tB)
    if isinstance(tA, (int, float)) and isinstance(tB, (int, float)):
        print(
            f"Total Time Delta: {tB - tA:+.4f}s ({pct:.2f}% if baseline A)"
            if pct is not None
            else f"Total Time Delta: {tB - tA:+.4f}s"
        )
    print()

    # Cache IO diff
    print("-- Cache IO --")
    ioA = a.get("cache_io", {}) or {}
    ioB = b.get("cache_io", {}) or {}
    keys = sorted(set(ioA) | set(ioB))
    for k in keys:
        vA = ioA.get(k, 0)
        vB = ioB.get(k, 0)
        print(f"{k:14s} A={vA:5} B={vB:5} Δ={vB - vA:+5}")
    print()

    # Per-system comparison
    print("-- Per-System --")
    psA = a.get("per_system", {}) or {}
    psB = b.get("per_system", {}) or {}
    systems = sorted(set(psA) | set(psB))
    header = f"{'system':10s} {'A_time':>10s} {'B_time':>10s} {'Δs':>9s} {'Δ%':>8s} {'A_sym':>7s} {'B_sym':>7s} {'A_cand':>7s} {'B_cand':>7s}"
    print(header)
    print("-" * len(header))
    for s in systems:
        eA = psA.get(s, {}) or {}
        eB = psB.get(s, {}) or {}
        a_time = eA.get("elapsed_sec")
        b_time = eB.get("elapsed_sec")
        delta = None
        if isinstance(a_time, (int, float)) and isinstance(b_time, (int, float)):
            delta = b_time - a_time
        pct2 = _percent(a_time, b_time)
        line = f"{s:10s} {_fmt_float(a_time):>10s} {_fmt_float(b_time):>10s} {_fmt_float(delta):>9s} {_fmt_float(pct2):>8s} {str(eA.get('symbol_count')):>7s} {str(eB.get('symbol_count')):>7s} {str(eA.get('candidate_count')):>7s} {str(eB.get('candidate_count')):>7s}"
        print(line)
    print("\nDone.")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(
            "Usage: python scripts/compare_perf_snapshots.py <perf_A.json> <perf_B.json>"
        )
        return 1
    a = _load(argv[1])
    b = _load(argv[2])
    compare(a, b)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))
