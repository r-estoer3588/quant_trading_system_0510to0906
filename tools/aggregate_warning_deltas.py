#!/usr/bin/env python3
"""Aggregate warning deltas between latest two JSONL entries produced by collect_warnings.

Usage:
  python tools/aggregate_warning_deltas.py --file logs/warnings_summary.jsonl

Outputs a small markdown-style diff to stdout. Safe if file missing or <2 lines.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True, help="warnings_summary.jsonl path")
    return p.parse_args()


def load_last_two(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    return [json.loads(line) for line in lines[-2:]]


def diff(prev: Dict, curr: Dict) -> List[Tuple[str, int, int, int]]:
    prev_map = prev.get("by_category", {}) if prev else {}
    curr_map = curr.get("by_category", {}) if curr else {}
    cats = set(prev_map) | set(curr_map)
    rows: List[Tuple[str, int, int, int]] = []
    for c in sorted(cats):
        before = int(prev_map.get(c, 0))
        after = int(curr_map.get(c, 0))
        delta = after - before
        rows.append((c, before, after, delta))
    return rows


def main() -> int:
    args = parse_args()
    entries = load_last_two(Path(args.file))
    if len(entries) < 2:
        print("(no delta: need at least two records)")
        return 0
    prev, curr = entries[0], entries[1]
    rows = diff(prev, curr)
    inc = [r for r in rows if r[3] > 0]
    dec = [r for r in rows if r[3] < 0]
    print("# Warning Delta Report")
    print(f"Prev ts: {prev.get('ts')} -> Curr ts: {curr.get('ts')}")
    if not rows:
        print("(no categories)")
        return 0

    def fmt(r):
        sign = "+" if r[3] > 0 else ""
        return f"| {r[0]:25} | {r[1]:5} | {r[2]:5} | {sign}{r[3]:+4} |"

    print(
        "\n| Category                  | Prev  | Curr  | Δ    |\n|---------------------------|-------|-------|------|"
    )
    for r in rows:
        print(fmt(r))
    print("\nIncreased:")
    for r in inc:
        print(f"  {r[0]}: +{r[3]}")
    print("Decreased:")
    for r in dec:
        print(f"  {r[0]}: {r[3]}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
