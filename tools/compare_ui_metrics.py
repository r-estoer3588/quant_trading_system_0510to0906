"""Simple JSON comparator for UI metrics snapshots.

Usage:
  python tools/compare_ui_metrics.py base.json new.json

Prints per-system differences in a compact form. Designed to be used by Playwright
scripts as a lightweight diff step.
"""
from __future__ import annotations

import argparse
import json
from typing import Any


def load(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_dicts(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    msgs: list[str] = []
    keys = sorted(set(a.keys()) | set(b.keys()))
    for k in keys:
        va = a.get(k)
        vb = b.get(k)
        if va == vb:
            continue
        msgs.append(f"[{k}]\n  - base: {va}\n  - new : {vb}")
    return msgs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("base")
    p.add_argument("new")
    args = p.parse_args()
    base = load(args.base)
    new = load(args.new)
    diffs = compare_dicts(base, new)
    if not diffs:
        print("NO_DIFFS")
        return
    print("DIFFS:")
    for d in diffs:
        print(d)


if __name__ == "__main__":
    main()
