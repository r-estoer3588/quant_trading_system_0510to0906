#!/usr/bin/env python3
"""Collect warnings from a pytest (or arbitrary) log file and append summary JSONL.

Usage:
  python tools/collect_warnings.py --log pytest_warnings.log \
      --out logs/warnings_summary.jsonl [--top 20]

Behavior:
  - Parses lines containing 'Warning:' (case sensitive as pytest output uses that)
  - Normalizes category token (e.g. 'DeprecationWarning')
  - Counts frequency per category
  - Appends one JSON object line with timestamp, total count, per-category counts
  - Prints a brief table to stdout

Design constraints:
  - No heavy deps; use stdlib only
  - Safe when log missing (warn + exit code 0)
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re

WARNING_REGEX = re.compile(r"([A-Za-z0-9_]+Warning):")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect warnings summary from log")
    p.add_argument("--log", required=True, help="Path to pytest warnings log file")
    p.add_argument("--out", required=True, help="JSONL output file path")
    p.add_argument(
        "--top", type=int, default=20, help="Show top-N categories on stdout"
    )
    return p.parse_args()


def collect(log_path: Path) -> Counter:
    counts: Counter[str] = Counter()
    if not log_path.exists():
        print(f"[collect_warnings] log file not found: {log_path}")
        return counts
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "Warning:" not in line:
                continue
            m = WARNING_REGEX.search(line)
            if m:
                counts[m.group(1)] += 1
    return counts


def append_jsonl(out_path: Path, counts: Counter) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = sum(counts.values())
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "total": total,
        "by_category": dict(counts.most_common()),
    }
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(
        f"[collect_warnings] Appended summary: total={total} categories={len(counts)} -> {out_path}"
    )


def main() -> int:
    args = parse_args()
    log_path = Path(args.log)
    out_path = Path(args.out)
    counts = collect(log_path)
    append_jsonl(out_path, counts)

    if counts:
        print("\nTop categories:")
        for cat, c in counts.most_common(args.top):
            print(f"  {cat:<24} {c}")
    else:
        print("[collect_warnings] No warnings found.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
