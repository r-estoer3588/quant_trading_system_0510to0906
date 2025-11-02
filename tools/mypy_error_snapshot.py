#!/usr/bin/env python3
"""Run mypy and append a compact JSONL snapshot of error metrics.

Usage:
  python tools/mypy_error_snapshot.py --out logs/mypy_errors.jsonl [--mypy-args "..."]

Behavior:
  - Executes `mypy` in a subprocess (non-failing; captures stdout+stderr)
  - Counts total error lines (`error:`) and groups by error code (if present as [code])
  - Appends JSON line: { ts, total_errors, by_code: {code: count}, sample: [first N lines] }
  - Prints one-line summary for CI Step Summary consumption.

Design:
  - Lightweight, no external dependencies
  - Resilient: if mypy not installed or command fails, records failure flag
  - Encoding safe for Windows (force utf-8, replacement errors)
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List

ERROR_CODE_RE = re.compile(r"error: .* \[(?P<code>[a-zA-Z0-9_-]+)\]")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Snapshot mypy errors into JSONL")
    p.add_argument("--out", required=True, help="Output JSONL file path")
    p.add_argument(
        "--mypy-args",
        default="--python-version 3.11 --show-error-codes",
        help="Extra arguments passed to mypy (string)",
    )
    p.add_argument("--sample", type=int, default=25, help="Sample first N error lines")
    return p.parse_args()


def run_mypy(extra: str) -> List[str]:
    cmd = f"mypy {extra}".strip()
    try:
        proc = subprocess.run(
            shlex.split(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return proc.stdout.splitlines()
    except FileNotFoundError:
        return ["mypy: command not found"]


def summarize(lines: List[str], sample: int):
    total = 0
    by_code: dict[str, int] = {}
    error_lines: List[str] = []
    for line in lines:
        if "error:" not in line:
            continue
        total += 1
        m = ERROR_CODE_RE.search(line)
        if m:
            code = m.group("code")
            by_code[code] = by_code.get(code, 0) + 1
        error_lines.append(line)
    return {
        "total_errors": total,
        "by_code": dict(sorted(by_code.items(), key=lambda x: (-x[1], x[0]))),
        "sample": error_lines[:sample],
    }


def append_jsonl(out_path: Path, snapshot: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        **snapshot,
    }
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    lines = run_mypy(args.mypy_args)
    snapshot = summarize(lines, args.sample)
    append_jsonl(Path(args.out), snapshot)
    print(
        f"[mypy_error_snapshot] total_errors={snapshot['total_errors']} unique_codes={len(snapshot['by_code'])} -> {args.out}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
