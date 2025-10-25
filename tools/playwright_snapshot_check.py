#!/usr/bin/env python3
"""Playwright helper to compare latest UI metrics snapshot against a baseline.

Usage:
  python tools/playwright_snapshot_check.py --baseline path/to/baseline.json [--new path/to/new.json]

If --new is omitted, the script finds the newest file matching
`results_csv/ui_metrics_*.json` and compares it to the baseline.

Exit codes:
  0 : no diffs
  1 : usage / missing files / other error
  2 : diffs found

Designed to be invoked from Playwright/CI as a post-step.
"""
from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from pathlib import Path


def find_latest_snapshot(results_dir: Path) -> Path | None:
    pattern = str(results_dir / "ui_metrics_*.json")
    files = glob.glob(pattern)
    if not files:
        return None
    files_sorted = sorted(files, key=lambda p: Path(p).stat().st_mtime)
    return Path(files_sorted[-1])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", "-b", default="results_csv/ui_metrics_baseline.json")
    p.add_argument("--new", "-n", default=None)
    p.add_argument(
        "--ignore-keys",
        "-i",
        default=None,
        help="comma-separated top-level keys to ignore when comparing",
    )
    args = p.parse_args()

    baseline = Path(args.baseline)
    if args.new is None:
        results_dir = Path("results_csv")
        new_fp = find_latest_snapshot(results_dir)
        if new_fp is None:
            print(
                "ERROR: no ui_metrics_*.json snapshots found in results_csv",
                file=sys.stderr,
            )
            return 1
        new = new_fp
    else:
        new = Path(args.new)

    if not baseline.exists():
        print(f"ERROR: baseline not found: {baseline}", file=sys.stderr)
        return 1
    if not new.exists():
        print(f"ERROR: new snapshot not found: {new}", file=sys.stderr)
        return 1

    # run the comparator script
    comp = Path(__file__).parent / "compare_ui_metrics.py"
    if not comp.exists():
        print(f"ERROR: comparator script not found: {comp}", file=sys.stderr)
        return 1

    cmd = [sys.executable, str(comp), str(baseline), str(new)]
    if args.ignore_keys:
        cmd += ["--ignore-keys", args.ignore_keys]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if stdout:
        print(stdout)
    if stderr:
        print(stderr, file=sys.stderr)

    # decide exit code based on comparator output
    if "NO_DIFFS" in (stdout or ""):
        return 0
    # comparator prints DIFFS: then details
    if proc.returncode == 0 and stdout:
        # comparator printed diffs but still returned 0; treat as diffs
        return 2
    return proc.returncode or 2


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
