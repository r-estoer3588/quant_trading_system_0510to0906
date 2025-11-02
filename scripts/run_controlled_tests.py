#!/usr/bin/env python3
"""Controlled pipeline test orchestrator (deterministic).

Actions:
    1) Generate test symbols deterministically (current set: 13 + SPY rolling)
    2) Run pipeline in test mode: --test-mode test_symbols --skip-external --save-csv
    3) Validate outputs:
         - validation_report_*.json has summary.errors == 0
             - signals_final_*.csv exists under settings.outputs.signals_dir
                 and has expected deterministic row count (currently 1)

Exit codes:
    0 = success
    2 = generation or pipeline failed
    3 = validation failed
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import get_settings


def _run_generator() -> None:
    # Import and run directly to avoid shell quirks on Windows
    try:
        from tools.generate_test_symbols import generate_test_symbols  # noqa: E402

        generate_test_symbols()
    except Exception:
        # fallback to subprocess execution if import path is problematic
        cmd = [sys.executable, "tools/generate_test_symbols.py"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout + "\n" + proc.stderr)
            raise SystemExit(2)


def _run_pipeline() -> None:
    cmd = [
        sys.executable,
        "scripts/run_all_systems_today.py",
        "--test-mode",
        "test_symbols",
        "--skip-external",
        "--save-csv",
        "--benchmark",
    ]
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        raise SystemExit(2)


def _latest_file(root: Path, pattern: str) -> Optional[Path]:
    files = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _validate_outputs() -> None:
    # Resolve directories from settings for robustness
    settings = get_settings(create_dirs=False)

    # validation_report lives under settings.RESULTS_DIR/validation
    val_dir = Path(settings.RESULTS_DIR) / "validation"
    val_file = _latest_file(val_dir, "validation_report_*.json")
    if not val_file or not val_file.exists():
        sys.stderr.write(f"validation report not found under {val_dir}\n")
        raise SystemExit(3)
    try:
        report = json.loads(val_file.read_text(encoding="utf-8"))
        errs = int(report.get("summary", {}).get("errors", 0))
        if errs != 0:
            sys.stderr.write(f"validation errors present: {errs} in {val_file.name}\n")
            raise SystemExit(3)
    except Exception as e:
        sys.stderr.write(f"failed to parse validation report: {e}\n")
        raise SystemExit(3)

    # final CSV lives under settings.outputs.signals_dir; glob by pattern
    sig_dir = Path(settings.outputs.signals_dir)
    final_csv = _latest_file(sig_dir, "signals_final_*.csv")
    if not final_csv or not final_csv.exists():
        sys.stderr.write(f"final CSV not found under {sig_dir}\n")
        raise SystemExit(3)
    try:
        df = pd.read_csv(final_csv)
        # Deterministic expectation for current test symbol set is 1
        expected_rows = 1
        if len(df) != expected_rows:
            sys.stderr.write(
                (
                    "unexpected final count: {} rows (expected {}) in {}\n".format(
                        len(df), expected_rows, final_csv.name
                    )
                )
            )
            raise SystemExit(3)
    except Exception as e:
        sys.stderr.write(f"failed to read final CSV: {e}\n")
        raise SystemExit(3)


def main() -> int:
    _run_generator()
    _run_pipeline()
    _validate_outputs()
    print("\n✅ Controlled pipeline test passed: validation OK (final rows validated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
