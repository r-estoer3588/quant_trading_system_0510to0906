#!/usr/bin/env python3
"""
Streamlit UI launcher wrapper with convenient flags for compact logging.

Usage examples:
  - Compact logs (suppress per-symbol) and run Today Signals UI
      python scripts/run_today_ui.py --compact

  - Compact + force verbose per-symbol (for debugging)
      python scripts/run_today_ui.py --compact --verbose-per-symbol

  - Integrated UI with compact logs
      python scripts/run_today_ui.py --compact --app apps/main.py

  - Pass options to streamlit (after --)
      python scripts/run_today_ui.py --compact -- --server.port 8502

Flags:
  --compact                       Set COMPACT_TODAY_LOGS=1
  --no-compact                    Unset/disable COMPACT_TODAY_LOGS
  --verbose-per-symbol            Set ROLLING_MANUAL_REBUILD_VERBOSE=1
  --suppress-per-symbol           Set ROLLING_MANUAL_REBUILD_SUPPRESS_PER_SYMBOL=1
  --verbose-limit N               Set ROLLING_MANUAL_REBUILD_VERBOSE_LIMIT=N
  --app PATH                      App to run (default: apps/app_today_signals.py)

Notes:
  - This wrapper only sets environment variables and delegates to
    `python -m streamlit run <app>`.
  - On Windows, it uses the current interpreter (sys.executable).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run Streamlit UI with compact log flags")
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Set COMPACT_TODAY_LOGS=1 (suppress per-symbol logs by default)",
    )
    parser.add_argument(
        "--no-compact",
        dest="no_compact",
        action="store_true",
        help="Disable compact mode (remove COMPACT_TODAY_LOGS)",
    )
    parser.add_argument(
        "--verbose-per-symbol",
        action="store_true",
        help="Set ROLLING_MANUAL_REBUILD_VERBOSE=1 (force per-symbol details)",
    )
    parser.add_argument(
        "--suppress-per-symbol",
        action="store_true",
        help="Set ROLLING_MANUAL_REBUILD_SUPPRESS_PER_SYMBOL=1",
    )
    parser.add_argument(
        "--verbose-limit",
        type=int,
        default=None,
        help="Set ROLLING_MANUAL_REBUILD_VERBOSE_LIMIT to cap per-symbol lines",
    )
    parser.add_argument(
        "--app",
        type=str,
        default=str(Path("apps/app_today_signals.py")),
        help="Path to Streamlit app to run (default: apps/app_today_signals.py)",
    )
    # Collect any remaining args to pass through to streamlit
    args, passthrough = parser.parse_known_args()
    return args, passthrough


def set_env_flags(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()

    if getattr(args, "compact", False):
        env["COMPACT_TODAY_LOGS"] = "1"
    if getattr(args, "no_compact", False):
        # Explicitly disable/remove
        env.pop("COMPACT_TODAY_LOGS", None)

    if getattr(args, "verbose_per_symbol", False):
        env["ROLLING_MANUAL_REBUILD_VERBOSE"] = "1"
    if getattr(args, "suppress_per_symbol", False):
        env["ROLLING_MANUAL_REBUILD_SUPPRESS_PER_SYMBOL"] = "1"
    if getattr(args, "verbose_limit", None) is not None:
        env["ROLLING_MANUAL_REBUILD_VERBOSE_LIMIT"] = str(int(args.verbose_limit))

    return env


def main() -> int:
    args, passthrough = parse_args()

    app_path = Path(args.app).as_posix()
    if not Path(app_path).exists():
        print(f"Error: app not found: {app_path}", file=sys.stderr)
        return 2

    env = set_env_flags(args)

    cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    if passthrough:
        # Forward additional args to streamlit
        cmd.extend(passthrough)

    try:
        return subprocess.call(cmd, env=env)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # pragma: no cover
        print(f"Failed to launch Streamlit: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
