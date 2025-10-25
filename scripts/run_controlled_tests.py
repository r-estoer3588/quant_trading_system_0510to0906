#!/usr/bin/env python3
"""Run the controlled systems tests (systems 1-6).

This script is a small convenience wrapper so developers can run the
single-file integration test quickly from the repo root:

    python scripts/run_controlled_tests.py

It will invoke pytest for `tests/test_systems_controlled_all.py` and
propagate the exit code.
"""
import sys
from typing import List, Optional

import pytest


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv or sys.argv[1:]
    # allow passing extra pytest args, but default to running the single test file
    args: List[str] = argv or ["-q", "tests/test_systems_controlled_all.py"]
    # pytest.main may return a non-int in some versions; coerce to int
    return int(pytest.main(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
