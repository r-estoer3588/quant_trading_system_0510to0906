#!/usr/bin/env python3
"""Validate files in repro_payloads for UTF-8 decodability and JSON validity.

This script is intended to be lightweight and safe to run in pre-commit. It
ensures that files under repro_payloads are encoded as UTF-8. For JSON files
it additionally validates that the content is parseable JSON.
"""
from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1] / "repro_payloads"
    if not root.exists():
        # Nothing to validate in a clean tree
        print("repro_payloads/ not present; skipping UTF-8 validation")
        return 0

    failed = False
    for f in sorted(root.iterdir()):
        if not f.is_file():
            continue
        try:
            data = f.read_bytes()
        except Exception as exc:  # pragma: no cover - defensive
            print(f"ERROR: could not read {f}: {exc}")
            failed = True
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            print(f"ERROR: {f} is not UTF-8 decodable: {exc}")
            failed = True
            continue
        if f.suffix.lower() == ".json":
            try:
                json.loads(text)
            except Exception as exc:
                print(f"ERROR: {f} is not valid JSON: {exc}")
                failed = True
    if failed:
        return 1
    print("All repro_payloads files are UTF-8 decodable and JSON files are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
