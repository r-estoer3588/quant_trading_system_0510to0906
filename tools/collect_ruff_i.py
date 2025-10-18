#!/usr/bin/env python3
import json
import os
import subprocess
import sys

# Run ruff to collect I-category violations and write JSON with utf-8
try:
    proc = subprocess.run(
        ["ruff", "check", ".", "--select", "I", "--output-format", "json"],
        capture_output=True,
    )
except FileNotFoundError:
    print("ERROR: 'ruff' not found in PATH", file=sys.stderr)
    sys.exit(2)

out = proc.stdout.decode("utf-8", errors="replace") if proc.stdout is not None else ""
with open(os.path.join(os.getcwd(), "tmp_ruff_i.json"), "w", encoding="utf-8") as f:
    f.write(out)

codes = set()
try:
    data = json.loads(out) if out.strip() else []
    for item in data:
        codes.add(item.get("code"))
except Exception as e:
    print("WARN: could not parse JSON output from ruff:", e, file=sys.stderr)

codes = sorted([c for c in codes if c])
if codes:
    print("Found I-category ruff codes:")
    for c in codes:
        print(c)
else:
    print("No I-category ruff codes found or no output from ruff.")

# Exit with ruff's exit code so non-zero doesn't fail callers unless ruff failed to run
sys.exit(proc.returncode)
