#!/usr/bin/env python3
"""Analyze F821 undefined name errors"""
import json
import subprocess
from collections import defaultdict

# Run ruff to get F821 errors
result = subprocess.run(
    ["python", "-m", "ruff", "check", ".", "--select", "F821", "--output-format", "json"],
    capture_output=True,
    text=True,
)

if result.returncode == 0:
    print("No F821 errors found!")
    exit(0)

try:
    errors = json.loads(result.stdout)
    print(f"Total F821 errors: {len(errors)}")

    # Group by file
    by_file = defaultdict(list)
    for error in errors:
        file_path = error.get("filename", "Unknown")
        by_file[file_path].append(error)

    print("\nErrors by file:")
    for file_path in sorted(by_file.keys()):
        error_count = len(by_file[file_path])
        print(f"  {file_path}: {error_count} errors")

    # Show the most problematic file details
    max_errors_file = max(by_file.items(), key=lambda x: len(x[1]))
    file_path, file_errors = max_errors_file

    print(f"\nMost problematic file: {file_path} ({len(file_errors)} errors)")
    print("First 10 errors:")

    for i, error in enumerate(file_errors[:10]):
        line = error.get("location", {}).get("row", "Unknown")
        message = error.get("message", "Unknown")
        print(f"  {i+1}. Line {line}: {message}")

    # Analyze undefined names
    undefined_names = defaultdict(int)
    for error in errors:
        message = error.get("message", "")
        if "Undefined name" in message:
            # Extract the undefined name
            parts = message.split("`")
            if len(parts) >= 2:
                name = parts[1]
                undefined_names[name] += 1

    print("\nMost common undefined names:")
    for name, count in sorted(undefined_names.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {name}: {count} occurrences")

except json.JSONDecodeError as e:
    print(f"Failed to parse JSON: {e}")
    print("Raw stdout:", result.stdout[:1000])
    print("Raw stderr:", result.stderr[:500])
