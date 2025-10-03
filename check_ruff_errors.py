#!/usr/bin/env python3
"""Check remaining ruff errors"""
import json
import subprocess

# Run ruff and capture output
result = subprocess.run(
    ["python", "-m", "ruff", "check", ".", "--output-format", "json"],
    capture_output=True,
    text=True,
)

if result.returncode == 0:
    print("No ruff errors found!")
else:
    try:
        errors = json.loads(result.stdout)
        print(f"Total errors: {len(errors)}")

        # Count by error type
        error_types = {}
        for error in errors:
            code = error.get("code", "Unknown")
            error_types[code] = error_types.get(code, 0) + 1

        print("\nError breakdown:")
        for code, count in sorted(error_types.items()):
            print(f"  {code}: {count}")

        # Show first few errors as examples
        print("\nFirst 5 errors:")
        for i, error in enumerate(errors[:5]):
            file_path = error.get("filename", "Unknown")
            line = error.get("location", {}).get("row", "Unknown")
            code = error.get("code", "Unknown")
            message = error.get("message", "Unknown")
            print(f"  {i+1}. {file_path}:{line} [{code}] {message}")

    except json.JSONDecodeError:
        print("Failed to parse JSON output")
        print("Raw output:", result.stdout[:500])
