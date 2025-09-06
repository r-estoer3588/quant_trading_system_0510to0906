"""
Walk all .py files and try importing them as modules relative to repo root.
Skips venv/.git/__pycache__ directories. Prints a concise summary.
Run with:  set PYTHONPATH=.&& python tools\\check_all_imports.py
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


def iter_module_names(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        # prune dirs
        dirnames[:] = [d for d in dirnames if d not in {'.git', 'venv', '__pycache__'}]
        for f in filenames:
            if not f.endswith('.py'):
                continue
            full = Path(dirpath) / f
            rel = full.relative_to(root)
            mod = str(rel.with_suffix('')).replace('\\', '.').replace('/', '.')
            yield mod


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    errors: list[tuple[str, str]] = []
    for mod in iter_module_names(root):
        try:
            importlib.import_module(mod)
            print('OK', mod)
        except Exception as e:
            print('ERR', mod, type(e).__name__, e)
            errors.append((mod, f"{type(e).__name__}: {e}"))

    print("\nSUMMARY:")
    print(f"errors: {len(errors)}")
    for m, e in errors:
        print('-', m, '->', e)
    return 0 if not errors else 1


if __name__ == '__main__':
    raise SystemExit(main())

