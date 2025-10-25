"""Simple JSON comparator for UI metrics snapshots.

Usage:
  python tools/compare_ui_metrics.py base.json new.json

Prints per-system differences in a compact form. Designed to be used by Playwright
scripts as a lightweight diff step.
"""

from __future__ import annotations

import argparse
import json
from typing import Any


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _normalize(obj: Any, float_round: int = 6) -> Any:
    """Recursively normalize JSON-like objects.

    - dicts: sort keys and normalize values
    - lists: normalize each element
    - floats: round to `float_round` decimals
    - ints/str/bool/None: unchanged
    """
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k in sorted(obj.keys()):
            out[k] = _normalize(obj[k], float_round=float_round)
        return out
    if isinstance(obj, list):
        return [_normalize(x, float_round=float_round) for x in obj]
    if isinstance(obj, float):
        # round floats to avoid tiny representation diffs across runs
        try:
            return round(obj, float_round)
        except Exception:
            return float(obj)
    # ints, str, bool, None
    return obj


def _format_value(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except Exception:
        try:
            return str(obj)
        except Exception:
            return "<unserializable>"


def compare_dicts(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    msgs: list[str] = []
    keys = sorted(set(a.keys()) | set(b.keys()))
    for k in keys:
        va = a.get(k)
        vb = b.get(k)
        if va == vb:
            continue
        msgs.append(
            f"[{k}]\n  - base: {_format_value(va)}\n  - new : {_format_value(vb)}"
        )
    return msgs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("base")
    p.add_argument("new")
    p.add_argument("--round", type=int, default=6, help="float rounding digits")
    p.add_argument(
        "--ignore-keys",
        type=str,
        default="",
        help="comma-separated top-level keys to ignore (case-insensitive)",
    )
    args = p.parse_args()
    base = _load_json(args.base)
    new = _load_json(args.new)
    # canonicalize top-level keys to lower-case to avoid System1 vs system1 diffs
    try:
        if isinstance(base, dict):
            base = {str(k).lower(): v for k, v in base.items()}
    except Exception:
        pass
    try:
        if isinstance(new, dict):
            new = {str(k).lower(): v for k, v in new.items()}
    except Exception:
        pass

    # handle ignore-keys: remove specified top-level keys from both structures
    try:
        ignore_raw = (args.ignore_keys or "").strip()
        ignore_set = {s.strip().lower() for s in ignore_raw.split(",") if s.strip()}
        if ignore_set and isinstance(base, dict):
            for k in list(base.keys()):
                if k.lower() in ignore_set:
                    base.pop(k, None)
        if ignore_set and isinstance(new, dict):
            for k in list(new.keys()):
                if k.lower() in ignore_set:
                    new.pop(k, None)
    except Exception:
        pass

    base_n = _normalize(base, float_round=args.round)
    new_n = _normalize(new, float_round=args.round)

    diffs = compare_dicts(base_n, new_n)
    if not diffs:
        print("NO_DIFFS")
        return
    print("DIFFS:")
    for d in diffs:
        print(d)


if __name__ == "__main__":
    main()
