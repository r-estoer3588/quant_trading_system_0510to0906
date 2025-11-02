"""
Benchmark JSON inspector

Usage:
  python tools/inspect_benchmark_json.py --latest
  python tools/inspect_benchmark_json.py <path-to-json>

This script summarizes benchmark JSON files produced by the pipeline.
It searches under results_csv_test/ and resuults_csv_test/ by default when
--latest is given.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable, cast

ROOTS = [Path("results_csv_test"), Path("resuults_csv_test")]


def iter_benchmark_files() -> Iterable[Path]:
    for root in ROOTS:
        if root.exists():
            yield from root.rglob("benchmark_mini_*.json")


def find_latest() -> Path | None:
    candidates = list(iter_benchmark_files())
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_json(p: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(p.read_text(encoding="utf-8")))


def summarize(data: dict[str, Any]) -> str:
    total = None
    for k in (
        "total_duration_sec",
        "total_seconds",
        "total_time",
        "total",
        "elapsed_seconds",
    ):
        v = data.get(k)
        if isinstance(v, (int, float)):
            total = float(v)
            break

    phases: list[tuple[str, float]] = []
    phases_obj = data.get("phases")
    if isinstance(phases_obj, list):
        for it in data["phases"]:
            name = str(it.get("name", "phase"))
            sec = it.get("seconds") or it.get("duration") or it.get("duration_secs")
            if isinstance(sec, (int, float)):
                phases.append((name, float(sec)))
    elif isinstance(phases_obj, dict):
        for name, info in phases_obj.items():
            if not isinstance(info, dict):
                continue
            sec = (
                info.get("duration_sec")
                or info.get("seconds")
                or info.get("duration")
                or info.get("duration_secs")
            )
            if isinstance(sec, (int, float)):
                phases.append((str(name), float(sec)))
    else:
        # flat keys like phase4_signal_generation: 123.4
        for k, v in data.items():
            if isinstance(v, (int, float)) and re.match(
                r"^phase\d+(_[a-zA-Z0-9]+)?$",
                k,
            ):
                phases.append((k, float(v)))

    lines: list[str] = []
    lines.append("=== Benchmark Summary ===")
    for k in ("timestamp", "mode", "test_mode", "parallel", "symbol_count"):
        if k in data:
            lines.append(f"{k}: {data[k]}")

    if total is None and phases:
        total = sum(sec for _, sec in phases)
    if total is not None:
        lines.append(f"total_seconds: {total:.2f}")

    if phases:
        phases.sort(key=lambda x: x[1], reverse=True)
        lines.append("\n-- Phases (desc) --")
        for name, sec in phases:
            pct = f"{(sec / total * 100):.1f}%" if total else "-"
            lines.append(f"{name:28s} {sec:8.2f}s  {pct}")
    else:
        lines.append("phases: not found")

    # extras (optional): show per-system breakdown if available
    try:
        extras = data.get("extras")
        if isinstance(extras, dict):
            per_sys = extras.get("phase4_per_system")
            if isinstance(per_sys, list) and per_sys:
                lines.append("\n-- Phase4 per-system --")
                # sort by total_sec desc if available
                try:
                    per_sys = sorted(
                        per_sys,
                        key=lambda d: float(d.get("total_sec", 0.0)),
                        reverse=True,
                    )
                except Exception:
                    pass
                for it in per_sys:
                    if not isinstance(it, dict):
                        continue
                    sysname = it.get("system", "system?")
                    tot = it.get("total_sec")
                    prep = it.get("prepare_sec")
                    gen = it.get("generate_candidates_sec")
                    cands = it.get("candidates")
                    lo = it.get("latest_only")
                    try:
                        total_s = float(tot or 0)
                        prep_s = float(prep or 0)
                        gen_s = float(gen or 0)
                        cands_i = int(cands or 0)
                        lo_b = bool(lo)
                        lines.append(
                            f"{sysname:8s} total={total_s:6.2f}s  "
                            f"prep={prep_s:6.2f}s  gen={gen_s:6.2f}s  "
                            f"cands={cands_i:4d}  latest_only={lo_b}"
                        )
                    except Exception:
                        lines.append(str(it))
    except Exception:
        pass

    for k in ("file", "run_id", "commit", "env", "notes"):
        if k in data:
            lines.append(f"{k}: {data[k]}")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="benchmark json path")
    ap.add_argument(
        "--latest",
        action="store_true",
        help="find latest benchmark file automatically",
    )
    args = ap.parse_args()

    if args.latest:
        target = find_latest()
        if not target:
            print("No benchmark json found under results_csv_test/resuults_csv_test")
            return
    else:
        if not args.path:
            ap.error("path or --latest is required")
            return
        target = Path(args.path)
        if not target.exists():
            ap.error(f"file not found: {target}")
            return

    data = load_json(target)
    print(f"file: {target}")
    print(summarize(data))


if __name__ == "__main__":
    main()
