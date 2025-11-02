"""Normalize legacy 6d_return column across cache files.

This script permanently removes the legacy column name '6d_return' by
rewriting cache files so that only the canonical 'return_6d' column
remains. After running it, the physical CSV/Parquet/Feather files will no
longer contain '6d_return'.

Design notes:
- Uses CacheManager.read() which already performs the in-memory rename.
  Writing the DataFrame back (write_atomic) is enough to drop the legacy
  column from disk.
- Does NOT recompute indicators (avoids extra load) – it simply rewrites
  the already-normalized frame.
- Supports profiles: rolling, full, or both (default).
- A --dry-run flag lists affected tickers without writing.
- Respects CacheManager's format detection (we only pass ticker + profile
  back to write_atomic).

Usage (PowerShell):
  python scripts/normalize_return_6d.py --profile both --workers 8
  python scripts/normalize_return_6d.py --profile rolling --dry-run

Safe to re-run (idempotent).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from common.cache_manager import CacheManager
from config.settings import get_settings


@dataclass
class Result:
    ticker: str
    profile: str
    changed: bool
    message: str


def _collect_tickers(dir_path: Path) -> list[str]:
    stems = set()
    if not dir_path.exists():  # pragma: no cover (defensive)
        return []
    for ext in ("*.csv", "*.parquet", "*.feather"):
        for p in dir_path.glob(ext):
            # skip temp / hidden files
            if p.name.startswith("_"):
                continue
            stems.add(p.stem)
    return sorted(stems)


def _process_one(cm: CacheManager, ticker: str, profile: str, dry_run: bool) -> Result:
    df = cm.read(ticker, profile)
    if df is None or df.empty:
        return Result(ticker, profile, False, "empty or missing")

    legacy_present = "6d_return" in df.columns
    canonical_present = "return_6d" in df.columns

    # If legacy still present (should not after read, but guard), drop it.
    if legacy_present:
        df = df.drop(columns=["6d_return"], errors="ignore")

    # If canonical missing but we can reconstruct from close prices, compute.
    if not canonical_present:
        if "close" in df.columns:
            try:
                df["return_6d"] = df["close"].pct_change(6)
                canonical_present = True
            except Exception:  # pragma: no cover - fallback only
                pass

    changed = legacy_present or (not canonical_present)

    if changed and not dry_run:
        cm.write_atomic(df, ticker, profile)
        return Result(ticker, profile, True, "rewritten (normalized)")

    if changed:
        return Result(ticker, profile, True, "would rewrite (dry-run)")
    return Result(ticker, profile, False, "ok")


def _iter_profiles(arg_profile: str) -> Iterable[str]:
    if arg_profile == "both":
        yield from ("rolling", "full")
    else:
        yield arg_profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize legacy 6d_return column")
    parser.add_argument(
        "--profile",
        choices=["rolling", "full", "both"],
        default="both",
        help="Target cache profile(s) (default: both)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write changes, just report what would happen",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Thread workers for processing tickers (default: 4)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N tickers per profile (for testing)",
    )
    args = parser.parse_args()

    settings = get_settings(create_dirs=True)
    cm = CacheManager(settings)

    total_rewritten = 0
    total_examined = 0

    for profile in _iter_profiles(args.profile):
        base_dir = cm.rolling_dir if profile == "rolling" else cm.full_dir
        tickers = _collect_tickers(base_dir)
        if args.limit:
            tickers = tickers[: args.limit]
        print(f"[{profile}] target tickers: {len(tickers)}")

        with cf.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
            futures = [
                ex.submit(_process_one, cm, t, profile, args.dry_run) for t in tickers
            ]
            for fut in cf.as_completed(futures):
                res = fut.result()
                total_examined += 1
                if res.changed:
                    total_rewritten += 1
                print(f"{res.profile}:{res.ticker} -> {res.message}")

    action = "would rewrite" if args.dry_run else "rewritten"
    print(
        f"Done. Examined={total_examined}, {action}={total_rewritten}, profile={args.profile}, dry_run={args.dry_run}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
