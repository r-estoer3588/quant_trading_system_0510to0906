"""Export per-symbol comparison: historical setup count vs latest-row setup.

Writes CSV to results_csv/historical_vs_latest_setup_{date}.csv and
prints the head (first 50 rows) to stdout in CSV form for easy copy/paste.
"""

from __future__ import annotations

import sys
from pathlib import Path

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from datetime import datetime
from typing import Any

import pandas as pd

from common.cache_manager import CacheManager
from common.symbols_manifest import load_symbol_manifest
from config.settings import get_settings
from core import system3


def read_raw_cache(
    cm: CacheManager, symbols: list[str], max_workers: int | None = None
) -> dict[str, pd.DataFrame]:
    try:
        return cm.read_batch_parallel(
            symbols, profile="rolling", max_workers=max_workers or 8
        )
    except Exception:
        out: dict[str, pd.DataFrame] = {}
        for s in symbols:
            try:
                df = cm.read(s, "rolling")
                if df is not None:
                    out[s] = df
            except Exception:
                continue
        return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=99999)
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--date", type=str, default=None)
    args = p.parse_args()

    settings = get_settings(create_dirs=True)
    cm = CacheManager(settings)

    manifest = load_symbol_manifest(cm.full_dir) or []
    symbols = manifest[: args.sample]

    raw = read_raw_cache(cm, symbols, max_workers=args.max_workers)
    prepared = system3.prepare_data_vectorized_system3(raw, reuse_indicators=True)

    drop_thr = 0.125
    rows: list[dict[str, Any]] = []
    for s in symbols:
        df = prepared.get(s)
        if df is None or getattr(df, "empty", True):
            rows.append(
                {
                    "symbol": s,
                    "prepared_rows": 0,
                    "historical_setup_count": 0,
                    "latest_date": None,
                    "latest_setup": False,
                    "latest_drop3d": None,
                    "latest_atr_ratio": None,
                }
            )
            continue

        prepared_rows = len(df)
        if "setup" in df.columns:
            historical_setup_count = int(df["setup"].astype(bool).sum())
        else:
            # fallback
            filt = df.get("filter")
            filt_bool = filt.fillna(False).astype(bool) if filt is not None else False
            mask = (df.get("drop3d") >= drop_thr) & filt_bool
            historical_setup_count = int(mask.sum())

        last = df.iloc[-1]
        latest_date = str(df.index[-1])
        latest_setup = bool(last.get("setup", False))
        try:
            latest_drop3d = float(last.get("drop3d", float("nan")))
        except Exception:
            latest_drop3d = None
        try:
            latest_atr = float(last.get("atr_ratio", float("nan")))
        except Exception:
            latest_atr = None

        rows.append(
            {
                "symbol": s,
                "prepared_rows": prepared_rows,
                "historical_setup_count": historical_setup_count,
                "latest_date": latest_date,
                "latest_setup": latest_setup,
                "latest_drop3d": latest_drop3d,
                "latest_atr_ratio": latest_atr,
            }
        )

    out_dir = Path(getattr(settings, "results_dir", None) or "results_csv")
    out_dir.mkdir(parents=True, exist_ok=True)
    date_tag = args.date or datetime.today().strftime("%Y-%m-%d")
    out_path = out_dir / f"historical_vs_latest_setup_{date_tag}.csv"

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_path, index=False)

    # Print head (first 50 rows) to stdout in CSV for easy copy/paste
    print(df_out.head(50).to_csv(index=False))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
