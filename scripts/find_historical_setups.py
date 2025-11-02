"""Find any historical rows where System3 'setup' was True (or drop3d >= threshold).

Usage: python scripts/find_historical_setups.py --date YYYY-MM-DD --sample 99999
"""

from __future__ import annotations

from pathlib import Path
import sys

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging
from typing import Dict

import pandas as pd

from common.cache_manager import CacheManager
from common.symbols_manifest import load_symbol_manifest
from config.settings import get_settings
from core import system3

logger = logging.getLogger("find.setup")


def _setup_logging():
    h = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


def read_raw_cache(
    cm: CacheManager, symbols: list[str], max_workers: int | None = None
) -> Dict[str, pd.DataFrame]:
    try:
        return cm.read_batch_parallel(
            symbols, profile="rolling", max_workers=max_workers or 8
        )
    except Exception:
        out = {}
        for s in symbols:
            try:
                df = cm.read(s, "rolling")
                if df is not None:
                    out[s] = df
            except Exception:
                continue
        return out


def main():
    _setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=99999)
    p.add_argument("--max-workers", type=int, default=8)
    args = p.parse_args()

    settings = get_settings(create_dirs=False)
    cm = CacheManager(settings)

    manifest = load_symbol_manifest(cm.full_dir) or []
    symbols = manifest[: args.sample]
    logger.info("symbols to check=%d", len(symbols))

    raw = read_raw_cache(cm, symbols, max_workers=args.max_workers)
    prepared = system3.prepare_data_vectorized_system3(raw, reuse_indicators=True)

    drop_thr = 0.125
    any_found = 0
    for s, df in prepared.items():
        if df is None or getattr(df, "empty", True):
            continue
        # Find rows where setup is true (preferred), otherwise where drop3d>=threshold and filter True
        try:
            if "setup" in df.columns:
                mask = df["setup"]
            else:
                drop = df.get("drop3d")
                filt = df.get("filter")
                if drop is not None:
                    left = drop >= drop_thr
                else:
                    left = pd.Series(False, index=df.index)
                if filt is not None:
                    right = filt
                else:
                    right = pd.Series(False, index=df.index)
                mask = left & right
            if mask.any():
                any_found += 1
                rows = df.loc[mask]
                # show up to 5 matching rows
                out = []
                for idx, r in rows.tail(5).iterrows():
                    d = idx if hasattr(idx, "strftime") else r.get("Date")
                    out.append(
                        {
                            "date": str(d),
                            "drop3d": float(r.get("drop3d", float("nan"))),
                            "atr_ratio": float(r.get("atr_ratio", float("nan"))),
                            "close": float(r.get("Close", float("nan"))),
                        }
                    )
                logger.info("%s: matches=%d sample_tail=%s", s, int(mask.sum()), out)
        except Exception as e:
            logger.exception("failed for %s: %s", s, e)

    logger.info("Total symbols with historical setup rows: %d", any_found)


if __name__ == "__main__":
    main()
