"""Quick debug helper for System3 candidate-generation issues.

Reads cached rolling data via CacheManager, prepares System3 inputs,
logs per-symbol indicator stats and filter pass rates, and runs the
candidate generator with diagnostics enabled.
"""
from __future__ import annotations

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import argparse
import json
import logging
from typing import Any

import pandas as pd

# Ensure repository root is on sys.path so local packages (common, core, etc.)
# can be imported when the script is executed with the venv python.
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from common.cache_manager import CacheManager
from common.symbols_manifest import load_symbol_manifest
from config.settings import get_settings

try:
    from config.environment import get_env_config
except Exception:
    get_env_config = None

from core import system3

try:
    from core.system1 import generate_candidates_system1
except Exception:
    generate_candidates_system1 = None
from common.system_setup_predicates import system3_setup_predicate

logger = logging.getLogger("debug.system3")


def _setup_logging(level: int = logging.INFO) -> None:
    h = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
    logger.setLevel(level)


def read_raw_cache(
    cm: CacheManager,
    symbols: list[str],
    max_workers: int | None = None,
) -> dict[str, pd.DataFrame]:
    logger.info("Reading %d symbols from rolling cache (parallel)", len(symbols))
    try:
        data = cm.read_batch_parallel(
            symbols, profile="rolling", max_workers=max_workers or 8
        )
    except Exception as e:
        logger.warning(
            "read_batch_parallel failed: %s - falling back to sequential read",
            e,
        )
        data = {}
        for s in symbols:
            try:
                df = cm.read(s, "rolling")
                if df is not None:
                    data[s] = df
            except Exception:
                continue
    return data


def log_prepared_summary(prepared: dict[str, pd.DataFrame], sample_n: int = 5) -> None:
    keys = list(prepared.keys())
    logger.info("prepared_dict keys=%d", len(keys))
    for sym in keys[:sample_n]:
        df = prepared.get(sym)
        if df is None or getattr(df, "empty", True):
            logger.warning("Symbol=%s has no data (empty or None)", sym)
            continue
        try:
            logger.info(
                "Symbol=%s Shape=%s Columns=%s", sym, df.shape, list(df.columns)
            )
            if "drop3d" in df.columns:
                logger.info(
                    "%s drop3d stats:\n%s", sym, df["drop3d"].describe()
                )
            else:
                logger.info("%s drop3d: MISSING", sym)
            if "atr_ratio" in df.columns:
                logger.info("%s atr_ratio stats:\n%s", sym, df["atr_ratio"].describe())
            else:
                logger.info("%s atr_ratio: MISSING", sym)
            if "dollarvolume20" in df.columns:
                logger.info(
                    "%s dollarvolume20 stats:\n%s",
                    sym,
                    df["dollarvolume20"].describe(),
                )
            else:
                logger.info("%s dollarvolume20: MISSING", sym)
        except Exception as e:
            logger.exception("Failed to log symbol %s stats: %s", sym, e)


def compute_filter_stats(prepared: dict[str, pd.DataFrame]) -> dict[str, Any]:
    stats = {
        "total": 0,
        "close_ok": 0,
        "vol_ok": 0,
        "atr_ok": 0,
        "setup_ok": 0,
        "drop3d_nan": 0,
    }
    # obtain potential override from env
    atr_thr = 0.05
    drop_thr = 0.125
    try:
        if get_env_config is not None:
            env = get_env_config()
            v = getattr(env, "min_atr_ratio_for_test", None)
            if v is not None:
                atr_thr = float(v)
            v2 = getattr(env, "min_drop3d_for_test", None)
            if v2 is not None:
                drop_thr = float(v2)
    except Exception:
        pass

    for s, df in prepared.items():
        if df is None or getattr(df, "empty", True):
            continue
        try:
            last = df.iloc[-1]
            stats["total"] += 1
            try:
                c = float(last.get("Close", float("nan")))
                if c >= 5.0:
                    stats["close_ok"] += 1
            except Exception:
                pass
            try:
                dv = float(last.get("dollarvolume20", float("nan")))
                if dv > 25_000_000:
                    stats["vol_ok"] += 1
            except Exception:
                pass
            try:
                av = float(last.get("atr_ratio", float("nan")))
                if av >= atr_thr:
                    stats["atr_ok"] += 1
            except Exception:
                pass
            try:
                d3 = last.get("drop3d")
                if d3 is None or pd.isna(d3):
                    stats["drop3d_nan"] += 1
                else:
                    if float(d3) >= drop_thr:
                        stats["setup_ok"] += 1
            except Exception:
                stats["drop3d_nan"] += 1
        except Exception:
            continue
    return {**stats, "atr_threshold": atr_thr, "drop3d_threshold": drop_thr}


def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(
        description=(
            "Debug System3 candidate generation (quick). Reads cached data and "
            "runs generator with diagnostics."
        )
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=30,
        help="sample first N symbols from manifest",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="comma separated symbol list to debug",
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="run latest_only mode",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="top_n to pass to generator",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="optional latest_mode_date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="max workers for cache reads",
    )
    args = parser.parse_args()

    settings = get_settings(create_dirs=True)
    cm = CacheManager(settings)

    symbols: list[str] = []
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        # load from manifest; fallback to a small set if manifest missing
        manifest = load_symbol_manifest(cm.full_dir)
        if manifest:
            symbols = manifest[: args.sample]
        else:
            # minimal default for local debugging
            symbols = ["SPY", "AAPL", "MSFT", "QQQ", "TSLA"][: args.sample]

    logger.info("Symbols count=%d sample=%s", len(symbols), symbols[:10])

    raw = read_raw_cache(cm, symbols, max_workers=args.max_workers)

    # Prepare using system3 vectorized prepare (reuse indicators fast-path)
    try:
        prepared = system3.prepare_data_vectorized_system3(raw, reuse_indicators=True)
    except Exception as e:
        logger.exception("prepare_data_vectorized_system3 failed: %s", e)
        prepared = {}

    # Log prepared summary (first few symbols)
    log_prepared_summary(prepared, sample_n=5)

    # Filter stats
    fstats = compute_filter_stats(prepared)
    logger.info("Filter stats: %s", json.dumps(fstats, ensure_ascii=False))

    # Environment config
    try:
        env = get_env_config() if get_env_config is not None else None
    except Exception:
        env = None
    logger.info(
        "Env config keys: %s",
        list(vars(env).keys()) if env is not None else None,
    )

    # Per-symbol compare: prepare.setup vs predicate
    try:
        logger.info("Per-symbol prepare vs predicate (latest row):")
        mismatches: list[str] = []
        for s, df in prepared.items():
            if df is None or getattr(df, "empty", True):
                continue
            last = df.iloc[-1]
            prep_setup = bool(last.get("setup", False))
            try:
                pred = system3_setup_predicate(last, return_reason=True)
                if isinstance(pred, tuple):
                    pred_bool, pred_reason = bool(pred[0]), pred[1]
                else:
                    pred_bool, pred_reason = bool(pred), None
            except Exception:
                pred_bool, pred_reason = False, "exception"
            if prep_setup != pred_bool:
                mismatches.append(s)
            logger.info(
                "%s: prep_setup=%s pred=%s reason=%s drop3d=%s atr_ratio=%s close=%s sma150=%s",
                s,
                prep_setup,
                pred_bool,
                pred_reason,
                last.get("drop3d"),
                last.get("atr_ratio"),
                last.get("Close"),
                last.get("sma150"),
            )
        logger.info("MISMATCHES (%d): %s", len(mismatches), mismatches)
    except Exception:
        logger.exception("Per-symbol predicate check failed")
    # List symbols where prepare.setup == True (for quick visibility)
    try:
        setups = [
            s
            for s, df in prepared.items()
            if df is not None
            and not getattr(df, "empty", True)
            and bool(df.iloc[-1].get("setup", False))
        ]
        logger.info("Symbols with prepare.setup True: %s", setups)
    except Exception:
        logger.exception("Failed to list setup symbols")

    # Call generate_candidates_system3 in requested mode
    try:
        kwargs: dict[str, Any] = {
            "top_n": args.top_n,
            "latest_only": bool(args.latest_only),
            "include_diagnostics": True,
            # pass logger as log_callback so core functions can emit debug lines
            "log_callback": logger.info,
        }
        if args.date:
            kwargs["latest_mode_date"] = args.date
        logger.info(
            "Calling generate_candidates_system3 latest_only=%s top_n=%s",
            kwargs.get("latest_only"),
            kwargs.get("top_n"),
        )
        res = system3.generate_candidates_system3(prepared, **kwargs)
        if len(res) == 3:
            by_date, df_all, diagnostics = res
        else:
            by_date, df_all = res
            diagnostics = None
        logger.info(
            "System3 result: candidates_by_date_keys=%s df_all_rows=%s",
            list(by_date.keys())[:5],
            len(df_all) if df_all is not None else 0,
        )
        if diagnostics is not None:
            logger.info(
                "Diagnostics:\n%s",
                json.dumps(diagnostics, default=str, ensure_ascii=False, indent=2),
            )
    except Exception as e:
        logger.exception("generate_candidates_system3 failed: %s", e)

    # Optional: compare with system1 if available
    if generate_candidates_system1 is not None:
        try:
            logger.info(
                "Running system1 for comparison (latest_only=%s)",
                bool(args.latest_only),
            )
            s1_res = generate_candidates_system1(
                prepared,
                top_n=args.top_n,
                latest_only=bool(args.latest_only),
                include_diagnostics=True,
            )
            if len(s1_res) == 3:
                s1_by_date, s1_df, s1_diag = s1_res
            else:
                s1_by_date, s1_df = s1_res
                s1_diag = None
            logger.info(
                "System1 candidates: dates=%s rows=%s",
                list(s1_by_date.keys())[:5],
                len(s1_df) if s1_df is not None else 0,
            )
            if s1_diag is not None:
                logger.info(
                    "System1 diagnostics:\n%s",
                    json.dumps(s1_diag, default=str, ensure_ascii=False, indent=2),
                )
        except Exception:
            logger.exception("generate_candidates_system1 failed")
    else:
        logger.info("system1 not importable; skipping comparison")

    logger.info("Debug run complete")


if __name__ == "__main__":
    main()
