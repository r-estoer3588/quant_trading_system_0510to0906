"""Build a validation report linking daily metrics and per-system signals.

This script joins ``results_csv/daily_metrics.csv`` with signal CSVs in
``outputs.signals_dir`` for the latest day, and saves a compact report to
``results_csv/daily_metrics_report.csv``. It includes per-system counts and
the first few symbols as a spot check.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from common.cache_manager import round_dataframe
from config.settings import get_settings


def _read_metrics() -> pd.DataFrame:
    try:
        settings = get_settings(create_dirs=True)
        fp = Path(settings.outputs.results_csv_dir) / "daily_metrics.csv"
    except Exception:
        fp = Path("results_csv") / "daily_metrics.csv"
    if not fp.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(fp)
    except Exception:
        logging.exception("failed to read metrics: %s", fp)
        return pd.DataFrame()
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        except Exception:
            pass
    return df


def _collect_signals_for_day(day_str: str) -> dict[str, pd.DataFrame]:
    systems: dict[str, pd.DataFrame] = {}
    settings = get_settings(create_dirs=True)
    sig_dir = Path(settings.outputs.signals_dir)
    if not sig_dir.exists():
        return systems
    for p in sig_dir.glob(f"signals_system*_{day_str}.csv"):
        try:
            df = pd.read_csv(p)
            # normalize system name from filename
            name = p.stem.replace("signals_", "").replace(f"_{day_str}", "")
            systems[name] = df
        except Exception:
            logging.exception("failed to read signal file: %s", p)
    return systems


def build_metrics_report() -> Path | None:
    metrics = _read_metrics()
    if metrics.empty:
        logging.info("no metrics; skip report")
        return None
    try:
        last_day = sorted(metrics["date"].dropna().unique())[-1]
    except Exception:
        return None
    day_str = str(last_day)
    per_sys = _collect_signals_for_day(day_str)

    rows: list[dict] = []
    for _, r in metrics[metrics["date"] == last_day].iterrows():
        sys_name = str(r.get("system"))
        pre = int(r.get("prefilter_pass", 0) or 0)
        cand = int(r.get("candidates", 0) or 0)
        sig_df = per_sys.get(sys_name) or pd.DataFrame()
        syms = []
        try:
            if not sig_df.empty and "symbol" in sig_df.columns:
                syms = sig_df["symbol"].astype(str).head(10).tolist()
        except Exception:
            pass
        rows.append(
            {
                "date": day_str,
                "system": sys_name,
                "prefilter_pass": pre,
                "candidates": cand,
                "signals_file": f"signals_{sys_name}_{day_str}.csv",
                "symbols_sample": ", ".join(syms),
            }
        )
    out_df = pd.DataFrame(rows)
    try:
        settings = get_settings(create_dirs=True)
        out_dir = Path(settings.outputs.results_csv_dir)
    except Exception:
        out_dir = Path("results_csv")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / "daily_metrics_report.csv"
    try:
        settings = get_settings(create_dirs=True)
        round_dec = getattr(settings.cache, "round_decimals", None)
    except Exception:
        round_dec = None
    try:
        out_write = round_dataframe(out_df, round_dec)
    except Exception:
        out_write = out_df
    out_write.to_csv(out_fp, index=False)
    logging.info("metrics report saved: %s", out_fp)
    return out_fp


if __name__ == "__main__":
    build_metrics_report()
