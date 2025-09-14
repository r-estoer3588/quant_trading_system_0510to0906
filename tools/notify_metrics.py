"""Notify daily metrics summary.

Reads ``results_csv/daily_metrics.csv`` and sends a compact summary for the
latest available date via ``common.notifier.Notifier``. Safe to run even when
no webhook is configured (logs only).
"""

from __future__ import annotations

from pathlib import Path
import logging
import pandas as pd

from common.notifier import Notifier
from config.settings import get_settings


def _load_latest_metrics() -> tuple[pd.DataFrame, str] | tuple[None, None]:
    try:
        settings = get_settings(create_dirs=True)
        fp = Path(settings.outputs.results_csv_dir) / "daily_metrics.csv"
    except Exception:
        fp = Path("results_csv") / "daily_metrics.csv"
    if not fp.exists():
        logging.info("metrics CSV not found: %s", fp)
        return None, None
    try:
        df = pd.read_csv(fp)
    except Exception:
        logging.exception("failed to read metrics: %s", fp)
        return None, None
    if df is None or df.empty or "date" not in df.columns:
        return None, None
    try:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    except Exception:
        pass
    try:
        last_date = sorted(df["date"].dropna().unique())[-1]
    except Exception:
        return None, None
    day_df = df[df["date"] == last_date].copy()
    return day_df, str(last_date)


def notify_metrics() -> None:
    day_df, day_str = _load_latest_metrics()
    if day_df is None or day_df.empty:
        logging.info("no metrics to notify")
        return
    fields = {}
    try:
        for _, r in day_df.iterrows():
            sys = str(r.get("system"))
            pre = int(r.get("prefilter_pass", 0) or 0)
            cand = int(r.get("candidates", 0) or 0)
            fields[sys] = f"pre={pre}, cand={cand}"
    except Exception:
        pass
    title = "\U0001F4C8 本日のメトリクス（事前フィルタ / 候補数）"
    msg = f"対象日: {day_str or ''}"
    try:
        Notifier(platform="auto").send(title, msg, fields=fields)
    except Exception:
        # 環境未設定でも処理継続（ログのみ）
        logging.info("metrics notified (log only)")


if __name__ == "__main__":
    notify_metrics()
