"""Notify daily metrics summary.

Reads ``results_csv/daily_metrics.csv`` and sends a compact summary for the
latest available date via ``common.notifier.Notifier``. Safe to run even when
no webhook is configured (logs only).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import logging
from pathlib import Path
from typing import Any

import pandas as pd

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


def send_metrics_notification(
    *,
    day_str: str | None,
    fields: Sequence[Mapping[str, Any]] | None = None,
    summary_pairs: Sequence[tuple[Any, Any]] | None = None,
    extra_lines: Sequence[str] | None = None,
    title: str = "\U0001f4c8 本日のメトリクス（system別）",
) -> None:
    """Send a metrics summary via the default notifier.

    Parameters
    ----------
    day_str:
        Target day label (e.g. ``"2024-05-01"``). ``None`` becomes an empty label.
    fields:
        Rich embed fields for Slack/Discord notifications.
    summary_pairs:
        Key/value pairs included in the message body (``key: value`` each line).
    extra_lines:
        Additional free-form lines appended to the body (e.g. code blocks).
    title:
        Notification title. Emoji default matches existing notifications.
    """

    body_lines: list[str] = []
    if day_str is not None:
        body_lines.append(f"対象日: {day_str}")
    elif summary_pairs or extra_lines:
        body_lines.append("対象日: ")

    if summary_pairs:
        for key, value in summary_pairs:
            body_lines.append(f"{key}: {value}")

    if extra_lines:
        body_lines.extend(str(line) for line in extra_lines if str(line).strip())

    if not body_lines:
        body_lines.append("対象日: -")

    msg = "\n".join(body_lines)

    try:
        from common.notifier import create_notifier
    except Exception:
        logging.info("metrics notified (log only)")
        return

    try:
        notifier = create_notifier(platform="auto", fallback=True)
        notifier.send(title, msg, fields=list(fields or []))
    except Exception:
        logging.exception("failed to send metrics notification")


def notify_metrics() -> None:
    day_df, day_str = _load_latest_metrics()
    if day_df is None or day_df.empty:
        logging.info("no metrics to notify")
        return
    fields: list[dict[str, str]] = []
    lines: list[str] = []
    try:
        for _, r in day_df.iterrows():
            sys = str(r.get("system"))
            pre = int(r.get("prefilter_pass", 0) or 0)
            cand = int(r.get("candidates", 0) or 0)
            fields.append({"name": sys, "value": f"pre {pre} / cand {cand}"})
            lines.append(f"{sys:<7} {pre:>4} {cand:>4}")
    except Exception:
        pass
    header = f"{'System':<7} {'pre':>4} {'cand':>4}"
    table = "\n".join([header] + lines)
    title = "\U0001f4c8 本日のメトリクス（事前フィルタ / 候補数）"
    send_metrics_notification(
        day_str=day_str,
        fields=fields,
        extra_lines=[f"```{table}```"],
        title=title,
    )


if __name__ == "__main__":
    notify_metrics()
