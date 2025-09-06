"""Simple signal notification utilities.

`notify_signals()` reads today's CSVs under ``settings.outputs.signals_dir``
and logs a summary. ``send_signal_notification`` posts a short text message to
webhooks defined by ``TEAMS_WEBHOOK_URL`` or ``SLACK_WEBHOOK_URL`` environment
variables.  The payload is compatible with Microsoft Teams and Slack incoming
webhooks. Discord webhooks are also supported via ``DISCORD_WEBHOOK_URL``.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from config.settings import get_settings
from common.notifier import create_notifier


def notify_signals():
    settings = get_settings(create_dirs=True)
    sig_dir = Path(settings.outputs.signals_dir)
    if not sig_dir.exists():
        logging.info("signals ディレクトリが存在しません: %s", sig_dir)
        return

    today = datetime.today().strftime("%Y-%m-%d")
    files = list(sig_dir.glob(f"*{today}*.csv"))
    if not files:
        logging.info("本日の新規シグナルCSVは見つかりませんでした。")
        return

    total = 0
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            n = len(df)
            total += n
            frames.append(df)
            logging.info("シグナル: %s (%d 件)", f.name, n)
        except Exception:
            logging.exception("シグナルCSVの読み込みに失敗: %s", f)

    logging.info("本日の合計シグナル件数: %d", total)
    if frames:
        try:
            send_signal_notification(pd.concat(frames, ignore_index=True))
        except Exception:
            logging.exception("signal notification failed")


def _send_via_notifier(text: str, symbols: list[str]) -> None:
    """Send notification using Slack (fallback to Discord) via Notifier.

    Uses create_notifier(fallback=True) and Notifier.send_signals for formatting.
    """
    n = create_notifier(platform="slack", fallback=True)
    try:
        # Use a neutral system name for aggregated signals
        n.send_signals("integrated", symbols)
    except Exception:
        logging.exception("signal notification failed (slack+discord)")


def send_signal_notification(df: pd.DataFrame) -> None:
    """Send a brief notification for the given signals DataFrame."""
    if df is None or df.empty:
        return
    syms = ", ".join(df["symbol"].astype(str).head(10))
    text = f"Today signals: {len(df)} picks\n{syms}"
    _send_via_notifier(text, df["symbol"].astype(str).tolist())


if __name__ == "__main__":
    notify_signals()

