"""Simple signal notification utilities.

`notify_signals()` reads today's CSVs under ``settings.outputs.signals_dir``
and logs a summary. ``send_signal_notification`` posts a short text message to
webhooks defined by ``TEAMS_WEBHOOK_URL`` or ``SLACK_WEBHOOK_URL`` environment
variables.  The payload is compatible with Microsoft Teams and Slack incoming
webhooks. Discord webhooks are also supported via ``DISCORD_WEBHOOK_URL``.
"""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path

import pandas as pd

from common.notifier import create_notifier
from common.price_chart import save_price_chart
from config.settings import get_settings


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


def _send_via_notifier(symbols: list[str]) -> None:
    """Send notification using Slack (fallback to Discord) via Notifier."""
    n = create_notifier(platform="slack", fallback=True)
    try:
        # Use a neutral system name for aggregated signals
        n.send_signals("integrated", symbols)
        for sym in symbols:
            try:
                img_path, img_url = save_price_chart(sym)
                if img_path:
                    # BroadcastNotifier には send_with_mention が存在しない場合があるためフォールバックする
                    send_with_mention = getattr(n, "send_with_mention", None)
                    if callable(send_with_mention):
                        send_with_mention(
                            f"📈 {sym} 日足チャート",
                            "",
                            image_url=img_url,
                            mention=False,
                            image_path=img_path,
                        )
                    else:
                        # 画像アップロード非対応の場合は URL を含む簡易メッセージで通知する
                        msg_symbols = [f"{sym} {img_url}" if img_url else sym]
                        n.send_signals("charts", msg_symbols)
            except Exception:
                logging.exception("failed to send chart for %s", sym)
    except Exception:
        logging.exception("signal notification failed (slack+discord)")


def send_signal_notification(df: pd.DataFrame) -> None:
    """Send a brief notification for the given signals DataFrame."""
    if df is None or df.empty:
        return
    logging.info("Today signals: %d picks", len(df))
    _send_via_notifier(df["symbol"].astype(str).tolist())


if __name__ == "__main__":
    notify_signals()
