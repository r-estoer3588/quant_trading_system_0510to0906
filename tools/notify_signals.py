"""Simple signal notification utilities.

`notify_signals()` reads today's CSVs under ``settings.outputs.signals_dir``
and logs a summary. ``send_signal_notification`` posts a short text message via
Microsoft Teams (``TEAMS_WEBHOOK_URL``) or Slack Web API (``SLACK_BOT_TOKEN``).
Discord webhooks are also supported via ``DISCORD_WEBHOOK_URL``.
"""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path

from PIL import Image
import pandas as pd

from common.notifier import create_notifier
from common.price_chart import save_price_chart
from common.trade_cache import pop_entry, store_entry
from config.settings import get_settings


def _combine_images(paths: list[str]) -> str:
    """Combine images vertically and return the output path."""
    images = [Image.open(p) for p in paths if p]
    if not images:
        return ""
    width = max(img.width for img in images)
    height = sum(img.height for img in images)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for img in images:
        canvas.paste(img, (0, y))
        y += img.height
        img.close()
    out_dir = Path(paths[0]).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    canvas.save(out_path)
    return str(out_path)


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


def _send_via_notifier(df: pd.DataFrame) -> None:
    """Send notification using Slack (fallback to Discord) via Notifier."""
    n = create_notifier(platform="slack", fallback=True)
    try:
        if "system" in df.columns:
            groups = df.groupby("system")
        else:
            groups = [("integrated", df)]
        for sys_name, g in groups:
            raw_symbols = g["symbol"].astype(str).tolist()
            if "close" in g.columns:
                closes = g["close"].tolist()
                symbols = []
                for sym, close in zip(raw_symbols, closes, strict=False):
                    try:
                        price = float(close)
                        symbols.append(f"{sym} ${price:.2f}")
                    except Exception:
                        symbols.append(sym)
            else:
                symbols = raw_symbols
            n.send_signals(str(sys_name), symbols)
            chart_paths: list[str] = []
            for sym in raw_symbols:
                try:
                    row = g[g["symbol"] == sym].iloc[0]
                except Exception:
                    row = None
                trades_df: pd.DataFrame | None = None
                if row is not None:
                    action = str(row.get("action") or row.get("side") or "").upper()
                    if action == "BUY":
                        entry_date = row.get("entry_date")
                        entry_price = row.get("entry_price")
                        if entry_date and entry_price:
                            store_entry(sym, str(entry_date), float(entry_price))
                        trades_df = pd.DataFrame([row])
                    elif action == "SELL":
                        entry_info = pop_entry(sym) or {}
                        if entry_info:
                            row = {**row.to_dict(), **entry_info}
                        trades_df = pd.DataFrame([row])
                try:
                    img_path, _ = save_price_chart(sym, trades=trades_df)
                    if img_path:
                        chart_paths.append(img_path)
                except Exception:
                    logging.exception("failed to generate chart for %s", sym)
            if chart_paths:
                try:
                    combined = _combine_images(chart_paths)
                    if combined:
                        send_with_mention = getattr(n, "send_with_mention", None)
                        if callable(send_with_mention):
                            msg = "\n".join(symbols)
                            send_with_mention(
                                "📈 日足チャート",
                                msg,
                                mention=False,
                                image_path=combined,
                            )
                        else:
                            n.send_signals("charts", ["\n".join(symbols)])
                except Exception:
                    logging.exception("failed to send combined chart")
    except Exception:
        logging.exception("signal notification failed (slack+discord)")


def send_signal_notification(df: pd.DataFrame) -> None:
    """Send a brief notification for the given signals DataFrame."""
    if df is None or df.empty:
        return
    logging.info("Today signals: %d picks", len(df))
    _send_via_notifier(df)


if __name__ == "__main__":
    notify_signals()
