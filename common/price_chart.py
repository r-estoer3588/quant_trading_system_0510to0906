from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common.data_loader import load_price

__all__ = ["save_price_chart"]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_price_chart(
    symbol: str,
    out_dir: Path | str = Path("logs") / "images",
    days: int = 90,
) -> tuple[str, str | None]:
    """Save a simple daily close price chart and return (path, url?)."""
    df = load_price(symbol)
    if df is None or getattr(df, "empty", True):
        return "", None
    if "date" not in df.columns or "close" not in df.columns:
        return "", None

    df = df.tail(days).copy()
    df["date"] = pd.to_datetime(df["date"])

    out_dir = Path(out_dir)
    _ensure_dir(out_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{symbol}_daily_{ts}.png"
    out_path = out_dir / fname

    try:
        plt.style.use("seaborn-v0_8")
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(df["date"], df["close"], color="#1f77b4", linewidth=1.5)
    ax.set_title(f"{symbol} Daily Close")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    base_url = os.getenv("IMAGE_BASE_URL", "").rstrip("/")
    url = f"{base_url}/{fname}" if base_url else None
    return str(out_path), url
