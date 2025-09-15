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
    trades: pd.DataFrame | None = None,
) -> tuple[str, str | None]:
    """日足終値チャートを保存し、(path, url) を返す。

    Parameters
    ----------
    symbol:
        対象銘柄シンボル。
    out_dir:
        出力先ディレクトリ。
    days:
        表示する過去日数。
    trades:
        ``symbol`` に対応するトレードを含む DataFrame（任意）。
        ``entry_date``/``exit_date`` と ``entry_price``/``exit_price`` が
        含まれている場合、チャート上にエントリー・エグジットを
        マーカー表示する。
    """
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

    if trades is not None and not getattr(trades, "empty", True):
        try:
            tdf = trades[trades["symbol"] == symbol]
        except Exception:
            tdf = pd.DataFrame()
        for _, row in tdf.iterrows():
            try:
                entry_date = pd.to_datetime(row.get("entry_date"))
                exit_date = pd.to_datetime(row.get("exit_date"))
                entry_price = float(row.get("entry_price"))
                exit_price = float(row.get("exit_price"))
            except Exception:
                continue
            ax.scatter(entry_date, entry_price, marker="^", color="#2ca02c", s=50)
            ax.scatter(exit_date, exit_price, marker="v", color="#d62728", s=50)
            ax.plot(
                [entry_date, exit_date],
                [entry_price, exit_price],
                color="#888888",
                linestyle="--",
                linewidth=1,
            )
            # Annotate profit or loss near the exit marker
            profit = exit_price - entry_price
            pct = profit / entry_price * 100 if entry_price else 0.0
            color = "#2ca02c" if profit >= 0 else "#d62728"
            ax.annotate(
                f"{profit:+.2f} ({pct:+.1f}%)",
                xy=(exit_date, exit_price),
                xytext=(0, -15),
                textcoords="offset points",
                ha="center",
                color=color,
                fontsize=8,
            )

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    base_url = os.getenv("IMAGE_BASE_URL", "").rstrip("/")
    url = f"{base_url}/{fname}" if base_url else None
    return str(out_path), url
