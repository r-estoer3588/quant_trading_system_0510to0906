from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import logging
import matplotlib.pyplot as plt
import pandas as pd

__all__ = ["save_equity_curve"]


@dataclass(frozen=True)
class _ImageResult:
    path: Path
    url: Optional[str]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_equity_curve(
    trades_df: pd.DataFrame,
    initial_capital: float,
    system_name: str,
    out_dir: Path | str = Path("logs") / "images",
) -> Tuple[str, Optional[str]]:
    """Save a simple equity curve PNG and return (path, url?).

    The curve is computed as initial_capital + cumulative sum of per-trade PnL
    ordered by exit_date if available, otherwise entry_date.

    If environment variable ``IMAGE_BASE_URL`` is set, a URL is constructed as
    ``{IMAGE_BASE_URL}/{filename}`` and returned alongside the local path.
    """
    if trades_df is None or getattr(trades_df, "empty", True):
        return "", None

    df = trades_df.copy()
    if "exit_date" in df.columns:
        df["_t"] = pd.to_datetime(df["exit_date"])  # type: ignore[assignment]
    elif "entry_date" in df.columns:
        df["_t"] = pd.to_datetime(df["entry_date"])  # type: ignore[assignment]
    else:
        df["_t"] = range(len(df))  # fallback order
    df = df.sort_values("_t")

    if "pnl" not in df.columns:
        logging.getLogger("equity_curve").warning("'pnl' column missing; using zeros for equity curve")
    pnl = pd.to_numeric(df.get("pnl", 0), errors="coerce").fillna(0.0)
    equity = float(initial_capital) + pnl.cumsum()

    out_dir = Path(out_dir)
    _ensure_dir(out_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{system_name}_equity_{ts}.png"
    out_path = out_dir / fname

    try:
        plt.style.use("seaborn-v0_8")
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(df["_t"], equity, color="#2b8a3e", linewidth=1.5)
    ax.set_title(f"{system_name} Equity Curve")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity (USD)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    base_url = os.getenv("IMAGE_BASE_URL", "").rstrip("/")
    url = f"{base_url}/{fname}" if base_url else None
    return str(out_path), url
