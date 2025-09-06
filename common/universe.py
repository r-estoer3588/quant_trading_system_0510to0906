from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd

from config.settings import get_settings


def build_universe_from_cache(
    *,
    min_price: float = 5.0,
    min_dollar_vol: float = 25_000_000.0,
    limit: int | None = 2000,
    prefer_base: bool = True,
) -> List[str]:
    """data_cache から最新行の Close と DollarVolume50 を見てユニバースを構築。
    prefer_base=True の場合は data_cache/base/*.csv を優先。
    """
    settings = get_settings(create_dirs=True)
    cache_dir = Path(settings.DATA_CACHE_DIR)
    base_dir = cache_dir / "base"

    files: Iterable[Path]
    if prefer_base and base_dir.exists():
        files = base_dir.glob("*.csv")
    else:
        files = cache_dir.glob("*.csv")

    symbols: List[str] = []
    for i, f in enumerate(files, 1):
        try:
            df = pd.read_csv(f)
            if "Date" not in df.columns or "Close" not in df.columns:
                continue
            df["Date"] = pd.to_datetime(df["Date"])  # ensure sorted
            df = df.sort_values("Date")
            last = df.iloc[-1]
            close = float(last.get("Close", 0.0))
            dv50 = last.get("DollarVolume50")
            if pd.isna(dv50):
                # フォールバック: 直近50本で再計算（軽量化のため列のみ）
                if {"Close", "Volume"}.issubset(df.columns):
                    tail = df[["Close", "Volume"]].tail(50)
                    dv50 = float((tail["Close"] * tail["Volume"]).mean())
                else:
                    dv50 = 0.0
            dv50 = float(dv50 or 0.0)
            if close >= min_price and dv50 >= min_dollar_vol:
                symbols.append(f.stem)
        except Exception:
            continue
        if limit and len(symbols) >= int(limit):
            break
    # 代表ETFを先頭に残す
    out = list(dict.fromkeys(["SPY"] + [s for s in symbols if s != "SPY"]))
    return out


def save_universe_file(symbols: List[str], path: str | None = None) -> str:
    settings = get_settings(create_dirs=True)
    if path is None:
        path = str(Path(settings.PROJECT_ROOT) / "data" / "universe_auto.txt")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(symbols), encoding="utf-8")
    return str(p)


def load_universe_file(path: str | None = None) -> List[str]:
    settings = get_settings(create_dirs=True)
    if path is None:
        path = str(Path(settings.PROJECT_ROOT) / "data" / "universe_auto.txt")
    p = Path(path)
    if not p.exists():
        return []
    txt = p.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    return [s.strip().upper() for s in txt.splitlines() if s.strip()]


__all__ = [
    "build_universe_from_cache",
    "save_universe_file",
    "load_universe_file",
]

