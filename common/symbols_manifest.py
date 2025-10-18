"""Helper utilities for persisting and loading symbol manifests.

``cache_daily_data.py`` fetches ticker universes from the network before
downloading price data.  Persisting that universe allows other scripts (for
example the rolling cache rebuild) to reuse the exact same symbol list without
relying on directory scans that may include stale files.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
import json
from pathlib import Path

MANIFEST_FILENAME = "_symbols.json"


def _manifest_path(base_dir: Path | str) -> Path:
    return Path(base_dir) / MANIFEST_FILENAME


def _normalize_symbols(symbols: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for sym in symbols:
        if sym is None:
            continue
        text = str(sym).strip()
        if not text:
            continue
        # cache ディレクトリは大文字シンボルで揃えるため upper() に統一する
        ticker = text.upper()
        if ticker in seen:
            continue
        seen.add(ticker)
        normalized.append(ticker)
    return normalized


def save_symbol_manifest(
    symbols: Iterable[str],
    base_dir: Path | str,
    *,
    generated_at: datetime | None = None,
) -> Path:
    """Persist the given symbol list to ``base_dir``.

    Parameters
    ----------
    symbols:
        Iterable of ticker strings.  Empty/None entries are ignored.
    base_dir:
        Directory where the manifest should be written (typically the
        ``data_cache/full_backup`` folder).
    generated_at:
        Optional timestamp for reproducibility; defaults to current UTC.
    """

    normalized = _normalize_symbols(symbols)
    payload = {
        "generated_at": (generated_at or datetime.now(timezone.utc)).isoformat(),
        "count": len(normalized),
        "symbols": normalized,
    }

    manifest_path = _manifest_path(base_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def load_symbol_manifest(base_dir: Path | str) -> list[str]:
    """Load the cached symbol universe from ``base_dir`` if available."""

    manifest_path = _manifest_path(base_dir)
    if not manifest_path.exists():
        return []
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    symbols = data.get("symbols")
    if not isinstance(symbols, list):
        return []
    return _normalize_symbols(symbols)


__all__ = [
    "MANIFEST_FILENAME",
    "load_symbol_manifest",
    "save_symbol_manifest",
]
