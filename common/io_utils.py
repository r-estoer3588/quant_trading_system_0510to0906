"""I/O helper utilities.

Provide UTF-8-safe write helpers so all tools and scripts write JSON/CSV/text
in a consistent and robust way. These helpers sanitize input and ensure
files are created with parent directories as needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


def _ensure_parent(p: Path) -> None:
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        # best-effort: if mkdir fails, let the subsequent write raise
        pass


def safe_unicode(text: Any) -> str:
    """Return a UTF-8-safe string representation of the input.

    Non-string inputs are converted to str(); any invalid Unicode sequences
    are replaced to guarantee the caller receives valid UTF-8 text.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    # Replace invalid sequences when encoding to UTF-8
    try:
        encoded = text.encode("utf-8", errors="replace")
        decoded: str = encoded.decode("utf-8")
        return decoded
    except Exception:
        # Fallback: use str() to ensure we always return a string
        return str(text)


def write_text(path: Path, text: Any, encoding: str = "utf-8") -> None:
    """Write text to path with UTF-8 sanitization and parent folder creation."""
    _ensure_parent(path)
    s = safe_unicode(text)
    path.write_text(s, encoding=encoding)


def write_json(
    path: Path,
    obj: Any,
    *,
    ensure_ascii: bool = False,
    indent: int = 2,
    encoding: str = "utf-8",
    **json_kwargs: Any,
) -> None:
    """Serialize obj as JSON and write to path using UTF-8.

    This centralizes JSON formatting and ensures output is always UTF-8
    decodable even if some strings contained invalid bytes.
    """
    _ensure_parent(path)
    # Allow callers to pass through other json.dumps kwargs such as
    # `default=str` so we don't need to special-case every call site.
    j = json.dumps(obj, ensure_ascii=ensure_ascii, indent=indent, **json_kwargs)
    # Ensure UTF-8 validity
    j = safe_unicode(j)
    path.write_text(j, encoding=encoding)


def write_bytes(path: Path, data: bytes) -> None:
    """Write raw bytes to disk, ensuring parent directories exist.

    Some callers need a bytes representation (for example when returning
    CSV bytes to a web response). This helper centralizes the parent
    directory creation and write behavior.
    """
    _ensure_parent(path)
    path.write_bytes(data)


def write_json_lines(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    ensure_ascii: bool = False,
    indent: int | None = None,
    encoding: str = "utf-8",
    append: bool = False,
    **json_kwargs: Any,
) -> None:
    """Write an iterable of objects as newline-delimited JSON (NDJSON).

    This mirrors the common pattern of writing json.dumps(row) + "\n"
    but centralizes encoding and sanitization.
    """
    _ensure_parent(path)
    mode = "a" if append else "w"
    with path.open(mode, encoding=encoding) as fh:
        for row in rows:
            dump_kwargs: dict[str, Any] = {
                "ensure_ascii": ensure_ascii,
                "indent": indent,
            }
            if json_kwargs:
                dump_kwargs.update(json_kwargs)
            line = json.dumps(row, **dump_kwargs)
            fh.write(safe_unicode(line) + "\n")


def df_to_bytes(
    df: pd.DataFrame,
    *,
    index: bool = False,
    encoding: str = "utf-8",
    **kwargs: Any,
) -> bytes:
    """Return CSV representation of a DataFrame as UTF-8 bytes.

    Useful where code previously called ``df.to_csv(...).encode('utf-8')``.
    We intentionally perform a best-effort UTF-8 sanitization so callers
    receive valid bytes.
    """
    # pandas returns a str when no path is provided; ensure the string is
    # UTF-8-safe before encoding to bytes.
    csv_text: str = df.to_csv(index=index, **kwargs)
    csv_text = safe_unicode(csv_text)
    return csv_text.encode(encoding)


def df_to_csv(
    df: pd.DataFrame,
    path: Path,
    *,
    index: bool = False,
    encoding: str = "utf-8",
    **kwargs: Any,
) -> None:
    """Write DataFrame to CSV on disk with UTF-8 encoding and parent dirs.

    Extra kwargs are forwarded to pandas.DataFrame.to_csv.
    """
    _ensure_parent(path)
    # pandas will handle encoding; ensure index default is False for our usage
    df.to_csv(path, index=index, encoding=encoding, **kwargs)
