"""Small diagnostics helper utilities shared across systems.

Provides a defensive helper to record exclusion reasons and (optionally)
associate symbols with those reasons. This allows older code that stores
diagnostics as plain dicts to gain the same symbol-tracking behaviour as
the System1Diagnostics dataclass.

The helper is intentionally defensive: it will not raise on malformed
inputs and will try to preserve existing diagnostics shape where possible.
"""

from __future__ import annotations

from typing import Any


def record_exclude(diagnostics: dict[str, Any], reason: str, symbol: str | None = None) -> None:
    """Record an exclusion in a diagnostics dict.

    Updates (in-place) the mapping keys:
      - "exclude_reasons": mapping reason -> int count
      - "exclude_symbols": mapping reason -> set of symbols (kept as set in-memory)

    Behaviour is best-effort and defensive: exceptions are swallowed so
    callers don't fail when diagnostics are partially malformed.
    """
    try:
        if diagnostics is None:
            return

        # ensure mapping exists
        ex = diagnostics.get("exclude_reasons")
        if ex is None or not isinstance(ex, dict):
            ex = {}
            diagnostics["exclude_reasons"] = ex
        key = str(reason)
        ex[key] = int(ex.get(key, 0)) + 1

        if symbol:
            ex_sym = diagnostics.get("exclude_symbols")
            if ex_sym is None or not isinstance(ex_sym, dict):
                ex_sym = {}
                diagnostics["exclude_symbols"] = ex_sym
            cur = ex_sym.get(key)
            # prefer sets in-memory to avoid duplicates; accept lists too
            if cur is None:
                ex_sym[key] = {str(symbol)}
            else:
                try:
                    if isinstance(cur, set):
                        cur.add(str(symbol))
                    else:
                        # coerce list/tuple/other to set
                        s = set(cur)
                        s.add(str(symbol))
                        ex_sym[key] = s
                except Exception:
                    # last resort: overwrite with a single-element set
                    ex_sym[key] = {str(symbol)}
    except Exception:
        try:
            diagnostics.setdefault("exclude_reasons", {})["exception"] = (
                diagnostics.setdefault("exclude_reasons", {}).get("exception", 0) + 1
            )
        except Exception:
            pass
