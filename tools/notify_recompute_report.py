"""Send a short notification if recompute bulk report contains failures.

This small utility is intended to be called from PowerShell after a recompute run.
It reads the JSON report and, if any symbol has message == 'recompute_failed' or
there are general errors, sends a compact notification using the project's
notifier machinery.

Usage:
  python tools/notify_recompute_report.py path/to/report.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# typing.Any intentionally unused here


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    if not argv:
        print("Usage: notify_recompute_report.py report.json")
        return 2
    p = Path(argv[0])
    if not p.exists():
        print("Report not found:", p)
        return 2
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logging.exception("failed to read JSON report: %s", e)
        return 2

    errors = []
    per = data.get("per_symbol", {}) or {}
    for sym, info in per.items():
        if not info.get("success"):
            errors.append((sym, info.get("message") or "failed"))
        else:
            # also check for would_update being False and message
            # telling recompute failed
            msg = info.get("message")
            if msg and str(msg).startswith("recompute_failed"):
                errors.append((sym, msg))

    # general errors
    gen_errs = data.get("errors") or {}
    if gen_errs:
        for k, v in gen_errs.items():
            errors.append((k, v))

    if not errors:
        # no failures -> nothing to notify
        return 0

    # Build a compact message
    lines = [f"recompute failures: {len(errors)} symbols"]
    for s, m in errors[:20]:
        lines.append(f"{s}: {m}")
    if len(errors) > 20:
        lines.append("...and more")

    try:
        from common.notifier import create_notifier

        notifier = create_notifier(platform="auto", fallback=True)
        title = "⚠️ Recompute failures detected"
        notifier.send(title, "\n".join(lines))
    except Exception:
        logging.exception(
            "failed to send recompute notification; writing to stderr instead"
        )
        print("\n".join(lines), file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
