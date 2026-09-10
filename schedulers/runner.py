"""Scheduler compatibility wrapper for S1/S4 intraday trailing protection.

The original scheduler implementation is preserved verbatim in
``runner_legacy.py``.  This wrapper fixes the two cron forms already present in
config.yaml (``*/30`` and overnight ``22-6``), wires the fractional S1/S4 soft
trailing monitor into the existing market-hours portfolio task, and retires the
old blanket cancel/recreate trailing task.
"""

from __future__ import annotations

from datetime import datetime
import logging
import sys
from typing import Literal

from schedulers import runner_legacy as _legacy

Field = tuple[int, ...] | Literal["*"]


def _parse_field(value: str, min_v: int, max_v: int) -> Field:
    value = value.strip()
    if value == "*":
        return "*"
    vals: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        step = 1
        base = token
        if "/" in token:
            base, step_s = token.split("/", 1)
            step = int(step_s)
            if step <= 0:
                raise ValueError(f"invalid cron step: {token}")
        if base == "*":
            seq = range(min_v, max_v + 1, step)
        elif "-" in base:
            a_s, b_s = base.split("-", 1)
            a, b = int(a_s), int(b_s)
            if not (min_v <= a <= max_v and min_v <= b <= max_v):
                raise ValueError(f"cron range outside bounds: {token}")
            if a <= b:
                seq = range(a, b + 1, step)
            else:
                # Overnight wrapped range, e.g. 22-6.
                joined = list(range(a, max_v + 1)) + list(range(min_v, b + 1))
                seq = joined[::step]
        else:
            n = int(base)
            if not (min_v <= n <= max_v):
                raise ValueError(f"cron value outside bounds: {token}")
            seq = (n,)
        vals.update(seq)
    if not vals:
        raise ValueError(f"empty cron field: {value}")
    return tuple(sorted(vals))


def _wrapped_hour_end(hour_field: str) -> int | None:
    """Return low-side endpoint for a wrapped hour range like 22-6."""
    for token in hour_field.split(","):
        base = token.split("/", 1)[0].strip()
        if "-" not in base:
            continue
        a_s, b_s = base.split("-", 1)
        try:
            a, b = int(a_s), int(b_s)
        except ValueError:
            continue
        if a > b:
            return b
    return None


def parse_cron(cron: str):
    """Parse scheduler cron including steps and overnight hour ranges.

    For an overnight range such as ``22-6 * * 1-5``, the 00:00-06:xx portion
    belongs to the session that started on the previous weekday. This makes a
    Friday US session continue into Saturday JST instead of silently losing its
    early-morning checks.
    """
    parts = cron.split()
    if len(parts) != 5:
        raise ValueError(f"Unsupported cron format: {cron}")
    minute_s, hour_s, _, _, dow_s = parts
    minutes = _parse_field(minute_s, 0, 59)
    hours = _parse_field(hour_s, 0, 23)
    dows = _parse_field(dow_s, 0, 7)
    wrapped_end = _wrapped_hour_end(hour_s)

    def _match(value: int, allowed: Field) -> bool:
        return True if allowed == "*" else value in allowed

    def pred(dt: datetime) -> bool:
        if not _match(dt.minute, minutes) or not _match(dt.hour, hours):
            return False
        effective = dt
        if wrapped_end is not None and dt.hour <= wrapped_end:
            from datetime import timedelta

            effective = dt - timedelta(days=1)
        dow = effective.weekday() + 1  # Mon=1 ... Sun=7
        if dow == 7:
            dow = 0
        return _match(dow, dows) or (dow == 0 and dows != "*" and 7 in dows)

    return pred


def task_monitor_portfolio() -> None:
    """Run risk protection first, then the existing PnL alert monitor."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
        from common.soft_trailing import run_soft_trailing

        result = run_soft_trailing(paper=True, dry_run=False)
        logging.info(
            "soft trailing pass: open=%s protected=%s closed=%s",
            result.get("market_open"),
            result.get("protected"),
            result.get("closed"),
        )
    except Exception:
        logging.exception("soft trailing monitor failed")
        try:
            _legacy._notify_task_error("soft_trailing_monitor")
        except Exception:
            logging.exception("soft trailing error notification failed")

    # Alerting failure must not prevent the protection pass above, and vice versa.
    _legacy.task_monitor_portfolio()


def task_update_trailing_stops_retired() -> None:
    """Retire the blanket cancel/recreate task now that protection is split.

    Whole-share positions keep broker-resident native trailing orders created at
    entry. Fractional S1/S4 positions are handled by ``task_monitor_portfolio``.
    No broker cancellation or order mutation happens here.
    """
    logging.info(
        "update_trailing_stops retired: native whole-share + fractional soft trailing are authoritative"
    )


# Inject only the intended behavior into the preserved scheduler.
_legacy.parse_cron = parse_cron
_legacy.TASKS["monitor_portfolio"] = task_monitor_portfolio
_legacy.TASKS["update_trailing_stops"] = task_update_trailing_stops_retired

# Re-export the existing task surface for imports/tests.
TASKS = _legacy.TASKS


def __getattr__(name: str):
    return getattr(_legacy, name)


def main():
    return _legacy.main()


if __name__ == "__main__":
    sys.exit(main())
