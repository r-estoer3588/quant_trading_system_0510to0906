"""Monitor recompute rolling reports for a short observation period.

Usage:
    # Start monitoring for 3 business days and perform an immediate check
    python tools/monitor_recompute_reports.py --start --days 3 --check-now

    # Do a single check (useful for cron or manual invocation)
    python tools/monitor_recompute_reports.py --check-now

Behavior:
    * Creates state in results_csv_test/recompute_monitor_state.json when
        --start is used.
    * Looks for files matching results_csv_test/
        recompute_rolling_bulk_report*.json.
    * For each new report it finds, it invokes the notify helper
        (tools/notify_recompute_report.py) which sends notification only when
        failures are present.
    * Writes a monitoring check record under results_csv_test/monitoring/ for
        auditing.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

LOG = logging.getLogger("monitor_recompute")
_JST = ZoneInfo("Asia/Tokyo")

STATE_PATH = Path("results_csv_test") / "recompute_monitor_state.json"
MONITOR_DIR = Path("results_csv_test") / "monitoring"
REPORT_GLOB = "recompute_rolling_bulk_report*.json"
NOTIFY_SCRIPT = Path("tools") / "notify_recompute_report.py"


def is_business_day(d: date) -> bool:
    return d.weekday() < 5  # Mon-Fri are business days


def add_business_days(start: date, days: int) -> date:
    if days <= 0:
        return start
    # inclusive: start counts as day 1
    count = 1
    cur = start
    while count < days:
        cur = cur + timedelta(days=1)
        if is_business_day(cur):
            count += 1
    return cur


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {}
    try:
        text = STATE_PATH.read_text(encoding="utf-8")
        return cast(dict[str, Any], json.loads(text))
    except Exception:
        LOG.exception("failed to read state file; ignoring")
        return {}


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(state, ensure_ascii=False, indent=2)
    STATE_PATH.write_text(payload, encoding="utf-8")


def find_reports() -> list[Path]:
    p = Path("results_csv_test")
    if not p.exists():
        return []
    return sorted(p.glob(REPORT_GLOB), key=lambda p: p.stat().st_mtime)


def call_notify(report_path: Path) -> int:
    if not NOTIFY_SCRIPT.exists():
        LOG.warning("notify helper not found: %s", NOTIFY_SCRIPT)
        return 0
    try:
        cmd = [sys.executable, str(NOTIFY_SCRIPT), str(report_path)]
        LOG.info("Invoking notifier: %s", " ".join(cmd))
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            LOG.warning(
                "notify helper returned %d stdout=%s stderr=%s",
                res.returncode,
                res.stdout[:200],
                res.stderr[:200],
            )
        return res.returncode
    except Exception:
        LOG.exception("failed to invoke notifier")
        return 2


def inspect_report(report_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        LOG.exception("failed to parse report: %s", report_path)
        return {"error": "parse_failed"}

    per = payload.get("per_symbol", {}) or {}
    processed = payload.get("processed", 0)
    would_update = payload.get("would_update", 0)
    updated = payload.get("updated", 0)
    errors = payload.get("errors", {}) or {}
    failures = []
    for sym, info in per.items():
        if not info.get("success"):
            failures.append({"symbol": sym, "message": info.get("message")})
        else:
            msg = info.get("message")
            if msg and str(msg).startswith("recompute_failed"):
                failures.append({"symbol": sym, "message": msg})

    return {
        "path": str(report_path),
        "processed": processed,
        "would_update": would_update,
        "updated": updated,
        "errors_count": len(errors),
        "failures": failures,
    }


def do_check(state: dict[str, Any]) -> dict[str, Any]:
    MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    reports = find_reports()
    seen = set(state.get("seen_files") or [])
    new_reports = [r for r in reports if str(r) not in seen]

    LOG.info("found %d reports, %d new", len(reports), len(new_reports))

    check_summary: dict[str, Any] = {
        "timestamp": datetime.now(tz=_JST).isoformat(),
        "total_reports": len(reports),
        "new_reports_count": len(new_reports),
        "new_reports": [],
    }

    any_failures = False
    for rpt in new_reports:
        info = inspect_report(rpt)
        check_summary["new_reports"].append(info)
        # If the report contains failures or errors, call notifier
        if info.get("failures") or info.get("errors_count", 0) > 0:
            any_failures = True
            rc = call_notify(rpt)
            info["notifier_exit_code"] = rc
        else:
            info["notifier_exit_code"] = 0
        # mark as seen
        seen.add(str(rpt))

    # write a monitoring record for auditing
    now_ts = datetime.now(tz=_JST).strftime("%Y%m%d_%H%M%S")
    out_path = MONITOR_DIR / f"recompute_monitor_check_{now_ts}.json"
    payload = json.dumps(check_summary, ensure_ascii=False, indent=2)
    out_path.write_text(payload, encoding="utf-8")

    # persist updated seen list
    state["seen_files"] = sorted(list(seen))
    state["last_checked_ts"] = datetime.now(tz=_JST).isoformat()
    save_state(state)

    check_summary["any_failures"] = any_failures
    return check_summary


def start_monitor(days: int) -> dict[str, Any]:
    today = datetime.now(tz=_JST).date()
    end_date = add_business_days(today, days)
    state = {
        "start_date": today.isoformat(),
        "start_ts": datetime.now(tz=_JST).isoformat(),
        "days": days,
        "end_date": end_date.isoformat(),
        "seen_files": [],
        "last_checked_ts": None,
    }
    save_state(state)
    LOG.info("monitor started: %s -> %s", state["start_date"], state["end_date"])
    return state


def stop_monitor() -> None:
    if STATE_PATH.exists():
        STATE_PATH.unlink()
        LOG.info("monitor stopped (state removed)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=("Monitor recompute rolling reports for a short observation period")
    )
    p.add_argument(
        "--start",
        action="store_true",
        help="Start monitoring (writes state file)",
    )
    p.add_argument(
        "--stop",
        action="store_true",
        help="Stop monitoring (removes state file)",
    )
    p.add_argument(
        "--days",
        type=int,
        default=3,
        help="Number of business days to observe (inclusive). Default: 3",
    )
    p.add_argument(
        "--check-now",
        action="store_true",
        help="Perform an immediate check now",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)
    state = load_state()

    if args.stop:
        stop_monitor()
        return 0

    if args.start:
        state = start_monitor(args.days)

    if args.check_now:
        summary = do_check(state)
        s = json.dumps(summary, ensure_ascii=False)
        LOG.info("check summary: %s", s)

    # If a start was requested and no immediate check requested, still perform
    # an initial check so the baseline is created.
    if args.start and not args.check_now:
        summary = do_check(state)
        s2 = json.dumps(summary, ensure_ascii=False)
        LOG.info("initial baseline created: %s", s2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
