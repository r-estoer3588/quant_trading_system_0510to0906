"""Create CSV summary from excluded_vs_final JSON and optionally notify Slack.

Behavior:
- Finds the latest `results_csv_test/excluded_vs_final_*.json` file.
- Writes a CSV summary to `results_csv_test/excluded_vs_final_summary_*.csv`.
- If environment variable `SLACK_WEBHOOK` is set, posts a short message with counts
  to the webhook URL. No webhook => only write files and print output.

Usage:
  python scripts/notify_excluded_vs_final.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def find_latest_json(dirpath: Path) -> Path | None:
    files = sorted(dirpath.glob("excluded_vs_final_*.json"))
    return files[-1] if files else None


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf8"))


def build_dataframe(summary: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for system, info in sorted(summary.items()):
        rows.append(
            {
                "system": system,
                "persisted_count": info.get("persisted_count", 0),
                "persisted_symbols": "|".join(info.get("persisted_symbols", [])),
                "kept_in_final": "|".join(info.get("kept_in_final", [])),
                "dropped_in_final": "|".join(info.get("dropped_in_final", [])),
                "extra_in_final": "|".join(info.get("extra_in_final", [])),
            }
        )
    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = outdir / f"excluded_vs_final_summary_{ts}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path


def post_slack(webhook: str, text: str) -> bool:
    # Try requests if available, else fallback to urllib
    try:
        import requests

        resp = requests.post(webhook, json={"text": text}, timeout=10)
        return resp.status_code == 200
    except Exception:
        try:
            import json as _json
            import urllib.request

            data = _json.dumps({"text": text}).encode("utf8")
            req = urllib.request.Request(
                webhook,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return 200 <= resp.getcode() < 300
        except Exception:
            return False


def main() -> int:
    results_dir = ROOT / "results_csv_test"
    latest = find_latest_json(results_dir)
    if latest is None:
        print("No excluded_vs_final_*.json found in results_csv_test")
        return 2

    print("Loading", latest)
    try:
        summary = load_summary(latest)
    except Exception as e:
        print("Failed to load JSON:", e)
        return 2

    df = build_dataframe(summary)
    out_csv = write_csv(df, results_dir)
    print("Wrote CSV summary:", out_csv)

    # Compose Slack text: one-line per system with counts
    lines = []
    for _, row in df.iterrows():
        dropped = row["dropped_in_final"]
        if not dropped:
            dropped_count = 0
        else:
            dropped_count = len(str(dropped).split("|"))
        lines.append(f"{row['system']}: persisted={row['persisted_count']} " f"dropped={dropped_count}")
    text = "Excluded vs Final summary:\n" + "\n".join(lines)

    webhook = os.environ.get("SLACK_WEBHOOK")
    if webhook:
        ok = post_slack(webhook, text)
        if ok:
            print("Slack notification sent")
        else:
            print("Slack notification failed")
    else:
        print("\nSlack webhook not set; skipping notification. Summary:")
        print(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
