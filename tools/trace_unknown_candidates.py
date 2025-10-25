#!/usr/bin/env python3
"""Trace unknown candidates from repro payload to find drop reason.

Reads repro_payloads/exclusion_report.csv and for each row with reason=="unknown"
re-runs finalize_allocation with ALLOCATION_DEBUG=1 while capturing logger
output, then extracts the first relevant log lines for that symbol to infer the
drop reason (e.g. already_selected, invalid_price, desired_shares_zero).

Outputs:
 - repro_payloads/unknown_trace_report.csv
 - repro_payloads/unknown_trace_report.json
"""
from __future__ import annotations

# json is not needed here; keep imports minimal
import logging
import os
import re
import sys
from io import StringIO
from pathlib import Path
from common.io_utils import safe_unicode, write_json, df_to_csv
from typing import Dict, List

import pandas as pd

# Ensure repo root is importable for 'core' and 'strategies' packages
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# ...existing code...

PAYLOAD_DIR = Path(__file__).resolve().parents[1] / "repro_payloads"


def capture_allocation_logs(
    per_system_map: Dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, object, str]:
    """Run finalize_allocation with ALLOCATION_DEBUG=1 and capture logs."""
    # Ensure debug env
    os.environ["ALLOCATION_DEBUG"] = "1"

    # Prepare logger capture
    sio = StringIO()
    handler = logging.StreamHandler(sio)
    handler.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(levelname)s: %(message)s")
    handler.setFormatter(fmt)

    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)

    try:
        from core.final_allocation import finalize_allocation

        final_df, summary = finalize_allocation(
            per_system=per_system_map,
            strategies=None,
            symbol_system_map=None,
            capital_long=100000.0,
            capital_short=100000.0,
        )
    finally:
        # flush and remove handler
        handler.flush()
        root.removeHandler(handler)

    logs = sio.getvalue()
    return final_df, summary, logs


def find_first_relevant_log(symbol: str, logs: str) -> List[str]:
    """Return lines from logs that mention symbol, up to first allocation decision."""
    lines = [line for line in logs.splitlines() if symbol in line]
    if not lines:
        # try lowercase symbol
        lsym = symbol.lower()
        lines = [line for line in logs.splitlines() if lsym in line.lower()]
    # return up to first ALLOC/Invalid/Already selected/No cash line
    out: List[str] = []
    for line in lines:
        out.append(line)
        if any(
            k in line
            for k in [
                "ALLOC shares",
                "Invalid",
                "Already selected",
                "No cash shares",
                "Invalid position_value",
                "Invalid shares",
                "skipped:",
            ]
        ):
            break
    return out


def infer_reason_from_logs(lines: List[str]) -> str:
    joined = "\n".join(lines)
    if not lines:
        return "no_log_found"
    if any("Already selected" in line for line in lines):
        return "already_selected"
    if any("Invalid entry/stop" in line for line in lines) or any(
        "Invalid prices" in line for line in lines
    ):
        return "invalid_price"
    if any("No cash shares" in line for line in lines) or re.search(
        r"No cash shares=\d+", joined
    ):
        return "no_cash_shares"
    if any("Invalid shares" in line for line in lines) or any(
        "Invalid shares" in line for line in lines
    ):
        return "desired_shares_zero"
    if any("skipped:" in line for line in lines):
        m = re.search(r"skipped: (.*)", joined)
        if m:
            return f"skipped:{m.group(1)}"
    if any("ALLOC shares" in line for line in lines):
        return "allocated"
    return "unknown_from_logs"


def _safe_unicode(text: object | None) -> str:
    """Return a UTF-8-safe string.

    Ensures the returned value is a str and that any bytes or invalid
    sequences are replaced so downstream JSON/CSV writers do not embed
    non-UTF-8 byte sequences into files.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    # Encode/decode with 'replace' to guarantee UTF-8 validity
    return text.encode("utf-8", errors="replace").decode("utf-8")


def load_per_system_payload(payload_dir: Path) -> Dict[str, pd.DataFrame]:
    per_files = sorted(payload_dir.glob("per_system_*.csv"))
    per_map: Dict[str, pd.DataFrame] = {}
    for p in per_files:
        name = p.stem.replace("per_system_", "").lower()
        try:
            df = pd.read_csv(p)
        except Exception:
            try:
                df = pd.read_csv(p, encoding="utf-8")
            except Exception:
                continue
        # normalize symbol/system
        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
        if "system" in df.columns:
            df["system"] = df["system"].astype(str).str.strip().str.lower()
        else:
            df["system"] = name
        per_map[name] = df
    return per_map


def main() -> int:
    payload_dir = PAYLOAD_DIR
    excl_csv = payload_dir / "exclusion_report.csv"
    if not excl_csv.exists():
        print("exclusion_report.csv not found; run analyze_repro_payload.py first")
        return 2

    excl = pd.read_csv(excl_csv)
    unknowns = excl[excl["reason"] == "unknown"]
    if unknowns.empty:
        print("No unknown candidates to trace")
        return 0

    per_map = load_per_system_payload(payload_dir)
    final_sig = None
    final_csv = next(payload_dir.glob("signals_final_*.csv"), None)
    if final_csv:
        final_sig = pd.read_csv(final_csv)

    # Run allocation once and capture logs
    final_df, summary, logs = capture_allocation_logs(per_map)

    results = []
    for _, row in unknowns.iterrows():
        system = str(row["system"]).strip().lower()
        symbol = str(row["symbol"]).strip().upper()
        # Safely coerce persisted_row_index to int with fallbacks
        persisted_idx = -1
        try:
            raw_idx = row.get("persisted_row_index", -1)
            if not pd.isna(raw_idx):
                persisted_idx = int(raw_idx)
        except Exception:
            persisted_idx = -1

        # Check if symbol exists in final under any system
        present_elsewhere = False
        present_system = None
        if final_sig is not None and not final_sig.empty:
            matches = final_sig[final_sig["symbol"].astype(str).str.upper() == symbol]
            if not matches.empty:
                # if present in final under different system
                systems_present = list(
                    matches["system"].astype(str).str.lower().unique()
                )
                if system not in systems_present:
                    present_elsewhere = True
                    present_system = systems_present[0]

        # Extract relevant logs for symbol
        relevant_lines = find_first_relevant_log(symbol, logs)
        inferred = infer_reason_from_logs(relevant_lines)

        # If present elsewhere, that explains the drop
        if present_elsewhere and inferred == "unknown_from_logs":
            inferred = f"selected_by_other_system:{present_system}"

        snippet = safe_unicode("\n".join(relevant_lines)[:4000])

        results.append(
            {
                "system": system,
                "symbol": symbol,
                "persisted_index": persisted_idx,
                "in_final_anywhere": bool(
                    (final_sig is not None)
                    and (symbol in final_sig["symbol"].astype(str).str.upper().tolist())
                ),
                "present_elsewhere_system": present_system or "",
                "inferred_reason": inferred,
                "log_snippet": snippet,
            }
        )

    out_df = pd.DataFrame(results)
    out_csv = payload_dir / "unknown_trace_report.csv"
    out_json = payload_dir / "unknown_trace_report.json"
    # Write files explicitly as UTF-8 to avoid platform default encodings
    # use helpers to guarantee UTF-8 safe output
    df_to_csv(out_df, out_csv, index=False)
    write_json(out_json, results, ensure_ascii=False, indent=2)

    print(f"Wrote unknown trace report CSV: {out_csv}")
    print(f"Wrote unknown trace report JSON: {out_json}")
    # print summary
    print("\nSummary:")
    summary_cols = [
        "system",
        "symbol",
        "inferred_reason",
        "present_elsewhere_system",
    ]
    print(out_df[summary_cols].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
