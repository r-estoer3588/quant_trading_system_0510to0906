#!/usr/bin/env python3
"""Analyze exported repro payload and produce per-candidate exclusion report.

Reads repro_payloads/meta.json, per-system CSVs, diagnostics snapshot and final
signals CSV. Runs finalize_allocation over the reconstructed per_system frames to
capture allocator-level exclusions, then produces a CSV report mapping each
persisted candidate to the reason it was dropped (or kept).

Output:
 - repro_payloads/exclusion_report.csv
 - prints summary to stdout
"""
from __future__ import annotations

from collections import defaultdict
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from common.io_utils import df_to_csv, write_json

LOG = logging.getLogger("analyze_repro_payload")


def _resolve_path(repo_root: Path, s: str | None, payload_dir: Path) -> Path | None:
    if not s:
        return None
    p = Path(s)
    if p.is_absolute() and p.exists():
        return p
    # try relative to repo_root
    cand = repo_root / s
    if cand.exists():
        return cand
    # try payload_dir with basename
    cand2 = payload_dir / p.name
    if cand2.exists():
        return cand2
    return None


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    payload_dir = repo_root / "repro_payloads"
    if not payload_dir.exists():
        print("repro_payloads directory not found - run export_run_repro.py first")
        return 2

    meta_path = payload_dir / "meta.json"
    if not meta_path.exists():
        print("meta.json not found in repro_payloads - cannot proceed")
        return 2

    meta = json.loads(meta_path.read_text(encoding="utf8"))

    # Snapshot
    snapshot_path = _resolve_path(repo_root, meta.get("snapshot"), payload_dir)
    snapshot = {}
    if snapshot_path is not None:
        try:
            snapshot = json.loads(snapshot_path.read_text(encoding="utf8"))
        except Exception as e:
            print(f"Could not load snapshot {snapshot_path}: {e}")

    # Load final signals CSV
    final_signals_path = _resolve_path(
        repo_root, meta.get("final_signals"), payload_dir
    )
    final_df = pd.DataFrame()
    if final_signals_path is not None:
        try:
            final_df = pd.read_csv(final_signals_path)
        except Exception as e:
            print(f"Could not read final signals CSV {final_signals_path}: {e}")

    # Build final set of (system, symbol)
    final_set: set[tuple[str, str]] = set()
    if (
        not final_df.empty
        and "system" in final_df.columns
        and "symbol" in final_df.columns
    ):
        for _, row in final_df.iterrows():
            final_set.add(
                (str(row["system"]).strip().lower(), str(row["symbol"]).strip().upper())
            )

    # Load per-system persisted CSVs (from meta if possible, else scan payload_dir)
    per_system_entries = meta.get("per_system_files") or []
    per_system_files = []
    for ent in per_system_entries:
        csv_path = _resolve_path(repo_root, ent.get("csv"), payload_dir)
        if csv_path is not None:
            per_system_files.append(csv_path)

    # fallback: find per_system_*.csv in payload dir
    if not per_system_files:
        per_system_files = list(sorted(payload_dir.glob("per_system_*.csv")))

    per_system_map: dict[str, pd.DataFrame] = {}
    for p in per_system_files:
        name = p.stem.replace("per_system_", "")
        try:
            df = pd.read_csv(p)
        except Exception:
            try:
                df = pd.read_csv(p, encoding="utf-8")
            except Exception:
                print(f"Could not parse per-system file: {p}")
                continue
        # Normalize column names to lower-case canonical keys
        rename = {}
        for c in df.columns:
            lc = c.strip()
            if lc.lower() == "symbol":
                rename[c] = "symbol"
            elif lc.lower() == "system":
                rename[c] = "system"
            elif lc.lower() in ("entry_price", "entryprice"):
                rename[c] = "entry_price"
            elif lc.lower() in ("stop_price", "stopprice"):
                rename[c] = "stop_price"
            elif lc.lower() == "close":
                rename[c] = "close"
            elif lc.lower() == "atr10":
                rename[c] = "atr10"
        try:
            if rename:
                df = df.rename(columns=rename)
        except Exception:
            pass
        # ensure symbol column exists
        if "symbol" not in df.columns:
            # try to find any column named like symbol
            possible = [c for c in df.columns if c.lower() == "symbol"]
            if possible:
                df = df.rename(columns={possible[0]: "symbol"})
        # normalize symbol/system
        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
        if "system" in df.columns:
            df["system"] = df["system"].astype(str).str.strip().str.lower()
        else:
            df["system"] = name.lower()

        per_system_map[name.lower()] = df

    # Build generation-level exclude map from snapshot
    gen_exclude_map: dict[str, dict[str, set[str]]] = defaultdict(dict)
    try:
        # prefer per-system entries in snapshot
        for s in snapshot.get("systems", []) if isinstance(snapshot, dict) else []:
            sid = str(s.get("system_id") or "").strip()
            if not sid or sid.startswith("__"):
                continue
            extra = s.get("diagnostics_extra") or {}
            ex = extra.get("exclude_symbols") or {}
            if isinstance(ex, dict):
                for reason, syms in ex.items():
                    gen_exclude_map[sid.lower()][reason] = set(
                        map(lambda x: str(x).upper(), syms or [])
                    )
        # fallback: top-level __allocation__ payload may contain
        # a diagnostics_extra mapping with per-system entries
        for s in snapshot.get("systems", []):
            if s.get("system_id") == "__allocation__":
                alloc_extra = s.get("diagnostics_extra") or {}
                # alloc_extra may itself contain per-system entries
                for key, val in (alloc_extra or {}).items():
                    if key.startswith("system") and isinstance(val, dict):
                        ex = val.get("exclude_symbols") or {}
                        for reason, syms in ex.items():
                            gen_exclude_map[key.lower()][reason] = set(
                                map(lambda x: str(x).upper(), syms or [])
                            )
    except Exception:
        pass

    # Run finalize_allocation to capture allocator-level excludes
    alloc_excludes: dict[str, dict[str, set[str]]] = {}
    try:
        from core.final_allocation import finalize_allocation

        print("Running finalize_allocation to capture excludes...")
        final_run_df, summary_run = finalize_allocation(
            per_system=per_system_map,
            strategies=None,
            symbol_system_map=None,
            capital_long=100000.0,
            capital_short=100000.0,
        )
        # summary_run.system_diagnostics may contain allocator_excludes
        sd = getattr(summary_run, "system_diagnostics", {}) or {}
        ae = sd.get("allocator_excludes") or {}
        # ae expected like {"long": {reason:set(sym)}, "short": {...}}
        if isinstance(ae, dict):
            for side, mapping in ae.items():
                for reason, syms in (mapping or {}).items():
                    try:
                        symlist = set(map(lambda x: str(x).upper(), syms or []))
                    except Exception:
                        symlist = set()
                    alloc_excludes.setdefault(side, {})[reason] = symlist
    except Exception as e:
        print(f"Could not run finalize_allocation: {e}")

    # Build per-candidate report rows
    rows: list[dict[str, Any]] = []
    for sysname, df in per_system_map.items():
        for idx, row in df.reset_index(drop=True).iterrows():
            sym = str(row.get("symbol") or "").strip().upper()
            system_key = str(row.get("system") or sysname).strip().lower()
            included = (system_key, sym) in final_set
            gen_reasons = []
            # collect generation reasons
            for reason, syms in gen_exclude_map.get(system_key, {}).items():
                if sym in syms:
                    gen_reasons.append(reason)
            alloc_reasons = []
            for side, mapping in alloc_excludes.items():
                for reason, syms in mapping.items():
                    if sym in syms:
                        alloc_reasons.append(f"{side}:{reason}")

            # entry/close/atr values
            entry_price = (
                row.get("entry_price")
                if "entry_price" in row
                else row.get("entryprice")
            )
            close_val = (
                row.get("Close")
                if "Close" in row
                else (row.get("close") if "close" in row else None)
            )
            atr_val = (
                row.get("atr10")
                if "atr10" in row
                else (row.get("ATR10") if "ATR10" in row else None)
            )

            if included:
                reason = "kept"
            elif gen_reasons:
                reason = ",".join(gen_reasons)
            elif alloc_reasons:
                reason = ",".join(alloc_reasons)
            else:
                reason = "unknown"

            # Safe conversion of index to integer for persistence
            try:
                # Coerce index to string then int to avoid type checker
                # complaints when idx is a Hashable/pandas index type.
                persisted_idx_val = int(str(idx))
            except Exception:
                try:
                    persisted_idx_val = int(float(str(idx)))
                except Exception:
                    persisted_idx_val = -1

            rows.append(
                {
                    "system": system_key,
                    "symbol": sym,
                    "persisted_row_index": persisted_idx_val,
                    "entry_price": entry_price,
                    "close": close_val,
                    "atr10": atr_val,
                    "included_in_final": bool(included),
                    "reason": reason,
                    "gen_reasons": ",".join(gen_reasons) if gen_reasons else "",
                    "alloc_reasons": ",".join(alloc_reasons) if alloc_reasons else "",
                }
            )

    out_df = pd.DataFrame(rows)
    out_csv = payload_dir / "exclusion_report.csv"
    # Ensure CSV is written using UTF-8 for cross-platform reproducibility
    df_to_csv(out_df, out_csv, index=False)
    print(f"Wrote exclusion report: {out_csv} ({len(out_df)} rows)")

    # Print concise summary
    print("\nSummary by reason:")
    reason_counts = out_df[~out_df["included_in_final"]]["reason"].value_counts()
    # Ensure mapping types are simple Python primitives for downstream use
    reason_counts_dict = {str(k): int(v) for k, v in reason_counts.to_dict().items()}
    for r, c in reason_counts_dict.items():
        try:
            cnt = int(c)
        except Exception:
            cnt = 0
        print(f"  {r}: {cnt}")

    print("\nTop systems with dropped symbols:")
    dropped = out_df[~out_df["included_in_final"]]
    sys_counts = dropped["system"].value_counts()
    sys_counts_dict = {str(k): int(v) for k, v in sys_counts.to_dict().items()}
    for s, c in sys_counts_dict.items():
        try:
            cnt = int(c)
        except Exception:
            cnt = 0
        print(f"  {s}: {cnt}")

    # Write a JSON summary for quick programmatic consumption
    summary_json = payload_dir / "exclusion_report_summary.json"
    try:
        summary_out = {
            "total_candidates": int(len(out_df)),
            "total_dropped": int((~out_df["included_in_final"]).sum()),
            "reasons": reason_counts_dict,
            "dropped_by_system": sys_counts_dict,
        }
        write_json(summary_json, summary_out, ensure_ascii=False, indent=2)
        print(f"Wrote summary JSON: {summary_json}")
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
