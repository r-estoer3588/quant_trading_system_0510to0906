"""Analyze repro payload snapshot + per-system CSVs and final CSV.

Writes results to results_csv_test/excluded_vs_final_repro.json and prints a concise summary
for system1 and system4.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "repro_payloads"
OUT = ROOT / "results_csv_test"
OUT.mkdir(parents=True, exist_ok=True)


def load_snapshot(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf8"))
    except Exception:
        return {}


def load_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, encoding="utf8")
        except Exception:
            return None


def main() -> int:
    snap_path = REPRO / "diagnostics_snapshot_20251021_204958.json"
    final_path = REPRO / "signals_final_2025-10-21.csv"

    snap = load_snapshot(snap_path) if snap_path.exists() else {}

    final_df = load_csv(final_path)
    if final_df is None:
        print("Final CSV not found in repro_payloads")
        return 2

    final_symbols_by_system: dict[str, set[str]] = {}
    if "system" in final_df.columns and "symbol" in final_df.columns:
        for _, r in final_df.iterrows():
            sysn = str(r["system"]) if r["system"] is not None else ""
            sym = str(r["symbol"]) if r["symbol"] is not None else ""
            final_symbols_by_system.setdefault(sysn, set()).add(sym)

    summary = {}
    for per in REPRO.glob("per_system_*.csv"):
        sysname = per.stem.replace("per_system_", "")
        df = load_csv(per)
        if df is None:
            continue
        if "symbol" in df.columns:
            syms = set(df["symbol"].astype(str).tolist())
        else:
            syms = set()

        final_syms = final_symbols_by_system.get(sysname, set())
        dropped = sorted(list(syms - final_syms))
        kept = sorted(list(syms & final_syms))
        extra = sorted(list(final_syms - syms))

        # diagnostics from snapshot
        diag_map = {}
        try:
            if isinstance(snap, dict) and "systems" in snap:
                for s in snap["systems"]:
                    sid = s.get("system_id")
                    if sid == sysname:
                        diag_map = s.get("diagnostics_extra") or s.get("diagnostics") or {}
                        break
                    if sid == "__allocation__":
                        # capture allocation-level diagnostics separately
                        alloc = s.get("diagnostics_extra") or s.get("diagnostics") or {}
                        # attach under special key
                        diag_map.setdefault("__allocation__", alloc)
        except Exception:
            diag_map = {}

        summary[sysname] = {
            "persisted_count": len(syms),
            "persisted_symbols": sorted(list(syms)),
            "kept_in_final": kept,
            "dropped_in_final": dropped,
            "extra_in_final": extra,
            "diagnostics": diag_map,
        }

    out_path = OUT / "excluded_vs_final_repro.json"
    with out_path.open("w", encoding="utf8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(f"Wrote repro report: {out_path}")
    # print concise for system1 and system4
    for s in ["system1", "system4"]:
        info = summary.get(s)
        if not info:
            print(f"No data for {s}")
            continue
        print("\n" + "=" * 40)
        print(f"System: {s}")
        print(f"Persisted: {info['persisted_count']} -> {', '.join(info['persisted_symbols'])}")
        print(f"Kept in final: {', '.join(info['kept_in_final']) or '(none)'}")
        print(f"Dropped: {', '.join(info['dropped_in_final']) or '(none)'}")
        if info["diagnostics"]:
            print("Diagnostics keys: " + ", ".join(sorted(info["diagnostics"].keys())))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
