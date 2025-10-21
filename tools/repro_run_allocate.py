"""Run finalize_allocation on repro per-system CSVs with mocked strategies.

Prints allocation summary and allocator_excludes so we can see why symbols were
dropped (e.g., already_selected).
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import json

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "repro_payloads"


class MockStrategy:
    def __init__(self, name: str, risk_pct: float = 0.02, max_pct: float = 0.1, max_positions: int = 10):
        self.SYSTEM_NAME = name
        self.config = {"risk_pct": risk_pct, "max_pct": max_pct, "max_positions": max_positions}

    def calculate_position_size(self, available_cap, entry_price, stop_price, risk_pct=0.02, max_pct=0.1):
        try:
            if not entry_price or entry_price <= 0:
                return 0
            max_by_cash = int(available_cap // abs(entry_price)) if available_cap and entry_price else 0
            # Simple policy: allocate at most 1% of available_cap per position in shares
            target_dollars = max(1.0, float(available_cap) * 0.01)
            target_shares = int(target_dollars // abs(entry_price))
            if target_shares <= 0:
                return max_by_cash if max_by_cash > 0 else 0
            return min(max_by_cash, target_shares)
        except Exception:
            return 0


def load_per_system() -> dict[str, pd.DataFrame]:
    out = {}
    for p in REPRO.glob("per_system_*.csv"):
        name = p.stem.replace("per_system_", "")
        try:
            df = pd.read_csv(p)
            out[name.lower()] = df
        except Exception:
            continue
    return out


def main() -> int:
    per_system = load_per_system()
    if not per_system:
        print("No per-system repro CSVs found in repro_payloads/")
        return 2

    # Build mocked strategies for systems found
    strategies = {name: MockStrategy(name) for name in per_system.keys()}

    # Import finalize_allocation lazily
    from core.final_allocation import finalize_allocation

    # Run allocation in capital mode with modest budgets to force realistic flows
    final_df, summary = finalize_allocation(
        per_system,
        strategies=strategies,
        positions=[],
        symbol_system_map={},
        capital_long=50000.0,
        capital_short=50000.0,
        include_trade_management=False,
    )

    print("Final rows:", len(final_df) if final_df is not None else 0)
    try:
        print("Final symbols:", summary.final_symbols)
    except Exception:
        pass

    # diagnostics
    diag = getattr(summary, "system_diagnostics", {}) or {}
    print("Diagnostics keys:", sorted(list(diag.keys())))

    alloc_ex = None
    try:
        alloc_ex = diag.get("allocator_excludes") if isinstance(diag, dict) else None
    except Exception:
        alloc_ex = None

    print("allocator_excludes (from summary.system_diagnostics):")
    print(json.dumps(alloc_ex, ensure_ascii=False, indent=2))

    # Also print per-system final_counts
    try:
        print("final_counts:", summary.final_counts)
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
