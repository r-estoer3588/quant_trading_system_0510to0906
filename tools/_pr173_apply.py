from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected exactly one match, found {count}: {old[:80]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def regex_once(path: Path, pattern: str, replacement: str) -> None:
    text = path.read_text(encoding="utf-8")
    new, count = re.subn(pattern, replacement, text, count=1, flags=re.S)
    if count != 1:
        raise SystemExit(f"{path}: expected exactly one regex match, found {count}")
    path.write_text(new, encoding="utf-8")


# 1) Reuse the execution-side canonical stop-floor logic from snapshot code via
#    a small public wrapper. This avoids copying the env-driven disaster floor.
alpaca = ROOT / "common" / "alpaca_trading.py"
marker = "\n\ndef _target_price_for(\n"
wrapper = '''\n\ndef protective_stop_price(\n    *,\n    side: str,\n    avg_entry_price: float,\n    rules: Any,\n    atr_value: float | None,\n    symbol: str = \"snapshot\",\n) -> float | None:\n    \"\"\"Public read-only wrapper around the canonical protective-stop formula.\n\n    Snapshot/reporting code must use the same env-driven floor semantics as the\n    execution path instead of reimplementing ``max(0.01, ...)``.\n    \"\"\"\n    snap = PositionSnapshot(\n        symbol=symbol,\n        qty=1.0,\n        side=side,\n        avg_entry_price=float(avg_entry_price),\n    )\n    return _stop_price_for(snap, rules, atr_value)\n\n\ndef _target_price_for(\n'''
replace_once(alpaca, marker, wrapper)

# 2) Snapshot exporter: import canonical helper and stop emitting the stale
#    $0.01 pseudo-stop. Also export trailing percentage from strategy rules.
exporter = ROOT / "scripts" / "export_alpaca_snapshot.py"
replace_once(
    exporter,
    "    compute_holding_days,\n    parse_entry_date_from_client_order_id,",
    "    compute_holding_days,\n    protective_stop_price,\n    parse_entry_date_from_client_order_id,",
)
regex_once(
    exporter,
    r'''    atr_stop = atr\.get\(int\(getattr\(rules, "stop_atr_period", 20\)\)\)\n    if atr_stop:\n        dist = atr_stop \* float\(getattr\(rules, "stop_atr_multiplier", 0\) or 0\)\n        if dist > 0:\n            stop = round\(\n                max\(0\.01, avg_entry - dist\) if side == "long" else avg_entry \+ dist, 4\n            \)''',
    '''    atr_stop = atr.get(int(getattr(rules, "stop_atr_period", 20)))\n    if atr_stop:\n        canonical_stop = protective_stop_price(\n            side=side,\n            avg_entry_price=avg_entry,\n            rules=rules,\n            atr_value=atr_stop,\n            symbol="dashboard-snapshot",\n        )\n        if canonical_stop is not None:\n            stop = round(canonical_stop, 4)''',
)
replace_once(
    exporter,
    '            "exit_type": _exit_type(sys_label, rules),\n            "exit_expected": exit_expected,',
    '            "exit_type": _exit_type(sys_label, rules),\n            "trailing_stop_pct": (\n                float(rules.trailing_stop_pct)\n                if rules is not None\n                and getattr(rules, "use_trailing_stop", False)\n                and getattr(rules, "trailing_stop_pct", 0) > 0\n                else None\n            ),\n            "exit_expected": exit_expected,',
)

# 3) Type contract.
types = ROOT / "apps" / "dashboards" / "alpaca-next" / "lib" / "types.ts"
replace_once(
    types,
    '  /** "time" | "trailing" | "stop" | "spy_hedge" | "unknown" */\n  exit_type: string;\n',
    '  /** "time" | "trailing" | "stop" | "spy_hedge" | "unknown" */\n  exit_type: string;\n  /** Strategy trailing width as a fraction (S1=0.25, S4=0.20). Intent only. */\n  trailing_stop_pct?: number | null;\n',
)

# 4) Dashboard badge: do not collapse trailing into ATR stop. Whole-share
#    positions show strategy trail intent without claiming broker residency;
#    legacy fractional positions show the synthetic-daily gap explicitly.
component = ROOT / "apps" / "dashboards" / "alpaca-next" / "components" / "AlpacaSection.tsx"
old = '''  if ((p.exit_type === 'trailing' || p.exit_type === 'stop') && p.stop_price_est != null) {\n    return {\n      text: `stop ${fmtPrice(p.stop_price_est)}`,\n      cls: 'bg-white/10 text-muted',\n      sub: p.distance_to_stop_pct != null ? `${fmtPct(p.distance_to_stop_pct, 1)}` : undefined,\n    };\n  }\n'''
new = '''  if (p.exit_type === 'trailing') {\n    const wholeShare = Math.abs(p.qty - Math.round(p.qty)) <= 1e-6;\n    const trailPct =\n      p.trailing_stop_pct != null && Number.isFinite(p.trailing_stop_pct)\n        ? p.trailing_stop_pct * 100\n        : null;\n    if (!wholeShare) {\n      return p.stop_price_est != null\n        ? {\n            text: `synthetic stop ${fmtPrice(p.stop_price_est)} ⚠`,\n            cls: 'bg-warn/15 text-warn',\n            sub: 'trail gap · 日次ATR',\n          }\n        : {\n            text: 'synthetic daily ⚠',\n            cls: 'bg-warn/15 text-warn',\n            sub: 'trail gap · threshold未計測',\n          };\n    }\n    return {\n      text: trailPct != null ? `trail ${trailPct.toFixed(0)}%` : 'trail',\n      cls: 'bg-sky-400/15 text-sky-300',\n      sub: 'broker未検証',\n    };\n  }\n  if (p.exit_type === 'stop' && p.stop_price_est != null) {\n    return {\n      text: `stop ${fmtPrice(p.stop_price_est)}`,\n      cls: 'bg-white/10 text-muted',\n      sub: p.distance_to_stop_pct != null ? `${fmtPct(p.distance_to_stop_pct, 1)}` : undefined,\n    };\n  }\n'''
replace_once(component, old, new)

# 5) Regression guard. Keep this deliberately narrow: UI intent vs broker state,
#    shared floor behavior, and no hard-coded S1/S4 trail constants in React.
test = ROOT / "tests" / "test_dashboard_trailing_display_20260908.py"
test.write_text('''from pathlib import Path\n\nimport pytest\n\nfrom common.trade_management import SYSTEM_TRADE_RULES\nfrom scripts import export_alpaca_snapshot as ex\n\n\nROOT = Path(__file__).resolve().parents[1]\nCOMPONENT = ROOT / "apps" / "dashboards" / "alpaca-next" / "components" / "AlpacaSection.tsx"\nTYPES = ROOT / "apps" / "dashboards" / "alpaca-next" / "lib" / "types.ts"\nEXPORTER = ROOT / "scripts" / "export_alpaca_snapshot.py"\n\n\ndef test_snapshot_uses_execution_stop_floor_not_one_cent(monkeypatch):\n    monkeypatch.delenv("PROTECT_STOP_FLOOR_ENABLED", raising=False)\n    monkeypatch.delenv("PROTECT_STOP_FLOOR_PCT", raising=False)\n    rules = SYSTEM_TRADE_RULES["system1"]\n    stop, _ = ex._estimate_stop_target(\n        side="long",\n        avg_entry=6.77,\n        rules=rules,\n        atr={int(rules.stop_atr_period): 10.0},\n    )\n    assert stop == pytest.approx(3.385)\n    assert stop != pytest.approx(0.01)\n\n\ndef test_dashboard_distinguishes_trailing_from_fixed_stop():\n    text = COMPONENT.read_text(encoding="utf-8")\n    assert "if (p.exit_type === 'trailing')" in text\n    assert "synthetic stop ${fmtPrice(p.stop_price_est)} ⚠" in text\n    assert "trail gap · 日次ATR" in text\n    assert "broker未検証" in text\n    assert "trail ${trailPct.toFixed(0)}%" in text\n    assert "p.exit_type === 'trailing' || p.exit_type === 'stop'" not in text\n\n\ndef test_trailing_width_is_exported_not_hard_coded_in_react():\n    component = COMPONENT.read_text(encoding="utf-8")\n    types = TYPES.read_text(encoding="utf-8")\n    exporter = EXPORTER.read_text(encoding="utf-8")\n    assert "trailing_stop_pct?: number | null" in types\n    assert '"trailing_stop_pct": (' in exporter\n    assert "p.trailing_stop_pct * 100" in component\n    assert "trail 25%" not in component\n    assert "trail 20%" not in component\n\n\ndef test_fractional_classification_uses_tolerance():\n    text = COMPONENT.read_text(encoding="utf-8")\n    assert "Math.abs(p.qty - Math.round(p.qty)) <= 1e-6" in text\n''', encoding="utf-8")

print("PR173 patch applied")
