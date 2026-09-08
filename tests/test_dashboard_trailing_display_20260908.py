from pathlib import Path

import pytest

from common.trade_management import SYSTEM_TRADE_RULES
from scripts import export_alpaca_snapshot as ex

# Presentation contract: strategy trailing intent is not broker-observed residency.
ROOT = Path(__file__).resolve().parents[1]
COMPONENT = (
    ROOT / "apps" / "dashboards" / "alpaca-next" / "components" / "AlpacaSection.tsx"
)
TYPES = ROOT / "apps" / "dashboards" / "alpaca-next" / "lib" / "types.ts"
EXPORTER = ROOT / "scripts" / "export_alpaca_snapshot.py"


def test_snapshot_uses_execution_stop_floor_not_one_cent(monkeypatch):
    monkeypatch.delenv("PROTECT_STOP_FLOOR_ENABLED", raising=False)
    monkeypatch.delenv("PROTECT_STOP_FLOOR_PCT", raising=False)
    rules = SYSTEM_TRADE_RULES["system1"]
    stop, _ = ex._estimate_stop_target(
        side="long",
        avg_entry=6.77,
        rules=rules,
        atr={int(rules.stop_atr_period): 10.0},
    )
    assert stop == pytest.approx(3.385)
    assert stop != pytest.approx(0.01)


def test_snapshot_stop_floor_follows_execution_configuration(monkeypatch):
    monkeypatch.delenv("PROTECT_STOP_FLOOR_ENABLED", raising=False)
    monkeypatch.setenv("PROTECT_STOP_FLOOR_PCT", "0.40")
    rules = SYSTEM_TRADE_RULES["system1"]
    stop, _ = ex._estimate_stop_target(
        side="long",
        avg_entry=6.77,
        rules=rules,
        atr={int(rules.stop_atr_period): 10.0},
    )
    assert stop == pytest.approx(4.062)


def test_dashboard_distinguishes_trailing_from_fixed_stop():
    text = COMPONENT.read_text(encoding="utf-8")
    assert "if (p.exit_type === 'trailing')" in text
    assert "synthetic stop ${fmtPrice(p.stop_price_est)} ⚠" in text
    assert "trail gap · 日次ATR" in text
    assert "broker未検証" in text
    assert "trail ${trailPct.toFixed(0)}%" in text
    assert "p.exit_type === 'trailing' || p.exit_type === 'stop'" not in text


def test_trailing_width_is_exported_not_hard_coded_in_react():
    component = COMPONENT.read_text(encoding="utf-8")
    types = TYPES.read_text(encoding="utf-8")
    exporter = EXPORTER.read_text(encoding="utf-8")
    assert "trailing_stop_pct?: number | null" in types
    assert '"trailing_stop_pct": (' in exporter
    assert "p.trailing_stop_pct * 100" in component
    assert "trail 25%" not in component
    assert "trail 20%" not in component


def test_strategy_trailing_widths_are_canonical():
    assert SYSTEM_TRADE_RULES["system1"].trailing_stop_pct == pytest.approx(0.25)
    assert SYSTEM_TRADE_RULES["system4"].trailing_stop_pct == pytest.approx(0.20)


def test_fractional_classification_uses_tolerance():
    text = COMPONENT.read_text(encoding="utf-8")
    assert "Math.abs(p.qty - Math.round(p.qty)) <= 1e-6" in text
