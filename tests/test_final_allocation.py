from types import SimpleNamespace

import pandas as pd

from core.final_allocation import finalize_allocation


class _DummyStrategy:
    SYSTEM_NAME = "system"

    def __init__(self, max_positions: int = 3) -> None:
        self.config = {"max_positions": max_positions, "risk_pct": 0.05, "max_pct": 0.2}

    def calculate_position_size(
        self,
        capital: float,
        entry_price: float,
        stop_price: float,
        *,
        risk_pct: float,
        max_pct: float,
    ) -> int:
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share <= 0:
            return 0
        risk_budget = capital * risk_pct
        max_shares = capital * max_pct / entry_price
        shares = min(risk_budget / risk_per_share, max_shares)
        return int(shares)


def _make_candidates(symbols: list[str], *, system: str, score_start: float) -> pd.DataFrame:
    data = []
    score = score_start
    for sym in symbols:
        data.append(
            {
                "symbol": sym,
                "score": score,
                "entry_price": 100.0 + score,
                "stop_price": 95.0,
                "system": system,
            }
        )
        score -= 1
    return pd.DataFrame(data)


def test_finalize_allocation_slot_mode() -> None:
    per_system = {
        "system1": _make_candidates(["AAA", "BBB", "CCC"], system="system1", score_start=10),
        "system6": _make_candidates(["XXX", "YYY", "ZZZ"], system="system6", score_start=5),
    }
    strategies = {
        "system1": _DummyStrategy(max_positions=3),
        "system6": _DummyStrategy(max_positions=4),
    }
    positions = [SimpleNamespace(symbol="AAA", qty=1, side="long")]
    symbol_map = {"AAA": "system1"}

    final_df, summary = finalize_allocation(
        per_system,
        strategies=strategies,
        positions=positions,
        symbol_system_map=symbol_map,
        long_allocations={"system1": 1.0},
        short_allocations={"system6": 1.0},
        slots_long=3,
        slots_short=2,
    )

    assert not final_df.empty
    assert list(final_df["system"].unique()) == ["system1", "system6"]
    assert summary.mode == "slot"
    # One active position reduces available long slots to 2
    assert summary.slot_allocation == {"system1": 2, "system6": 2}
    assert summary.final_counts["system1"] == 2
    assert summary.final_counts["system6"] == 2
    assert list(final_df["no"]) == list(range(1, len(final_df) + 1))


def test_finalize_allocation_capital_mode() -> None:
    per_system = {
        "system1": _make_candidates(["AAA", "BBB"], system="system1", score_start=3),
        "system6": _make_candidates(["XXX", "YYY"], system="system6", score_start=2),
    }
    strategies = {
        "system1": _DummyStrategy(),
        "system6": _DummyStrategy(),
    }

    final_df, summary = finalize_allocation(
        per_system,
        strategies=strategies,
        symbol_system_map={},
        long_allocations={"system1": 1.0},
        short_allocations={"system6": 1.0},
        capital_long=10000,
        capital_short=8000,
    )

    assert not final_df.empty
    assert "shares" in final_df.columns
    assert "position_value" in final_df.columns
    assert set(final_df["side"]) == {"long", "short"}
    assert summary.mode == "capital"
    assert summary.budgets == {"system1": 10000.0, "system6": 8000.0}
    for system, remaining in summary.budget_remaining.items():
        assert 0 <= remaining <= summary.budgets[system]
    assert list(final_df["no"]) == list(range(1, len(final_df) + 1))
