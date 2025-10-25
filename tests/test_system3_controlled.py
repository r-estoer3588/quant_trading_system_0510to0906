import pandas as pd

from core.system3 import generate_candidates_system3


def _make_row(date, **vals):
    # single-row DataFrame with given indicator values
    df = pd.DataFrame([vals], index=[pd.Timestamp(date)])
    return df


def test_system3_controlled_top10_latest_only():
    # Controlled symbols:
    # - A*: pass Phase2 filter but fail setup (Close <= sma150)
    # - B*: pass Phase2 and setup (drop3d >= 0.125, Close > sma150)
    # - C*: fail Phase2
    date = pd.Timestamp("2025-10-21")

    symbols_A = [f"A{i}" for i in range(1, 3)]
    symbols_B = [f"B{i}" for i in range(1, 12)]  # 11 B symbols -> top_n 10 selection
    symbols_C = [f"C{i}" for i in range(1, 3)]

    prepared = {}

    # A: Phase2 OK (low>=1, AvgVolume50>=1e6, atr_ratio>=0.05)
    # but setup fails (close <= sma150)
    for s in symbols_A:
        prepared[s] = _make_row(
            date,
            Low=2.0,
            AvgVolume50=1_500_000,
            atr_ratio=0.06,
            Close=10.0,
            sma150=11.0,  # close <= sma150 -> setup False
            drop3d=0.10,
            dollarvolume20=30_000_000,
        )

    # B: fully passing symbols (setup True)
    # give descending drop3d so ranking picks top ones deterministically
    drop_vals = [0.30 - i * 0.01 for i in range(len(symbols_B))]
    for s, dv in zip(symbols_B, drop_vals):
        prepared[s] = _make_row(
            date,
            Low=2.0,
            AvgVolume50=2_000_000,
            atr_ratio=0.08,
            Close=20.0,
            sma150=10.0,  # close > sma150 -> setup True
            drop3d=float(dv),
            dollarvolume20=50_000_000,
        )

    # C: fail Phase2 (low < 1 or avgvol < 1e6)
    for s in symbols_C:
        prepared[s] = _make_row(
            date,
            Low=0.5,
            AvgVolume50=500_000,
            atr_ratio=0.01,
            Close=3.0,
            sma150=5.0,
            drop3d=0.05,
            dollarvolume20=1_000_000,
        )

    # Run System3 generator in latest_only mode with top_n=10
    by_date, df_all, diagnostics = generate_candidates_system3(
        prepared,
        top_n=10,
        latest_only=True,
        include_diagnostics=True,
        latest_mode_date=date,
    )

    # Expect one label date and 10 candidates
    assert isinstance(by_date, dict), "by_date should be a dict"
    assert len(by_date) == 1, f"Expected 1 date key, got {len(by_date)}"
    vals = list(by_date.values())[0]
    assert len(vals) == 10, f"Expected 10 candidates, got {len(vals)}"

    # Diagnostics should reflect 10 ranked_top_n_count
    ranked_val = diagnostics.get("ranked_top_n_count")
    assert ranked_val == 10, f"Diagnostics ranked_top_n_count mismatch: {ranked_val}"

    # Ensure all returned symbols belong to the B group
    returned_syms = {x["symbol"] for x in vals}
    assert returned_syms.issubset(set(symbols_B)), (
        f"Returned symbols contain non-B symbols: {returned_syms - set(symbols_B)}"
    )
