# tests/test_system5_option_b_parity.py
from __future__ import annotations

import pandas as pd

from strategies.system5_strategy import System5Strategy


def _make_df(rows: int = 5, last_adx7: float = 60.0) -> pd.DataFrame:
    dates = pd.date_range("2025-10-20", periods=rows, freq="B")
    df = pd.DataFrame(
        {
            "Close": [10.0] * rows,
            "adx7": [10.0] * (rows - 1) + [last_adx7],
            "atr_pct": [0.03] * rows,
            "atr10": [1.0] * rows,
        },
        index=dates,
    )
    # setup/filter compatibility (core recalculates but ensure columns exist)
    df["filter"] = True
    df["setup"] = True
    return df


essential_keys = ("adx7", "atr_pct", "close")


def test_system5_option_b_parity_latest_only():
    # Arrange: minimal prepared dict with one clear candidate
    prepared = {"AAA": _make_df()}

    strat = System5Strategy()

    # Act (OFF): default path (Option-B disabled via kwargs)
    by_date_off, df_off = strat.generate_candidates(
        prepared,
        latest_only=True,
        top_n=5,
        use_option_b_utils=False,
    )
    diag_off = getattr(strat, "last_diagnostics", {}) or {}

    # Act (ON): Option-B path enabled via kwargs
    by_date_on, df_on = strat.generate_candidates(
        prepared,
        latest_only=True,
        top_n=5,
        use_option_b_utils=True,
    )
    diag_on = getattr(strat, "last_diagnostics", {}) or {}

    # Assert: candidates equivalence
    assert (df_off is not None) and (df_on is not None)
    assert len(df_off) == len(df_on)

    # by-date structure should match
    assert set(by_date_off.keys()) == set(by_date_on.keys())
    for dt in by_date_off.keys():
        syms_off = set(by_date_off[dt].keys())
        syms_on = set(by_date_on[dt].keys())
        assert syms_off == syms_on
        # Ensure major fields are present
        for sym in syms_off:
            payload_off = by_date_off[dt][sym]
            payload_on = by_date_on[dt][sym]
            for k in essential_keys:
                assert k in payload_off and k in payload_on

    # diagnostics key counts should be consistent
    assert int(diag_off.get("ranked_top_n_count", 0)) == int(
        diag_on.get("ranked_top_n_count", 0)
    )
    # When Option-B is ON, final_top_n_count mirrors ranked_top_n_count
    if "final_top_n_count" in diag_on:
        assert int(diag_on["final_top_n_count"]) == int(
            diag_on.get("ranked_top_n_count", 0)
        )
