# tests/test_system3_option_b_parity.py
"""Parity test for System3 Option-B (prepare_ranking_input, apply_thresholds)."""
from __future__ import annotations

import pandas as pd

from core.system3 import generate_candidates_system3


def _make_df(rows: int = 5, last_drop3d: float = 0.2) -> pd.DataFrame:
    """Create test DataFrame with proper System3 format."""
    dates = pd.date_range("2025-10-20", periods=rows, freq="B")
    df = pd.DataFrame(
        {
            "Close": [10.0] * rows,
            "dollarvolume20": [30_000_000.0] * rows,
            "atr_ratio": [0.06] * rows,
            "drop3d": [0.0] * (rows - 1) + [last_drop3d],
            "atr10": [1.0] * rows,
        },
        index=dates,
    )
    # setup/filter compatibility (core recalculates but ensure columns exist)
    df["filter"] = True
    df["setup"] = df["drop3d"] >= 0.125
    return df


class TestSystem3OptionBParity:
    """Verify that Option-B utilities produce equivalent results."""

    def test_system3_latest_only_parity(self):
        """Test parity: with/without Option-B should yield equivalent results."""
        # Arrange: minimal prepared dict with clear candidate
        prepared = {"AAA": _make_df()}

        # Act (OFF): Option-B disabled explicitly
        result_baseline = generate_candidates_system3(
            prepared,
            latest_only=True,
            top_n=5,
            include_diagnostics=False,
            use_option_b_utils=False,
        )
        by_date_baseline, df_baseline = result_baseline[:2]

        # Act (ON): Option-B enabled explicitly
        result_option_b = generate_candidates_system3(
            prepared,
            latest_only=True,
            top_n=5,
            include_diagnostics=False,
            use_option_b_utils=True,
        )
        by_date_option_b, df_option_b = result_option_b[:2]

        # Assert: candidates equivalence
        assert df_baseline is not None
        assert df_option_b is not None
        assert len(df_baseline) == len(df_option_b)

        # by-date structure should match
        assert set(by_date_baseline.keys()) == set(by_date_option_b.keys())

    def test_system3_diagnostics_consistency(self):
        """Test diagnostics consistency across ranked paths."""
        prepared = {"BBB": _make_df(rows=10, last_drop3d=0.15)}

        result_base = generate_candidates_system3(
            prepared,
            latest_only=True,
            top_n=5,
            include_diagnostics=True,
        )
        if isinstance(result_base, tuple) and len(result_base) == 3:
            by_date, df, diag = result_base
        else:
            by_date, df = result_base  # type: ignore[misc]
            diag = {}

        # Assert: diagnostics shape
        assert "ranked_top_n_count" in diag
        assert "ranking_source" in diag
        assert "setup_predicate_count" in diag

        # ranked >= 0, ranking_source should be set
        assert diag["ranked_top_n_count"] >= 0
        assert diag["ranking_source"] in ("latest_only", "full_scan")

        # if ranked > 0, df should match count
        if diag["ranked_top_n_count"] > 0:
            assert df is not None
            assert len(df) == diag["ranked_top_n_count"]

    def test_system3_zero_candidates(self):
        """Test handling of zero candidates (no setup pass)."""
        # Create data that fails setup (drop3d too low)
        df_no_setup = pd.DataFrame(
            {
                "Close": [10.0] * 5,
                "dollarvolume20": [30_000_000.0] * 5,
                "atr_ratio": [0.06] * 5,
                "drop3d": [0.05] * 5,  # < 0.125 threshold
                "atr10": [1.0] * 5,
                "filter": [True] * 5,
                "setup": [False] * 5,
            },
            index=pd.date_range("2025-10-20", periods=5, freq="B"),
        )
        prepared = {"CCC": df_no_setup}

        result_zero = generate_candidates_system3(
            prepared,
            latest_only=True,
            top_n=5,
            include_diagnostics=True,
        )
        if isinstance(result_zero, tuple) and len(result_zero) == 3:
            by_date, df, diag = result_zero
        else:
            by_date, df = result_zero  # type: ignore[misc]
            diag = {}

        # Assert: zero candidates handled gracefully
        assert diag["ranked_top_n_count"] == 0
        assert diag["ranking_source"] is not None
        assert df is None or len(df) == 0
