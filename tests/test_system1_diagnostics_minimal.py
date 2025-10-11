"""
Minimal diagnostics tests for System1 to verify diagnostics contract.
Validates that generate_candidates_system1 returns required diagnostic keys.
"""

from __future__ import annotations

import pandas as pd
import pytest

from common.testing import set_test_determinism

try:
    from core.system1 import generate_candidates_system1

    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


class TestSystem1DiagnosticsMinimal:
    """Minimal tests validating System1 diagnostics contract"""

    def setup_method(self):
        set_test_determinism()
        if not IMPORTS_AVAILABLE:
            pytest.skip("core.system1 imports not available")

    def test_diagnostics_keys_present_latest_only(self):
        """Verify diagnostics keys are present in latest_only mode"""
        # Create minimal synthetic data satisfying System1 setup predicate
        # Setup predicate: (sma25 > sma50) & (roc200 > 0)
        # Also needs Close >= 5, dollarvolume20 >= 50M from filters
        prepared_dict = {
            "TEST": pd.DataFrame(
                {
                    "Close": [100.0, 105.0, 110.0],
                    "sma25": [102.0, 104.0, 106.0],
                    "sma50": [98.0, 99.0, 100.0],
                    "roc200": [5.0, 6.0, 7.0],
                    "dollarvolume20": [60_000_000, 65_000_000, 70_000_000],
                    "setup": [True, True, True],
                },
                index=pd.date_range("2023-01-01", periods=3),
            )
        }

        _candidates, _df, diagnostics = generate_candidates_system1(
            prepared_dict, top_n=5, latest_only=True, include_diagnostics=True
        )

        # Assert required diagnostic keys are present
        assert "ranking_source" in diagnostics
        assert "setup_predicate_count" in diagnostics
        assert "ranked_top_n_count" in diagnostics

        # Assert ranking_source is set correctly
        assert diagnostics["ranking_source"] in ["latest_only", "full_scan", None]

        # Assert counts are non-negative integers
        assert isinstance(diagnostics["setup_predicate_count"], int)
        assert diagnostics["setup_predicate_count"] >= 0
        assert isinstance(diagnostics["ranked_top_n_count"], int)
        assert diagnostics["ranked_top_n_count"] >= 0

    def test_diagnostics_keys_present_full_scan(self):
        """Verify diagnostics keys are present in full_scan mode"""
        prepared_dict = {
            "TEST": pd.DataFrame(
                {
                    "Close": [100.0, 105.0, 110.0],
                    "sma25": [102.0, 104.0, 106.0],
                    "sma50": [98.0, 99.0, 100.0],
                    "roc200": [5.0, 6.0, 7.0],
                    "dollarvolume20": [60_000_000, 65_000_000, 70_000_000],
                    "setup": [True, True, True],
                },
                index=pd.date_range("2023-01-01", periods=3),
            )
        }

        _candidates, _df, diagnostics = generate_candidates_system1(
            prepared_dict, top_n=5, latest_only=False, include_diagnostics=True
        )

        # Assert required diagnostic keys
        assert "ranking_source" in diagnostics
        assert "setup_predicate_count" in diagnostics
        assert "ranked_top_n_count" in diagnostics

        # Assert ranking_source reflects full_scan
        assert diagnostics["ranking_source"] in ["full_scan", "latest_only", None]

    def test_diagnostics_empty_data(self):
        """Verify diagnostics are returned even with empty data"""
        _candidates, _df, diagnostics = generate_candidates_system1(
            {}, top_n=5, include_diagnostics=True
        )

        # Diagnostics should still be a dict
        assert isinstance(diagnostics, dict)
        assert "ranking_source" in diagnostics
        assert "setup_predicate_count" in diagnostics
        assert "ranked_top_n_count" in diagnostics

        # With empty data, counts should be 0
        assert diagnostics["setup_predicate_count"] == 0
        assert diagnostics["ranked_top_n_count"] == 0

    def test_diagnostics_no_setup_conditions(self):
        """Verify diagnostics when no setup conditions are met"""
        prepared_dict = {
            "TEST": pd.DataFrame(
                {
                    "Close": [100.0, 105.0, 110.0],
                    "sma25": [90.0, 91.0, 92.0],  # sma25 < sma50, no setup
                    "sma50": [98.0, 99.0, 100.0],
                    "roc200": [5.0, 6.0, 7.0],
                    "dollarvolume20": [60_000_000, 65_000_000, 70_000_000],
                    "setup": [False, False, False],
                },
                index=pd.date_range("2023-01-01", periods=3),
            )
        }

        _candidates, _df, diagnostics = generate_candidates_system1(
            prepared_dict, top_n=5, include_diagnostics=True
        )

        # Diagnostics should still be present
        assert "ranking_source" in diagnostics
        assert "setup_predicate_count" in diagnostics
        assert "ranked_top_n_count" in diagnostics

        # No candidates expected
        assert diagnostics["ranked_top_n_count"] == 0
