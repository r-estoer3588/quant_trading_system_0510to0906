"""
Minimal diagnostics tests for System6 to verify diagnostics contract.
Validates that generate_candidates_system6 returns required diagnostic keys.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from common.testing import set_test_determinism

generate_candidates_system6: Any = None
try:
    from core.system6 import generate_candidates_system6 as _gc6

    generate_candidates_system6 = _gc6
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


class TestSystem6DiagnosticsMinimal:
    """Minimal tests validating System6 diagnostics contract"""

    def setup_method(self):
        set_test_determinism()
        if not IMPORTS_AVAILABLE:
            pytest.skip("core.system6 imports not available")

    def test_diagnostics_keys_present_latest_only(self):
        """Verify diagnostics keys are present in latest_only mode"""
        # System6 setup predicate: (return_6d > 0.20) & uptwodays
        # Also needs: Close>=5, DV50>=10M, HV50 in bounds
        prepared_dict = {
            "TEST": pd.DataFrame(
                {
                    "Close": [100.0, 110.0, 120.0],
                    "return_6d": [0.25, 0.28, 0.30],  # Ranking key (descending)
                    "atr10": [2.5, 2.7, 2.9],
                    "dollarvolume50": [15_000_000, 16_000_000, 17_000_000],
                    "hv50": [15.0, 18.0, 20.0],
                    "uptwodays": [True, True, True],
                    "setup": [True, True, True],
                },
                index=pd.date_range("2023-01-01", periods=3),
            )
        }

        _candidates, _df, diagnostics = generate_candidates_system6(
            prepared_dict, top_n=5, latest_only=True, include_diagnostics=True
        )

        # Assert required diagnostic keys
        assert "ranking_source" in diagnostics
        assert "setup_predicate_count" in diagnostics
        assert "ranked_top_n_count" in diagnostics

        # Assert ranking_source is set
        assert diagnostics["ranking_source"] in ["latest_only", "full_scan", None]

        # Assert counts are valid
        assert isinstance(diagnostics["setup_predicate_count"], int)
        assert diagnostics["setup_predicate_count"] >= 0
        assert isinstance(diagnostics["ranked_top_n_count"], int)
        assert diagnostics["ranked_top_n_count"] >= 0

    def test_diagnostics_keys_present_full_scan(self):
        """Verify diagnostics keys are present in full_scan mode"""
        prepared_dict = {
            "TEST": pd.DataFrame(
                {
                    "Close": [100.0, 110.0, 120.0],
                    "return_6d": [0.25, 0.28, 0.30],
                    "atr10": [2.5, 2.7, 2.9],
                    "dollarvolume50": [15_000_000, 16_000_000, 17_000_000],
                    "hv50": [15.0, 18.0, 20.0],
                    "uptwodays": [True, True, True],
                    "setup": [True, True, True],
                },
                index=pd.date_range("2023-01-01", periods=3),
            )
        }

        _candidates, _df, diagnostics = generate_candidates_system6(
            prepared_dict, top_n=5, latest_only=False, include_diagnostics=True
        )

        # Assert required diagnostic keys
        assert "ranking_source" in diagnostics
        assert "setup_predicate_count" in diagnostics
        assert "ranked_top_n_count" in diagnostics

    def test_diagnostics_empty_data(self):
        """Verify diagnostics are returned even with empty data"""
        _candidates, _df, diagnostics = generate_candidates_system6({}, top_n=5, include_diagnostics=True)

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
                    "Close": [100.0, 110.0, 120.0],
                    "return_6d": [0.10, 0.12, 0.15],  # Below 0.20 threshold
                    "atr10": [2.5, 2.7, 2.9],
                    "dollarvolume50": [15_000_000, 16_000_000, 17_000_000],
                    "hv50": [15.0, 18.0, 20.0],
                    "uptwodays": [False, False, False],
                    "setup": [False, False, False],
                },
                index=pd.date_range("2023-01-01", periods=3),
            )
        }

        _candidates, _df, diagnostics = generate_candidates_system6(prepared_dict, top_n=5, include_diagnostics=True)

        # Diagnostics should still be present
        assert "ranking_source" in diagnostics
        assert "setup_predicate_count" in diagnostics
        assert "ranked_top_n_count" in diagnostics

        # No candidates expected
        assert diagnostics["ranked_top_n_count"] == 0
