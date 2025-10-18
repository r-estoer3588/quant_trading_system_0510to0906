"""
Minimal diagnostics tests for System2 to verify diagnostics contract.
Validates that generate_candidates_system2 returns required diagnostic keys.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from common.testing import set_test_determinism

generate_candidates_system2: Any = None
try:
    from core.system2 import generate_candidates_system2 as _gc2

    generate_candidates_system2 = _gc2
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


class TestSystem2DiagnosticsMinimal:
    """Minimal tests validating System2 diagnostics contract"""

    def setup_method(self):
        set_test_determinism()
        if not IMPORTS_AVAILABLE:
            pytest.skip("core.system2 imports not available")

    def test_diagnostics_keys_present_latest_only(self):
        """Verify diagnostics keys are present in latest_only mode"""
        # System2 setup predicate: Close>=5, DV20>25M, atr_ratio>0.03, rsi3>90, twodayup
        prepared_dict = {
            "TEST": pd.DataFrame(
                {
                    "Close": [10.0, 11.0, 12.0],
                    "rsi3": [85.0, 92.0, 95.0],
                    "adx7": [55.0, 60.0, 65.0],  # Ranking key (descending)
                    "dollarvolume20": [30_000_000, 32_000_000, 35_000_000],
                    "atr_ratio": [0.04, 0.045, 0.05],
                    "twodayup": [False, True, True],
                    "setup": [False, True, True],
                },
                index=pd.date_range("2023-01-01", periods=3),
            )
        }

        res = generate_candidates_system2(prepared_dict, top_n=5, latest_only=True, include_diagnostics=True)
        if isinstance(res, tuple) and len(res) == 3:
            _candidates, _df, diagnostics = res
        else:
            _candidates, _df = res
            diagnostics = {}

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
                    "Close": [10.0, 11.0, 12.0],
                    "rsi3": [85.0, 92.0, 95.0],
                    "adx7": [55.0, 60.0, 65.0],
                    "dollarvolume20": [30_000_000, 32_000_000, 35_000_000],
                    "atr_ratio": [0.04, 0.045, 0.05],
                    "twodayup": [False, True, True],
                    "setup": [False, True, True],
                },
                index=pd.date_range("2023-01-01", periods=3),
            )
        }

        res = generate_candidates_system2(prepared_dict, top_n=5, latest_only=False, include_diagnostics=True)
        if isinstance(res, tuple) and len(res) == 3:
            _candidates, _df, diagnostics = res
        else:
            _candidates, _df = res
            diagnostics = {}

        # Assert required diagnostic keys
        assert "ranking_source" in diagnostics
        assert "setup_predicate_count" in diagnostics
        assert "ranked_top_n_count" in diagnostics

    def test_diagnostics_empty_data(self):
        """Verify diagnostics are returned even with empty data"""
        res = generate_candidates_system2({}, top_n=5, include_diagnostics=True)
        if isinstance(res, tuple) and len(res) == 3:
            _candidates, _df, diagnostics = res
        else:
            _candidates, _df = res
            diagnostics = {}

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
                    "Close": [10.0, 11.0, 12.0],
                    "rsi3": [70.0, 75.0, 80.0],  # Below 90 threshold
                    "adx7": [55.0, 60.0, 65.0],
                    "dollarvolume20": [30_000_000, 32_000_000, 35_000_000],
                    "atr_ratio": [0.04, 0.045, 0.05],
                    "twodayup": [False, False, False],
                    "setup": [False, False, False],
                },
                index=pd.date_range("2023-01-01", periods=3),
            )
        }

        res = generate_candidates_system2(prepared_dict, top_n=5, include_diagnostics=True)
        if isinstance(res, tuple) and len(res) == 3:
            _candidates, _df, diagnostics = res
        else:
            _candidates, _df = res
            diagnostics = {}

        # Diagnostics should still be present
        assert "ranking_source" in diagnostics
        assert "setup_predicate_count" in diagnostics
        assert "ranked_top_n_count" in diagnostics

        # No candidates expected
        assert diagnostics["ranked_top_n_count"] == 0
