"""
Unit tests for validate_predicate_equivalence function.
Validates Two-Phase consistency check (Filter vs Setup alignment).
"""

from __future__ import annotations

import os

import pandas as pd

from common.testing import set_test_determinism


class TestValidatePredicateEquivalence:
    """Test validate_predicate_equivalence for Two-Phase consistency"""

    def setup_method(self):
        set_test_determinism()
        # Enable validation for all tests in this class
        os.environ["VALIDATE_SETUP_PREDICATE"] = "1"

    def teardown_method(self):
        # Clean up environment variable
        os.environ.pop("VALIDATE_SETUP_PREDICATE", None)

    def test_system1_predicate_match(self):
        """Test System1 predicate matches setup column"""
        from common.system_setup_predicates import validate_predicate_equivalence

        # Create data where setup column matches predicate
        # System1 setup: (sma25 > sma50) & (roc200 > 0)
        prepared_dict = {
            "TEST1": pd.DataFrame(
                {
                    "Close": [100.0, 105.0, 110.0],
                    "sma25": [102.0, 104.0, 106.0],
                    "sma50": [98.0, 99.0, 100.0],
                    "roc200": [5.0, 6.0, 7.0],
                    "dollarvolume20": [60_000_000, 65_000_000, 70_000_000],
                    "setup": [True, True, True],  # All should pass
                },
                index=pd.date_range("2023-01-01", periods=3),
            ),
            "TEST2": pd.DataFrame(
                {
                    "Close": [100.0, 105.0, 110.0],
                    "sma25": [90.0, 91.0, 92.0],  # sma25 < sma50
                    "sma50": [98.0, 99.0, 100.0],
                    "roc200": [5.0, 6.0, 7.0],
                    "dollarvolume20": [60_000_000, 65_000_000, 70_000_000],
                    "setup": [False, False, False],  # All should fail
                },
                index=pd.date_range("2023-01-01", periods=3),
            ),
        }

        # Collect logs
        log_messages = []

        def log_fn(msg):
            log_messages.append(msg)

        # Should not raise any errors, and no mismatch logs
        validate_predicate_equivalence(prepared_dict, "System1", log_fn=log_fn)

        # No mismatch messages expected
        assert not any("mismatch" in msg for msg in log_messages)

    def test_system2_predicate_match(self):
        """Test System2 predicate matches setup column"""
        from common.system_setup_predicates import validate_predicate_equivalence

        # System2 setup: Close>=5, DV20>25M, atr_ratio>0.03, rsi3>90, twodayup
        prepared_dict = {
            "TEST": pd.DataFrame(
                {
                    "Close": [10.0, 11.0, 12.0],
                    "rsi3": [85.0, 92.0, 95.0],
                    "adx7": [55.0, 60.0, 65.0],
                    "dollarvolume20": [30_000_000, 32_000_000, 35_000_000],
                    "atr_ratio": [0.04, 0.045, 0.05],
                    "twodayup": [False, True, True],
                    "setup": [False, True, True],  # Match predicate
                },
                index=pd.date_range("2023-01-01", periods=3),
            )
        }

        log_messages = []

        def log_fn(msg):
            log_messages.append(msg)

        validate_predicate_equivalence(prepared_dict, "System2", log_fn=log_fn)

        # No mismatch messages expected
        assert not any("mismatch" in msg for msg in log_messages)

    def test_system6_predicate_match(self):
        """Test System6 predicate matches setup column"""
        from common.system_setup_predicates import validate_predicate_equivalence

        # System6 setup: (return_6d > 0.20) & uptwodays
        prepared_dict = {
            "TEST": pd.DataFrame(
                {
                    "Close": [100.0, 110.0, 120.0],
                    "return_6d": [0.25, 0.28, 0.30],
                    "atr10": [2.5, 2.7, 2.9],
                    "dollarvolume50": [15_000_000, 16_000_000, 17_000_000],
                    "hv50": [15.0, 18.0, 20.0],
                    "uptwodays": [True, True, True],
                    "UpTwoDays": [True, True, True],
                    "setup": [True, True, True],  # Match predicate
                },
                index=pd.date_range("2023-01-01", periods=3),
            )
        }

        log_messages = []

        def log_fn(msg):
            log_messages.append(msg)

        validate_predicate_equivalence(prepared_dict, "System6", log_fn=log_fn)

        # No mismatch messages expected
        assert not any("mismatch" in msg for msg in log_messages)

    def test_predicate_mismatch_detected(self):
        """Test that mismatches are detected and logged"""
        from common.system_setup_predicates import validate_predicate_equivalence

        # Intentional mismatch: setup=True but predicate should be False
        prepared_dict = {
            "MISMATCH": pd.DataFrame(
                {
                    "Close": [100.0],
                    "sma25": [90.0],  # sma25 < sma50 -> predicate False
                    "sma50": [98.0],
                    "roc200": [5.0],
                    "dollarvolume20": [60_000_000],
                    "setup": [True],  # But setup column says True (mismatch!)
                },
                index=pd.date_range("2023-01-01", periods=1),
            )
        }

        log_messages = []

        def log_fn(msg):
            log_messages.append(msg)

        validate_predicate_equivalence(prepared_dict, "System1", log_fn=log_fn)

        # Mismatch should be detected
        assert any("mismatch" in msg.lower() for msg in log_messages)
        assert any("MISMATCH" in msg for msg in log_messages)

    def test_validation_disabled(self):
        """Test that validation is skipped when env var is not set"""
        from common.system_setup_predicates import validate_predicate_equivalence

        # Temporarily disable validation
        os.environ.pop("VALIDATE_SETUP_PREDICATE", None)

        prepared_dict = {
            "TEST": pd.DataFrame(
                {
                    "Close": [100.0],
                    "sma25": [90.0],
                    "sma50": [98.0],
                    "roc200": [5.0],
                    "setup": [True],  # Mismatch, but validation disabled
                },
                index=pd.date_range("2023-01-01", periods=1),
            )
        }

        log_messages = []

        def log_fn(msg):
            log_messages.append(msg)

        # Should not log anything (validation disabled)
        validate_predicate_equivalence(prepared_dict, "System1", log_fn=log_fn)

        assert len(log_messages) == 0

    def test_empty_data(self):
        """Test validation with empty data"""
        from common.system_setup_predicates import validate_predicate_equivalence

        log_messages = []

        def log_fn(msg):
            log_messages.append(msg)

        # Empty dict
        validate_predicate_equivalence({}, "System1", log_fn=log_fn)
        assert len(log_messages) == 0

        # Dict with empty DataFrame
        validate_predicate_equivalence(
            {"TEST": pd.DataFrame()}, "System1", log_fn=log_fn
        )
        assert len(log_messages) == 0

    def test_missing_setup_column(self):
        """Test validation when setup column is missing"""
        from common.system_setup_predicates import validate_predicate_equivalence

        # DataFrame without setup column (should be silently skipped)
        prepared_dict = {
            "TEST": pd.DataFrame(
                {
                    "Close": [100.0],
                    "sma25": [102.0],
                    "sma50": [98.0],
                    "roc200": [5.0],
                    # No setup column
                },
                index=pd.date_range("2023-01-01", periods=1),
            )
        }

        log_messages = []

        def log_fn(msg):
            log_messages.append(msg)

        # Should not raise error, just skip validation
        validate_predicate_equivalence(prepared_dict, "System1", log_fn=log_fn)

        # No logs expected (setup column missing, skip validation)
        assert len(log_messages) == 0
