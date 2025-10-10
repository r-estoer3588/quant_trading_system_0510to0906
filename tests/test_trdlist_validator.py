"""
Tests for common.trdlist_validator module
"""

from __future__ import annotations

import pandas as pd

from common.trdlist_validator import (
    ValidationResult,
    build_validation_report,
    summarize_trd_frame,
    validate_trd_frame,
)


class TestValidateTrdFrame:
    """Test validate_trd_frame function"""

    def test_validate_none_frame(self):
        """Test validation with None frame"""
        result = validate_trd_frame(None, name="test")
        assert len(result.errors) == 1
        assert "None" in result.errors[0]

    def test_validate_empty_frame(self):
        """Test validation with empty frame"""
        df = pd.DataFrame()
        result = validate_trd_frame(df, name="test")
        assert len(result.warnings) == 1
        assert "empty" in result.warnings[0]

    def test_validate_missing_required_columns(self):
        """Test validation with missing required columns"""
        df = pd.DataFrame({"symbol": ["AAPL"], "system": ["system1"]})
        result = validate_trd_frame(df, name="test")
        assert len(result.errors) >= 1
        assert any("missing required columns" in e for e in result.errors)

    def test_validate_duplicate_symbol_system_pairs(self):
        """Test detection of duplicate symbol/system pairs"""
        df = pd.DataFrame(
            {
                "symbol": ["AAPL", "AAPL"],
                "system": ["system1", "system1"],
                "entry_date": pd.to_datetime(["2023-01-01", "2023-01-01"]),
                "entry_price": [100.0, 100.0],
                "stop_price": [95.0, 95.0],
            }
        )
        result = validate_trd_frame(df, name="test")
        assert len(result.errors) >= 1
        assert any("duplicate symbol/system pairs" in e for e in result.errors)

    def test_validate_invalid_entry_price(self):
        """Test detection of invalid entry prices"""
        df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "system": ["system1"],
                "entry_date": pd.to_datetime(["2023-01-01"]),
                "entry_price": [0.0],  # Invalid: zero
                "stop_price": [95.0],
            }
        )
        result = validate_trd_frame(df, name="test")
        assert len(result.errors) >= 1
        assert any("invalid entry_price" in e for e in result.errors)

    def test_validate_invalid_stop_price(self):
        """Test detection of invalid stop prices"""
        df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "system": ["system1"],
                "entry_date": pd.to_datetime(["2023-01-01"]),
                "entry_price": [100.0],
                "stop_price": [-5.0],  # Invalid: negative
            }
        )
        result = validate_trd_frame(df, name="test")
        assert len(result.errors) >= 1
        assert any("invalid stop_price" in e for e in result.errors)

    def test_validate_invalid_shares(self):
        """Test detection of invalid shares"""
        df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "system": ["system1"],
                "entry_date": pd.to_datetime(["2023-01-01"]),
                "entry_price": [100.0],
                "stop_price": [95.0],
                "shares": [-10],  # Invalid: negative
            }
        )
        result = validate_trd_frame(df, name="test")
        assert len(result.errors) >= 1
        assert any("invalid shares" in e for e in result.errors)

    def test_validate_valid_frame(self):
        """Test validation with valid frame"""
        df = pd.DataFrame(
            {
                "symbol": ["AAPL", "MSFT"],
                "system": ["system1", "system2"],
                "entry_date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
                "entry_price": [100.0, 200.0],
                "stop_price": [95.0, 190.0],
                "side": ["long", "short"],
                "shares": [10, 5],
            }
        )
        result = validate_trd_frame(df, name="test")
        # Should have no critical errors
        assert len(result.errors) == 0

    def test_validate_unknown_side(self):
        """Test warning for unknown side values"""
        df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "system": ["system1"],
                "entry_date": pd.to_datetime(["2023-01-01"]),
                "entry_price": [100.0],
                "stop_price": [95.0],
                "side": ["unknown"],  # Invalid side
            }
        )
        result = validate_trd_frame(df, name="test")
        assert len(result.warnings) >= 1
        assert any("unknown side" in w for w in result.warnings)

    def test_validate_duplicate_symbols_warning(self):
        """Test warning for duplicate symbols across systems"""
        df = pd.DataFrame(
            {
                "symbol": ["AAPL", "AAPL"],
                "system": ["system1", "system2"],  # Different systems
                "entry_date": pd.to_datetime(["2023-01-01", "2023-01-01"]),
                "entry_price": [100.0, 100.0],
                "stop_price": [95.0, 95.0],
            }
        )
        result = validate_trd_frame(df, name="test")
        # Should warn about duplicate symbols
        assert len(result.warnings) >= 1
        assert any("duplicate symbols" in w for w in result.warnings)


class TestSummarizeTrdFrame:
    """Test summarize_trd_frame function"""

    def test_summarize_none_frame(self):
        """Test summary with None frame"""
        summary = summarize_trd_frame(None)
        assert summary["rows"] == 0
        assert summary["unique_symbols"] == 0

    def test_summarize_empty_frame(self):
        """Test summary with empty frame"""
        df = pd.DataFrame()
        summary = summarize_trd_frame(df)
        assert summary["rows"] == 0

    def test_summarize_basic_frame(self):
        """Test summary with basic frame"""
        df = pd.DataFrame(
            {
                "symbol": ["AAPL", "MSFT", "AAPL"],
                "system": ["system1", "system2", "system3"],
                "side": ["long", "short", "long"],
            }
        )
        summary = summarize_trd_frame(df)
        assert summary["rows"] == 3
        assert summary["unique_symbols"] == 2
        assert summary["side_counts"]["long"] == 2
        assert summary["side_counts"]["short"] == 1
        assert summary["system_counts"]["system1"] == 1


class TestBuildValidationReport:
    """Test build_validation_report function"""

    def test_report_with_none_inputs(self):
        """Test report generation with None inputs"""
        report = build_validation_report(None, None)
        assert "final" in report
        assert "systems" in report
        assert "summary" in report
        assert report["summary"]["warnings"] >= 1

    def test_report_with_valid_final_df(self):
        """Test report generation with valid final DataFrame"""
        df = pd.DataFrame(
            {
                "symbol": ["AAPL", "MSFT"],
                "system": ["system1", "system2"],
                "entry_date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
                "entry_price": [100.0, 200.0],
                "stop_price": [95.0, 190.0],
            }
        )
        report = build_validation_report(df, None)
        assert "final" in report
        assert report["summary"]["errors"] == 0
        assert "final_stats" in report
        assert report["final_stats"]["rows"] == 2

    def test_report_with_per_system(self):
        """Test report generation with per-system DataFrames"""
        final_df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "system": ["system1"],
                "entry_date": pd.to_datetime(["2023-01-01"]),
                "entry_price": [100.0],
                "stop_price": [95.0],
            }
        )
        per_system = {
            "system1": pd.DataFrame(
                {
                    "symbol": ["AAPL"],
                    "system": ["system1"],
                    "entry_date": pd.to_datetime(["2023-01-01"]),
                    "entry_price": [100.0],
                    "stop_price": [95.0],
                }
            ),
            "system2": pd.DataFrame(
                {
                    "symbol": ["MSFT"],
                    "system": ["system2"],
                    "entry_date": pd.to_datetime(["2023-01-02"]),
                    "entry_price": [200.0],
                    "stop_price": [190.0],
                }
            ),
        }
        report = build_validation_report(final_df, per_system)
        assert "systems" in report
        assert "system1" in report["systems"]
        assert "system2" in report["systems"]
        assert "system_stats" in report
        assert report["system_stats"]["system1"]["rows"] == 1

    def test_report_with_errors(self):
        """Test report generation with validation errors"""
        df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "system": ["system1"],
                "entry_date": pd.to_datetime(["2023-01-01"]),
                "entry_price": [0.0],  # Invalid
                "stop_price": [95.0],
            }
        )
        report = build_validation_report(df, None)
        assert report["summary"]["errors"] >= 1


class TestValidationResult:
    """Test ValidationResult dataclass"""

    def test_to_dict(self):
        """Test to_dict method"""
        result = ValidationResult(errors=["error1", "error2"], warnings=["warning1"])
        d = result.to_dict()
        assert d["errors"] == ["error1", "error2"]
        assert d["warnings"] == ["warning1"]
