"""Enhanced unit tests for final_allocation module."""

import json
from unittest.mock import patch

import pandas as pd
import pytest

from core.final_allocation import (
    AllocationConstants,
    AllocationSummary,
    _get_position_attr,
    _normalize_allocations,
    _safe_positive_float,
    count_active_positions_by_system,
    finalize_allocation,
    load_symbol_system_map,
)


class TestAllocationConstants:
    """Test the AllocationConstants class."""

    def test_constants_exist(self):
        """Test that all expected constants are defined."""
        assert hasattr(AllocationConstants, "DEFAULT_RISK_PCT")
        assert hasattr(AllocationConstants, "DEFAULT_MAX_PCT")
        assert hasattr(AllocationConstants, "DEFAULT_MAX_POSITIONS")
        assert hasattr(AllocationConstants, "DEFAULT_CAPITAL")
        assert hasattr(AllocationConstants, "DEFAULT_LONG_RATIO")
        assert hasattr(AllocationConstants, "MAX_ITERATIONS")

    def test_constant_values(self):
        """Test that constants have expected values."""
        assert AllocationConstants.DEFAULT_RISK_PCT == 0.02
        assert AllocationConstants.DEFAULT_MAX_PCT == 0.10
        assert AllocationConstants.DEFAULT_MAX_POSITIONS == 10
        assert AllocationConstants.DEFAULT_CAPITAL == 100_000.0
        assert AllocationConstants.DEFAULT_LONG_RATIO == 0.5
        assert AllocationConstants.MAX_ITERATIONS == 10_000


class TestSafePositiveFloat:
    """Comprehensive tests for _safe_positive_float function."""

    @pytest.mark.parametrize(
        "value,allow_zero,expected",
        [
            # 正常系
            ("123.45", False, 123.45),
            (123.45, False, 123.45),
            (0, True, 0.0),
            ("0", True, 0.0),
            # 境界値
            (1e-10, False, 1e-10),
            (1e308, False, 1e308),
            # 異常系
            (None, False, None),
            ("", False, None),
            (-1, False, None),
            (0, False, None),
            (float("inf"), False, None),
            (float("nan"), False, None),
            ("abc", False, None),
            ({}, False, None),
            ([], False, None),
        ],
    )
    def test_safe_positive_float_variations(self, value, allow_zero, expected):
        """Test various input types and edge cases."""
        result = _safe_positive_float(value, allow_zero=allow_zero)
        if expected is None:
            assert result is None
        else:
            assert result == pytest.approx(expected)

    def test_safe_positive_float_logging(self, caplog):
        """Test that appropriate debug messages are logged."""
        import logging

        caplog.set_level(logging.DEBUG)

        # Test empty value logging
        _safe_positive_float(None)
        assert "Empty value provided" in caplog.text

        # Test conversion error logging
        _safe_positive_float("invalid")
        assert "Failed to convert" in caplog.text

        # Test negative value logging
        _safe_positive_float(-5)
        assert "Negative value rejected" in caplog.text


class TestGetPositionAttr:
    """Tests for _get_position_attr function."""

    def test_object_with_attributes(self):
        """Test with object that has attributes."""

        class Position:
            def __init__(self):
                self.symbol = "AAPL"
                self.qty = 100

        pos = Position()
        assert _get_position_attr(pos, "symbol") == "AAPL"
        assert _get_position_attr(pos, "qty") == 100
        assert _get_position_attr(pos, "missing") is None

    def test_dictionary_object(self):
        """Test with dictionary-like object."""
        pos = {"symbol": "GOOGL", "qty": -50}
        assert _get_position_attr(pos, "symbol") == "GOOGL"
        assert _get_position_attr(pos, "qty") == -50
        assert _get_position_attr(pos, "missing") is None

    def test_invalid_object(self):
        """Test with invalid object types."""
        assert _get_position_attr("string", "attr") is None
        assert _get_position_attr(123, "attr") is None
        assert _get_position_attr(None, "attr") is None


class TestNormalizeAllocations:
    """Tests for _normalize_allocations function."""

    def test_valid_weights(self):
        """Test normalization with valid weights."""
        weights = {"system1": 0.6, "system2": 0.4}
        defaults = {"system1": 0.5, "system2": 0.5}

        result = _normalize_allocations(weights, defaults)

        assert result == {"system1": 0.6, "system2": 0.4}
        assert abs(sum(result.values()) - 1.0) < 1e-10

    def test_unnormalized_weights(self):
        """Test normalization with weights that don't sum to 1."""
        weights = {"system1": 2.0, "system2": 3.0}
        defaults = {"system1": 0.5, "system2": 0.5}

        result = _normalize_allocations(weights, defaults)

        assert result == {"system1": 0.4, "system2": 0.6}
        assert abs(sum(result.values()) - 1.0) < 1e-10

    def test_invalid_weights_fallback_to_defaults(self):
        """Test fallback to defaults when weights are invalid."""
        weights = {"system1": -1.0, "system2": 0.0}  # Invalid weights
        defaults = {"system1": 0.3, "system2": 0.7}

        result = _normalize_allocations(weights, defaults)

        assert result == {"system1": 0.3, "system2": 0.7}

    def test_empty_weights_use_defaults(self):
        """Test that empty weights use defaults."""
        weights = None
        defaults = {"system1": 0.4, "system2": 0.6}

        result = _normalize_allocations(weights, defaults)

        assert result == {"system1": 0.4, "system2": 0.6}

    def test_equal_weights_fallback(self):
        """Test equal weights when all else fails."""
        weights = None
        defaults = {"system1": -1.0, "system2": 0.0}  # Invalid defaults

        result = _normalize_allocations(weights, defaults)

        assert result == {"system1": 0.5, "system2": 0.5}


class TestCountActivePositions:
    """Tests for count_active_positions_by_system function."""

    def test_count_with_object_positions(self):
        """Test counting with position objects."""

        class Position:
            def __init__(self, symbol, qty, side="long"):
                self.symbol = symbol
                self.qty = qty
                self.side = side

        positions = [
            Position("AAPL", 100),
            Position("GOOGL", -50, "short"),
            Position("SPY", -200, "short"),  # Special system7 case
            Position("MSFT", 0),  # Zero qty, should be ignored
        ]

        symbol_map = {
            "AAPL": "system1",
            "GOOGL": "system2",
            "MSFT": "system3",
        }

        result = count_active_positions_by_system(positions, symbol_map)

        assert result["system1"] == 1  # AAPL
        assert result["system2"] == 1  # GOOGL
        assert result["system7"] == 1  # SPY (special case)
        assert "system3" not in result  # MSFT has 0 qty

    def test_count_with_dict_positions(self):
        """Test counting with dictionary positions."""
        positions = [
            {"symbol": "TSLA", "qty": 150, "side": "long"},
            {"symbol": "NVDA", "qty": -75, "side": "short"},
            {"symbol": "UNKNOWN", "qty": 100, "side": "long"},  # Not in map
        ]

        symbol_map = {
            "TSLA": "system4",
            "NVDA": "system6",
        }

        result = count_active_positions_by_system(positions, symbol_map)

        assert result["system4"] == 1  # TSLA
        assert result["system6"] == 1  # NVDA
        assert "unknown" not in result  # Unknown symbol ignored

    def test_empty_inputs(self):
        """Test with empty/None inputs."""
        result = count_active_positions_by_system(None, None)
        assert result == {}

        result = count_active_positions_by_system([], {})
        assert result == {}

    def test_invalid_positions_ignored(self):
        """Test that invalid positions are gracefully ignored."""
        positions = [
            {"symbol": "AAPL", "qty": "invalid"},  # Invalid qty
            {"qty": 100},  # Missing symbol
            "invalid_position",  # Invalid object type
            None,  # None position
        ]

        symbol_map = {"AAPL": "system1"}

        result = count_active_positions_by_system(positions, symbol_map)

        assert result == {}  # All positions should be ignored


class TestLoadSymbolSystemMap:
    """Tests for load_symbol_system_map function."""

    def test_load_valid_file(self, tmp_path):
        """Test loading a valid JSON file."""
        test_file = tmp_path / "test_map.json"
        test_data = {"AAPL": "system1", "GOOGL": "system2"}
        test_file.write_text(json.dumps(test_data))

        result = load_symbol_system_map(test_file)

        assert result == {"aapl": "system1", "googl": "system2"}

    def test_missing_file(self, tmp_path):
        """Test with missing file."""
        missing_file = tmp_path / "missing.json"
        result = load_symbol_system_map(missing_file)
        assert result == {}

    def test_invalid_json(self, tmp_path):
        """Test with invalid JSON."""
        test_file = tmp_path / "invalid.json"
        test_file.write_text("invalid json content")

        result = load_symbol_system_map(test_file)
        assert result == {}

    def test_non_dict_json(self, tmp_path):
        """Test with JSON that's not a dictionary."""
        test_file = tmp_path / "non_dict.json"
        test_file.write_text('["not", "a", "dict"]')

        result = load_symbol_system_map(test_file)
        assert result == {}

    def test_case_normalization(self, tmp_path):
        """Test that keys and values are normalized to lowercase."""
        test_file = tmp_path / "case_test.json"
        test_data = {"AAPL": "SYSTEM1", "googl": "System2"}
        test_file.write_text(json.dumps(test_data))

        result = load_symbol_system_map(test_file)

        assert result == {"aapl": "system1", "googl": "system2"}


class TestAllocationSummary:
    """Tests for AllocationSummary dataclass."""

    def test_basic_creation(self):
        """Test basic AllocationSummary creation."""
        summary = AllocationSummary(
            mode="slot",
            long_allocations={"system1": 1.0},
            short_allocations={"system2": 1.0},
            active_positions={"system1": 2},
            available_slots={"system1": 8, "system2": 10},
            final_counts={"system1": 5, "system2": 3},
        )

        assert summary.mode == "slot"
        assert summary.long_allocations == {"system1": 1.0}
        assert summary.short_allocations == {"system2": 1.0}

    def test_post_init_validation(self, caplog):
        """Test post-initialization validation warnings."""
        import logging

        caplog.set_level(logging.WARNING)

        # Test invalid mode warning
        AllocationSummary(
            mode="invalid_mode",
            long_allocations={},
            short_allocations={},
            active_positions={},
            available_slots={},
            final_counts={},
        )
        assert "Unknown allocation mode" in caplog.text

        # Clear log for next test
        caplog.clear()

        # Test missing slot data in slot mode
        AllocationSummary(
            mode="slot",
            long_allocations={},
            short_allocations={},
            active_positions={},
            available_slots={},
            final_counts={},
            slot_allocation=None,  # Missing slot data
        )
        assert "missing slot_allocation data" in caplog.text


class TestFinalizeAllocationBasic:
    """Basic integration tests for finalize_allocation function."""

    def test_empty_input(self):
        """Test with empty input."""
        per_system = {}

        result_df, summary = finalize_allocation(per_system)

        assert result_df.empty
        assert isinstance(summary, AllocationSummary)

    def test_slot_mode_basic(self):
        """Test basic slot mode allocation."""
        per_system = {
            "system1": pd.DataFrame(
                {
                    "symbol": ["AAPL", "MSFT"],
                    "entry_price": [150.0, 300.0],
                    "score": [0.8, 0.9],
                }
            ),
            "system2": pd.DataFrame(
                {
                    "symbol": ["GOOGL", "TSLA"],
                    "entry_price": [2800.0, 800.0],
                    "score": [0.7, 0.85],
                }
            ),
        }

        result_df, summary = finalize_allocation(
            per_system,
            slots_long=2,
            slots_short=2,
        )

        assert not result_df.empty
        assert summary.mode == "slot"
        assert len(result_df) <= 4  # Max 4 positions total

    @patch("core.final_allocation._load_allocations_from_settings")
    def test_settings_fallback(self, mock_load_settings):
        """Test fallback to settings when allocations not provided."""
        mock_load_settings.return_value = (
            {"system1": 0.5, "system3": 0.5},
            {"system2": 0.5, "system6": 0.5},
        )

        per_system: dict[str, pd.DataFrame] = {
            "system1": pd.DataFrame(
                {
                    "symbol": ["AAPL"],
                    "entry_price": [150.0],
                    "score": [0.8],
                }
            ),
        }

        result_df, summary = finalize_allocation(per_system)

        mock_load_settings.assert_called_once()
        assert summary.long_allocations == {"system1": 0.5, "system3": 0.5}


# Performance and edge case tests would go here...
# Benchmark tests would also be separate modules

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
