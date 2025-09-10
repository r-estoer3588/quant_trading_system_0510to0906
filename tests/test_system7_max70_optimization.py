"""
Test for System7 max_70 optimization to ensure redundant calculations are avoided.
"""

import pandas as pd

from core.system7 import prepare_data_vectorized_system7


def spy_data_without_max70():
    """Create test SPY data without max_70 column."""
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    df = pd.DataFrame(
        {
            "Open": [100] * 100,
            "High": [101] * 100,
            "Low": [99] * 100,
            "Close": [100] * 100,
            "Volume": [1_000_000] * 100,
        },
        index=dates,
    )
    return {"SPY": df}


def spy_data_with_max70():
    """Create test SPY data with pre-calculated max_70 column."""
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    df = pd.DataFrame(
        {
            "Open": [100] * 100,
            "High": [101] * 100,
            "Low": [99] * 100,
            "Close": [100] * 100,
            "Volume": [1_000_000] * 100,
            "max_70": [110] * 100,  # Pre-calculated value
        },
        index=dates,
    )
    return {"SPY": df}


def test_max70_calculated_when_missing():
    """Test that max_70 is calculated when not present in the data."""
    spy_data = spy_data_without_max70()
    result = prepare_data_vectorized_system7(spy_data)
    
    # Should have max_70 column
    assert "max_70" in result["SPY"].columns
    
    # Should be calculated correctly (rolling max of constant 100 should be 100)
    expected_max70 = spy_data["SPY"]["Close"].rolling(window=70).max()
    pd.testing.assert_series_equal(
        result["SPY"]["max_70"], expected_max70, check_names=False
    )


def test_max70_preserved_when_present():
    """Test that existing max_70 is preserved and not recalculated."""
    spy_data = spy_data_with_max70()
    original_max70 = spy_data["SPY"]["max_70"].copy()
    
    result = prepare_data_vectorized_system7(spy_data)
    
    # Should still have max_70 column
    assert "max_70" in result["SPY"].columns
    
    # Should preserve the original values (110), not recalculate to 100
    pd.testing.assert_series_equal(
        result["SPY"]["max_70"], original_max70, check_names=False
    )
    
    # Verify it wasn't recalculated (would be 100 if recalculated)
    assert all(result["SPY"]["max_70"] == 110)


def test_optimization_prevents_redundant_calculation():
    """Test that the optimization prevents redundant calculation in real scenarios."""
    # Create data that would have different max_70 if recalculated
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    prices = list(range(100, 200))  # Increasing prices
    
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": [p + 1 for p in prices],
            "Low": [p - 1 for p in prices],
            "Close": prices,
            "Volume": [1_000_000] * 100,
            "max_70": [500] * 100,  # Artificially high cached value
        },
        index=dates,
    )
    
    raw_data = {"SPY": df}
    result = prepare_data_vectorized_system7(raw_data)
    
    # Should preserve the cached value (500), not recalculate
    assert all(result["SPY"]["max_70"] == 500)
    
    # Verify it's different from what would be calculated
    actual_rolling_max = df["Close"].rolling(window=70).max()
    assert not all(result["SPY"]["max_70"] == actual_rolling_max)


if __name__ == "__main__":
    print("Testing max_70 optimization...")
    
    test_max70_calculated_when_missing()
    print("✓ Test 1 passed: max_70 calculated when missing")
    
    test_max70_preserved_when_present()
    print("✓ Test 2 passed: max_70 preserved when present")
    
    test_optimization_prevents_redundant_calculation()
    print("✓ Test 3 passed: optimization prevents redundant calculation")
    
    print("✅ All optimization tests passed!")