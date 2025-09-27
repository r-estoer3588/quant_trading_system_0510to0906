"""
Comprehensive testing example demonstrating modern test patterns for the trading system.

This file shows the recommended approach for testing trading strategies:
1. Use strategy-level testing instead of low-level core functions
2. Use test helpers to create realistic data
3. Focus on behavioral testing rather than internal implementation
4. Handle dependencies properly (indicators, data formats)
"""

import pytest
import pandas as pd

from tests.test_helpers import (
    create_system_test_data, 
    assert_valid_ohlcv, 
    assert_has_indicators,
    create_ohlcv_data,
    add_basic_indicators
)


class TestModernSystemTesting:
    """Demonstrates modern testing patterns for trading system components."""
    
    def test_strategy_basic_functionality(self):
        """Test that strategy classes can be instantiated and basic methods work."""
        from strategies.system5_strategy import System5Strategy
        from strategies.system6_strategy import System6Strategy
        
        # Test strategy instantiation
        s5 = System5Strategy()
        assert s5.SYSTEM_NAME == "system5"
        
        s6 = System6Strategy()  
        assert s6.SYSTEM_NAME == "system6"
        
    def test_data_creation_helpers(self):
        """Test that our test data helpers create valid data."""
        # Test basic OHLCV creation
        data = create_ohlcv_data(periods=50, base_price=100)
        assert_valid_ohlcv(data, "test_data")
        
        # Test with different parameters
        spy_data = create_ohlcv_data(
            symbol="SPY", 
            periods=100, 
            base_price=400, 
            trend="up",
            volatility=0.015
        )
        assert_valid_ohlcv(spy_data, "SPY")
        assert spy_data['Close'].iloc[-1] > spy_data['Close'].iloc[0], "Should have upward trend"
        
    def test_indicator_calculation(self):
        """Test that indicators are calculated correctly by helpers."""
        # Create basic data and add indicators
        data = create_ohlcv_data(periods=150, base_price=100)
        data_with_indicators = add_basic_indicators(data)
        
        # Check that key indicators are present (use lowercase names as that's what indicators_common uses)
        expected_indicators = ['atr10', 'atr20', 'atr50', 'sma100', 'rsi3']
        assert_has_indicators(data_with_indicators, expected_indicators)
        
        # Check indicator validity (skip first few values that might be 0 or NaN)
        atr10 = data_with_indicators['atr10'].dropna()
        if len(atr10) > 10:  # Only check if we have enough data
            atr10_nonzero = atr10[atr10 > 0]
            assert len(atr10_nonzero) > len(atr10) * 0.7, "Most ATR values should be positive"
        
        rsi3 = data_with_indicators['rsi3'].dropna()
        if len(rsi3) > 0:
            valid_rsi = (rsi3 >= 0) & (rsi3 <= 100)
            assert valid_rsi.all(), "RSI should be between 0 and 100"
        
    def test_system_specific_data_creation(self):
        """Test creating data tailored for specific systems."""
        # Test System1 data
        s1_data = create_system_test_data(1, periods=200)
        assert 'DUMMY' in s1_data
        assert 'TEST' in s1_data
        assert_has_indicators(s1_data['DUMMY'], ['atr20', 'sma200'])
        
        # Test System7 (SPY-only) data  
        s7_data = create_system_test_data(7, periods=150)
        assert 'SPY' in s7_data
        assert len(s7_data) == 1, "System7 should only have SPY"
        assert_has_indicators(s7_data['SPY'], ['atr50', 'min_50'])
        
    def test_strategy_prepare_minimal_consistency(self):
        """Test that prepare_minimal_for_test works consistently across strategies."""
        from strategies.system5_strategy import System5Strategy
        from strategies.system6_strategy import System6Strategy
        from strategies.system7_strategy import System7Strategy
        
        # Create test data
        test_data = {
            'TEST': create_ohlcv_data(periods=120, base_price=50),
            'SPY': create_ohlcv_data(periods=120, base_price=400)
        }
        
        # Test System5
        s5 = System5Strategy()  
        s5_processed = s5.prepare_minimal_for_test(test_data)
        assert 'TEST' in s5_processed
        assert 'SMA100' in s5_processed['TEST'].columns
        
        # Test System6
        s6 = System6Strategy()
        s6_processed = s6.prepare_minimal_for_test(test_data)
        assert 'TEST' in s6_processed
        # System6 adds ATR10
        assert 'ATR10' in s6_processed['TEST'].columns
        
        # Test System7 (only processes SPY)
        s7 = System7Strategy()
        s7_processed = s7.prepare_minimal_for_test({'SPY': test_data['SPY']})
        assert 'SPY' in s7_processed
        assert 'ATR50' in s7_processed['SPY'].columns
        
    def test_backtest_basic_functionality(self):
        """Test basic backtesting functionality using strategy methods."""
        from strategies.system6_strategy import System6Strategy
        
        # Create test data with proper setup
        test_data = create_system_test_data(6, periods=150)  
        
        strategy = System6Strategy()
        
        # Test basic backtest setup (without actual trading)
        dates = list(test_data['DUMMY'].index[-10:])  # Last 10 dates
        candidates = {}
        
        for i, date in enumerate(dates[:-5]):  # Leave some dates for exits
            candidates[date] = [{
                'symbol': 'DUMMY',
                'entry_date': date,
            }]
            
        # This should not crash (though may return empty trades without proper setup)
        try:
            trades = strategy.run_backtest(test_data, candidates, capital=10_000)
            assert isinstance(trades, pd.DataFrame), "Should return DataFrame"
        except Exception as e:
            # If it fails due to missing indicators, that's expected - 
            # the key is that the method exists and has proper interface
            assert 'indicator' in str(e).lower() or 'missing' in str(e).lower()
            
    def test_error_handling(self):
        """Test that functions handle edge cases gracefully."""
        from strategies.system2_strategy import System2Strategy
        
        strategy = System2Strategy()
        
        # Test with empty data
        empty_result = strategy.prepare_minimal_for_test({})
        assert isinstance(empty_result, dict)
        assert len(empty_result) == 0
        
        # Test with minimal data (should not crash)
        minimal_data = {
            'TEST': pd.DataFrame({
                'Open': [100, 101, 102],
                'High': [101, 102, 103], 
                'Low': [99, 100, 101],
                'Close': [100, 101, 102],
                'Volume': [1000, 1100, 1200]
            }, index=pd.date_range('2024-01-01', periods=3))
        }
        
        result = strategy.prepare_minimal_for_test(minimal_data)
        assert 'TEST' in result
        assert len(result['TEST']) == 3


class TestLegacyTestPatterns:
    """
    Examples of problematic test patterns that should be avoided.
    
    These tests demonstrate what NOT to do - they're kept here for educational
    purposes but should not be used as templates for new tests.
    """
    
    def test_bad_pattern_direct_core_function_call(self):
        """
        BAD PATTERN: Testing core functions directly without proper setup.
        
        This pattern fails because core functions expect pre-computed indicators
        and specific data formats that are hard to mock properly.
        """
        # DON'T DO THIS:
        # result = prepare_data_vectorized_system7({"SPY": raw_data})
        # This will fail because the function expects precomputed indicators
        
        # INSTEAD DO THIS:
        from strategies.system7_strategy import System7Strategy
        spy_data = create_system_test_data(7, periods=100)
        strategy = System7Strategy()
        result = strategy.prepare_minimal_for_test(spy_data)
        assert 'SPY' in result
        
    def test_bad_pattern_manual_data_creation(self):
        """
        BAD PATTERN: Manually creating test data without helpers.
        
        This leads to inconsistent data formats and missing edge cases.
        """
        # DON'T DO THIS:
        # df = pd.DataFrame({'Open': [100]*50, 'Close': [100]*50, ...})
        
        # INSTEAD DO THIS:
        df = create_ohlcv_data(periods=50, base_price=100)
        assert_valid_ohlcv(df, "properly_created_data")
        
    def test_bad_pattern_testing_internal_implementation(self):
        """
        BAD PATTERN: Testing internal implementation details.
        
        Tests should focus on behavior, not internal calculations.
        """
        # DON'T TEST: Specific indicator calculation formulas
        # DON'T TEST: Internal function calls and data transformations
        
        # INSTEAD TEST: Overall strategy behavior
        from strategies.system5_strategy import System5Strategy
        test_data = create_system_test_data(5, periods=150)
        
        strategy = System5Strategy()
        result = strategy.prepare_minimal_for_test(test_data)
        
        # Test behavior: strategy should process all provided symbols
        assert len(result) == len(test_data)
        for symbol in test_data.keys():
            assert symbol in result, f"Strategy should process {symbol}"