"""Test helper utilities for creating consistent test data and mocking dependencies.

This module provides utilities to create test data that matches the expectations
of the trading system, including pre-computed indicators and proper data formats.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from typing import Dict, Any, Optional


def create_ohlcv_data(
    symbol: str = "TEST", 
    periods: int = 100, 
    start_date: str = "2024-01-01",
    freq: str = "B",
    base_price: float = 100.0,
    trend: Optional[str] = None,
    volatility: float = 0.02
) -> pd.DataFrame:
    """Create realistic OHLCV data for testing.
    
    Args:
        symbol: Symbol name (for identification)
        periods: Number of periods to generate
        start_date: Start date for the data
        freq: Frequency (B=business days, D=calendar days)
        base_price: Starting price level
        trend: None, 'up', 'down', or 'volatile' for price patterns
        volatility: Daily volatility as a fraction (0.02 = 2%)
        
    Returns:
        DataFrame with OHLC+Volume data and proper datetime index
    """
    dates = pd.date_range(start_date, periods=periods, freq=freq)
    np.random.seed(42)  # Deterministic for testing
    
    # Generate price series based on trend
    prices = []
    current_price = base_price
    
    for i in range(periods):
        # Add trend component
        trend_component = 0
        if trend == 'up':
            trend_component = 0.001 * i  # Gradual uptrend
        elif trend == 'down':
            trend_component = -0.001 * i  # Gradual downtrend  
        elif trend == 'volatile':
            trend_component = 0.01 * np.sin(i * 0.1)  # Oscillating
            
        # Add random walk
        random_change = np.random.normal(0, volatility)
        current_price = current_price * (1 + trend_component + random_change)
        current_price = max(current_price, 1.0)  # Prevent negative prices
        prices.append(current_price)
    
    # Generate OHLCV from close prices
    df = pd.DataFrame(index=dates)
    df['Close'] = prices
    df['Open'] = df['Close'].shift(1).fillna(df['Close'].iloc[0])
    df['High'] = df[['Open', 'Close']].max(axis=1) * (1 + np.random.uniform(0, 0.005, periods))
    df['Low'] = df[['Open', 'Close']].min(axis=1) * (1 - np.random.uniform(0, 0.005, periods))
    df['Volume'] = np.random.randint(500_000, 2_000_000, periods)
    
    # Ensure High >= Low
    df['High'] = df[['High', 'Low']].max(axis=1)
    
    return df


def add_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic technical indicators that many tests expect.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added indicators
    """
    from indicators_common import add_indicators
    
    # Use the common indicators module if available
    try:
        return add_indicators(df.copy())
    except Exception:
        # Fallback: add minimal indicators manually
        result = df.copy()
        close = result['Close']
        high = result['High']
        low = result['Low']
        volume = result['Volume']
        
        # ATR variants
        true_range = np.maximum(
            high - low,
            np.maximum(
                abs(high - close.shift(1)),
                abs(low - close.shift(1))
            )
        )
        result['ATR10'] = true_range.rolling(10).mean()
        result['ATR20'] = true_range.rolling(20).mean()
        result['ATR50'] = true_range.rolling(50).mean()
        result['atr10'] = result['ATR10']  # lowercase version
        result['atr20'] = result['ATR20']  # lowercase version  
        result['atr50'] = result['ATR50']  # lowercase version
        
        # Moving averages
        result['SMA100'] = close.rolling(100).mean()
        result['SMA200'] = close.rolling(200).mean()
        
        # RSI (simplified)
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(3).mean()
        loss = -delta.where(delta < 0, 0).rolling(3).mean()
        rs = gain / loss
        result['RSI3'] = 100 - (100 / (1 + rs))
        
        # Volume indicators
        result['AvgVolume50'] = volume.rolling(50).mean()
        result['DollarVolume50'] = (close * volume).rolling(50).mean()
        
        # System-specific indicators
        result['ADX7'] = np.random.uniform(20, 80, len(result))  # Mock ADX
        result['min_50'] = low.rolling(50).min()
        result['max_70'] = high.rolling(70).max()
        
        # ATR percentage
        result['ATR_Pct'] = result['ATR10'] / close
        result['atr_pct'] = result['ATR_Pct']  # lowercase version
        
        return result


def create_system_test_data(
    system_num: int,
    symbols: list[str] = None,
    periods: int = 150,
    **kwargs
) -> dict[str, pd.DataFrame]:
    """Create test data tailored for specific systems.
    
    Args:
        system_num: System number (1-7)
        symbols: List of symbols to create data for
        periods: Number of data points
        **kwargs: Additional arguments passed to create_ohlcv_data
        
    Returns:
        Dictionary of symbol -> DataFrame with appropriate indicators
    """
    if symbols is None:
        if system_num == 7:
            symbols = ['SPY']  # System7 is SPY-only
        else:
            symbols = ['DUMMY', 'TEST']
    
    result = {}
    
    for symbol in symbols:
        # Adjust parameters based on system
        base_price = 400.0 if symbol == 'SPY' else 100.0
        volume_multiplier = 50 if symbol == 'SPY' else 1
        
        df = create_ohlcv_data(
            symbol=symbol,
            periods=periods,
            base_price=base_price,
            **kwargs
        )
        
        # Adjust volume for SPY
        if symbol == 'SPY':
            df['Volume'] *= volume_multiplier
            
        # Add indicators
        df = add_basic_indicators(df)
        
        # System-specific adjustments
        if system_num in [2, 6]:  # Short systems
            df['setup'] = 1  # Mock setup signals
        elif system_num == 7:  # SPY catastrophe hedge
            df['setup'] = (df['Low'] <= df['min_50']).astype(int)
            
        result[symbol] = df
        
    return result


@pytest.fixture
def mock_spy_data():
    """Fixture providing SPY data for System7 tests."""
    return create_system_test_data(7, periods=100)


@pytest.fixture
def mock_system_data():
    """Fixture providing generic system test data."""
    return create_system_test_data(1, periods=150)


class MockStrategy:
    """Mock strategy class for testing base functionality."""
    
    def __init__(self, system_name: str = "test"):
        self.SYSTEM_NAME = system_name
        
    def prepare_minimal_for_test(self, raw_data_dict: dict) -> dict:
        """Mock prepare_minimal_for_test that adds basic indicators."""
        result = {}
        for symbol, df in raw_data_dict.items():
            processed = add_basic_indicators(df.copy())
            result[symbol] = processed
        return result
        

def assert_valid_ohlcv(df: pd.DataFrame, symbol: str = "test"):
    """Assert that a DataFrame has valid OHLCV structure."""
    assert not df.empty, f"{symbol} data should not be empty"
    
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        assert col in df.columns, f"{symbol} should have {col} column"
        assert not df[col].isna().all(), f"{symbol} {col} should not be all NaN"
    
    # Basic OHLC validity
    valid_high = (df['High'] >= df[['Open', 'Close']].max(axis=1)).all()
    assert valid_high, f"{symbol} High should be >= max(Open, Close)"
    
    valid_low = (df['Low'] <= df[['Open', 'Close']].min(axis=1)).all()  
    assert valid_low, f"{symbol} Low should be <= min(Open, Close)"
    
    assert (df['Volume'] > 0).all(), f"{symbol} Volume should be positive"


def assert_has_indicators(df: pd.DataFrame, indicators: list[str], symbol: str = "test"):
    """Assert that a DataFrame has the expected technical indicators."""
    for indicator in indicators:
        assert indicator in df.columns, f"{symbol} should have {indicator} indicator"
        # Allow many NaN values at the beginning due to rolling calculations
        non_null_count = df[indicator].notna().sum()
        # For rolling indicators, we only need some valid values (at least 10% of data)
        min_required = max(10, len(df) * 0.1)  # At least 10 values or 10% of data
        assert non_null_count >= min_required, f"{symbol} {indicator} should have at least {min_required} non-NaN values, got {non_null_count}"