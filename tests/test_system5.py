import numpy as np
import pandas as pd
import pytest

from core.system5 import DEFAULT_ATR_PCT_THRESHOLD
from strategies.system5_strategy import System5Strategy


@pytest.fixture
def dummy_data():
    dates = pd.date_range("2024-01-01", periods=150, freq="B")
    df = pd.DataFrame(
        {
            "Open": [100] * 150,
            "High": [101] * 150,
            "Low": [99] * 150,
            "Close": [100] * 150,
            "Volume": [1_000_000] * 150,
        },
        index=dates,
    )
    return {"DUMMY": df}


def test_minimal_indicators(dummy_data):
    """Test that System5Strategy can prepare minimal data with required indicators"""
    strategy = System5Strategy()
    processed = strategy.prepare_minimal_for_test(dummy_data)
    assert "SMA100" in processed["DUMMY"].columns


def test_default_atr_pct_threshold():
    """Test that DEFAULT_ATR_PCT_THRESHOLD is properly defined"""
    assert DEFAULT_ATR_PCT_THRESHOLD == 0.025
    assert isinstance(DEFAULT_ATR_PCT_THRESHOLD, float)


def test_strategy_initialization():
    """Test that System5Strategy can be initialized properly"""
    strategy = System5Strategy()
    assert strategy is not None
    

def test_placeholder_run(dummy_data):
    """Test basic functionality of System5Strategy"""
    strategy = System5Strategy()
    # 戦略オブジェクトが正常に作成できることをテスト
    assert strategy is not None, "Strategy should be created successfully"

    # prepare_minimal_for_testが動作することを確認
    processed = strategy.prepare_minimal_for_test(dummy_data)
    assert isinstance(processed, dict), "prepare_minimal_for_test should return a dictionary"
    assert len(processed) > 0, "Processed data should not be empty"
    
    # Check that minimal indicators are present (SMA100 is what prepare_minimal_for_test adds)
    dummy_processed = processed["DUMMY"]
    assert "SMA100" in dummy_processed.columns, "SMA100 should be in processed data"
    assert len(dummy_processed.columns) == 6, "Should have original 5 columns + SMA100"
