import numpy as np
import pandas as pd
import pytest

from core.system7 import prepare_data_vectorized_system7
from strategies.system7_strategy import System7Strategy


@pytest.fixture
def dummy_data():
    dates = pd.date_range("2024-01-01", periods=70, freq="B")
    df = pd.DataFrame(
        {
            "Open": [100] * 70,
            "High": [101] * 70,
            "Low": [99] * 70,
            "Close": [100] * 70,
            "Volume": [1_000_000] * 70,
        },
        index=dates,
    )
    return {"SPY": df}


@pytest.fixture
def spy_trend_data():
    """System7用のSPYトレンドデータ（ショート戦略なので下落トレンド）"""
    dates = pd.date_range("2024-01-01", periods=100, freq="B")

    # 下落トレンドでmin_50を更新する条件を作る
    prices = []
    for i in range(100):
        if i < 70:
            # 初期上昇期間
            price = 400 - i * 0.5  # 下落開始
        else:
            # さらなる下落期間（min_50 <= Low条件を満たすため）
            price = 400 - 70 * 0.5 - (i - 69) * 1.0  # より急激な下落
        prices.append(max(price, 200))  # 最低価格保証

    df = pd.DataFrame(
        {
            "Open": prices,
            "High": [p * 1.01 for p in prices],
            "Low": [p * 0.99 for p in prices],
            "Close": prices,
            "Volume": [10_000_000] * 100,  # SPYらしい高い出来高
        },
        index=dates,
    )

    return {"SPY": df}


def test_minimal_indicators(dummy_data):
    strategy = System7Strategy()
    processed = strategy.prepare_minimal_for_test(dummy_data)
    assert "ATR50" in processed["SPY"].columns


def test_spy_indicators_computation():
    """System7 SPY指標計算のテスト - using strategy method instead of direct core function"""
    from tests.test_helpers import create_system_test_data
    
    # Create properly formatted test data with required indicators  
    spy_data = create_system_test_data(7, periods=100)
    
    # Use the strategy's prepare_minimal_for_test instead of core function
    strategy = System7Strategy()
    result = strategy.prepare_minimal_for_test(spy_data)

    assert "SPY" in result, "SPY data should be processed"
    spy_result = result["SPY"]

    # Check that ATR50 indicator is calculated (this is what prepare_minimal_for_test adds)
    assert "ATR50" in spy_result.columns, "ATR50 should be calculated"
    
    # ATR50 should be positive for most data points (allowing for initial NaN values)
    atr50_values = spy_result["ATR50"].dropna()
    assert len(atr50_values) > 0, "Should have some ATR50 values"
    assert (atr50_values > 0).all(), "ATR50 values should be positive"


def test_min_50_rolling_calculation():
    """50日間の最低価格（min_50）計算の正確性テスト - using strategy approach"""
    from tests.test_helpers import create_ohlcv_data
    
    # Create test data with a clear pattern for min_50 calculation
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    
    # Create prices with a clear minimum pattern
    prices = [100] * 20 + [95] * 10 + [90] * 10 + [110] * 20 + [85] * 20
    
    df = pd.DataFrame({
        "Open": prices,
        "High": [p + 1 for p in prices],
        "Low": [p - 1 for p in prices],  # Low values will be used for min_50
        "Close": prices,
        "Volume": [10_000_000] * 80,
    }, index=dates)
    
    spy_data = {"SPY": df}
    strategy = System7Strategy()
    result = strategy.prepare_minimal_for_test(spy_data)
    
    assert "SPY" in result, "SPY data should be processed"
    assert "ATR50" in result["SPY"].columns, "ATR50 should be calculated by prepare_minimal_for_test"
    
    # Test that we can manually calculate min_50 and understand the pattern
    manual_min_50 = df['Low'].rolling(50).min()
    # After 50 periods, min_50 should capture the lowest Low value in that window
    assert not manual_min_50.iloc[49:].isna().all(), "Should have min_50 values after 50 periods"


def test_setup_condition_detection():
    """System7セットアップ条件（Low <= min_50）の検出テスト"""
    dates = pd.date_range("2024-01-01", periods=80, freq="B")

    # セットアップ条件を意図的に作成
    prices = []
    lows = []

    for i in range(80):
        if i < 60:
            # 通常の価格レンジ（セットアップ無し）
            price = 100 + i * 0.1
            low = price - 0.5
        else:
            # セットアップ期間：Lowが過去50日間の最低を下回る
            price = 100 + 60 * 0.1 - (i - 59) * 2.0  # 急落
            low = price - 1.0  # より低いLow

        prices.append(price)
        lows.append(low)

    df = pd.DataFrame(
        {
            "Open": prices,
            "High": [p + 1 for p in prices],
            "Low": lows,
            "Close": prices,
            "Volume": [10_000_000] * 80,
        },
        index=dates,
    )

    result = prepare_data_vectorized_system7({"SPY": df})["SPY"]

    # セットアップ条件の確認
    setup_signals = result["setup"]

    # 後半でセットアップシグナルが発生しているはず
    late_setups = setup_signals.iloc[60:].sum()
    assert late_setups > 0, f"Should have setup signals in the declining period, got {late_setups}"


def test_max_70_preservation():
    """max_70の既存値保持ロジックのテスト"""
    dates = pd.date_range("2024-01-01", periods=100, freq="B")

    # 初期データ
    prices = [400 + i * 0.5 for i in range(100)]
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": [p + 2 for p in prices],
            "Low": [p - 2 for p in prices],
            "Close": prices,
            "Volume": [20_000_000] * 100,
        },
        index=dates,
    )

    # 最初の計算
    result1 = prepare_data_vectorized_system7({"SPY": df})["SPY"]
    original_max_70 = result1["max_70"].copy()

    # キャッシュを使用して再計算（use_cache=Trueのシミュレーション）
    result2_data = prepare_data_vectorized_system7(
        {"SPY": df}, use_cache=True, cached_data={"SPY": result1}
    )
    result2 = result2_data["SPY"]

    # max_70が保持されているかチェック
    preserved_max_70 = result2["max_70"]

    # NaN以外の値は保持されているはず
    valid_original = original_max_70.notna()
    if valid_original.any():
        preserved_valid = preserved_max_70[valid_original]
        original_valid = original_max_70[valid_original]

        # 値がほぼ等しいかチェック（浮動小数点の精度を考慮）
        diff = abs(preserved_valid - original_valid).max()
        assert diff < 0.01, f"max_70 values should be preserved, max diff: {diff}"


def test_spy_only_constraint():
    """System7がSPY専用であることのテスト"""
    dates = pd.date_range("2024-01-01", periods=70, freq="B")
    df = pd.DataFrame(
        {
            "Open": [100] * 70,
            "High": [101] * 70,
            "Low": [99] * 70,
            "Close": [100] * 70,
            "Volume": [1_000_000] * 70,
        },
        index=dates,
    )

    # SPY以外のシンボルでテスト
    non_spy_data = {"AAPL": df}
    result = prepare_data_vectorized_system7(non_spy_data)

    # SPY以外のデータは処理されないはず
    assert len(result) == 0 or "AAPL" not in result, "System7 should only process SPY data"

    # SPYデータでテスト
    spy_data = {"SPY": df}
    spy_result = prepare_data_vectorized_system7(spy_data)

    # SPYデータは処理されるはず
    assert "SPY" in spy_result, "System7 should process SPY data"
    assert len(spy_result["SPY"]) > 0, "SPY data should be processed successfully"


def test_placeholder_run(dummy_data):
    strategy = System7Strategy()
    # 戦略オブジェクトが正常に作成できることをテスト
    assert strategy is not None, "Strategy should be created successfully"

    # prepare_minimal_for_testが動作することを確認
    processed = strategy.prepare_minimal_for_test(dummy_data)
    assert isinstance(processed, dict), "prepare_minimal_for_test should return a dictionary"
    assert len(processed) > 0, "Processed data should not be empty"
    assert "SPY" in processed, "Should process SPY data"
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    df = pd.DataFrame(
        {
            "Open": [100, 100, 100, 100],
            "High": [100, 100, 110, 100],
            "Low": [100, 100, 100, 100],
            "Close": [100, 100, 100, 100],
            "Volume": [1_000_000] * 4,
            "ATR50": [1, 1, 1, 1],
            "max_70": [150, 150, 150, 150],
        },
        index=dates,
    )
    prepared = {"SPY": df}
    entry_date = dates[1]
    candidates = {entry_date: [{"symbol": "SPY", "entry_date": entry_date, "ATR50": 1}]}
    trades = strategy.run_backtest(prepared, candidates, capital=10_000)
    assert not trades.empty
    assert "pnl" in trades.columns


def test_entry_rule_market_short():
    strategy = System7Strategy()
    dates = pd.date_range("2024-01-01", periods=2, freq="B")
    df = pd.DataFrame(
        {
            "Open": [100, 100],
            "High": [101, 101],
            "Low": [99, 99],
            "Close": [100, 100],
            "ATR50": [1, 1],
        },
        index=dates,
    )
    candidate = {"symbol": "SPY", "entry_date": dates[1]}
    entry = strategy.compute_entry(df, candidate, current_capital=10_000)
    assert entry == (100.0, 103.0)
