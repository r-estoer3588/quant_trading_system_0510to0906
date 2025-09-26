import numpy as np
import pandas as pd
import pytest

from core.system4 import _compute_indicators_frame, _rename_ohlcv
from strategies.system4_strategy import System4Strategy


@pytest.fixture
def dummy_data():
    dates = pd.date_range("2024-01-01", periods=250, freq="B")
    df = pd.DataFrame(
        {
            "Open": [100] * 250,
            "High": [101] * 250,
            "Low": [99] * 250,
            "Close": [100] * 250,
            "Volume": [2_000_000] * 250,
        },
        index=dates,
    )
    return {"DUMMY": df}


@pytest.fixture
def volatility_trend_data():
    """System4用のボラティリティとトレンドデータ"""
    dates = pd.date_range("2024-01-01", periods=250, freq="B")

    # 上昇トレンド（SMA200を上回る）+ 低ボラティリティ
    prices = []
    volumes = []

    base_price = 50.0
    for i in range(250):
        # 長期上昇トレンド with occasional pullbacks
        if i < 200:
            # 上昇トレンド期間
            price = base_price + i * 0.25 + np.sin(i * 0.1) * 2
        else:
            # RSI4が低くなるプルバック期間
            price = base_price + 200 * 0.25 - (i - 199) * 0.5

        # 低ボラティリティ設定
        price += np.random.normal(0, 0.1)  # 小さなノイズ

        prices.append(max(price, 10))  # 最低価格保証
        volumes.append(3_000_000 + np.random.randint(-500_000, 500_000))

    df = pd.DataFrame(
        {
            "Open": prices,
            "High": [p * 1.01 for p in prices],
            "Low": [p * 0.99 for p in prices],
            "Close": prices,
            "Volume": volumes,
        },
        index=dates,
    )

    return {"VOL_TREND": df}


def test_minimal_indicators(dummy_data):
    strategy = System4Strategy()
    processed = strategy.prepare_minimal_for_test(dummy_data)
    assert "SMA200" in processed["DUMMY"].columns


def test_core_indicators_computation():
    """System4コア指標計算のテスト"""
    dates = pd.date_range("2024-01-01", periods=250, freq="B")

    # より変動のあるデータでテスト
    price_data = []
    for i in range(250):
        base = 100 + i * 0.1
        volatility = np.sin(i * 0.05) * 2  # 周期的な変動
        price_data.append(base + volatility)

    df = pd.DataFrame(
        {
            "Open": price_data,
            "High": [p * 1.02 for p in price_data],
            "Low": [p * 0.98 for p in price_data],
            "Close": price_data,
            "Volume": [2_500_000] * 250,
        },
        index=dates,
    )

    result = _compute_indicators_frame(df)

    # 必要な指標が計算されているかチェック
    expected_columns = ["SMA200", "ATR40", "HV50", "RSI4", "DollarVolume50"]
    for col in expected_columns:
        assert col in result.columns, f"{col} column missing"

    # 指標の妥当性チェック
    assert result["ATR40"].min() >= 0, "ATR should be positive"
    assert result["HV50"].min() >= 0, "Historical volatility should be positive"

    # RSI4の範囲チェック（NaN値を除外）
    rsi_values = result["RSI4"].dropna()
    assert len(rsi_values) > 0, "Should have some RSI values"
    assert (rsi_values >= 0).all() and (rsi_values <= 100).all(), "RSI should be in [0,100]"

    assert result["DollarVolume50"].min() >= 0, "Dollar volume should be positive"


def test_historical_volatility_calculation():
    """HV50（歴史的ボラティリティ）の計算テスト"""
    dates = pd.date_range("2024-01-01", periods=100, freq="B")

    # 高ボラティリティデータ
    high_vol_prices = [100 + np.random.normal(0, 5) for _ in range(100)]
    high_vol_df = pd.DataFrame(
        {
            "Open": high_vol_prices,
            "High": [p + 1 for p in high_vol_prices],
            "Low": [p - 1 for p in high_vol_prices],
            "Close": high_vol_prices,
            "Volume": [1_000_000] * 100,
        },
        index=dates,
    )

    # 低ボラティリティデータ
    low_vol_prices = [100 + i * 0.01 for i in range(100)]  # 非常に安定
    low_vol_df = pd.DataFrame(
        {
            "Open": low_vol_prices,
            "High": [p + 0.1 for p in low_vol_prices],
            "Low": [p - 0.1 for p in low_vol_prices],
            "Close": low_vol_prices,
            "Volume": [1_000_000] * 100,
        },
        index=dates,
    )

    high_vol_result = _compute_indicators_frame(high_vol_df)
    low_vol_result = _compute_indicators_frame(low_vol_df)

    # 高ボラティリティデータのHV50が低ボラティリティデータより大きいはず
    high_vol_avg = high_vol_result["HV50"].dropna().mean()
    low_vol_avg = low_vol_result["HV50"].dropna().mean()

    assert high_vol_avg > low_vol_avg, (
        f"High vol HV50 ({high_vol_avg:.2f}) should be > " f"low vol HV50 ({low_vol_avg:.2f})"
    )


def test_rsi4_pullback_detection():
    """RSI4によるプルバック検出のテスト"""
    dates = pd.date_range("2024-01-01", periods=50, freq="B")

    # プルバック：上昇トレンド後の短期下落
    prices = []
    for i in range(50):
        if i < 30:
            # 上昇期間
            price = 100 + i * 1.0
        else:
            # プルバック期間（RSI4が低下）
            price = 130 - (i - 29) * 0.5
        prices.append(price)

    df = pd.DataFrame(
        {
            "Open": prices,
            "High": [p + 0.5 for p in prices],
            "Low": [p - 0.5 for p in prices],
            "Close": prices,
            "Volume": [1_500_000] * 50,
        },
        index=dates,
    )

    result = _compute_indicators_frame(df)

    # プルバック期間のRSI4が上昇期間より低いことを確認
    uptrend_rsi = result["RSI4"].iloc[25:30].mean()  # 上昇期間終盤
    pullback_rsi = result["RSI4"].iloc[45:].mean()  # プルバック期間

    assert pullback_rsi < uptrend_rsi, (
        f"Pullback RSI ({pullback_rsi:.1f}) should be < " f"uptrend RSI ({uptrend_rsi:.1f})"
    )


def test_ohlcv_column_normalization():
    """OHLCV列名の正規化テスト"""
    # 小文字の列名データ
    df_lower = pd.DataFrame(
        {
            "open": [100, 101, 102],
            "high": [101, 102, 103],
            "low": [99, 100, 101],
            "close": [100, 101, 102],
            "volume": [1000, 1100, 1200],
        }
    )

    result = _rename_ohlcv(df_lower)

    # 大文字に正規化されているかチェック
    expected_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in expected_cols:
        assert col in result.columns, f"{col} should be present after normalization"


def test_placeholder_run(dummy_data):
    strategy = System4Strategy()
    # 戦略オブジェクトが正常に作成できることをテスト
    assert strategy is not None, "Strategy should be created successfully"

    # prepare_minimal_for_testが動作することを確認
    processed = strategy.prepare_minimal_for_test(dummy_data)
    assert isinstance(processed, dict), "prepare_minimal_for_test should return a dictionary"
    assert len(processed) > 0, "Processed data should not be empty"
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame(
        {
            "Open": [100, 100, 100],
            "High": [101, 101, 101],
            "Low": [99, 99, 90],
            "Close": [100, 100, 90],
            "Volume": [2_000_000] * 3,
            "ATR40": [1, 1, 1],
        },
        index=dates,
    )
    prepared = {"DUMMY": df}
    entry_date = dates[1]
    candidates = {entry_date: [{"symbol": "DUMMY", "entry_date": entry_date}]}
    trades = strategy.run_backtest(prepared, candidates, capital=10_000)
    assert not trades.empty
    assert "pnl" in trades.columns


def test_entry_rule_market_open():
    strategy = System4Strategy()
    dates = pd.date_range("2024-01-01", periods=2, freq="B")
    df = pd.DataFrame(
        {
            "Open": [100, 100],
            "High": [101, 101],
            "Low": [99, 99],
            "Close": [100, 100],
            "ATR40": [1, 1],
        },
        index=dates,
    )
    candidate = {"symbol": "DUMMY", "entry_date": dates[1]}
    entry = strategy.compute_entry(df, candidate, current_capital=10_000)
    assert entry == (100.0, pytest.approx(98.5))
