#!/usr/bin/env python3
"""
テスト用架空銘柄データ生成スクリプト

System1-7の各段階（フィルター・セットアップ・シグナル）をテストするための
架空銘柄データを生成します。

使用方法:
    python tools/generate_test_symbols.py

生成されるファイル:
    data_cache/test_symbols/FAIL_ALL.feather
    data_cache/test_symbols/FILTER_ONLY_S1.feather
    ... (他の架空銘柄)
"""

from datetime import datetime, timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings


def create_base_dates(days: int = 300) -> pd.DatetimeIndex:
    """基準となる日付インデックスを作成"""
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days - 1)
    return pd.date_range(start=start_date, end=end_date, freq="D")


def create_base_ohlcv(dates: pd.DatetimeIndex, base_price: float, volatility: float = 0.02) -> pd.DataFrame:
    """基本的なOHLCVデータを生成"""
    np.random.seed(42)  # 再現性のため

    n_days = len(dates)
    returns = np.random.normal(0, volatility, n_days)
    prices = [base_price]

    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    prices = np.array(prices)

    # OHLC生成（終値ベース）
    close_prices = prices
    high_noise = np.random.uniform(0, 0.01, n_days)
    low_noise = np.random.uniform(-0.01, 0, n_days)

    high_prices = close_prices * (1 + high_noise)
    low_prices = close_prices * (1 + low_noise)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]

    # Volume生成
    base_volume = 1000000
    volume_noise = np.random.uniform(0.5, 2.0, n_days)
    volumes = (base_volume * volume_noise).astype(int)

    return pd.DataFrame(
        {
            "Date": dates,
            "Open": open_prices,
            "High": high_prices,
            "Low": low_prices,
            "Close": close_prices,
            "Volume": volumes,
        }
    )


def add_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """指標を追加"""
    df = df.copy()

    # 移動平均
    for period in [25, 50, 100, 150, 200]:
        df[f"SMA{period}"] = config.get(f"SMA{period}", df["Close"].rolling(period).mean())

    # ATR計算
    df["HL"] = df["High"] - df["Low"]
    df["HC"] = abs(df["High"] - df["Close"].shift(1))
    df["LC"] = abs(df["Low"] - df["Close"].shift(1))
    df["TrueRange"] = df[["HL", "HC", "LC"]].max(axis=1)

    for period in [10, 20, 50]:
        df[f"ATR{period}"] = df["TrueRange"].rolling(period).mean()

    # Min/Max計算
    for period in [20, 50]:
        df[f"Min_{period}"] = df["Low"].rolling(period).min()
        df[f"Max_{period}"] = df["High"].rolling(period).max()

    # RSI（簡易版）
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=3).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=3).mean()
    rs = gain / loss
    df["RSI3"] = 100 - (100 / (1 + rs))

    # RSI4
    gain4 = (delta.where(delta > 0, 0)).rolling(window=4).mean()
    loss4 = (-delta.where(delta < 0, 0)).rolling(window=4).mean()
    rs4 = gain4 / loss4
    df["RSI4"] = 100 - (100 / (1 + rs4))

    # ADX（簡易版）
    df["ADX7"] = config.get("ADX7", 30.0)  # デフォルト値

    # DollarVolume計算
    df["DollarVolume"] = df["Close"] * df["Volume"]
    for period in [20, 50]:
        df[f"DollarVolume_{period}"] = df["DollarVolume"].rolling(period).mean()

    # ATR比率計算
    df["ATR_Ratio"] = (df["ATR20"] / df["Close"]) * 100

    # HV（Historical Volatility）簡易版
    returns = df["Close"].pct_change()
    df["HV50"] = returns.rolling(50).std() * np.sqrt(252) * 100

    # ボリューム平均
    for period in [50]:
        df[f"AvgVol{period}"] = df["Volume"].rolling(period).mean()

    # カスタム指標を適用
    for key, value in config.items():
        if key not in df.columns and isinstance(value, (int, float)):
            df[key] = value

    # 一時的な計算列を削除
    df = df.drop(["HL", "HC", "LC", "TrueRange"], axis=1, errors="ignore")

    return df
    df["ADX7"] = config.get("ADX7", np.random.uniform(20, 80, len(df)))

    # ATR関連
    high_low = df["High"] - df["Low"]
    df["ATR10"] = high_low.rolling(10).mean()
    df["ATR_Ratio"] = config.get("ATR_Ratio", df["ATR10"] / df["Close"])
    df["ATR_Pct"] = df["ATR_Ratio"]

    # 出来高指標
    df["AvgVolume50"] = config.get("AvgVolume50", df["Volume"].rolling(50).mean())
    df["DollarVolume20"] = config.get("DollarVolume20", (df["Close"] * df["Volume"]).rolling(20).mean())
    df["DollarVolume50"] = config.get("DollarVolume50", (df["Close"] * df["Volume"]).rolling(50).mean())

    # HV50（ボラティリティ）
    returns = df["Close"].pct_change()
    df["HV50"] = config.get("HV50", returns.rolling(50).std() * 100)

    # ROC200
    df["ROC200"] = config.get("ROC200", (df["Close"] / df["Close"].shift(200) - 1))

    # return_6d
    df["return_6d"] = config.get("return_6d", (df["Close"] / df["Close"].shift(6) - 1) * 100)

    # パターン検出（簡易版）
    up_days = df["Close"] > df["Close"].shift(1)
    df["TwoDayUp"] = config.get("TwoDayUp", up_days & up_days.shift(1))
    df["UpTwoDays"] = df["TwoDayUp"]

    # 3日下落率（簡易計算）
    down_3d = (df["Close"].shift(3) - df["Close"]) / df["Close"].shift(3) * 100
    df["3日下落率"] = config.get("3日下落率", down_3d)

    # 6日上昇率
    up_6d = (df["Close"] - df["Close"].shift(6)) / df["Close"].shift(6) * 100
    df["6日上昇率"] = config.get("6日上昇率", up_6d)

    # 最終行の値を設定値で上書き（最新データとして使用）
    for key, value in config.items():
        if key in df.columns and isinstance(value, (int, float, bool)):
            df.loc[df.index[-1], key] = value

    return df


def create_test_symbol_configs():
    """各架空銘柄の設定を定義"""
    return {
        "FAIL_ALL": {
            "base_price": 2.0,
            "Close": 2.0,
            "Volume": 100000,
            "SMA25": 2.1,
            "SMA50": 2.0,
            "RSI3": 50,
            "ATR_Ratio": 0.01,
            "DollarVolume20": 200000,  # 2.0 * 100000
            "DollarVolume50": 200000,
            "HV50": 5,
        },
        "FILTER_ONLY_S1": {
            "base_price": 50.0,
            "Close": 50.0,
            "Volume": 2000000,
            "SMA25": 52.0,
            "SMA50": 51.0,  # SMA25 < SMA50でセットアップで落ちる
            "DollarVolume20": 100000000,  # 50 * 2M = 100M
            "ATR_Ratio": 0.01,  # 他システムで落ちる
            "ROC200": 0.05,
        },
        "FILTER_ONLY_S2": {
            "base_price": 25.0,
            "Close": 25.0,
            "Volume": 1500000,
            "DollarVolume20": 37500000,  # 25 * 1.5M = 37.5M
            "ATR_Ratio": 0.04,  # 3%以上
            "RSI3": 85,  # 90未満でセットアップで落ちる
            "ADX7": 30,
        },
        "FILTER_ONLY_S3": {
            "base_price": 22.0,
            "Close": 22.0,
            "Low": 20.0,
            "Volume": 1500000,
            "AvgVolume50": 1500000,  # 1M以上
            "ATR_Ratio": 0.06,  # 5%以上
            "SMA150": 23.0,  # Close < SMA150でセットアップで落ちる
            "3日下落率": 10.0,  # 12.5%未満でセットアップで落ちる
        },
        "FILTER_ONLY_S4": {
            "base_price": 100.0,
            "Close": 100.0,
            "Volume": 1200000,
            "DollarVolume50": 120000000,  # 100 * 1.2M = 120M
            "HV50": 25,  # 10-40範囲内
            "SMA200": 105.0,  # Close < SMA200でセットアップで落ちる
            "RSI4": 30,
        },
        "FILTER_ONLY_S5": {
            "base_price": 15.0,
            "Close": 15.0,
            "Volume": 600000,
            "AvgVolume50": 600000,  # 500k以上
            "DollarVolume50": 9000000,  # 15 * 600k = 9M
            "ATR_Pct": 0.03,  # 2.5%以上
            "SMA100": 14.0,
            "ATR10": 0.8,  # Close < SMA100+ATR10でセットアップで落ちる（15 < 14.8）
            "ADX7": 65,
        },
        "FILTER_ONLY_S6": {
            "base_price": 20.0,
            "Close": 20.0,
            "Low": 18.0,
            "Volume": 800000,
            "DollarVolume50": 16000000,  # 20 * 800k = 16M
            "return_6d": 15.0,  # 20%未満でセットアップで落ちる
            "UpTwoDays": False,
        },
        "SETUP_PASS_S1": {
            "base_price": 50.0,
            "Close": 50.0,
            "Volume": 2000000,
            "SMA25": 51.0,  # SMA25 > SMA50
            "SMA50": 49.0,
            "DollarVolume20": 100000000,
            "ROC200": 0.05,  # シグナル生成用
        },
        "SETUP_PASS_S2": {
            "base_price": 25.0,
            "Close": 25.0,
            "Volume": 1500000,
            "DollarVolume20": 37500000,
            "ATR_Ratio": 0.04,
            "RSI3": 95,  # 90以上
            "TwoDayUp": True,
            "ADX7": 30,  # シグナル生成用（低めに設定）
        },
        "SETUP_PASS_S3": {
            "base_price": 22.0,
            "Close": 22.0,
            "Low": 20.0,
            "Volume": 1500000,
            "AvgVolume50": 1500000,
            "ATR_Ratio": 0.06,
            "SMA150": 21.0,  # Close > SMA150
            "3日下落率": 15.0,  # 12.5%以上
        },
        "SETUP_PASS_S4": {
            "base_price": 100.0,
            "Close": 100.0,
            "Volume": 1200000,
            "DollarVolume50": 120000000,
            "HV50": 25,
            "SMA200": 95.0,  # Close > SMA200
            "RSI4": 30,  # シグナル生成用
        },
        "SETUP_PASS_S5": {
            "base_price": 15.0,
            "Close": 15.0,
            "Volume": 600000,
            "AvgVolume50": 600000,
            "DollarVolume50": 9000000,
            "ATR_Pct": 0.03,
            "SMA100": 13.0,
            "ATR10": 0.5,  # Close > SMA100+ATR10 (15 > 13.5)
            "ADX7": 60,  # 55以上
            "RSI3": 40,  # 50未満
        },
        "SETUP_PASS_S6": {
            "base_price": 20.0,
            "Close": 20.0,
            "Low": 18.0,
            "Volume": 800000,
            "DollarVolume50": 16000000,
            "return_6d": 25.0,  # 20%以上
            "UpTwoDays": True,
            "6日上昇率": 30.0,  # シグナル生成用
        },
    }


def generate_test_symbols():
    """架空銘柄データを生成"""
    settings = get_settings()
    output_dir = Path(settings.data.cache_dir) / "test_symbols"
    output_dir.mkdir(parents=True, exist_ok=True)

    dates = create_base_dates()
    configs = create_test_symbol_configs()

    print(f"架空銘柄データを生成中... 出力先: {output_dir}")

    for symbol_name, config in configs.items():
        print(f"  {symbol_name}を生成中...")

        # 基本OHLCVデータ生成
        df = create_base_ohlcv(dates, config["base_price"])

        # 指標追加
        df = add_indicators(df, config)

        # インデックスをDateに設定
        df.set_index("Date", inplace=True)

        # Feather形式で保存
        output_path = output_dir / f"{symbol_name}.feather"
        df.reset_index().to_feather(output_path)

        print(f"    保存完了: {output_path}")

        # 最新行の重要な値を表示（デバッグ用）
        last_row = df.iloc[-1]
        print(f"    最新データ: Close={last_row['Close']:.2f}, Volume={last_row['Volume']:,}")
        if "SMA25" in last_row and "SMA50" in last_row:
            print(f"    SMA25={last_row['SMA25']:.2f}, SMA50={last_row['SMA50']:.2f}")
        if "RSI3" in last_row:
            print(f"    RSI3={last_row['RSI3']:.1f}")

    print(f"\n✅ 架空銘柄データ生成完了: {len(configs)}銘柄")
    print(f"📁 出力ディレクトリ: {output_dir}")

    # 使用方法の表示
    print("\n📖 使用方法:")
    print("  python scripts/run_all_systems_today.py --test-mode test_symbols --skip-external")


if __name__ == "__main__":
    generate_test_symbols()
