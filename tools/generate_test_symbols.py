#!/usr/bin/env python3
"""
テスト用架空銘柄データ生成スクリプト

System1-7 の各段階（フィルター・セットアップ・シグナル）をテストするための
架空銘柄データを生成します。

使用方法:
    python tools/generate_test_symbols.py

生成されるファイル:
    data_cache/test_symbols/FAIL_ALL.feather
    data_cache/test_symbols/FILTER_ONLY_S1.feather
    ... (他の架空銘柄)
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.indicators_common import add_indicators as compute_all_indicators
from config.settings import get_settings

DEFAULT_LOOKBACK_DAYS = 300
DEFAULT_VOLATILITY = 0.02
DEFAULT_RANDOM_SEED = 42

ALIAS_COLUMN_MAP: dict[str, str] = {
    "sma25": "SMA25",
    "sma50": "SMA50",
    "sma100": "SMA100",
    "sma150": "SMA150",
    "sma200": "SMA200",
    "atr10": "ATR10",
    "atr20": "ATR20",
    "atr40": "ATR40",
    "atr50": "ATR50",
    "atr_ratio": "ATR_Ratio",
    "atr_pct": "ATR_Pct",
    "dollarvolume20": "DollarVolume20",
    "dollarvolume50": "DollarVolume50",
    "avgvolume50": "AvgVolume50",
    "roc200": "ROC200",
    "rsi3": "RSI3",
    "rsi4": "RSI4",
    "hv50": "HV50",
    "adx7": "ADX7",
}


def create_base_dates(days: int = DEFAULT_LOOKBACK_DAYS) -> pd.DatetimeIndex:
    """営業日（NYSE）ベースの日付インデックスを作成"""

    nyse = mcal.get_calendar("NYSE")
    end_date = pd.Timestamp.utcnow().normalize()
    start_candidate = end_date - pd.Timedelta(days=int(days * 1.5))
    schedule = nyse.schedule(start_date=start_candidate, end_date=end_date)
    trading_days = pd.DatetimeIndex(schedule.index.tz_localize(None)).normalize()

    if len(trading_days) >= days:
        trading_days = trading_days[-days:]

    return pd.DatetimeIndex(trading_days, name="Date")


def create_base_ohlcv(
    dates: pd.DatetimeIndex,
    base_price: float,
    volatility: float = DEFAULT_VOLATILITY,
    seed: int | None = None,
) -> pd.DataFrame:
    """基本的な OHLCV データを生成"""

    rng = np.random.default_rng(seed if seed is not None else DEFAULT_RANDOM_SEED)
    n_days = len(dates)

    prices = np.empty(n_days)
    prices[0] = base_price
    returns = rng.normal(0, volatility, n_days)
    for idx in range(1, n_days):
        prices[idx] = prices[idx - 1] * (1 + returns[idx])

    high_noise = rng.uniform(0, 0.01, n_days)
    low_noise = rng.uniform(-0.01, 0, n_days)
    high_prices = prices * (1 + high_noise)
    low_prices = prices * (1 + low_noise)
    open_prices = np.roll(prices, 1)
    open_prices[0] = prices[0]

    base_volume = 1_000_000
    volume_noise = rng.uniform(0.5, 2.0, n_days)
    volumes = (base_volume * volume_noise).astype(int)

    return pd.DataFrame(
        {
            "Date": dates,
            "Open": open_prices,
            "High": high_prices,
            "Low": low_prices,
            "Close": prices,
            "Volume": volumes,
        }
    )


def ensure_custom_columns(df: pd.DataFrame) -> pd.DataFrame:
    """戦略が参照する追加カラムを確実に持たせる"""

    enriched = df.copy()
    if "Close" not in enriched:
        return enriched

    close = enriched["Close"]
    up_days = close.gt(close.shift(1))
    enriched["TwoDayUp"] = up_days & up_days.shift(1)
    enriched["UpTwoDays"] = enriched["TwoDayUp"]

    with np.errstate(divide="ignore", invalid="ignore"):
        enriched["3日下落率"] = (close.shift(3) - close) / close.shift(3) * 100
        enriched["6日上昇率"] = (close - close.shift(6)) / close.shift(6) * 100

    return enriched


def add_alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    """主要列に従来表記（大文字）を付与する"""

    enriched = df.copy()
    for source, alias in ALIAS_COLUMN_MAP.items():
        if source in enriched.columns and alias not in enriched.columns:
            enriched[alias] = enriched[source]
    return enriched


def apply_symbol_config(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """共通指標を計算しつつ設定に基づいて最終行を上書き"""

    enriched = compute_all_indicators(df)
    enriched = ensure_custom_columns(enriched)
    enriched = add_alias_columns(enriched)

    col_map = {
        col.lower(): col
        for col in enriched.columns
        if col.lower() not in {"date"}
    }

    skip_keys = {"base_price", "volatility"}

    for key, value in config.items():
        if key.lower() in skip_keys:
            continue
        if not isinstance(value, (int, float, bool)):
            continue

        lookup_key = key.lower()
        actual_col = col_map.get(lookup_key)
        if actual_col is None:
            actual_col = key
            enriched[actual_col] = np.nan
            col_map[lookup_key] = actual_col

        enriched.loc[enriched.index[-1], actual_col] = value

    return enriched


def create_test_symbol_configs() -> dict[str, dict[str, Any]]:
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
            "DollarVolume20": 200000,
            "DollarVolume50": 200000,
            "HV50": 5,
        },
        "FILTER_ONLY_S1": {
            "base_price": 50.0,
            "Close": 50.0,
            "Volume": 2000000,
            "SMA25": 52.0,
            "SMA50": 51.0,
            "DollarVolume20": 100000000,
            "ATR_Ratio": 0.01,
            "ROC200": 0.05,
        },
        "FILTER_ONLY_S2": {
            "base_price": 25.0,
            "Close": 25.0,
            "Volume": 1500000,
            "DollarVolume20": 37500000,
            "ATR_Ratio": 0.04,
            "RSI3": 85,
            "ADX7": 30,
        },
        "FILTER_ONLY_S3": {
            "base_price": 22.0,
            "Close": 22.0,
            "Low": 20.0,
            "Volume": 1500000,
            "AvgVolume50": 1500000,
            "ATR_Ratio": 0.06,
            "SMA150": 23.0,
            "3日下落率": 10.0,
        },
        "FILTER_ONLY_S4": {
            "base_price": 100.0,
            "Close": 100.0,
            "Volume": 1200000,
            "DollarVolume50": 120000000,
            "HV50": 25,
            "SMA200": 105.0,
            "RSI4": 30,
        },
        "FILTER_ONLY_S5": {
            "base_price": 15.0,
            "Close": 15.0,
            "Volume": 600000,
            "AvgVolume50": 600000,
            "DollarVolume50": 9000000,
            "ATR_Pct": 0.03,
            "SMA100": 14.0,
            "ATR10": 0.8,
            "ADX7": 65,
        },
        "FILTER_ONLY_S6": {
            "base_price": 20.0,
            "Close": 20.0,
            "Low": 18.0,
            "Volume": 800000,
            "DollarVolume50": 16000000,
            "return_6d": 15.0,
            "UpTwoDays": False,
        },
        "SETUP_PASS_S1": {
            "base_price": 50.0,
            "Close": 50.0,
            "Volume": 2000000,
            "SMA25": 51.0,
            "SMA50": 49.0,
            "DollarVolume20": 100000000,
            "ROC200": 0.05,
        },
        "SETUP_PASS_S2": {
            "base_price": 25.0,
            "Close": 25.0,
            "Volume": 1500000,
            "DollarVolume20": 37500000,
            "ATR_Ratio": 0.04,
            "RSI3": 95,
            "TwoDayUp": True,
            "ADX7": 30,
        },
        "SETUP_PASS_S3": {
            "base_price": 22.0,
            "Close": 22.0,
            "Low": 20.0,
            "Volume": 1500000,
            "AvgVolume50": 1500000,
            "ATR_Ratio": 0.06,
            "SMA150": 21.0,
            "3日下落率": 15.0,
        },
        "SETUP_PASS_S4": {
            "base_price": 100.0,
            "Close": 100.0,
            "Volume": 1200000,
            "DollarVolume50": 120000000,
            "HV50": 25,
            "SMA200": 95.0,
            "RSI4": 30,
        },
        "SETUP_PASS_S5": {
            "base_price": 15.0,
            "Close": 15.0,
            "Volume": 600000,
            "AvgVolume50": 600000,
            "DollarVolume50": 9000000,
            "ATR_Pct": 0.03,
            "SMA100": 13.0,
            "ATR10": 0.5,
            "ADX7": 60,
            "RSI3": 40,
        },
        "SETUP_PASS_S6": {
            "base_price": 20.0,
            "Close": 20.0,
            "Low": 18.0,
            "Volume": 800000,
            "DollarVolume50": 16000000,
            "return_6d": 25.0,
            "UpTwoDays": True,
            "6日上昇率": 30.0,
        },
    }


def generate_test_symbols() -> None:
    """架空銘柄データを生成"""

    settings = get_settings()
    output_dir = Path(settings.data.cache_dir) / "test_symbols"
    output_dir.mkdir(parents=True, exist_ok=True)

    dates = create_base_dates()
    configs = create_test_symbol_configs()

    print(f"架空銘柄データを生成中... 出力先: {output_dir}")

    for idx, (symbol_name, config) in enumerate(configs.items()):
        print(f"  {symbol_name}を生成中...")

        df = create_base_ohlcv(
            dates=dates,
            base_price=float(config["base_price"]),
            volatility=float(config.get("volatility", DEFAULT_VOLATILITY)),
            seed=DEFAULT_RANDOM_SEED + idx,
        )

        df = apply_symbol_config(df, config)

        df.set_index("Date", inplace=True)

        output_path = output_dir / f"{symbol_name}.feather"
        df.reset_index().to_feather(output_path)

        last_row = df.iloc[-1]
        print(
            f"    保存完了: {output_path}\n"
            f"    最新データ: Close={last_row['Close']:.2f},"
            f" Volume={last_row['Volume']:,}"
        )
        if {"SMA25", "SMA50"}.issubset(last_row.index):
            print(f"    SMA25={last_row['SMA25']:.2f}, SMA50={last_row['SMA50']:.2f}")
        if "RSI3" in last_row.index:
            print(f"    RSI3={last_row['RSI3']:.1f}")

    print(f"\n✅ 架空銘柄データ生成完了: {len(configs)}銘柄")
    print(f"📁 出力ディレクトリ: {output_dir}")
    print("\n📖 使用方法:")
    print(
        "  python scripts/run_all_systems_today.py"
        " --test-mode test_symbols --skip-external"
    )


if __name__ == "__main__":
    generate_test_symbols()
