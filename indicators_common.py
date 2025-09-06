from ta.trend import SMAIndicator, ADXIndicator
from ta.momentum import ROCIndicator, RSIIndicator
from ta.volatility import AverageTrueRange
import numpy as np
import pandas as pd


def add_indicators(df):
    df = df.copy()

    # 0終値をNaNに変換（HV20計算用）
    close_nozero = df["Close"].replace(0, np.nan)

    # === 基本 ===
    # ATR
    for w in [10, 20, 40, 50]:
        if len(df) >= w + 1:
            df[f"ATR{w}"] = AverageTrueRange(
                df["High"], df["Low"], df["Close"], window=w
            ).average_true_range()
        else:
            df[f"ATR{w}"] = np.nan

    # SMA
    for w in [25, 50, 100, 150, 200]:
        if len(df) >= w + 1:
            df[f"SMA{w}"] = SMAIndicator(df["Close"], window=w).sma_indicator()
        else:
            df[f"SMA{w}"] = np.nan

    # ROC
    if len(df) >= 200:
        df["ROC200"] = ROCIndicator(df["Close"], window=200).roc()
    else:
        df["ROC200"] = np.nan

    # RSI
    for w in [3, 4]:
        if len(df) >= w + 1:
            df[f"RSI{w}"] = RSIIndicator(df["Close"], window=w).rsi()
        else:
            df[f"RSI{w}"] = np.nan

    # ADX
    for w in [7]:
        if len(df) >= w * 2:
            df[f"ADX{w}"] = ADXIndicator(
                df["High"], df["Low"], df["Close"], window=w
            ).adx()
        else:
            df[f"ADX{w}"] = np.nan

    # 売買代金
    for w in [20, 50]:
        if len(df) >= w + 1:
            df[f"DollarVolume{w}"] = (
                (df["Close"] * df["Volume"]).rolling(window=w).mean()
            )
        else:
            df[f"DollarVolume{w}"] = np.nan

    # 平均出来高
    for w in [50]:
        if len(df) >= w + 1:
            df[f"AvgVolume{w}"] = df["Volume"].rolling(window=w).mean()
        else:
            df[f"AvgVolume{w}"] = np.nan

    # ATR割合
    if "ATR10" in df:
        df[f"ATR_Ratio"] = df[f"ATR10"] / df["Close"]
        df[f"ATR_Pct"] = df[f"ATR10"] / df["Close"]
    else:
        df[f"ATR_Ratio"] = np.nan
        df[f"ATR_Pct"] = np.nan

    # その他戦略固有
    df["Return_3D"] = df["Close"].pct_change(3) if len(df) >= 3 else np.nan
    df["6D_Return"] = df["Close"].pct_change(6) if len(df) >= 6 else np.nan
    df["UpTwoDays"] = (
        (df["Close"] > df["Close"].shift(1))
        & (df["Close"].shift(1) > df["Close"].shift(2))
        if len(df) >= 3
        else False
    )
    df["TwoDayUp"] = df["UpTwoDays"]
    if len(df) >= 20:
        df["HV20"] = (
            np.log(close_nozero / close_nozero.shift(1)).rolling(window=20).std()
            * np.sqrt(252)
            * 100
        )
    else:
        df["HV20"] = np.nan
    df["min_50"] = df["Close"].rolling(window=50).min() if len(df) >= 50 else np.nan
    df["max_70"] = df["Close"].rolling(window=70).max() if len(df) >= 70 else np.nan

    return df
