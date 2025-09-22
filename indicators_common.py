import numpy as np
from ta.momentum import ROCIndicator, RSIIndicator
from ta.trend import ADXIndicator, SMAIndicator
from ta.volatility import AverageTrueRange


def add_indicators(df):
    df = df.copy()

    # 0終値をNaNに変換（HV50計算用）
    close = df["Close"] if "Close" in df else None
    high = df["High"] if "High" in df else None
    low = df["Low"] if "Low" in df else None
    volume = df["Volume"] if "Volume" in df else None
    close_nozero = close.replace(0, np.nan) if close is not None else None

    # === 基本 ===
    # ATR
    for w in [10, 20, 40, 50]:
        if close is None or high is None or low is None:
            df[f"ATR{w}"] = np.nan
            continue
        try:
            df[f"ATR{w}"] = AverageTrueRange(high, low, close, window=w).average_true_range()
        except Exception:
            df[f"ATR{w}"] = np.nan

    # SMA
    for w in [25, 50, 100, 150, 200]:
        if close is None:
            df[f"SMA{w}"] = np.nan
            continue
        try:
            df[f"SMA{w}"] = SMAIndicator(close, window=w).sma_indicator()
        except Exception:
            df[f"SMA{w}"] = np.nan

    # ROC
    if close is None:
        df["ROC200"] = np.nan
    else:
        try:
            df["ROC200"] = ROCIndicator(close, window=200).roc()
        except Exception:
            df["ROC200"] = np.nan

    # RSI
    for w in [3, 4]:
        if close is None:
            df[f"RSI{w}"] = np.nan
            continue
        try:
            df[f"RSI{w}"] = RSIIndicator(close, window=w).rsi()
        except Exception:
            df[f"RSI{w}"] = np.nan

    # ADX
    for w in [7]:
        if close is None or high is None or low is None:
            df[f"ADX{w}"] = np.nan
            continue
        try:
            df[f"ADX{w}"] = ADXIndicator(high, low, close, window=w).adx()
        except Exception:
            df[f"ADX{w}"] = np.nan

    # 売買代金
    for w in [20, 50]:
        if close is None or volume is None:
            df[f"DollarVolume{w}"] = np.nan
            continue
        try:
            df[f"DollarVolume{w}"] = (close * volume).rolling(window=w).mean()
        except Exception:
            df[f"DollarVolume{w}"] = np.nan

    # 平均出来高
    for w in [50]:
        if volume is None:
            df[f"AvgVolume{w}"] = np.nan
            continue
        try:
            df[f"AvgVolume{w}"] = volume.rolling(window=w).mean()
        except Exception:
            df[f"AvgVolume{w}"] = np.nan

    # ATR割合
    if "ATR10" in df and close_nozero is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = df["ATR10"].div(close_nozero)
        df["ATR_Ratio"] = ratio
        df["ATR_Pct"] = ratio
    else:
        df["ATR_Ratio"] = np.nan
        df["ATR_Pct"] = np.nan

    # その他戦略固有
    if close is not None:
        try:
            df["Return_3D"] = close.pct_change(3)
        except Exception:
            df["Return_3D"] = np.nan
        try:
            df["6D_Return"] = close.pct_change(6)
        except Exception:
            df["6D_Return"] = np.nan
        try:
            up_two = close.gt(close.shift(1)) & close.shift(1).gt(close.shift(2))
            df["UpTwoDays"] = up_two.fillna(False).astype(bool)
        except Exception:
            df["UpTwoDays"] = False
        df["TwoDayUp"] = df["UpTwoDays"]
        if close_nozero is not None:
            try:
                log_ret = np.log(close_nozero / close_nozero.shift(1))
                df["HV50"] = log_ret.rolling(window=50).std() * np.sqrt(252) * 100
            except Exception:
                df["HV50"] = np.nan
        else:
            df["HV50"] = np.nan
        try:
            df["min_50"] = close.rolling(window=50).min()
        except Exception:
            df["min_50"] = np.nan
        try:
            df["max_70"] = close.rolling(window=70).max()
        except Exception:
            df["max_70"] = np.nan
    else:
        df["Return_3D"] = np.nan
        df["6D_Return"] = np.nan
        df["UpTwoDays"] = False
        df["TwoDayUp"] = df["UpTwoDays"]
        df["HV50"] = np.nan
        df["min_50"] = np.nan
        df["max_70"] = np.nan

    return df
