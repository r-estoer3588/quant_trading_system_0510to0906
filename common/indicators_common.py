from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import ROCIndicator, RSIIndicator
from ta.trend import ADXIndicator, SMAIndicator
from ta.volatility import AverageTrueRange


def add_indicators(df):
    """単一DataFrameに指標を追加する（従来互換）

    事前計算済み指標が存在する場合は再計算をスキップして高速化。
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # 0終値をNaNに変換（HV50計算用）
    close = df["Close"] if "Close" in df else None
    high = df["High"] if "High" in df else None
    low = df["Low"] if "Low" in df else None
    volume = df["Volume"] if "Volume" in df else None
    close_nozero = close.replace(0, np.nan) if close is not None else None

    # === 基本 ===
    # ATR（事前計算済みの場合はスキップ）
    for w in [10, 20, 40, 50]:
        atr_col = f"atr{w}"
        if atr_col in df.columns:
            continue  # 事前計算済みなのでスキップ
        if close is None or high is None or low is None:
            df[atr_col] = np.nan
            continue
        try:
            df[atr_col] = AverageTrueRange(high, low, close, window=w).average_true_range()
        except Exception:
            df[atr_col] = np.nan

    # SMA（事前計算済みの場合はスキップ）
    for w in [25, 50, 100, 150, 200]:
        sma_col = f"sma{w}"
        if sma_col in df.columns:
            continue  # 事前計算済みなのでスキップ
        if close is None:
            df[sma_col] = np.nan
            continue
        try:
            df[sma_col] = SMAIndicator(close, window=w).sma_indicator()
        except Exception:
            df[sma_col] = np.nan

    # ROC（事前計算済みの場合はスキップ）
    if "roc200" not in df.columns:
        if close is None:
            df["roc200"] = np.nan
        else:
            try:
                df["roc200"] = ROCIndicator(close, window=200).roc()
            except Exception:
                df["roc200"] = np.nan

    # RSI（事前計算済みの場合はスキップ）
    for w in [3, 4]:
        rsi_col = f"rsi{w}"
        if rsi_col in df.columns:
            continue  # 事前計算済みなのでスキップ
        if close is None:
            df[rsi_col] = np.nan
            continue
        try:
            df[rsi_col] = RSIIndicator(close, window=w).rsi()
        except Exception:
            df[rsi_col] = np.nan

    # ADX（事前計算済みの場合はスキップ）
    for w in [7]:
        adx_col = f"adx{w}"
        if adx_col in df.columns:
            continue  # 事前計算済みなのでスキップ
        if close is None or high is None or low is None:
            df[adx_col] = np.nan
            continue
        try:
            df[adx_col] = ADXIndicator(high, low, close, window=w).adx()
        except Exception:
            df[adx_col] = np.nan

    # 売買代金（事前計算済みの場合はスキップ）
    for w in [20, 50]:
        dv_col = f"dollarvolume{w}"
        if dv_col in df.columns:
            continue  # 事前計算済みなのでスキップ
        if close is None or volume is None:
            df[dv_col] = np.nan
            continue
        try:
            df[dv_col] = (close * volume).rolling(window=w).mean()
        except Exception:
            df[dv_col] = np.nan

    # 平均出来高（事前計算済みの場合はスキップ）
    for w in [50]:
        avg_vol_col = f"avgvolume{w}"
        if avg_vol_col in df.columns:
            continue  # 事前計算済みなのでスキップ
        if volume is None:
            df[avg_vol_col] = np.nan
            continue
        try:
            df[avg_vol_col] = volume.rolling(window=w).mean()
        except Exception:
            df[avg_vol_col] = np.nan

    # ATR割合（事前計算済みの場合はスキップ）
    if "atr_ratio" not in df.columns:
        if "atr10" in df and close_nozero is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = df["atr10"].div(close_nozero)
            df["atr_ratio"] = ratio
            df["atr_pct"] = ratio
        else:
            df["atr_ratio"] = np.nan
            df["atr_pct"] = np.nan
    elif "atr_pct" not in df.columns:
        # atr_ratioは存在するがatr_pctがない場合
        df["atr_pct"] = df["atr_ratio"]

    # その他戦略固有（事前計算済みの場合はスキップ）
    if close is not None:
        # Return 3D（事前計算済みの場合はスキップ）
        if "return_3d" not in df.columns:
            try:
                df["return_3d"] = close.pct_change(3)
            except Exception:
                df["return_3d"] = np.nan

        # Return 6D（事前計算済みの場合はスキップ）
        if "return_6d" not in df.columns:
            try:
                df["return_6d"] = close.pct_change(6)
            except Exception:
                df["return_6d"] = np.nan

        # Return Pct（事前計算済みの場合はスキップ）
        if "return_pct" not in df.columns:
            try:
                df["return_pct"] = close.pct_change()
            except Exception:
                df["return_pct"] = np.nan

        # UpTwoDays（事前計算済みの場合はスキップ）
        if "uptwodays" not in df.columns:
            try:
                up_two = close.gt(close.shift(1)) & close.shift(1).gt(close.shift(2))
                df["uptwodays"] = up_two.fillna(False).astype(bool)
            except Exception:
                df["uptwodays"] = False

        # TwoDayUpは常にUpTwoDaysと同じ値
        if "twodayup" not in df.columns:
            df["twodayup"] = df["uptwodays"]

        # Drop3D（事前計算済みの場合はスキップ）
        if "drop3d" not in df.columns:
            try:
                df["drop3d"] = -(close.pct_change(3))
            except Exception:
                df["drop3d"] = np.nan

        # HV50（事前計算済みの場合はスキップ）
        if "hv50" not in df.columns:
            if close_nozero is not None:
                try:
                    log_ret = np.log(close_nozero / close_nozero.shift(1))
                    df["hv50"] = log_ret.rolling(window=50).std() * np.sqrt(252) * 100
                except Exception:
                    df["hv50"] = np.nan
            else:
                df["hv50"] = np.nan

        # Min 50（事前計算済みの場合はスキップ）
        if "min_50" not in df.columns:
            try:
                df["min_50"] = close.rolling(window=50).min()
            except Exception:
                df["min_50"] = np.nan

        # Max 70（事前計算済みの場合はスキップ）
        if "max_70" not in df.columns:
            try:
                df["max_70"] = close.rolling(window=70).max()
            except Exception:
                df["max_70"] = np.nan
    else:
        # closeがNoneの場合でも、既存の値がある場合は保持
        if "return_3d" not in df.columns:
            df["return_3d"] = np.nan
        if "return_6d" not in df.columns:
            df["return_6d"] = np.nan
        if "return_pct" not in df.columns:
            df["return_pct"] = np.nan
        if "uptwodays" not in df.columns:
            df["uptwodays"] = False
        if "twodayup" not in df.columns:
            df["twodayup"] = df.get("uptwodays", False)
        if "drop3d" not in df.columns:
            df["drop3d"] = np.nan
        if "hv50" not in df.columns:
            df["hv50"] = np.nan
        if "min_50" not in df.columns:
            df["min_50"] = np.nan
        if "max_70" not in df.columns:
            df["max_70"] = np.nan

    return df


def add_indicators_batch(data_dict: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    複数シンボルのDataFrameに対して指標を一括計算する。
    Phase2ボトルネック解消のため、重複計算を削減し、
    メモリ効率とベクトル化を重視した実装。

    Args:
        data_dict: {symbol: DataFrame} の辞書

    Returns:
        指標付きDataFrameの辞書
    """
    if not data_dict:
        return {}

    result = {}

    # 各シンボルの指標を並列計算可能だが、
    # メモリ効率のため逐次処理しつつ最適化
    for symbol, df in data_dict.items():
        if df is None or df.empty:
            continue

        try:
            # メモリ効率のため、必要最小限のコピーのみ
            df_with_indicators = _add_indicators_optimized(df)
            if df_with_indicators is not None:
                result[symbol] = df_with_indicators
        except Exception:
            # 失敗時は元のadd_indicatorsにフォールバック
            try:
                result[symbol] = add_indicators(df)
            except Exception:
                # それでも失敗なら元のDataFrameを保持
                result[symbol] = df

    return result


def _add_indicators_optimized(df: pd.DataFrame) -> pd.DataFrame:
    """
    最適化された指標計算（メモリ効率・ベクトル化重視）
    """

    if df is None or df.empty:
        return df

    # 元DataFrameを変更せずに作業用コピー作成
    work = df.copy()

    # 基本カラムの取得と型最適化
    price_cols = {}
    for col_name, standard_name in [
        ("Close", "close"),
        ("High", "high"),
        ("Low", "low"),
        ("Open", "open"),
        ("Volume", "volume"),
    ]:
        if col_name in work.columns:
            try:
                # 数値型への変換を一括実行
                series = pd.to_numeric(work[col_name], errors="coerce")
                price_cols[standard_name] = series
            except Exception:
                price_cols[standard_name] = None
        else:
            price_cols[standard_name] = None

    close = price_cols["close"]
    high = price_cols["high"]
    low = price_cols["low"]
    volume = price_cols["volume"]

    # 0を除去した終値（HV計算用）
    close_nozero = close.replace(0, np.nan) if close is not None else None

    # === 効率化された指標計算 ===

    # ATR計算（複数ウィンドウを一括）
    if close is not None and high is not None and low is not None:
        try:
            from ta.volatility import AverageTrueRange

            atr_calculator = AverageTrueRange(high, low, close, window=10)  # 基準を10で作成
            base_atr = atr_calculator.average_true_range()

            # 10期間ATRから他期間を効率計算
            for w in [10, 20, 40, 50]:
                if w == 10:
                    work[f"atr{w}"] = base_atr
                else:
                    try:
                        # 真の値幅を再利用
                        true_range = np.maximum(
                            high - low,
                            np.maximum(np.abs(high - close.shift(1)), np.abs(low - close.shift(1))),
                        )
                        work[f"atr{w}"] = true_range.rolling(window=w, min_periods=1).mean()
                    except Exception:
                        work[f"atr{w}"] = np.nan
        except Exception:
            for w in [10, 20, 40, 50]:
                work[f"atr{w}"] = np.nan
    else:
        for w in [10, 20, 40, 50]:
            work[f"atr{w}"] = np.nan

    # SMA計算（ベクトル化）
    if close is not None:
        try:
            for w in [25, 50, 100, 150, 200]:
                work[f"sma{w}"] = close.rolling(window=w, min_periods=1).mean()
        except Exception:
            for w in [25, 50, 100, 150, 200]:
                work[f"sma{w}"] = np.nan
    else:
        for w in [25, 50, 100, 150, 200]:
            work[f"sma{w}"] = np.nan

    # ROC, RSI, ADX計算
    if close is not None:
        try:
            work["roc200"] = close.pct_change(200) * 100  # ta.momentumより高速
        except Exception:
            work["roc200"] = np.nan

        # RSI計算（3, 4期間）
        for w in [3, 4]:
            try:
                from ta.momentum import RSIIndicator

                work[f"rsi{w}"] = RSIIndicator(close, window=w).rsi()
            except Exception:
                work[f"rsi{w}"] = np.nan
    else:
        work["roc200"] = np.nan
        for w in [3, 4]:
            work[f"rsi{w}"] = np.nan

    # ADX計算
    if close is not None and high is not None and low is not None:
        try:
            from ta.trend import ADXIndicator

            work["adx7"] = ADXIndicator(high, low, close, window=7).adx()
        except Exception:
            work["adx7"] = np.nan
    else:
        work["adx7"] = np.nan

    # 売買代金・出来高関連
    if close is not None and volume is not None:
        try:
            dollar_volume = close * volume
            for w in [20, 50]:
                work[f"dollarvolume{w}"] = dollar_volume.rolling(window=w, min_periods=1).mean()
        except Exception:
            for w in [20, 50]:
                work[f"dollarvolume{w}"] = np.nan

        try:
            work["avgvolume50"] = volume.rolling(window=50, min_periods=1).mean()
        except Exception:
            work["avgvolume50"] = np.nan
    else:
        for w in [20, 50]:
            work[f"dollarvolume{w}"] = np.nan
        work["avgvolume50"] = np.nan

    # ATR比率計算
    if "atr10" in work and close_nozero is not None:
        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = work["atr10"] / close_nozero
            work["atr_ratio"] = ratio
            work["atr_pct"] = ratio
        except Exception:
            work["atr_ratio"] = np.nan
            work["atr_pct"] = np.nan
    else:
        work["atr_ratio"] = np.nan
        work["atr_pct"] = np.nan

    # その他の戦略固有指標
    if close is not None:
        try:
            work["return_3d"] = close.pct_change(3)
            work["return_6d"] = close.pct_change(6)

            # 2日連続上昇フラグ
            up_two = close.gt(close.shift(1)) & close.shift(1).gt(close.shift(2))
            work["uptwodays"] = up_two.fillna(False)
            work["twodayup"] = work["uptwodays"]

            # ローリング統計
            work["min_50"] = close.rolling(window=50, min_periods=1).min()
            work["max_70"] = close.rolling(window=70, min_periods=1).max()

        except Exception:
            work["return_3d"] = np.nan
            work["return_6d"] = np.nan
            work["uptwodays"] = False
            work["twodayup"] = False
            work["min_50"] = np.nan
            work["max_70"] = np.nan

        # HV50計算（対数収益率ベース）
        if close_nozero is not None:
            try:
                log_ret = np.log(close_nozero / close_nozero.shift(1))
                work["hv50"] = log_ret.rolling(window=50, min_periods=1).std() * np.sqrt(252) * 100
            except Exception:
                work["hv50"] = np.nan
        else:
            work["hv50"] = np.nan
    else:
        for col in ["return_3d", "return_6d", "min_50", "max_70", "hv50"]:
            work[col] = np.nan
        work["uptwodays"] = False
        work["twodayup"] = False

    return work
