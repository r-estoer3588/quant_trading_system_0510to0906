"""今日のシグナル抽出パイプラインで用いる共通フィルター/条件判定ユーティリティ。

run_all_systems_today.py からロジックを分離（責務分割）:
  - 価格/出来高/ATR 比率などの低レベル指標抽出ヘルパ
  - System1〜System6 の事前フィルター条件
  - filter_systemX 関数群

注意: 公開 API (関数シグネチャ・戻り値) は run_all_systems_today.py と互換。
      依存: pandas のみ（外部 I/O なし / CacheManager 依存なし）
"""
from __future__ import annotations

from typing import Any, Sequence
import pandas as pd
from core.system5 import DEFAULT_ATR_PCT_THRESHOLD  # 振る舞い維持: 元スクリプトの閾値

__all__ = [
    "_pick_series",
    "_last_scalar",
    "_calc_dollar_volume_from_series",
    "_calc_average_volume_from_series",
    "_resolve_atr_ratio",
    "_system1_conditions",
    "_system2_conditions",
    "_system3_conditions",
    "_system4_conditions",
    "_system5_conditions",
    "_system6_conditions",
    "filter_system1",
    "filter_system2",
    "filter_system3",
    "filter_system4",
    "filter_system5",
    "filter_system6",
]

# ----------------------------- 基本ヘルパ ----------------------------- #

def _pick_series(df: pd.DataFrame, names: Sequence[str]):
    """候補列名リストから最初に存在する列を Series として返す。

    ・DataFrame（二重階層化）であれば先頭列を抽出
    ・数値化できる場合は to_numeric で変換（エラーは握りつぶし）
    見つからなければ None
    """
    try:
        for nm in names:
            if nm in df.columns:
                s = df[nm]
                if isinstance(s, pd.DataFrame):  # 2D -> 1D へ簡約
                    try:
                        if getattr(s, "ndim", None) == 2 and hasattr(s, "iloc"):
                            s = s.iloc[:, 0]
                        else:
                            cols = list(s.columns or [])
                            if cols:
                                s = s[cols[0]]
                            else:
                                continue
                    except Exception:
                        continue
                try:
                    s = pd.to_numeric(s, errors="coerce")
                except Exception:
                    pass
                return s
    except Exception:
        pass
    return None


def _last_scalar(series):
    """終値などの Series 末尾スカラを float で返す (NaN / 空は None)。"""
    try:
        if series is None:
            return None
        s2 = series.dropna()
        if s2.empty:
            return None
        return float(s2.iloc[-1])
    except Exception:
        return None


def _calc_dollar_volume_from_series(close_series, volume_series, window: int) -> float | None:
    if close_series is None or volume_series is None:
        return None
    try:
        product = close_series * volume_series
    except Exception:
        return None
    try:
        tail = product.tail(window) if hasattr(product, "tail") else product
    except Exception:
        tail = product
    try:
        tail = pd.to_numeric(tail, errors="coerce")
    except Exception:
        pass
    try:
        if hasattr(tail, "dropna"):
            tail = tail.dropna()
        if getattr(tail, "empty", False):
            return None
        return float(tail.mean())
    except Exception:
        return None


def _calc_average_volume_from_series(volume_series, window: int) -> float | None:
    if volume_series is None:
        return None
    try:
        tail = volume_series.tail(window) if hasattr(volume_series, "tail") else volume_series
    except Exception:
        tail = volume_series
    try:
        tail = pd.to_numeric(tail, errors="coerce")
    except Exception:
        pass
    try:
        if hasattr(tail, "dropna"):
            tail = tail.dropna()
        if getattr(tail, "empty", False):
            return None
        return float(tail.mean())
    except Exception:
        return None


def _resolve_atr_ratio(
    df: pd.DataFrame,
    close_series=None,
    last_close: float | None = None,
) -> float | None:
    """ATR_Ratio / ATR_Pct -> ATR10/20 -> High-Low から平均 True Range 比率を推定。"""
    ratio_series = _pick_series(df, ["ATR_Ratio", "ATR_Pct"])
    ratio_val = _last_scalar(ratio_series)
    if ratio_val is not None:
        return ratio_val

    atr_series = _pick_series(df, ["ATR10", "ATR20"])
    atr_val = _last_scalar(atr_series)
    if atr_val is None:
        high_series = _pick_series(df, ["High", "high"])
        low_series = _pick_series(df, ["Low", "low"])
        if high_series is not None and low_series is not None:
            try:
                tr = high_series - low_series
                if hasattr(tr, "dropna"):
                    tr = tr.dropna()
                tr_tail = tr.tail(10) if hasattr(tr, "tail") else tr
                if getattr(tr_tail, "empty", False):
                    atr_val = None
                else:
                    atr_val = float(tr_tail.mean())
            except Exception:
                atr_val = None
    if atr_val is None:
        return None

    if last_close is None:
        try:
            close_series = close_series or _pick_series(df, ["Close", "close"])
        except Exception:
            close_series = None
        last_close = _last_scalar(close_series)
    try:
        if last_close is None:
            return None
        close_val = float(last_close)
        if close_val == 0:
            return None
    except Exception:
        return None

    try:
        return float(atr_val) / close_val
    except Exception:
        return None


# ------------------------- System 条件関数群 ------------------------- #

def _system1_conditions(df: pd.DataFrame) -> tuple[bool, bool]:
    close_series = _pick_series(df, ["Close", "close"])
    last_close = _last_scalar(close_series)
    price_ok = bool(last_close is not None and last_close >= 5)

    dv_series = _pick_series(df, ["DollarVolume20"])
    dv20 = _last_scalar(dv_series)
    if dv20 is None:
        volume_series = _pick_series(df, ["Volume", "volume"])
        dv20 = _calc_dollar_volume_from_series(close_series, volume_series, 20)
    dv_ok = bool(dv20 is not None and dv20 >= 50_000_000)

    return price_ok, dv_ok


def _system2_conditions(df: pd.DataFrame) -> tuple[bool, bool, bool]:
    close_series = _pick_series(df, ["Close", "close"])
    last_close = _last_scalar(close_series)
    price_ok = bool(last_close is not None and last_close >= 5)

    dv_series = _pick_series(df, ["DollarVolume20"])
    dv20 = _last_scalar(dv_series)
    if dv20 is None:
        volume_series = _pick_series(df, ["Volume", "volume"])
        dv20 = _calc_dollar_volume_from_series(close_series, volume_series, 20)
    dv_ok = bool(dv20 is not None and dv20 >= 25_000_000)

    atr_ratio = _resolve_atr_ratio(df, close_series, last_close)
    atr_ok = bool(atr_ratio is not None and atr_ratio >= 0.03)

    return price_ok, dv_ok, atr_ok


def _system3_conditions(df: pd.DataFrame) -> tuple[bool, bool, bool]:
    low_series = _pick_series(df, ["Low", "low"])
    low_val = _last_scalar(low_series)
    low_ok = bool(low_val is not None and low_val >= 1)

    av_series = _pick_series(df, ["AvgVolume50"])
    av_val = _last_scalar(av_series)
    if av_val is None:
        volume_series = _pick_series(df, ["Volume", "volume"])
        av_val = _calc_average_volume_from_series(volume_series, 50)
    av_ok = bool(av_val is not None and av_val >= 1_000_000)

    atr_ratio = _resolve_atr_ratio(df)
    atr_ok = bool(atr_ratio is not None and atr_ratio >= 0.05)

    return low_ok, av_ok, atr_ok


def _system4_conditions(df: pd.DataFrame) -> tuple[bool, bool]:
    close_series = _pick_series(df, ["Close", "close"])
    volume_series = _pick_series(df, ["Volume", "volume"])
    dv_series = _pick_series(df, ["DollarVolume50"])
    dv50 = _last_scalar(dv_series)
    if dv50 is None:
        dv50 = _calc_dollar_volume_from_series(close_series, volume_series, 50)
    dv_ok = bool(dv50 is not None and dv50 > 100_000_000)

    hv_series = _pick_series(df, ["HV50"])
    hv_val = _last_scalar(hv_series)
    hv_ok = bool(hv_val is not None and 10 <= hv_val <= 40)

    return dv_ok, hv_ok


def _system5_conditions(df: pd.DataFrame) -> tuple[bool, bool, bool]:
    volume_series = _pick_series(df, ["Volume", "volume"])
    av_series = _pick_series(df, ["AvgVolume50"])
    av_val = _last_scalar(av_series)
    if av_val is None:
        av_val = _calc_average_volume_from_series(volume_series, 50)
    av_ok = bool(av_val is not None and av_val > 500_000)

    close_series = _pick_series(df, ["Close", "close"])
    dv_series = _pick_series(df, ["DollarVolume50"])
    dv50 = _last_scalar(dv_series)
    if dv50 is None:
        dv50 = _calc_dollar_volume_from_series(close_series, volume_series, 50)
    dv_ok = bool(dv50 is not None and dv50 > 2_500_000)

    atr_series = _pick_series(df, ["ATR_Pct", "ATR_Ratio"])
    atr_val = _last_scalar(atr_series)
    if atr_val is None:
        atr_val = _resolve_atr_ratio(df, close_series)
    atr_ok = bool(atr_val is not None and atr_val > DEFAULT_ATR_PCT_THRESHOLD)

    return av_ok, dv_ok, atr_ok


def _system6_conditions(df: pd.DataFrame) -> tuple[bool, bool]:
    low_series = _pick_series(df, ["Low", "low"])
    low_val = _last_scalar(low_series)
    low_ok = bool(low_val is not None and low_val >= 5)

    close_series = _pick_series(df, ["Close", "close"])
    volume_series = _pick_series(df, ["Volume", "volume"])
    dv_series = _pick_series(df, ["DollarVolume50"])
    dv50 = _last_scalar(dv_series)
    if dv50 is None:
        dv50 = _calc_dollar_volume_from_series(close_series, volume_series, 50)
    dv_ok = bool(dv50 is not None and dv50 > 10_000_000)

    return low_ok, dv_ok


# ------------------------- filter_systemX 群 ------------------------- #

def filter_system1(symbols, data, stats: dict[str, int] | None = None):
    result: list[str] = []
    total = len(symbols or [])
    price_pass = 0
    dv_pass = 0
    for sym in symbols or []:
        df = data.get(sym)
        if df is None or df.empty:
            continue
        price_ok, dv_ok = _system1_conditions(df)
        if not price_ok:
            continue
        price_pass += 1
        if not dv_ok:
            continue
        dv_pass += 1
        result.append(sym)
    if stats is not None:
        stats["total"] = total
        stats["price_pass"] = price_pass
        stats["dv_pass"] = dv_pass
    return result


def filter_system2(symbols, data, stats: dict[str, int] | None = None):
    result: list[str] = []
    total = len(symbols or [])
    price_pass = 0
    dv_pass = 0
    atr_pass = 0
    for sym in symbols or []:
        df = data.get(sym)
        if df is None or df.empty:
            continue
        price_ok, dv_ok, atr_ok = _system2_conditions(df)
        if not price_ok:
            continue
        price_pass += 1
        if not dv_ok:
            continue
        dv_pass += 1
        if not atr_ok:
            continue
        atr_pass += 1
        result.append(sym)
    if stats is not None:
        stats["total"] = total
        stats["price_pass"] = price_pass
        stats["dv_pass"] = dv_pass
        stats["atr_pass"] = atr_pass
    return result


def filter_system3(symbols, data, stats: dict[str, int] | None = None):
    result: list[str] = []
    total = len(symbols or [])
    low_pass = 0
    av_pass = 0
    atr_pass = 0
    for sym in symbols or []:
        df = data.get(sym)
        if df is None or df.empty:
            continue
        low_ok, av_ok, atr_ok = _system3_conditions(df)
        if not low_ok:
            continue
        low_pass += 1
        if not av_ok:
            continue
        av_pass += 1
        if not atr_ok:
            continue
        atr_pass += 1
        result.append(sym)
    if stats is not None:
        stats["total"] = total
        stats["low_pass"] = low_pass
        stats["avgvol_pass"] = av_pass
        stats["atr_pass"] = atr_pass
    return result


def filter_system4(symbols, data, stats: dict[str, int] | None = None):
    result: list[str] = []
    total = len(symbols or [])
    dv_pass = 0
    hv_pass = 0
    for sym in symbols or []:
        df = data.get(sym)
        if df is None or df.empty:
            continue
        dv_ok, hv_ok = _system4_conditions(df)
        if not dv_ok:
            continue
        dv_pass += 1
        if not hv_ok:
            continue
        hv_pass += 1
        result.append(sym)
    if stats is not None:
        stats["total"] = total
        stats["dv_pass"] = dv_pass
        stats["hv_pass"] = hv_pass
    return result


def filter_system5(symbols, data, stats: dict[str, int] | None = None):
    result: list[str] = []
    total = len(symbols or [])
    av_pass = 0
    dv_pass = 0
    atr_pass = 0
    for sym in symbols or []:
        df = data.get(sym)
        if df is None or df.empty:
            continue
        av_ok, dv_ok, atr_ok = _system5_conditions(df)
        if not av_ok:
            continue
        av_pass += 1
        if not dv_ok:
            continue
        dv_pass += 1
        if not atr_ok:
            continue
        atr_pass += 1
        result.append(sym)
    if stats is not None:
        stats["total"] = total
        stats["avgvol_pass"] = av_pass
        stats["dv_pass"] = dv_pass
        stats["atr_pass"] = atr_pass
    return result


def filter_system6(symbols, data, stats: dict[str, int] | None = None):
    result: list[str] = []
    total = len(symbols or [])
    low_pass = 0
    dv_pass = 0
    for sym in symbols or []:
        df = data.get(sym)
        if df is None or df.empty:
            continue
        low_ok, dv_ok = _system6_conditions(df)
        if not low_ok:
            continue
        low_pass += 1
        if not dv_ok:
            continue
        dv_pass += 1
        result.append(sym)
    if stats is not None:
        stats["total"] = total
        stats["low_pass"] = low_pass
        stats["dv_pass"] = dv_pass
    return result
