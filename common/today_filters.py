"""今日のシグナル抽出パイプラインで用いる共通フィルター/条件判定ユーティリティ。

run_all_systems_today.py からロジックを分離（責務分割）:
  - 価格/出来高/ATR 比率などの低レベル指標抽出ヘルパ
  - System1〜System6 の事前フィルター条件
  - filter_systemX 関数群

注意: 公開 API (関数シグネチャ・戻り値) は run_all_systems_today.py と互換。
      依存: pandas のみ（外部 I/O なし / CacheManager 依存なし）
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import pandas as pd

from core.system5 import DEFAULT_ATR_PCT_THRESHOLD  # 振る舞い維持: 元スクリプトの閾値
from core.system6 import (  # System6 のフィルター閾値と HV 範囲を共有
    HV50_BOUNDS_FRACTION,
    HV50_BOUNDS_PERCENT,
    MIN_DOLLAR_VOLUME_50,
    MIN_PRICE,
)

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


# ------------------------------------------------------------------
# Optional debug logging (FILTER_DEBUG=1) to trace pass counts quickly
# ノイズ抑制のため環境変数で明示的に有効化された時のみ出力。
# run_all_systems_today の log_callback が未配線でも print フォールバックで可視化可能。
# ------------------------------------------------------------------
def _filter_debug_enabled() -> bool:
    try:
        return os.getenv("FILTER_DEBUG", "0") == "1"
    except Exception:
        return False


def _emit_filter_debug(system_tag: str, stats: dict[str, int] | None, final_list_len: int) -> None:
    if not _filter_debug_enabled():
        return
    try:
        # 初回ヘッダ: 環境変数が有効であることを示す（system_tag 毎に1回）
        marker_name = f"_FDBG_INIT_{system_tag}"
        if marker_name not in globals():  # type: ignore
            try:
                globals()[marker_name] = True  # type: ignore
                print(f"[FDBG {system_tag}] debug-enabled FILTER_DEBUG=1 (init)")
            except Exception:
                pass
        st = stats or {}
        # 共通で出すキー順を簡易に整える（存在するものだけ）
        order = [
            "total",
            "price_pass",
            "low_pass",
            "avgvol_pass",
            "dv_pass",
            "hv_pass",
            "atr_pass",
            # reason counters (will appear only if collected)
            "atr_missing",
            "atr_below",
            "hv_missing",
            "hv_range_fail",
        ]
        parts: list[str] = []
        label_map = {
            "atr_missing": "atrMiss",
            "atr_below": "atrBelow",
            "hv_missing": "hvMiss",
            "hv_range_fail": "hvRange",
        }
        for k in order:
            if k in st:
                base = label_map.get(k, k.split("_")[0])
                parts.append(f"{base}={st[k]}")
        parts.append(f"final={final_list_len}")
        ratio_parts: list[str] = []
        try:
            total = float(st.get("total", 0) or 0)
            if total > 0:
                for k in [
                    "price_pass",
                    "low_pass",
                    "avgvol_pass",
                    "dv_pass",
                    "hv_pass",
                    "atr_pass",
                ]:
                    if k in st and st[k] >= 0:
                        ratio_parts.append(f"{k.split('_')[0]}%={st[k]/total*100:.1f}")
        except Exception:
            pass
        msg = f"[FDBG {system_tag}] " + " ".join(parts)
        if ratio_parts:
            msg += " | " + " ".join(ratio_parts)
        print(msg)
    except Exception:
        pass


# ----------------------------- 基本ヘルパ ----------------------------- #


def _pick_series(df: pd.DataFrame, names: Sequence[str]):
    """候補列名から Series を返す。大小文字やアンダースコア差異を吸収する正規化検索を追加。

    優先順:
      1. 与えられた順 (完全一致)
      2. 正規化一致 (lower + '_' 除去)
    見つかったら数値化 (coerce)。失敗時 None。
    """
    if df is None:
        return None
    try:
        cols = list(df.columns)
        if not cols:
            return None

        # 正規化マップ: key = lower + アンダースコア除去
        def norm(s: str):
            return s.replace("_", "").lower()

        norm_map: dict[str, str] = {}
        for c in cols:
            n = norm(c)
            # 先勝ち（最初の列を保持）
            if n not in norm_map:
                norm_map[n] = c

        # 1) 完全一致探索
        for nm in names:
            if nm in df.columns:
                s_any = df[nm]
                try:
                    if isinstance(s_any, pd.DataFrame) and getattr(s_any, "ndim", None) == 2:
                        # 先頭列のみ使用
                        s_any = s_any.iloc[:, 0]  # type: ignore[index]
                except Exception:
                    pass
                try:
                    s_any = pd.to_numeric(s_any, errors="coerce")
                except Exception:
                    pass
                return s_any

        # 2) 正規化一致
        for nm in names:
            key = norm(nm)
            real = norm_map.get(key)
            if real is None:
                continue
            try:
                s_any = df[real]
                if isinstance(s_any, pd.DataFrame) and getattr(s_any, "ndim", None) == 2:
                    s_any = s_any.iloc[:, 0]  # type: ignore[index]
                try:
                    s_any = pd.to_numeric(s_any, errors="coerce")
                except Exception:
                    pass
                return s_any
            except Exception:
                continue
    except Exception:
        return None
    return None


def _last_non_nan(series, lookback: int = 5):
    """末尾から最大 lookback 件遡って最初に非 NaN の値を返す。なければ None。

    指標末尾が一時的に NaN（インジ計算直後 / 欠損穴埋め前）でも直近の実値を利用して
    不要な False 判定を避けるためのフォールバック。
    """
    try:
        if series is None:
            return None
        tail = series.tail(lookback) if hasattr(series, "tail") else series
        if hasattr(tail, "tolist"):
            values = tail.tolist()
        else:
            values = list(tail)
        for v in reversed(values):
            try:
                if v == v:  # not NaN
                    return float(v)
            except Exception:
                continue
    except Exception:
        return None
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


def _calc_hv50_from_series(close_series) -> float | None:
    if close_series is None:
        return None
    try:
        returns = pd.Series(close_series).pct_change()
    except Exception:
        return None
    try:
        hv = returns.rolling(50).std() * (252**0.5) * 100
    except Exception:
        return None
    hv_val = _last_scalar(hv)
    if hv_val is None:
        hv_val = _last_non_nan(hv)
    return hv_val


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
    close_series = _pick_series(df, ["Close", "close", "CLOSE"])
    last_close = _last_scalar(close_series)
    if last_close is None:
        last_close = _last_non_nan(close_series)
    price_ok = bool(last_close is not None and last_close >= 5)

    dv_series = _pick_series(df, ["DollarVolume20", "dollarvolume20", "dollar_volume20", "DV20"])
    dv20 = _last_scalar(dv_series)
    if dv20 is None:
        volume_series = _pick_series(df, ["Volume", "volume"])
        dv20 = _calc_dollar_volume_from_series(close_series, volume_series, 20)
        if dv20 is None:
            dv20 = _last_non_nan(dv_series)
    dv_ok = bool(dv20 is not None and dv20 >= 50_000_000)

    return price_ok, dv_ok


def _system2_conditions(df: pd.DataFrame) -> tuple[bool, bool, bool]:
    close_series = _pick_series(df, ["Close", "close", "CLOSE"])
    last_close = _last_scalar(close_series)
    if last_close is None:
        last_close = _last_non_nan(close_series)
    price_ok = bool(last_close is not None and last_close >= 5)

    dv_series = _pick_series(df, ["DollarVolume20", "dollarvolume20", "dollar_volume20", "DV20"])
    dv20 = _last_scalar(dv_series)
    if dv20 is None:
        volume_series = _pick_series(df, ["Volume", "volume"])
        dv20 = _calc_dollar_volume_from_series(close_series, volume_series, 20)
        if dv20 is None:
            dv20 = _last_non_nan(dv_series)
    dv_ok = bool(dv20 is not None and dv20 >= 25_000_000)

    atr_ratio = _resolve_atr_ratio(df, close_series, last_close)
    atr_ok = bool(atr_ratio is not None and atr_ratio >= 0.03)
    if _filter_debug_enabled():  # 理由分類カウンタ用タグを添付（外部副作用なし）
        reason = None
        if atr_ok:
            reason = "pass"
        elif atr_ratio is None:
            reason = "atr_missing"
        else:
            reason = "atr_below"
        (df.attrs.setdefault("_fdbg_reasons2", []).append(reason) if hasattr(df, "attrs") else None)

    return price_ok, dv_ok, atr_ok


def _system3_conditions(df: pd.DataFrame) -> tuple[bool, bool, bool]:
    low_series = _pick_series(df, ["Low", "low"])
    low_val = _last_scalar(low_series)
    if low_val is None:
        low_val = _last_non_nan(low_series)
    low_ok = bool(low_val is not None and low_val >= 1)

    av_series = _pick_series(df, ["AvgVolume50", "avgvolume50", "avg_volume50", "AVGVOL50"])
    av_val = _last_scalar(av_series)
    if av_val is None:
        volume_series = _pick_series(df, ["Volume", "volume"])
        av_val = _calc_average_volume_from_series(volume_series, 50)
        if av_val is None:
            av_val = _last_non_nan(av_series)
    av_ok = bool(av_val is not None and av_val >= 1_000_000)

    atr_ratio = _resolve_atr_ratio(df)
    atr_ok = bool(atr_ratio is not None and atr_ratio >= 0.05)
    if _filter_debug_enabled():
        reason = None
        if atr_ok:
            reason = "pass"
        elif atr_ratio is None:
            reason = "atr_missing"
        else:
            reason = "atr_below"
        (df.attrs.setdefault("_fdbg_reasons3", []).append(reason) if hasattr(df, "attrs") else None)

    return low_ok, av_ok, atr_ok


def _system4_conditions(df: pd.DataFrame) -> tuple[bool, bool]:
    close_series = _pick_series(df, ["Close", "close", "CLOSE"])
    volume_series = _pick_series(df, ["Volume", "volume", "VOLUME"])
    dv_series = _pick_series(df, ["DollarVolume50", "dollarvolume50", "dollar_volume50", "DV50"])
    dv50 = _last_scalar(dv_series)
    if dv50 is None:
        dv50 = _calc_dollar_volume_from_series(close_series, volume_series, 50)
        if dv50 is None:
            dv50 = _last_non_nan(dv_series)
    dv_ok = bool(dv50 is not None and dv50 > 100_000_000)

    hv_series = _pick_series(df, ["HV50", "hv50", "HV_50"])
    hv_val = _last_scalar(hv_series)
    if hv_val is None:
        hv_val = _last_non_nan(hv_series)
    hv_ok = bool(hv_val is not None and 10 <= hv_val <= 40)
    if _filter_debug_enabled():
        reason = None
        if hv_ok:
            reason = "pass"
        elif hv_val is None:
            reason = "hv_missing"
        else:
            reason = "hv_range_fail"
        (df.attrs.setdefault("_fdbg_reasons4", []).append(reason) if hasattr(df, "attrs") else None)

    return dv_ok, hv_ok


def _system5_conditions(df: pd.DataFrame) -> tuple[bool, bool, bool]:
    volume_series = _pick_series(df, ["Volume", "volume", "VOLUME"])
    av_series = _pick_series(df, ["AvgVolume50", "avgvolume50", "avg_volume50", "AVGVOL50"])
    av_val = _last_scalar(av_series)
    if av_val is None:
        av_val = _calc_average_volume_from_series(volume_series, 50)
        if av_val is None:
            av_val = _last_non_nan(av_series)
    av_ok = bool(av_val is not None and av_val > 500_000)

    close_series = _pick_series(df, ["Close", "close", "CLOSE"])
    dv_series = _pick_series(df, ["DollarVolume50", "dollarvolume50", "dollar_volume50", "DV50"])
    dv50 = _last_scalar(dv_series)
    if dv50 is None:
        dv50 = _calc_dollar_volume_from_series(close_series, volume_series, 50)
        if dv50 is None:
            dv50 = _last_non_nan(dv_series)
    dv_ok = bool(dv50 is not None and dv50 > 2_500_000)

    atr_series = _pick_series(df, ["ATR_Pct", "ATR_Ratio"])
    atr_val = _last_scalar(atr_series)
    if atr_val is None:
        atr_val = _resolve_atr_ratio(df, close_series)
    atr_ok = bool(atr_val is not None and atr_val > DEFAULT_ATR_PCT_THRESHOLD)
    if _filter_debug_enabled():
        reason = None
        if atr_ok:
            reason = "pass"
        elif atr_val is None:
            reason = "atr_missing"
        else:
            reason = "atr_below"
        (df.attrs.setdefault("_fdbg_reasons5", []).append(reason) if hasattr(df, "attrs") else None)

    return av_ok, dv_ok, atr_ok


def _system6_conditions(df: pd.DataFrame) -> tuple[bool, bool, bool]:
    low_series = _pick_series(df, ["Low", "low"])
    low_val = _last_scalar(low_series)
    if low_val is None:
        low_val = _last_non_nan(low_series)
    low_ok = bool(low_val is not None and low_val >= MIN_PRICE)

    close_series = _pick_series(df, ["Close", "close", "CLOSE"])
    volume_series = _pick_series(df, ["Volume", "volume", "VOLUME"])
    dv_series = _pick_series(df, ["DollarVolume50", "dollarvolume50", "dollar_volume50", "DV50"])
    dv50 = _last_scalar(dv_series)
    if dv50 is None:
        dv50 = _calc_dollar_volume_from_series(close_series, volume_series, 50)
        if dv50 is None:
            dv50 = _last_non_nan(dv_series)
    dv_ok = bool(dv50 is not None and dv50 > MIN_DOLLAR_VOLUME_50)

    hv_series = _pick_series(df, ["HV50", "hv50", "HV_50"])
    hv_val = _last_scalar(hv_series)
    if hv_val is None:
        hv_val = _last_non_nan(hv_series)
    if hv_val is None:
        hv_val = _calc_hv50_from_series(close_series)
    hv_ok = False
    if hv_val is not None:
        try:
            hv_float = float(hv_val)
        except Exception:
            hv_float = None
        if hv_float is not None:
            hv_ok = bool(
                HV50_BOUNDS_PERCENT[0] <= hv_float <= HV50_BOUNDS_PERCENT[1]
                or HV50_BOUNDS_FRACTION[0] <= hv_float <= HV50_BOUNDS_FRACTION[1]
            )
    if _filter_debug_enabled():
        reason = None
        if hv_ok:
            reason = "pass"
        elif hv_val is None:
            reason = "hv_missing"
        else:
            reason = "hv_range_fail"
        if hasattr(df, "attrs"):
            try:
                df.attrs.setdefault("_fdbg_reasons6", []).append(reason)  # type: ignore[attr-defined]
            except Exception:
                pass

    return low_ok, dv_ok, hv_ok


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
    _emit_filter_debug("system1", stats, len(result))
    return result


def filter_system2(symbols, data, stats: dict[str, int] | None = None):
    result: list[str] = []
    total = len(symbols or [])
    price_pass = 0
    dv_pass = 0
    atr_pass = 0
    atr_missing = 0
    atr_below = 0
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
            # 理由カウンタ: _system2_conditions 内で df.attrs に付与された最新理由を参照
            try:
                reasons = df.attrs.get("_fdbg_reasons2")  # type: ignore[attr-defined]
                if isinstance(reasons, list) and reasons:
                    r = reasons[-1]
                    if r == "atr_missing":
                        atr_missing += 1
                    elif r == "atr_below":
                        atr_below += 1
            except Exception:
                pass
            continue
        atr_pass += 1
        result.append(sym)
    if stats is not None:
        stats["total"] = total
        stats["price_pass"] = price_pass
        stats["dv_pass"] = dv_pass
        stats["atr_pass"] = atr_pass
        if atr_missing or atr_below:
            stats["atr_missing"] = atr_missing
            stats["atr_below"] = atr_below
    _emit_filter_debug("system2", stats, len(result))
    return result


def filter_system3(symbols, data, stats: dict[str, int] | None = None):
    result: list[str] = []
    total = len(symbols or [])
    low_pass = 0
    av_pass = 0
    atr_pass = 0
    atr_missing = 0
    atr_below = 0
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
            try:
                reasons = df.attrs.get("_fdbg_reasons3")  # type: ignore[attr-defined]
                if isinstance(reasons, list) and reasons:
                    r = reasons[-1]
                    if r == "atr_missing":
                        atr_missing += 1
                    elif r == "atr_below":
                        atr_below += 1
            except Exception:
                pass
            continue
        atr_pass += 1
        result.append(sym)
    if stats is not None:
        stats["total"] = total
        stats["low_pass"] = low_pass
        stats["avgvol_pass"] = av_pass
        stats["atr_pass"] = atr_pass
        if atr_missing or atr_below:
            stats["atr_missing"] = atr_missing
            stats["atr_below"] = atr_below
    _emit_filter_debug("system3", stats, len(result))
    return result


def filter_system4(symbols, data, stats: dict[str, int] | None = None):
    result: list[str] = []
    total = len(symbols or [])
    dv_pass = 0
    hv_pass = 0
    hv_missing = 0
    hv_range_fail = 0
    debug_enabled = os.getenv("DEBUG_SYSTEM_FILTERS") == "1"
    debug_limit = 10
    debug_count = 0
    for sym in symbols or []:
        df = data.get(sym)
        if df is None or df.empty:
            continue
        dv_ok, hv_ok = _system4_conditions(df)
        if debug_enabled and debug_count < debug_limit:
            try:
                # 末尾値を直接参照（可能なら）
                dv_series = _pick_series(df, ["DollarVolume50", "dollar_volume50", "DV50"])  # type: ignore[misc]
                dv_val = _last_scalar(dv_series)
                hv_series = _pick_series(df, ["HV50", "hv50", "HV_50"])  # type: ignore[misc]
                hv_val = _last_scalar(hv_series)
            except Exception:
                dv_val = None
                hv_val = None
            print(
                f"[DBG system4] sym={sym} dv_val={dv_val} dv_ok={dv_ok} hv_val={hv_val} hv_ok={hv_ok}"
            )
            debug_count += 1
        if not dv_ok:
            continue
        dv_pass += 1
        if not hv_ok:
            try:
                reasons = df.attrs.get("_fdbg_reasons4")  # type: ignore[attr-defined]
                if isinstance(reasons, list) and reasons:
                    r = reasons[-1]
                    if r == "hv_missing":
                        hv_missing += 1
                    elif r == "hv_range_fail":
                        hv_range_fail += 1
            except Exception:
                pass
            continue
        hv_pass += 1
        result.append(sym)
    if stats is not None:
        stats["total"] = total
        stats["dv_pass"] = dv_pass
        stats["hv_pass"] = hv_pass
        if hv_missing or hv_range_fail:
            stats["hv_missing"] = hv_missing
            stats["hv_range_fail"] = hv_range_fail
    _emit_filter_debug("system4", stats, len(result))
    return result


def filter_system5(symbols, data, stats: dict[str, int] | None = None):
    result: list[str] = []
    total = len(symbols or [])
    av_pass = 0
    dv_pass = 0
    atr_pass = 0
    atr_missing = 0
    atr_below = 0
    debug_enabled = os.getenv("DEBUG_SYSTEM_FILTERS") == "1"
    debug_limit = 10
    debug_count = 0
    for sym in symbols or []:
        df = data.get(sym)
        if df is None or df.empty:
            continue
        av_ok, dv_ok, atr_ok = _system5_conditions(df)
        if debug_enabled and debug_count < debug_limit:
            try:
                av_series = _pick_series(df, ["AvgVolume50", "avgvolume50", "AVGVOL50"])  # type: ignore[misc]
                av_val = _last_scalar(av_series)
                dv_series = _pick_series(df, ["DollarVolume50", "dollar_volume50", "DV50"])  # type: ignore[misc]
                dv_val = _last_scalar(dv_series)
                atr_series = _pick_series(df, ["ATR_Pct", "atr_pct", "ATR_Ratio", "atr_ratio"])  # type: ignore[misc]
                atr_val = _last_scalar(atr_series)
            except Exception:
                av_val = dv_val = atr_val = None
            print(
                f"[DBG system5] sym={sym} av_val={av_val} av_ok={av_ok} dv_val={dv_val} dv_ok={dv_ok} atr_val={atr_val} atr_ok={atr_ok}"
            )
            debug_count += 1
        if not av_ok:
            continue
        av_pass += 1
        if not dv_ok:
            continue
        dv_pass += 1
        if not atr_ok:
            try:
                reasons = df.attrs.get("_fdbg_reasons5")  # type: ignore[attr-defined]
                if isinstance(reasons, list) and reasons:
                    r = reasons[-1]
                    if r == "atr_missing":
                        atr_missing += 1
                    elif r == "atr_below":
                        atr_below += 1
            except Exception:
                pass
            continue
        atr_pass += 1
        result.append(sym)
    if stats is not None:
        stats["total"] = total
        stats["avgvol_pass"] = av_pass
        stats["dv_pass"] = dv_pass
        stats["atr_pass"] = atr_pass
        if atr_missing or atr_below:
            stats["atr_missing"] = atr_missing
            stats["atr_below"] = atr_below
    _emit_filter_debug("system5", stats, len(result))
    return result


def filter_system6(symbols, data, stats: dict[str, int] | None = None):
    result: list[str] = []
    total = len(symbols or [])
    low_pass = 0
    dv_pass = 0
    hv_pass = 0
    hv_missing = 0
    hv_range_fail = 0
    for sym in symbols or []:
        df = data.get(sym)
        if df is None or df.empty:
            continue
        low_ok, dv_ok, hv_ok = _system6_conditions(df)
        if not low_ok:
            continue
        low_pass += 1
        if not dv_ok:
            continue
        dv_pass += 1
        if not hv_ok:
            try:
                reasons = df.attrs.get("_fdbg_reasons6")  # type: ignore[attr-defined]
                if isinstance(reasons, list) and reasons:
                    r = reasons[-1]
                    if r == "hv_missing":
                        hv_missing += 1
                    elif r == "hv_range_fail":
                        hv_range_fail += 1
            except Exception:
                pass
            continue
        hv_pass += 1
        result.append(sym)
    if stats is not None:
        stats["total"] = total
        stats["low_pass"] = low_pass
        stats["dv_pass"] = dv_pass
        stats["hv_pass"] = hv_pass
        if hv_missing or hv_range_fail:
            stats["hv_missing"] = hv_missing
            stats["hv_range_fail"] = hv_range_fail
    _emit_filter_debug("system6", stats, len(result))
    return result
