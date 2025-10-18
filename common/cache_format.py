# common/cache_format.py
"""
キャッシュ関連の共通フォーマット処理
- ファイル名の正規化
- 数値の丸め処理
- CSV 書式設定
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

# Windows 予約語（ファイル名用）
RESERVED_WORDS = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
}


def safe_filename(symbol: str) -> str:
    """Windows 予約語を避けたファイル名を返す。"""
    if symbol.upper() in RESERVED_WORDS:
        return symbol + "_RESV"
    return symbol


def round_dataframe(df: pd.DataFrame, decimals: int | None) -> pd.DataFrame:
    """DataFrame を指定小数点で丸めて返す。

    pandas.DataFrame.round は数値列のみを対象とし、日付や文字列列には影響しない。
    ただし decimals が不正値の場合や丸め処理が例外を送出した場合は、
    元の DataFrame をそのまま返す。
    """
    if df is None or decimals is None:
        return df

    try:
        decimals = int(decimals)
    except (ValueError, TypeError):
        return df

    # カテゴリ別の丸め設定（列名小文字ベース）
    price_atr_cols = {
        "open",
        "close",
        "high",
        "low",
        "atr10",
        "atr14",
        "atr20",
        "atr40",
        "atr50",
        "adjusted_close",
        "adjclose",
        "adj_close",
    }
    volume_cols = {"volume", "dollarvolume20", "dollarvolume50", "avgvolume50"}
    oscillator_cols = {"rsi3", "rsi4", "rsi14", "adx7"}
    pct_cols = {
        "roc200",
        "return_3d",
        "return_6d",
        "atr_ratio",
        "atr_pct",
        "hv50",
        "return_pct",
    }

    out = df.copy()
    lc_map = {c.lower(): c for c in out.columns}

    def _safe_round(series: pd.Series, ndigits: int) -> pd.Series:
        try:
            return series.round(ndigits)
        except Exception:
            return series

    # グループ別丸め
    groups = {
        2: price_atr_cols | oscillator_cols,
        4: pct_cols,
    }
    rounded_cols = set()

    for ndigits, names in groups.items():
        cols_to_round = [lc_map[lname] for lname in names if lname in lc_map]
        for col in cols_to_round:
            out[col] = _safe_round(out[col], ndigits)
            rounded_cols.add(col)

    # ボリューム系列（整数化）
    vol_cols_to_round = [lc_map[lname] for lname in volume_cols if lname in lc_map]
    for col in vol_cols_to_round:
        try:
            out[col] = out[col].round(0).astype("Int64", errors="ignore")
        except Exception:
            out[col] = _safe_round(out[col], 0)
    rounded_cols.update(vol_cols_to_round)

    # その他数値列（デフォルト小数点数で丸め）
    for col in out.columns:
        if col not in rounded_cols and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = _safe_round(out[col], decimals)

    return out


def make_csv_formatters(frame: pd.DataFrame, dec_point: str = ".", thous_sep: str | None = None) -> dict:
    """CSV 出力用のフォーマッター辞書を作成。

    Returns: 列名 -> callable の辞書
    """
    lc_map = {c.lower(): c for c in frame.columns}
    fmt: dict = {}

    def _add_thousands_sep(int_str: str, sep: str) -> str:
        """整数文字列に3桁区切りを追加。"""
        if not int_str or not int_str.isdigit():
            return int_str
        reversed_str = int_str[::-1]
        groups = [reversed_str[i : i + 3] for i in range(0, len(reversed_str), 3)]
        return sep.join(groups)[::-1]

    def _num_formatter(nd: int):
        """数値用フォーマッター生成。"""

        def formatter(x):
            try:
                if pd.isna(x):
                    return ""
                formatted = f"{float(x):.{nd}f}"
                if dec_point != ".":
                    formatted = formatted.replace(".", dec_point)
                return formatted
            except Exception:
                return str(x)

        return formatter

    def _int_formatter():
        """整数用フォーマッター生成。"""

        def formatter(x):
            try:
                if pd.isna(x):
                    return ""
                int_str = str(int(float(x)))
                if thous_sep:
                    int_str = _add_thousands_sep(int_str, thous_sep)
                return int_str
            except Exception:
                return str(x)

        return formatter

    # フォーマット規則定義
    rules = {
        _num_formatter(2): [
            "open",
            "close",
            "high",
            "low",
            "atr10",
            "atr14",
            "atr20",
            "atr40",
            "atr50",
            "rsi3",
            "rsi4",
            "rsi14",
            "adx7",
        ],
        _num_formatter(4): [
            "roc200",
            "return_3d",
            "return_6d",
            "atr_ratio",
            "atr_pct",
            "hv50",
        ],
        _int_formatter(): ["volume", "dollarvolume20", "dollarvolume50", "avgvolume50"],
    }

    for formatter, names in rules.items():
        cols = [lc_map[name] for name in names if name in lc_map]
        for col in cols:
            fmt[col] = formatter
    return fmt


def write_dataframe_to_csv(df: pd.DataFrame, path: Path, settings: Any) -> None:
    """DataFrame を設定に従って CSV に書き込む。"""
    # 設定からフォーマット情報を取得（フォールバック付き）
    try:
        dec_point = getattr(settings.cache.csv, "decimal_point", ".")
        thous_sep = getattr(settings.cache.csv, "thousands_sep", None)
        field_sep = getattr(settings.cache.csv, "field_sep", ",")
        round_dec = getattr(settings.cache, "round_decimals", None)
    except Exception:
        dec_point, thous_sep, field_sep, round_dec = ".", None, ",", None

    try:
        # 丸め処理
        df_to_write = round_dataframe(df, round_dec)

        # フォーマッター作成
        formatters = make_csv_formatters(df_to_write, dec_point, thous_sep)

        # CSV 書き込み
        path.parent.mkdir(parents=True, exist_ok=True)

        # 型の問題を回避するため、引数を明示的にキャスト
        csv_kwargs = {
            "path_or_buf": str(path),
            "index": False,
            "sep": str(field_sep),
            "decimal": str(dec_point),
            "formatters": formatters,
        }
        df_to_write.to_csv(**csv_kwargs)
    except Exception as e:
        # フォールバック：基本的な書き込み
        try:
            df.to_csv(path, index=False)
        except Exception:
            raise RuntimeError(f"CSV 書き込みに失敗しました: {path}") from e
