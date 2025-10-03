from __future__ import annotations

# DEPRECATED: このファイルは共有指標前計算フェーズの削除により一時的に無効化されています
# rolling cache での事前計算を使用してください
raise NotImplementedError(
    "indicators_precompute.py は無効化されています。"
    "rolling cache (scripts/build_rolling_with_indicators.py) を使用してください。"
)

# 以下は保持のため残しますが使用されません

import time as _t
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from common.indicators_common import add_indicators  # Used in deprecated code paths

# Note: add_indicators is referenced but deprecated, using placeholder

try:
    from config.settings import get_settings
except Exception:  # pragma: no cover
    get_settings = None  # type: ignore

try:
    from common.cache_manager import standardize_indicator_columns
except Exception:  # pragma: no cover
    standardize_indicator_columns = None  # type: ignore

# 共有前計算で付与する主な指標（説明用）。
# インポート時点で参照可能にし、呼び出し側の from ... import を安全化する。
PRECOMPUTED_INDICATORS = (
    # ATR 系
    "ATR10",
    "ATR20",
    "ATR40",
    "ATR50",
    # 移動平均
    "SMA25",
    "SMA50",
    "SMA100",
    "SMA150",
    "SMA200",
    # モメンタム/オシレーター
    "ROC200",
    "RSI3",
    "RSI4",
    "ADX7",
    # 流動性・ボラティリティ等
    "DollarVolume20",
    "DollarVolume50",
    "AvgVolume50",
    "ATR_Ratio",
    "ATR_Pct",
    # 派生・補助指標
    "Return_3D",
    "Return_6D",
    "Return_Pct",  # 新規追加：リターン率
    "UpTwoDays",
    "TwoDayUp",
    "Drop3D",  # 新規追加：3日間下落率
    "HV50",
    "min_50",
    "max_70",
)


def _ensure_price_columns_upper(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    # 既に大文字があれば尊重し、無い場合のみ小文字から補完
    if "Open" not in x.columns and "open" in x.columns:
        x["Open"] = x["open"]
    if "High" not in x.columns and "high" in x.columns:
        x["High"] = x["high"]
    if "Low" not in x.columns and "low" in x.columns:
        x["Low"] = x["low"]
    if "Close" not in x.columns and "close" in x.columns:
        x["Close"] = x["close"]
    if "Volume" not in x.columns and "volume" in x.columns:
        x["Volume"] = x["volume"]
    return x


def precompute_shared_indicators(
    basic_data: dict[str, pd.DataFrame],
    *,
    log: Callable[[str], None] | None = None,
    parallel: bool = False,
    max_workers: int | None = None,
) -> dict[str, pd.DataFrame]:
    """
    basic_data の各 DataFrame に共有インジケータ列を付与して返す。

    - 入力はローリング/ベース由来の最小カラム（小文字）でも可。
    - 価格系カラム（Open/High/Low/Close/Volume）は大文字を補完してから計算。
    - 出力は元の DataFrame に `add_indicators` で追加された列を結合。
    - 既存列は上書きしない方針（同名が存在すればそのまま残す）。
    """
    if not basic_data:
        return basic_data
    out: dict[str, pd.DataFrame] = {}
    total = len(basic_data)
    start_ts = _t.time()
    CHUNK = 500

    # 初回ログを即時出力（起動確認用）
    if callable(log):
        try:
            log(f"🧮 共有指標 前計算: 0/{total} | 起動中…")
        except Exception:
            pass

    # 共有前計算で付与する主な指標の名称一覧（ログ表示用）
    # add_indicators() が実際の計算を担うため、この一覧は説明用に限定
    # し、挙動の切り替えには影響しません。
    global PRECOMPUTED_INDICATORS
    PRECOMPUTED_INDICATORS = (
        # ATR 系
        "ATR10",
        "ATR20",
        "ATR40",
        "ATR50",
        # 移動平均
        "SMA25",
        "SMA50",
        "SMA100",
        "SMA150",
        "SMA200",
        # モメンタム/オシレーター
        "ROC200",
        "RSI3",
        "RSI4",
        "ADX7",
        # 流動性・ボラティリティ等
        "DollarVolume20",
        "DollarVolume50",
        "AvgVolume50",
        "ATR_Ratio",
        "ATR_Pct",
        # 派生・補助指標
        "Return_3D",
        "Return_6D",
        "Return_Pct",  # 新規追加：リターン率
        "UpTwoDays",
        "TwoDayUp",
        "Drop3D",  # 新規追加：3日間下落率
        "HV50",
        "min_50",
        "max_70",
    )

    # 共有インジケータのキャッシュ格納場所（設定 > 既定）
    def _cache_dir() -> Path:
        try:
            settings = get_settings(create_dirs=True) if get_settings else None
            base = Path(settings.outputs.signals_dir) if settings else Path("data_cache/signals")
        except Exception:
            base = Path("data_cache/signals")
        p = base / "shared_indicators"
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return p

    cdir = _cache_dir()

    def _read_cache(sym: str) -> pd.DataFrame | None:
        for ext in (".feather", ".parquet"):
            fp = cdir / f"{sym}{ext}"
            if fp.exists():
                try:
                    if ext == ".feather":
                        df = pd.read_feather(fp)
                    else:
                        df = pd.read_parquet(fp)
                    if df is not None and not df.empty:
                        # Date 正規化
                        col = "Date" if "Date" in df.columns else None
                        if col:
                            df[col] = pd.to_datetime(df[col], errors="coerce").dt.normalize()
                        return df
                except Exception:
                    continue
        return None

    def _write_cache(sym: str, df: pd.DataFrame) -> None:
        try:
            # Feather を優先、Parquet をフォールバック保存
            fp = cdir / f"{sym}.feather"
            df.reset_index(drop=True).to_feather(fp)
        except Exception:
            try:
                fp2 = cdir / f"{sym}.parquet"
                df.to_parquet(fp2, index=False)
            except Exception:
                pass

    def _calc(sym_df: tuple[str, pd.DataFrame]) -> tuple[str, pd.DataFrame]:
        sym, df = sym_df
        try:
            if df is None or getattr(df, "empty", True):
                return sym, df
            work = _ensure_price_columns_upper(df)
            # 既存キャッシュを読み込み、差分行だけ再計算
            cached = _read_cache(sym)
            if cached is not None and not cached.empty:
                # Date 基準で差分
                try:
                    src = work.copy()
                    if "Date" in src.columns:
                        src_dates = pd.to_datetime(src["Date"], errors="coerce").dt.normalize()
                    else:
                        src_dates = pd.to_datetime(src.index, errors="coerce").normalize()
                        src = src.reset_index(drop=True)
                        src["Date"] = src_dates
                    cached_local = cached.copy()
                    if "Date" in cached_local.columns:
                        cached_dates = pd.to_datetime(
                            cached_local["Date"], errors="coerce"
                        ).dt.normalize()
                    else:
                        cached_dates = pd.to_datetime(
                            cached_local.index, errors="coerce"
                        ).normalize()
                        cached_local = cached_local.reset_index(drop=True)
                        cached_local["Date"] = cached_dates
                    last = cached_dates.max()
                    src_latest = src_dates.max()
                    use_cached_only = (
                        pd.notna(last)
                        and pd.notna(src_latest)
                        and src_latest <= last
                        and len(cached_local) == len(src)
                    )
                    if use_cached_only:
                        ind_df = cached_local
                        ind_df.attrs["_precompute_skip_cache"] = True
                    else:
                        # 安全に文脈を付けて再計算（最大の必要窓は 200 と想定 + 10% 余裕）
                        ctx_days = 220
                        src_recent = src[src["Date"] >= (last - pd.Timedelta(days=ctx_days))]
                        # 差分再計算
                        recomputed = add_indicators(src_recent)
                        # 以前の最終日より新しい行だけを採用
                        recomputed_new = recomputed[recomputed["Date"] > last]
                        # FutureWarning 回避: 空/全NAのフレームは concat から除外
                        is_empty = recomputed_new is None or getattr(recomputed_new, "empty", True)
                        is_all_na = False
                        try:
                            if not is_empty:
                                is_all_na = bool(recomputed_new.count().sum() == 0)
                        except Exception:
                            is_all_na = False
                        if is_empty or is_all_na:
                            ind_df = cached_local
                        else:
                            merged = pd.concat([cached_local, recomputed_new], ignore_index=True)
                            ind_df = merged
                except Exception:
                    ind_df = add_indicators(work)
            else:
                ind_df = add_indicators(work)
            new_cols = [c for c in ind_df.columns if c not in df.columns]
            if new_cols:
                merged = df.copy()
                for c in new_cols:
                    merged[c] = ind_df[c]
                # 指標列の標準化を適用
                if standardize_indicator_columns:
                    merged = standardize_indicator_columns(merged)
                if getattr(ind_df, "attrs", {}).get("_precompute_skip_cache"):
                    try:
                        merged.attrs["_precompute_skip_cache"] = True
                    except Exception:
                        pass
                return sym, merged
            # 新規指標がない場合でも標準化を適用
            if standardize_indicator_columns:
                df = standardize_indicator_columns(df)
            return sym, df
        except Exception:
            # エラー時も標準化を適用
            if standardize_indicator_columns:
                df = standardize_indicator_columns(df)
            return sym, df

    # 並列指定があれば件数に関わらず並列実行する（ワーカー数は銘柄数を超えない）
    if parallel:
        workers = max_workers or min(32, (total // 1000) + 8)
        workers = max(1, min(int(workers), int(total)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_calc, item): item[0] for item in basic_data.items()}
            done = 0
            for fut in as_completed(futures):
                sym, res = fut.result()
                out[sym] = res
                # キャッシュ書き込み（新規列も含むテーブル）
                try:
                    skip_cache = bool(getattr(res, "attrs", {}).get("_precompute_skip_cache"))
                except Exception:
                    skip_cache = False
                try:
                    if not skip_cache and res is not None and not getattr(res, "empty", True):
                        _write_cache(sym, res)
                except Exception:
                    pass
                done += 1
                if log and (done % CHUNK == 0 or done == total):
                    try:
                        elapsed = max(0.001, _t.time() - start_ts)
                        rate = done / elapsed
                        remain = max(0, total - done)
                        eta_sec = int(remain / rate) if rate > 0 else 0
                        m, s = divmod(eta_sec, 60)
                        log(f"🧮 共有指標 前計算: {done}/{total} | ETA {m}分{s}秒")
                    except Exception:
                        try:
                            log(f"🧮 共有指標 前計算: {done}/{total}")
                        except Exception:
                            pass
    else:
        for idx, item in enumerate(basic_data.items(), start=1):
            sym, res = _calc(item)
            out[sym] = res
            try:
                skip_cache = bool(getattr(res, "attrs", {}).get("_precompute_skip_cache"))
            except Exception:
                skip_cache = False
            try:
                if not skip_cache and res is not None and not getattr(res, "empty", True):
                    _write_cache(sym, res)
            except Exception:
                pass
            if log and (idx % CHUNK == 0 or idx == total):
                try:
                    elapsed = max(0.001, _t.time() - start_ts)
                    rate = idx / elapsed
                    remain = max(0, total - idx)
                    eta_sec = int(remain / rate) if rate > 0 else 0
                    m, s = divmod(eta_sec, 60)
                    log(f"🧮 共有指標 前計算: {idx}/{total} | ETA {m}分{s}秒")
                except Exception:
                    try:
                        log(f"🧮 共有指標 前計算: {idx}/{total}")
                    except Exception:
                        pass
    return out


__all__ = ["precompute_shared_indicators", "PRECOMPUTED_INDICATORS"]
