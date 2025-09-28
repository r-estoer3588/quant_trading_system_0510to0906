"""
結合処理での値の消失を詳細調査
"""

import pandas as pd

from common.cache_manager import CacheManager
from common.indicators_common import add_indicators
from config.settings import get_settings


def debug_value_loss():
    """結合処理での値の消失を調査"""

    settings = get_settings()
    cache_manager = CacheManager(settings)

    # 短いテストデータで実行
    base_and_tail = cache_manager._read_base_and_tail("SPY", 50)

    if base_and_tail is None or base_and_tail.empty:
        print("ERROR: base_and_tail データが取得できません")
        return

    # 元のOHLCVを保持
    print("=== 元のOHLCVデータ ===")
    ohlcv = {"date", "open", "high", "low", "close", "volume", "raw_close"}
    ohlcv_cols = [col for col in ohlcv if col in base_and_tail.columns]
    combined = base_and_tail[ohlcv_cols].copy()
    print(f"OHLCV抽出後の行数: {len(combined)}")
    print(f"Close値の最初の3個: {combined['close'].head(3).tolist()}")

    # _recompute_indicators の enriched 部分だけを取得
    df = base_and_tail.copy()
    base = df.copy()
    base["date"] = pd.to_datetime(base["date"], errors="coerce")
    base = base.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for col in ("open", "high", "low", "close", "volume"):
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce")

    case_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    base_renamed = base.rename(columns={k: v for k, v in case_map.items() if k in base.columns})
    base_renamed["Date"] = base_renamed["date"]

    # 既存指標列を削除
    basic_cols = {
        "date",
        "Date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "raw_close",
    }
    indicator_cols = [col for col in base_renamed.columns if col not in basic_cols]
    if indicator_cols:
        base_renamed = base_renamed.drop(columns=indicator_cols)

    # add_indicators実行
    enriched = add_indicators(base_renamed)
    print("\n=== add_indicators 結果 ===")
    print(f"enriched行数: {len(enriched)}")
    print(f"sma25の値 (最初の5個): {enriched['sma25'].head().tolist()}")
    print(f"sma25のNaN率: {enriched['sma25'].isna().mean() * 100:.1f}%")

    # 後処理
    enriched = enriched.drop(columns=["Date"], errors="ignore")
    basic_cols = {"open", "high", "low", "close", "volume", "date"}
    enriched.columns = [c.lower() if c.lower() in basic_cols else c for c in enriched.columns]
    enriched["date"] = pd.to_datetime(enriched.get("date", base["date"]), errors="coerce")

    print("\n=== 後処理後のenriched ===")
    print(f"sma25の値 (最初の5個): {enriched['sma25'].head().tolist()}")
    print(f"sma25のNaN率: {enriched['sma25'].isna().mean() * 100:.1f}%")
    print(f"enrichedのインデックス: {enriched.index[:5].tolist()}")
    print(f"combinedのインデックス: {combined.index[:5].tolist()}")

    # 結合処理を段階的に実行
    print("\n=== 結合処理 ===")
    added_count = 0
    for col, series in enriched.items():
        if col in ohlcv:
            print(f"スキップ: {col} (OHLCV列)")
            continue
        print(
            f"追加中: {col}, サイズ: {len(series)}, 最初の値: {series.iloc[0] if len(series) > 0 else 'N/A'}"
        )
        combined[col] = series
        added_count += 1

        # sma25を追加した後の状態をチェック
        if col == "sma25":
            print(f"  sma25追加後のcombinedでのsma25値: {combined['sma25'].head().tolist()}")
            print(f"  sma25追加後のNaN率: {combined['sma25'].isna().mean() * 100:.1f}%")
            break  # デバッグのため最初のだけで停止

    print(f"\n最終combined行数: {len(combined)}")
    print(f"最終sma25のNaN率: {combined['sma25'].isna().mean() * 100:.1f}%")


if __name__ == "__main__":
    debug_value_loss()
