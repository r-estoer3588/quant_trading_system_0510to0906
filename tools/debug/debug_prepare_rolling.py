#!/usr/bin/env python3
"""Debug tool for analyzing the _prepare_rolling_frame function step by step."""

import sys
from pathlib import Path

import pandas as pd

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common.cache_manager import CacheManager
from common.indicators_common import add_indicators
from config.settings import get_settings


def debug_prepare_rolling_frame():
    """Debug the _prepare_rolling_frame function step by step."""
    print("🔍 _prepare_rolling_frame 関数の動作解析")

    settings = get_settings(create_dirs=True)
    cm = CacheManager(settings)

    # Get a sample symbol - skip if no data available
    csv_files = list(cm.full_dir.glob("*.csv"))
    if not csv_files:
        print("❌ CSVファイルが見つかりません。データがない可能性があります。")
        return

    csv_path = csv_files[0]
    symbol = csv_path.stem
    print(f"\n📊 シンボル: {symbol}")

    # Read raw data
    df = cm.read(symbol, "full")
    if df is None or df.empty:
        print(f"❌ シンボル {symbol} のデータが取得できませんでした。")
        return

    print(f"🔴 Raw データ: {len(df.columns)} 列")
    print(f"📝 Raw 列名: {list(df.columns)}")

    # Step 1: Copy and basic processing
    work = df.copy()
    print(f"\n1️⃣ Copy後: {len(work.columns)} 列")

    # Step 2: Date processing
    if "date" not in work.columns:
        if "Date" in work.columns:
            work = work.rename(columns={"Date": "date"})
        print(f"2️⃣ Date処理後: {len(work.columns)} 列")

    # Step 3: Date normalization
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"])
    work = (
        work.sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    print(f"3️⃣ Date正規化後: {len(work.columns)} 列")

    # Step 4: Create calc copy
    calc = work.copy()
    calc["Date"] = pd.to_datetime(calc["date"], errors="coerce").dt.normalize()
    print(f"4️⃣ calc作成(Date追加)後: {len(calc.columns)} 列")
    print(f"   📝 列名: {list(calc.columns)}")

    # Step 5: Column conversion (this is where the problem likely occurs)
    print("\n5️⃣ OHLCV 列変換処理:")
    col_pairs = (
        ("open", "Open"),
        ("high", "High"),
        ("low", "Low"),
        ("close", "Close"),
        ("volume", "Volume"),
    )

    for src, dst in col_pairs:
        if src in calc.columns and dst not in calc.columns:
            print(f"   ✅ {src} -> {dst} (変換実行)")
            calc[dst] = calc[src]
        elif src in calc.columns and dst in calc.columns:
            print(f"   ⚠️  {src} と {dst} 両方存在 -> {src}削除")
            calc = calc.drop(columns=[src])
        else:
            print(
                f"   ⏭️ {src}({src in calc.columns}) -> {dst}({dst in calc.columns}) スキップ"
            )

    print(f"   🔄 変換後: {len(calc.columns)} 列")
    print(f"   📝 列名: {list(calc.columns)}")

    # Step 6: AdjClose processing
    if "AdjClose" not in calc.columns:
        for cand in ("adjusted_close", "adj_close", "adjclose"):
            if cand in calc.columns:
                print(f"6️⃣ AdjClose変換: {cand} -> AdjClose")
                calc["AdjClose"] = calc[cand]
                calc = calc.drop(columns=[cand])
                break

    print(f"6️⃣ AdjClose処理後: {len(calc.columns)} 列")

    # Step 7: Indicator calculation
    print(f"\n7️⃣ 指標計算前: {len(calc.columns)} 列")
    try:
        enriched = add_indicators(calc)
        print(f"7️⃣ 指標計算後: {len(enriched.columns)} 列")
    except Exception as e:
        print(f"❌ 指標計算でエラー: {e}")
        return

    # Find what changed
    added = set(enriched.columns) - set(calc.columns)
    removed = set(calc.columns) - set(enriched.columns)

    if added:
        print(f"   🆕 追加列: {sorted(added)}")
    if removed:
        print(f"   🗑️ 削除列: {sorted(removed)}")

    # Step 8: Final cleanup (simulate _clean_duplicate_columns)
    print("\n8️⃣ 重複クリーンアップ前の列:")
    all_cols = enriched.columns.tolist()
    col_mapping = {}
    for col in all_cols:
        key = col.lower()
        if key not in col_mapping:
            col_mapping[key] = []
        col_mapping[key].append(col)

    duplicates_to_remove = []
    for key, similar_cols in col_mapping.items():
        if len(similar_cols) > 1:
            print(f"   🔄 {key}: {similar_cols}")
            # Keep the best one
            priority_scores = []
            for col in similar_cols:
                if col.isupper():
                    score = 3
                elif col[0].isupper():
                    score = 2
                elif "_" in col:
                    score = 1
                else:
                    score = 0
                priority_scores.append((score, col))

            priority_scores.sort(reverse=True)
            best_col = priority_scores[0][1]
            print(f"     ✅ 保持: {best_col}")

            for _, col in priority_scores[1:]:
                duplicates_to_remove.append(col)
                print(f"     ❌ 削除: {col}")

    print(f"\n🧹 削除対象: {duplicates_to_remove}")

    # Summary
    print("\n📊 要約:")
    print(f"   • Raw データ: {len(df.columns)} 列")
    print(f"   • 指標計算後: {len(enriched.columns)} 列")
    print(f"   • 追加された列: {len(added)}")
    print(f"   • 重複削除対象: {len(duplicates_to_remove)}")

    if len(enriched) > 0:
        print(f"   • データ行数: {len(enriched)}")
        print("✅ デバッグ完了")
    else:
        print("❌ データが空です")


if __name__ == "__main__":
    debug_prepare_rolling_frame()
