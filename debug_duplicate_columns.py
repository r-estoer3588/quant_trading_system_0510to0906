#!/usr/bin/env python3
"""Debug script to analyze duplicate column issue in cache data."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from common.cache_manager import CacheManager
from common.indicators_common import add_indicators
from config.settings import get_settings


def analyze_duplicate_columns():
    """Analyze the source of duplicate columns."""
    print("🔍 重複列問題の解析開始")

    settings = get_settings(create_dirs=True)
    cm = CacheManager(settings)

    # Get a sample symbol
    available_symbols = list(cm.full_dir.glob("*.csv"))[:3]  # First 3 symbols

    for csv_path in available_symbols:
        symbol = csv_path.stem
        print(f"\n📊 シンボル: {symbol}")

        # Read raw data
        try:
            df_raw = cm.read(symbol, "full")
            print(f"  🔴 Raw データ列数: {len(df_raw.columns)}")
            print(f"  📝 Raw 列名: {list(df_raw.columns)[:10]}...")  # First 10 columns

            # Check for existing duplicates
            lowercase_cols = {col.lower(): col for col in df_raw.columns}
            uppercase_cols = {col for col in df_raw.columns if col[0].isupper() if col}

            duplicates = []
            for low_col, orig_col in lowercase_cols.items():
                if any(
                    up_col.lower() == low_col for up_col in uppercase_cols if up_col != orig_col
                ):
                    duplicates.append(low_col)

            if duplicates:
                print(f"  ⚠️  既存重複: {duplicates}")
            else:
                print("  ✅ 重複なし")

            # Test indicator addition
            print(f"  🧮 指標計算前の列数: {len(df_raw.columns)}")

            # Make a copy and add Date column like in the real code
            work = df_raw.copy()
            if "date" not in work.columns and "Date" not in work.columns:
                work = work.reset_index()
                if "Date" not in work.columns and work.index.name:
                    work = work.rename_axis("Date").reset_index()

            print(f"  📅 Date処理後の列数: {len(work.columns)}")
            print(f"  📝 処理後列名: {list(work.columns)[:10]}...")

            # Apply indicator calculation
            enriched = add_indicators(work)
            print(f"  🧮 指標計算後の列数: {len(enriched.columns)}")

            # Find what was added
            new_cols = set(enriched.columns) - set(work.columns)
            print(f"  🆕 新規追加列: {len(new_cols)} 個")
            if len(new_cols) < 20:  # Only show if not too many
                print(f"     {sorted(new_cols)}")

            # Find exact duplicates
            all_cols = enriched.columns.tolist()
            seen = set()
            duplicates = []
            for col in all_cols:
                if col.lower() in seen:
                    duplicates.append(col)
                else:
                    seen.add(col.lower())

            if duplicates:
                print(f"  🔴 重複列発見: {duplicates}")
            else:
                print("  ✅ 指標計算後も重複なし")

        except Exception as e:
            print(f"  ❌ エラー: {e}")
            continue

        break  # Only analyze first symbol for now


if __name__ == "__main__":
    analyze_duplicate_columns()
