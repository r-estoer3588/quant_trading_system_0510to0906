#!/usr/bin/env python3
"""架空銘柄の基本的な動作確認スクリプト"""

import pandas as pd

from config.settings import get_settings


def check_fake_symbols():
    """架空銘柄データの基本チェック"""
    settings = get_settings()
    test_symbols_dir = settings.DATA_CACHE_DIR / "test_symbols"

    if not test_symbols_dir.exists():
        print("❌ 架空銘柄ディレクトリが見つかりません")
        return False

    print(f"✅ 架空銘柄ディレクトリ: {test_symbols_dir}")

    # ファイル一覧
    feather_files = list(test_symbols_dir.glob("*.feather"))
    print(f"📁 見つかったファイル数: {len(feather_files)}")

    for file in feather_files:
        try:
            df = pd.read_feather(file)
            print(f"📊 {file.stem}: {len(df)}行, 列={list(df.columns[:5])}...")

            # 最新データの確認
            if len(df) > 0:
                last_row = df.iloc[-1]
                print(
                    f"   最新: Close={last_row.get('Close', 'N/A')}, Volume={last_row.get('Volume', 'N/A')}"
                )
                print(
                    f"   指標: SMA25={last_row.get('SMA25', 'N/A')}, RSI3={last_row.get('RSI3', 'N/A')}"
                )
                print()

        except Exception as e:
            print(f"❌ {file.stem}: 読み込みエラー - {e}")

    return True


if __name__ == "__main__":
    check_fake_symbols()
