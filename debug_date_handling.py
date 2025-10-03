"""
日付データの問題を診断するスクリプト
CacheManagerの各段階で日付がどのように変化するかを追跡
"""

from common.cache_manager import CacheManager
from config.settings import get_settings


def debug_date_handling():
    """日付データの処理を診断する"""

    # CacheManagerの各段階のデータを確認
    settings = get_settings()
    cache_manager = CacheManager(settings)

    print("=== Full backup データの診断 ===")
    full_df = cache_manager.read("SPY", "full")
    if full_df is not None:
        print(f"Full backup 行数: {len(full_df)}")
        print(f"日付カラムの型: {full_df['date'].dtype}")
        print("最初の5行の日付:")
        print(full_df["date"].head())
        print(f"日付のNaN数: {full_df['date'].isna().sum()}")
        print(f"日付の最小値: {full_df['date'].min()}")
        print(f"日付の最大値: {full_df['date'].max()}")
    else:
        print("Full backup データが読み込めません")

    print("\n=== Base データの診断 ===")
    try:
        # baseディレクトリから直接読む
        base_dir = cache_manager.full_dir.parent / "base"
        path = cache_manager.file_manager.detect_path(base_dir, "SPY")
        if path.exists():
            base_df = cache_manager.file_manager.read_with_fallback(path, "SPY", "base")
            if base_df is not None and not base_df.empty:
                print(f"Base データ 行数: {len(base_df)}")
                print(f"日付カラムの型: {base_df['date'].dtype}")
                print("最初の5行の日付:")
                print(base_df["date"].head())
            else:
                print("Base データが空またはNone")
        else:
            print("Base データファイルが存在しません")
    except Exception as e:
        print(f"Base データの読み込みでエラー: {e}")

    print("\n=== _read_base_and_tail の結果 ===")
    base_and_tail = cache_manager._read_base_and_tail("SPY", 330)
    if base_and_tail is not None:
        print(f"Base and tail 行数: {len(base_and_tail)}")
        print(f"日付カラムの型: {base_and_tail['date'].dtype}")
        print("最初の5行の日付:")
        print(base_and_tail["date"].head())
        print(f"日付のNaN数: {base_and_tail['date'].isna().sum()}")
    else:
        print("Base and tail データが取得できません")


if __name__ == "__main__":
    debug_date_handling()
