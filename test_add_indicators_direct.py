"""
add_indicators 関数の直接テストスクリプト
CacheManager を通さずに直接テスト
"""

from common.cache_manager import CacheManager
from common.indicators_common import add_indicators
from config.settings import get_settings


def test_add_indicators_directly():
    """add_indicators を直接テストして問題を特定"""

    settings = get_settings()
    cache_manager = CacheManager(settings)

    # Full backup データから少量のデータを抽出
    full_data = cache_manager.read("SPY", "full")
    if full_data is None or full_data.empty:
        print("ERROR: Full backup データが取得できません")
        return

    # 最新の50行のみを使用してテスト
    test_data = full_data.tail(50).copy()
    print("=== テストデータの準備 ===")
    print(f"テストデータ行数: {len(test_data)}")
    print(f"列名: {list(test_data.columns)}")
    print("最初の3行:")
    print(test_data[["date", "open", "high", "low", "close", "volume"]].head(3))

    # 基本カラムのみを残して指標をクリア
    basic_cols = ["date", "open", "high", "low", "close", "volume", "raw_close"]
    clean_data = test_data[basic_cols].copy()

    print("\n=== 指標削除後のデータ ===")
    print(f"クリーンデータ列数: {len(clean_data.columns)}")
    print(f"クリーンデータ列名: {list(clean_data.columns)}")

    # カラム名を大文字に変換（add_indicators の要求に合わせる）
    case_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    clean_data = clean_data.rename(
        columns={k: v for k, v in case_map.items() if k in clean_data.columns}
    )
    clean_data["Date"] = clean_data["date"]

    print("\n=== 大文字変換後のデータ ===")
    print(f"変換後列名: {list(clean_data.columns)}")
    print("Close列の最初の5個の値:")
    print(clean_data["Close"].head().tolist())
    print(f"Close列のデータ型: {clean_data['Close'].dtype}")
    print(f"Close列のNaN数: {clean_data['Close'].isna().sum()}")

    # add_indicators を直接実行
    print("\n=== add_indicators 直接実行テスト ===")
    try:
        enriched_data = add_indicators(clean_data)
        print("✅ add_indicators 成功!")
        print(f"結果の行数: {len(enriched_data)}")
        print(f"結果の列数: {len(enriched_data.columns)}")

        # 追加された指標列をチェック
        original_cols = set(clean_data.columns)
        new_cols = set(enriched_data.columns) - original_cols
        print(f"新しく追加された指標列 ({len(new_cols)}個): {list(new_cols)[:10]}")

        # サンプル指標の値を確認
        sample_indicators = ["sma25", "rsi4", "atr10"]
        for col in sample_indicators:
            if col in enriched_data.columns:
                nan_rate = enriched_data[col].isna().mean() * 100
                print(f"{col}: NaN率 {nan_rate:.1f}%")
                if nan_rate < 100:
                    print(f"  最後の5個: {enriched_data[col].tail().tolist()}")

    except Exception as e:
        print(f"❌ add_indicators でエラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_add_indicators_directly()
