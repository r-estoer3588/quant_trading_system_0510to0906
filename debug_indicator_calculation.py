"""
指標計算の問題を診断するスクリプト
rolling データで指標が全て NaN になる原因を特定する
"""

from common.cache_manager import CacheManager
from common.indicators_common import add_indicators
from config.settings import get_settings


def debug_indicator_calculation():
    """指標計算の問題を診断する"""

    # CacheManagerからrollingデータを読み込み
    settings = get_settings()
    cache_manager = CacheManager(settings)
    rolling_df = cache_manager.read("SPY", "rolling")

    if rolling_df is None or rolling_df.empty:
        print("ERROR: Rolling データが取得できませんでした")
        return

    print("=== Rolling データの基本情報 ===")
    print(f"行数: {len(rolling_df)}")
    print(f"列名: {list(rolling_df.columns)}")
    print("データ型:")
    print(rolling_df.dtypes)

    print("\n=== 最初の5行のOHLCVデータ ===")
    basic_cols = ["date", "open", "high", "low", "close", "volume"]
    print(rolling_df[basic_cols].head())

    print("\n=== 指標計算前のテスト ===")
    # add_indicators は "Close" などの大文字を期待している
    test_df = rolling_df.copy()

    # カラム名をチェック
    has_uppercase = any(
        col in test_df.columns for col in ["Open", "High", "Low", "Close", "Volume"]
    )
    has_lowercase = any(
        col in test_df.columns for col in ["open", "high", "low", "close", "volume"]
    )

    print(f"大文字カラムの存在: {has_uppercase}")
    print(f"小文字カラムの存在: {has_lowercase}")

    if has_lowercase and not has_uppercase:
        print("\n=== カラム名を大文字に変換して指標計算テスト ===")
        test_df_upper = test_df.copy()
        col_mapping = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
        test_df_upper = test_df_upper.rename(columns=col_mapping)

        print("変換後のカラム名:", list(test_df_upper.columns))

        # 指標計算を実行
        try:
            enriched_df = add_indicators(test_df_upper)
            print(f"\n指標計算成功! 結果の行数: {len(enriched_df)}")
            print(f"結果の列数: {len(enriched_df.columns)}")

            # SMAとRSIの値をチェック
            sma_cols = [col for col in enriched_df.columns if col.startswith("sma")]
            rsi_cols = [col for col in enriched_df.columns if col.startswith("rsi")]

            print(f"\nSMA列: {sma_cols}")
            print(f"RSI列: {rsi_cols}")

            if sma_cols:
                print(f"\n{sma_cols[0]} の最後の10個の値:")
                print(enriched_df[sma_cols[0]].tail(10))
                print(f"NaN率: {enriched_df[sma_cols[0]].isna().mean():.2%}")

            if rsi_cols:
                print(f"\n{rsi_cols[0]} の最後の10個の値:")
                print(enriched_df[rsi_cols[0]].tail(10))
                print(f"NaN率: {enriched_df[rsi_cols[0]].isna().mean():.2%}")

        except Exception as e:
            print(f"指標計算でエラー: {e}")
            import traceback

            traceback.print_exc()

    print("\n=== _recompute_indicators メソッドのテスト ===")
    try:
        recomputed_df = cache_manager._recompute_indicators(rolling_df)
        print(f"_recompute_indicators 成功! 結果の行数: {len(recomputed_df)}")

        # 指標列をチェック
        indicator_cols = [col for col in recomputed_df.columns if col not in basic_cols]
        print(f"指標列数: {len(indicator_cols)}")

        if indicator_cols:
            sample_col = indicator_cols[0]
            print(f"\n{sample_col} の最後の10個の値:")
            print(recomputed_df[sample_col].tail(10))
            print(f"NaN率: {recomputed_df[sample_col].isna().mean():.2%}")

    except Exception as e:
        print(f"_recompute_indicators でエラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_indicator_calculation()
