from common.cache_manager import CacheManager
from config.settings import get_settings

settings = get_settings()
cache_manager = CacheManager(settings)

# rollingデータを読み込み
rolling_data = cache_manager.read("SPY", "rolling")
if rolling_data is not None:
    print("Rolling SPY データ:")
    print(f"行数: {len(rolling_data)}")
    print(f'日付範囲: {rolling_data["date"].min()} ～ {rolling_data["date"].max()}')
    print(f"列数: {len(rolling_data.columns)}")
    print(
        f'指標列の例: {[col for col in rolling_data.columns if col in ["sma25", "sma50", "rsi3", "rsi4", "atr10"]]}'
    )
    print("最近の指標値サンプル (sma25):")
    print(rolling_data[["date", "close", "sma25"]].tail(5))

    # 指標のNaN率をチェック
    indicators = ["sma25", "sma50", "rsi3", "rsi4", "atr10"]
    print("\n指標のNaN率:")
    for col in indicators:
        if col in rolling_data.columns:
            # 重複カラムがある場合は最初のものを選択
            col_data = rolling_data[col]
            if hasattr(col_data, "iloc"):  # DataFrame の場合
                col_data = col_data.iloc[:, 0]  # 最初のカラムを選択
            nan_count = col_data.isnull().sum()
            total_count = len(rolling_data)
            nan_rate = nan_count / total_count * 100
            print(f"{col}: {nan_rate:.1f}%")

    # 列の重複状況をチェック
    duplicate_cols = rolling_data.columns[rolling_data.columns.duplicated()].tolist()
    if duplicate_cols:
        print(f"\n重複カラム: {duplicate_cols}")

    # 日付の詳細確認
    print(f"\n日付カラムの型: {rolling_data['date'].dtype}")
    print(f"日付のユニーク値数: {rolling_data['date'].nunique()}")
    print(f"日付の最初の10個: {rolling_data['date'].head(10).tolist()}")

    print("\n列の重複状況:")
    duplicated_cols = rolling_data.columns.duplicated()
    if duplicated_cols.any():
        print(f"重複列が {duplicated_cols.sum()} 個あります")
        print(f"重複している列: {rolling_data.columns[duplicated_cols].tolist()}")
