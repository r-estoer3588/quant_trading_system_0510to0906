"""候補データの内容を確認するスクリプト"""

import sys

sys.path.insert(0, ".")

from strategies.system1_strategy import System1Strategy

# System1を実行
strategy = System1Strategy()

# テスト用シンボル
test_symbols = ["AAPL", "MSFT", "GOOGL"]

# データ準備
data = strategy.prepare_data(test_symbols, reuse_indicators=True)

# 候補生成
result = strategy.generate_candidates(data, latest_only=True)

if isinstance(result, tuple):
    candidates_by_date, merged_df = result[:2]
else:
    candidates_by_date = result
    merged_df = None

print("\n=== 候補データ確認 ===")
print(f"候補日数: {len(candidates_by_date)}")

for date, df in candidates_by_date.items():
    print(f"\n日付: {date}")
    print(f"候補数: {len(df)}")
    if not df.empty:
        print(f"列: {list(df.columns)}")
        print("\nサンプル:")
        for _, row in df.head(3).iterrows():
            print(
                f"  {row['symbol']}: entry_price={row.get('entry_price', 'N/A')}, "
                f"stop_price={row.get('stop_price', 'N/A')}, close={row.get('close', 'N/A')}"
            )
