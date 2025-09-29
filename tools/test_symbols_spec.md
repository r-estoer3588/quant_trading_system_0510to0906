# テスト用架空銘柄データ仕様書

## 目的

System1-7 の各段階（フィルター・セットアップ・シグナル）をテストするための架空銘柄データを定義する。

## 架空銘柄パターン

### FAIL_ALL: すべての条件を満たさない銘柄

- Close: 2.0 (価格 5 ドル未満で System1,2 のフィルターで落ちる)
- Volume: 100,000
- 全指標: 最小値または条件を満たさない値

### FILTER_ONLY_S1: System1 のフィルターのみ通過

- Close: 50.0
- Volume: 2,000,000 (DV20 = 100M)
- SMA25: 52.0, SMA50: 51.0 (SMA25>SMA50 でセットアップで落ちる)
- その他システムの条件は満たさない

### FILTER_ONLY_S2: System2 のフィルターのみ通過

- Close: 25.0
- Volume: 1,500,000 (DV20 = 37.5M)
- ATR_Ratio: 0.04 (3%以上)
- RSI3: 85 (90 未満でセットアップで落ちる)

### FILTER_ONLY_S3: System3 のフィルターのみ通過

- Low: 20.0
- Close: 22.0
- Volume: 1,500,000 (AvgVolume50 = 1.5M)
- ATR_Ratio: 0.06 (5%以上)
- 3 日下落率: 10% (12.5%未満でセットアップで落ちる)

### FILTER_ONLY_S4: System4 のフィルターのみ通過

- Close: 100.0
- Volume: 1,200,000 (DV50 = 120M)
- HV50: 25 (10-40 範囲内)
- SMA200: 105.0 (Close<SMA200 でセットアップで落ちる)

### FILTER_ONLY_S5: System5 のフィルターのみ通過

- Close: 15.0
- Volume: 600,000 (AvgVolume50 = 600k, DV50 = 9M)
- ATR_Pct: 0.03 (2.5%以上)
- SMA100: 14.0, ATR10: 0.8 (Close<SMA100+ATR10 でセットアップで落ちる)

### FILTER_ONLY_S6: System6 のフィルターのみ通過

- Low: 18.0
- Close: 20.0
- Volume: 800,000 (DV50 = 16M)
- return_6d: 15% (20%未満でセットアップで落ちる)

### SETUP_PASS_S1: System1 のセットアップも通過

- Close: 50.0
- Volume: 2,000,000
- SMA25: 49.0, SMA50: 51.0 (SMA25>SMA50)
- SPY 条件は別途考慮
- ROC200: 0.05 (シグナル生成のため低めに設定)

### SETUP_PASS_S2: System2 のセットアップも通過

- Close: 25.0
- Volume: 1,500,000
- ATR_Ratio: 0.04
- RSI3: 95 (90 以上)
- TwoDayUp: True
- ADX7: 30 (シグナル生成のため低めに設定)

### SETUP_PASS_S3: System3 のセットアップも通過

- Low: 20.0
- Close: 22.0
- Volume: 1,500,000
- ATR_Ratio: 0.06
- Close vs SMA150: 22.0 > 21.0
- 3 日下落率: 15% (12.5%以上)
- 3 日下落率: 20% (シグナル生成のため)

### SETUP_PASS_S4: System4 のセットアップも通過

- Close: 100.0
- Volume: 1,200,000
- HV50: 25
- SMA200: 95.0 (Close>SMA200)
- RSI4: 30 (シグナル生成のため低めに設定)

### SETUP_PASS_S5: System5 のセットアップも通過

- Close: 15.0
- Volume: 600,000
- ATR_Pct: 0.03
- SMA100: 13.0, ATR10: 0.5 (Close>SMA100+ATR10)
- ADX7: 60 (55 以上)
- RSI3: 40 (50 未満)
- ADX7: 65 (シグナル生成のため)

### SETUP_PASS_S6: System6 のセットアップも通過

- Low: 18.0
- Close: 20.0
- Volume: 800,000
- return_6d: 25% (20%以上)
- UpTwoDays: True
- 6 日上昇率: 30% (シグナル生成のため)

## 必要な指標列

- 価格: Close, High, Low, Open
- 出来高: Volume
- 移動平均: SMA25, SMA50, SMA100, SMA150, SMA200
- 技術指標: RSI3, RSI4, ADX7, ATR10, ATR_Ratio, ATR_Pct
- 出来高指標: AvgVolume50, DollarVolume20, DollarVolume50
- ボラティリティ: HV50
- リターン: ROC200, return_6d, 3 日下落率, 6 日上昇率
- パターン: TwoDayUp, UpTwoDays

## データ期間

- 直近 300 日分のデータ（rolling キャッシュ対応）
- 指標計算に十分な履歴データを含む
