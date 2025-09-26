# 忁E��指標リスチE

本シスチE���E�Edocs` フォルダおよび `app_system1.py`〜`app_system7.py`�E�で参�Eされる指標をまとめた、E

1. **SMA25**�E�E5 日単純移動平坁E
2. **SMA50**�E�E0 日単純移動平坁E
3. **SMA100**�E�E00 日単純移動平坁E
4. **SMA150**�E�E50 日単純移動平坁E
5. **SMA200**�E�E00 日単純移動平坁E
6. **ATR3**�E�EATR。過去 10 日・40 日・50 日など褁E��期間で使用
7. **ATR1.5**�E�E.5ATR。過去 40 日などで使用
8. **ATR1**�E�EATR。過去 10 日などで使用
9. **ATR2.5**�E�E.5ATR。過去 10 日などで使用
10. **ATR**�E�過去 10 日、E��去 50 日、E��去 40 日、E%ATR など褁E��期間で使用
11. **ADX7**�E�E 日 ADX
12. **return_6d�E�旧称 RETURN6�E�E*�E�E 日リターン
13. **return_pct**�E�総リターン
14. **Drop3D**�E�E 日ドロチE�E
15. **HV50**�E�E0 日ヒストリカルボラチE��リチE���E�年玁E��算！E

## 持E��と使用シスチE��対応表

| 持E��E                                 | 使用シスチE��                                                  | 実裁E��況E                                     |
| ------------------------------------- | ------------------------------------------------------------- | --------------------------------------------- |
| SMA25                                 | System1                                                       | 列として実裁E��E(`SMA25`)                      |
| SMA50                                 | System1                                                       | 列として実裁E��E(`SMA50`)                      |
| SMA100                                | System1, System5                                              | 列として実裁E��E(`SMA100`)                     |
| SMA150                                | System3                                                       | 列として実裁E��E(`SMA150`)                     |
| SMA200                                | System4                                                       | 列として実裁E��E(`SMA200`)                     |
| ATR3                                  | System2, System5, System6, System7                            | ストップ計算で `stop_atr_multiple=3` を使用   |
| ATR1.5                                | System4                                                       | ストップ計算で `stop_atr_multiple=1.5` を使用 |
| ATR1                                  | System5                                                       | `Close > SMA100 + ATR10` 判定で使用           |
| ATR2.5                                | System3                                                       | ストップ計算で `stop_atr_multiple=2.5` を使用 |
| ATR�E�E0 日・20 日・40 日・50 日など�E�E| System1, System2, System3, System4, System5, System6, System7 | 列として実裁E��E(`ATR10` 筁E                   |
| ADX7                                  | System2, System5                                              | 列として実裁E��E(`ADX7`)                       |
| return_6d�E�旧称 RETURN6�E�E             | System6                                                       | 列として実裁E��E(`return_6d`)                   |
| return_pct                            | System1, System2, System3, System4, System5, System6, System7 | 列として実裁E��E(`return_pct`)                 |
| Drop3D                                | System3                                                       | 列として実裁E��E(`Drop3D`)                     |
| HV50                                  | System4                                                       | 列として実裁E��E(`HV50`)                       |

## 補足

- ATR は「過去 10 日」「過去 40 日」「過去 50 日」、EATR」、E.5ATR」、E.5ATR」、EATR」、E%ATR」など褁E��パターンが存在する、E
- SMA は、E5 日」、E0 日」、E00 日」、E50 日」、E00 日」など褁E��の期間を参照する、E
- ADX は 7 日値を使用し、忁E��に応じて高い頁E��ンキング�E�EADX7_High`�E�や 55 以上�E閾値判定を行う、E
- Return 系持E���E「Return6D�E�旧称 RETURN6�E�」「return_pct」などを含む、E
- Drop3D は、E 日ドロチE�E」として使用される、E

## シスチE��別フィルター

### System1

- `avg_dollar_volume_20 > 50_000_000`
- `low >= 5`

### System2

- `low >= 5`
- `avg_dollar_volume_20 > 25_000_000`
- `ATR10 / close > 0.03`

### System3

- `low >= 1`
- `avg_volume_50 >= 1_000_000`
- `ATR10 / close >= 0.05`

### System4

- `avg_dollar_volume_50 > 100_000_000`
- `0.10 <= HV50 <= 0.40`

### System5

- `avg_volume_50 > 500_000`
- `avg_dollar_volume_50 > 2_500_000`
- `ATR10 / close > 0.04`

### System6

- `low >= 5`
- `avg_dollar_volume_50 > 10_000_000`

### System7

- フィルターなぁE

## シスチE��別セチE��アチE�E

### System1

- `SPY_close > SPY_SMA100`
- `SMA25 > SMA50`

### System2

- `RSI3 > 90`
- `close[-1] > close[-2]` かつ `close[-2] > close[-3]`

### System3

- `close > SMA150`
- `(close[-3] - close) / close[-3] <= -0.125`

### System4

- `SPX_close > SPX_SMA200`
- `close > SMA200`

### System5

- `close > SMA100 + ATR10`
- `ADX7 > 55`
- `RSI3 < 50`

### System6

- `(close / close[-6]) - 1 >= 0.20`
- `close[-1] > close[-2]` かつ `close[-2] > close[-3]`

### System7

- `SPY_low == rolling_min(SPY_low, window=50)`
